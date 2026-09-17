# SPDX-License-Identifier: Apache-2.0
"""Eight-rank ElasticBuffer Engram FP8 smoke test for Ascend 950DT.

Run with::

    torchrun --standalone --nproc-per-node=8 diagnose_elastic_buffer_engram_tp8.py

Every rank requests a different number of global rows. The test fetches local
and remote rows, checks FP8 payload and replicated E8M0 scale bits, validates
MXFP8 dequantization, and includes an idle rank in the collective.
"""

import argparse
import sys
from pathlib import Path

# This directory has a ``triton/`` folder. Avoid shadowing the Triton package
# when torch_npu imports torch._dynamo.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != _SCRIPT_DIR]

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch_npu  # noqa: E402


def _load_elastic_buffer():
    from cann_ops_transformer.ops.mc2.common import ElasticBuffer

    return ElasticBuffer


def _expected_data_bits(indices, entries_per_rank, hidden):
    owners = torch.div(indices, entries_per_rank, rounding_mode="floor")
    local_rows = indices.remainder(entries_per_rank)
    columns = torch.arange(hidden, dtype=torch.int64)
    return ((owners[:, None] * 17 + local_rows[:, None] * 3 + columns) % 64).to(torch.uint8)


def _expected_scale_bits(indices, groups):
    columns = torch.arange(groups, dtype=torch.int64)
    return (125 + (indices[:, None] + columns) % 5).to(torch.uint8)


def _expected_bf16(indices, entries_per_rank, hidden):
    owners = torch.div(indices, entries_per_rank, rounding_mode="floor")
    local_rows = indices.remainder(entries_per_rank)
    columns = torch.arange(hidden, dtype=torch.int64)
    return ((owners[:, None] * 17 + local_rows[:, None] * 3 + columns) % 64).to(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries-per-rank", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--storage-dtype", choices=("mxfp8", "bf16"), default="mxfp8")
    args = parser.parse_args()

    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.npu.set_device(rank)
    device = torch.device("npu", rank)
    if world_size != 8:
        raise RuntimeError(f"This probe requires TP8, got world_size={world_size}")
    if args.hidden % 32:
        raise RuntimeError("MXFP8 hidden size must be divisible by 32")

    ElasticBuffer = _load_elastic_buffer()
    entries = args.entries_per_rank
    hidden = args.hidden
    groups = hidden // 32

    local_rows = torch.arange(entries, dtype=torch.int64)
    columns = torch.arange(hidden, dtype=torch.int64)
    storage_values = rank * 17 + local_rows[:, None] * 3 + columns
    storage_values.remainder_(64)
    if args.storage_dtype == "mxfp8":
        storage = storage_values.to(torch.uint8).view(torch.float8_e4m3fn).contiguous()
    else:
        storage = storage_values.to(torch.bfloat16).contiguous()

    global_rows = torch.arange(entries * world_size, dtype=torch.int64)
    group_columns = torch.arange(groups, dtype=torch.int64)
    scale_bits = (125 + (global_rows[:, None] + group_columns) % 5).to(torch.uint8)
    scales = scale_bits.to(device).view(torch.float8_e8m0fnu).contiguous() if args.storage_dtype == "mxfp8" else None

    num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(entries, hidden, storage.dtype)
    buffer = ElasticBuffer(
        dist.group.WORLD,
        num_cpu_bytes=num_cpu_bytes,
        explicitly_destroy=True,
    )
    try:
        buffer.engram_write(storage, scales)

        query = (
            []
            if rank == world_size - 1
            else [
                rank * entries + 1,
                ((rank + 1) % world_size) * entries + 2,
                ((rank - 1) % world_size) * entries + 3,
            ]
        )
        if query:
            query.extend(((rank + offset + 2) % world_size) * entries + offset + 4 for offset in range(rank % 3))
        indices_cpu = torch.tensor(query, dtype=torch.int64)
        indices = indices_cpu.to(device=device, dtype=torch.int32)

        result = buffer.engram_fetch(indices)()
        torch.npu.synchronize()
        if args.storage_dtype == "bf16":
            expected = _expected_bf16(indices_cpu, entries, hidden)
            if not torch.equal(result.cpu(), expected):
                delta = (result.cpu().float() - expected.float()).abs().max()
                raise AssertionError(f"rank {rank}: BF16 payload mismatch, max_abs={delta}")
        else:
            fetched, fetched_scales = result
            expected_data = _expected_data_bits(indices_cpu, entries, hidden)
            expected_scales = _expected_scale_bits(indices_cpu, groups)
            actual_data = fetched.view(torch.uint8).cpu()
            actual_scales = fetched_scales.view(torch.uint8).cpu()
            if not torch.equal(actual_data, expected_data):
                mismatch = (actual_data != expected_data).nonzero()[0].tolist()
                raise AssertionError(f"rank {rank}: FP8 payload mismatch at {mismatch}")
            if not torch.equal(actual_scales, expected_scales):
                mismatch = (actual_scales != expected_scales).nonzero()[0].tolist()
                raise AssertionError(f"rank {rank}: E8M0 scale mismatch at {mismatch}")

            expected_values = expected_data.view(torch.float8_e4m3fn).float().unflatten(-1, (-1, 32))
            expected_powers = torch.pow(2.0, expected_scales.float() - 127.0).unsqueeze(-1)
            expected_dequantized = (expected_values * expected_powers).flatten(-2).bfloat16()
            if fetched.shape[0]:
                dequantized = torch_npu.npu_anti_mx_quant(
                    fetched,
                    fetched_scales.unflatten(-1, (-1, 2)),
                    axis=-1,
                    dst_type=torch.bfloat16,
                    src_type=torch.float8_e4m3fn,
                ).reshape(-1, hidden)
                if not torch.equal(dequantized.cpu(), expected_dequantized):
                    delta = (dequantized.cpu().float() - expected_dequantized.float()).abs().max()
                    raise AssertionError(f"rank {rank}: MXFP8 dequant mismatch, max_abs={delta}")

        second = torch.tensor([((rank + 4) % world_size) * entries], dtype=torch.int32, device=device)
        second_result = buffer.engram_fetch(second)()
        torch.npu.synchronize()
        second_cpu = second.cpu().to(torch.int64)
        if args.storage_dtype == "bf16":
            assert torch.equal(second_result.cpu(), _expected_bf16(second_cpu, entries, hidden))
        else:
            second_data, second_scales = second_result
            assert torch.equal(
                second_data.view(torch.uint8).cpu(),
                _expected_data_bits(second_cpu, entries, hidden),
            )
            assert torch.equal(
                second_scales.view(torch.uint8).cpu(),
                _expected_scale_bits(second_cpu, groups),
            )
        print(
            f"rank={rank} PASS dtype={args.storage_dtype} query_rows={len(query)} num_cpu_bytes={num_cpu_bytes}",
            flush=True,
        )
        dist.barrier()
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

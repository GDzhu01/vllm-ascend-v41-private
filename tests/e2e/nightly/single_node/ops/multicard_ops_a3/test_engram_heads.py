# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401


@torch.inference_mode()
def _worker(rank, world_size, rendezvous):
    # Worker-local setup avoids initializing NPU contexts in the parent.
    import tests.ut.conftest  # noqa: F401
    from vllm_ascend.models.deepseek_v41.engram_hbm import (
        EngramQueryGroup,
        NodeShardedEngram,
        dequantize_engram_rows,
        quantize_engram_rows,
    )

    torch.set_num_threads(1)
    torch.npu.set_device(rank)
    dist.init_process_group(
        "hccl", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=180)
    )
    cpu = dist.new_group(list(range(world_size)), backend="gloo")
    tp = dp = None
    for start in range(0, world_size, 8):
        ranks = list(range(start, start + 8))
        group = dist.new_group(ranks, backend="hccl")
        if rank in ranks:
            tp = group
    for index in range(8):
        ranks = list(range(index, world_size, 8))
        group = dist.new_group(ranks, backend="hccl")
        if rank in ranks:
            dp = group
    q = EngramQueryGroup(dist.group.WORLD, cpu, tp, rank // 8 * 8, dp)
    sizes = tuple(range(7, 31))
    references, tables = [], []
    for storage in ("bf16", "int8"):
        full = (torch.arange(sum(sizes) * 256).reshape(-1, 256).float() % 79 - 39).bfloat16()
        table = NodeShardedEngram(sum(sizes), 256, q, f"npu:{rank}", storage, sizes)
        table.set_rows(0, full[table.start : table.end])
        tables.append(table)
        references.append(dequantize_engram_rows(*quantize_engram_rows(full)) if storage == "int8" else full)
    for counts in ((129, 257), (2048, 513), (0, 3), (0, 0), (1, 1)):
        count = counts[q.dp_rank]
        hashes = torch.empty((count, len(tables), len(sizes)), dtype=torch.int64)
        start = 0
        for head, size in enumerate(sizes):
            for layer in range(len(tables)):
                hashes[:, layer, head] = start + (torch.arange(count) * 3 + head + q.dp_rank + layer) % size
            start += size
        ids = hashes.to(f"npu:{rank}")
        outputs = tables[0].route_heads(tables, ids, max(counts[: q.dp_size]))
        for layer, output in enumerate(outputs):
            torch.testing.assert_close(output.cpu(), references[layer][hashes[:, layer]], rtol=0, atol=0)
        if count == 1:
            graph = torch.npu.NPUGraph()
            persistent = [value.clone() for value in outputs]
            with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
                consumed = [value * 2 for value in persistent]
            for step in (1, 2):
                ids.copy_((hashes + step).to(ids.device))
                refreshed = tables[0].route_heads(tables, ids, 1)
                for fixed, value in zip(persistent, refreshed):
                    fixed.copy_(value)
                graph.replay()
                for actual, expected in zip(consumed, refreshed):
                    torch.testing.assert_close(actual.cpu(), (expected * 2).cpu(), rtol=0, atol=0)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [8, 16])
def test_hbm_engram_tp8_node_dp_and_decode_graph(world_size, tmp_path):
    if torch.npu.device_count() < world_size:
        pytest.skip(f"Requires {world_size} visible NPUs")
    mp.spawn(_worker, args=(world_size, f"file://{tmp_path / 'engram'}"), nprocs=world_size, join=True)

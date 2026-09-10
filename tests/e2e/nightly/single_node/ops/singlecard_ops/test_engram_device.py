# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.dsa_v41 import DeepseekV41Metadata
from vllm_ascend.models.deepseek_v41.engram_hash import DeviceNgramHistory, PagedNgramHistory, compute_hash_multipliers
from vllm_ascend.models.deepseek_v41.engram_hbm import NodeShardedEngram, dequantize_engram_rows, quantize_engram_rows
from vllm_ascend.ops.triton.engram import lookup_engram_heads, select_engram_rows

BLOCK_SIZE = 128


@pytest.mark.parametrize("block", [0, -1, 100000])
@torch.inference_mode()
def test_device_hash_masked_history_address(block):
    """Run in a fresh process with PYTORCH_NPU_ALLOC_CONF=expandable_segments:True.

    A large history allocation can begin at a mapped segment boundary. The
    graph warmup's inactive cache reads must not address the preceding page.
    """
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    reference = _history()
    state = DeviceNgramHistory(reference, device)
    state.cache = torch.full((2 * 1024 * 1024,), -1, dtype=torch.int32, device=device)
    ids = torch.zeros(192, dtype=torch.int32, device=device)
    positions = torch.full((192,), 127, dtype=torch.int64, device=device)
    metadata = SimpleNamespace(
        query_start_loc=torch.arange(0, 193, 6, dtype=torch.int32, device=device),
        block_table=torch.full((32, 1024), block, dtype=torch.int32, device=device),
        slot_mapping=torch.full((192, 2), -1, dtype=torch.int32, device=device),
        num_actual_reqs=32,
        num_actual_tokens=192,
        storage_block_size=BLOCK_SIZE,
    )
    actual, mask = state.update(ids, positions, metadata)
    rolling = torch.zeros(2, dtype=torch.int64)
    expected = []
    for shift in range(4):
        value = 0 if shift == 0 else reference.pad_id
        rolling ^= value * reference.multipliers[:, shift]
        if shift:
            expected.append(rolling[:, None] % reference.primes[:, shift - 1])
    expected = torch.cat(expected, dim=-1) + reference.offsets
    torch.testing.assert_close(actual.cpu(), expected.expand(192, -1, -1), rtol=0, atol=0)
    assert mask.cpu().all()
    assert (state.cache.cpu() == -1).all()


def _history():
    history = PagedNgramHistory.__new__(PagedNgramHistory)
    history.token_map = torch.arange(100, dtype=torch.int64)
    history.pad_id = 2
    history.image_token_id = 98
    history.image_pad_token_id = 99
    history.lookback = 4
    history.primes = torch.tensor(
        [
            [[101, 103, 107, 109], [113, 127, 131, 137], [139, 149, 151, 157]],
            [[163, 167, 173, 179], [181, 191, 193, 197], [199, 211, 223, 227]],
        ]
    )
    sizes = history.primes.flatten(1)
    history.offsets = sizes.cumsum(-1) - sizes
    history.multipliers = compute_hash_multipliers((1, 14), 4, 100)
    history.pages = {}
    return history


@torch.inference_mode()
def test_engram_dynamic_batches_reuse_jit_kernels():
    from vllm_ascend.ops.triton import engram as kernels

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    reference = _history()
    state = DeviceNgramHistory(reference, device)
    state.cache = torch.full((2 * 1024 * 1024,), -1, dtype=torch.int32, device=device)
    sizes = reference.primes[0].flatten().tolist()
    table = NodeShardedEngram(sum(sizes), 256, SimpleNamespace(size=1, head_shard_rank=0), device, "int8", sizes)
    rows = (torch.arange(sum(sizes) * 256).reshape(-1, 256).float() % 31 - 15).bfloat16()
    table.set_rows(0, rows)
    functions = (
        kernels._hash_kernel,
        kernels._write_history_kernel,
        kernels._lookup_heads_kernel,
        kernels._select_rows_kernel,
    )
    baseline = None
    for requests, length in ((1, 1), (2, 17), (3, 33), (7, 7), (8, 16), (16, 3), (32, 6), (1, 127)):
        tokens = requests * length
        # Shift addresses to exercise pointer-alignment specialization as well.
        offset = requests % 2
        ids = torch.zeros(tokens + offset, dtype=torch.int32, device=device)[offset:]
        positions = torch.full((tokens + offset,), 127, dtype=torch.int64, device=device)[offset:]
        metadata = SimpleNamespace(
            query_start_loc=torch.arange(0, tokens + 1, length, dtype=torch.int32, device=device),
            block_table=torch.zeros((requests, 1024), dtype=torch.int32, device=device),
            slot_mapping=torch.full((tokens, 2), -1, dtype=torch.int32, device=device),
            num_actual_reqs=requests,
            num_actual_tokens=tokens,
            storage_block_size=BLOCK_SIZE,
        )
        hashes, mask = state.update(ids, positions, metadata)
        looked_up = kernels.lookup_engram_heads(table, hashes[:, 0])
        # Change both the source token count and rank-local token offset.
        gathered = torch.cat((looked_up, looked_up), dim=0).unsqueeze(0)
        actual = kernels.select_engram_rows(gathered, tokens, tokens, len(sizes))
        torch.testing.assert_close(actual.cpu(), looked_up.cpu(), rtol=0, atol=0)
        assert mask.cpu().all()
        counts = tuple(len(fn.cache[torch.npu.current_device()]) for fn in functions)
        if baseline is None:
            baseline = counts
        else:
            assert counts == baseline, (requests, length, baseline, counts)


def _inputs(start, length, device):
    positions_cpu = torch.cat((torch.arange(start, start + length), torch.arange(start, start + length)))
    ids_cpu = (positions_cpu * 7 + torch.arange(2).repeat_interleave(length) * 11) % 97
    ids_cpu[positions_cpu % 131 == 33] = 98
    ids_cpu[positions_cpu % 131 == 34] = 99
    blocks_cpu = torch.stack((torch.arange(64) * 2 + 1, torch.arange(64) * 2 + 2)).int()
    requests = torch.arange(2).repeat_interleave(length)
    slots_cpu = torch.stack(
        (blocks_cpu[requests, positions_cpu // BLOCK_SIZE], positions_cpu % BLOCK_SIZE), dim=1
    ).int()
    metadata = DeepseekV41Metadata(
        query_start_loc=torch.tensor([0, length, 2 * length], device=device, dtype=torch.int32),
        block_table=blocks_cpu.to(device),
        slot_mapping=torch.cat((slots_cpu, torch.tensor([[-1, -1]], dtype=torch.int32))).to(device),
        seq_lens=torch.full((2,), start + length, device=device, dtype=torch.int32),
        storage_block_size=BLOCK_SIZE,
        compress_ratio=1,
        is_compressor_state=False,
        cache_kind="swa",
        num_actual_tokens=2 * length,
        num_input_tokens=2 * length + 1,
        num_reqs=2,
        num_actual_reqs=2,
    )
    ids = torch.cat((ids_cpu, torch.tensor([99]))).to(device)
    positions = torch.cat((positions_cpu, torch.tensor([0]))).to(device)
    return ids, positions, metadata, (ids_cpu, positions_cpu, requests, blocks_cpu, BLOCK_SIZE)


@pytest.mark.parametrize("length,wide_hash", [(1, False), (129, False), (2048, False), (129, True)])
@torch.inference_mode()
def test_device_hash_prefill_and_chunked_prefill(length, wide_hash):
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    reference = _history()
    if wide_hash:
        # Exercise nonidentity compression and bucket indices beyond exact FP32 integers.
        reference.token_map = reference.token_map * 13 % 100
        reference.primes = reference.primes.repeat(1, 1, 2) + (1 << 25)
        sizes = reference.primes.flatten(1)
        reference.offsets = sizes.cumsum(-1) - sizes
    state = DeviceNgramHistory(reference, device)
    kv = torch.empty((130, 1), device=device)
    assert state.ensure_cache(kv, BLOCK_SIZE)
    pointer = state.cache.data_ptr()
    start = 0
    for count in (length, 3, 1, 1):
        ids, positions, metadata, cpu = _inputs(start, count, device)
        expected, expected_mask = reference.update(*cpu)
        actual, mask = state.update(ids, positions, metadata)
        torch.testing.assert_close(actual[:-1].cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(mask[:-1].cpu(), expected_mask)
        assert (actual[-1].cpu() == -1).all()
        assert not mask[-1].cpu()
        assert state.cache.data_ptr() == pointer
        start += count


@torch.inference_mode()
def test_device_hash_decode_changed_input_graph_and_idle():
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    reference = _history()
    state = DeviceNgramHistory(reference, device)
    kv = torch.empty((130, 1), device=device)
    state.ensure_cache(kv, BLOCK_SIZE)
    ids, positions, metadata, cpu = _inputs(0, 127, device)
    reference.update(*cpu)
    state.update(ids, positions, metadata)
    ids, positions, metadata, _ = _inputs(127, 1, device)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        actual, mask = state.update(ids, positions, metadata)
    for start in (127, 128, 129):
        new_ids, new_positions, new_metadata, cpu = _inputs(start, 1, device)
        ids.copy_(new_ids)
        positions.copy_(new_positions)
        metadata.slot_mapping.copy_(new_metadata.slot_mapping)
        expected, expected_mask = reference.update(*cpu)
        graph.replay()
        torch.testing.assert_close(actual[:-1].cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(mask[:-1].cpu(), expected_mask)
    before = state.cache.clone()
    metadata.query_start_loc.zero_()
    metadata.slot_mapping.fill_(-1)
    graph.replay()
    torch.testing.assert_close(state.cache.cpu(), before.cpu(), rtol=0, atol=0)
    assert (actual.cpu() == -1).all()
    assert not mask.cpu().any()


@pytest.mark.parametrize("storage", ["bf16", "int8"])
@torch.inference_mode()
def test_head_lookup_and_fused_reorder(storage):
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    heads = (7, 11, 13, 17, 19, 23, 29)
    reference = (torch.arange(sum(heads) * 256).reshape(-1, 256).float() % 79 - 39).bfloat16()
    if storage == "int8":
        reference = dequantize_engram_rows(*quantize_engram_rows(reference))
    ids = torch.empty((259, len(heads)), dtype=torch.int64)
    start = 0
    for head, size in enumerate(heads):
        ids[:, head] = start + torch.arange(ids.shape[0]) % size
        start += size
    ids[-1] = -1
    ids_device = ids.to(device)
    parts = []
    for rank in range(4):
        table = NodeShardedEngram(
            sum(heads), 256, SimpleNamespace(size=4, head_shard_rank=rank), device, storage, heads
        )
        table.set_rows(0, reference[table.start : table.end])
        parts.append(lookup_engram_heads(table, ids_device))
    reordered = select_engram_rows(torch.stack(parts), ids.shape[0], 0, len(heads), table._head_gather_indices)
    expected = reference[ids.clamp_min(0)].masked_fill((ids < 0).unsqueeze(-1), 0)
    torch.testing.assert_close(reordered.cpu(), expected, rtol=0, atol=0)


@torch.inference_mode()
def test_model_hbm_prepare_and_persistent_decode_inputs():
    from vllm_ascend.models.deepseek_v41 import model as model_module

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    reference = _history()
    history = DeviceNgramHistory(reference, device)
    q = SimpleNamespace(size=1, head_shard_rank=0, tp_size=1, dp_size=1, tp_group=None, dp_group=None)
    tables, expected_tables = [], []
    for sizes in reference.primes.flatten(1).tolist():
        table = NodeShardedEngram(sum(sizes), 256, q, device, "bf16", sizes)
        full = (torch.arange(sum(sizes) * 256).reshape(-1, 256).float() % 79 - 39).bfloat16()
        table.set_rows(0, full)
        tables.append(table)
        expected_tables.append(full)
    cache = SimpleNamespace(prefix="swa", kv_cache=[torch.empty((130, 1), device=device)])
    layers = [SimpleNamespace(self_attn=SimpleNamespace(dsa_attn=SimpleNamespace(swa_cache_layer=cache)))]
    layers.extend(SimpleNamespace(engram=SimpleNamespace(embed=table)) for table in tables)
    model = SimpleNamespace(
        config=SimpleNamespace(engram_layer_ids=(1, 2), engram_max_ngram_size=4, engram_n_heads=4),
        layers=layers,
        engram_history=history,
        _engram_max_tokens=512,
        _engram_input_buffers=None,
    )
    model.prepare_engram = MethodType(model_module.DeepseekV41Model._prepare_engram_hbm, model)
    prepare = MethodType(model_module.DeepseekV41Model.prepare_engram_inputs, model)
    context = SimpleNamespace(attn_metadata=None, dp_metadata=None)
    pointers = None
    graph = None
    for start, count in ((0, 129), (129, 3), (132, 1), (133, 1)):
        ids, positions, metadata, cpu = _inputs(start, count, device)
        expected_hashes, expected_mask = reference.update(*cpu)
        context.attn_metadata = {"swa": metadata}
        with (
            patch.object(model_module, "get_forward_context", return_value=context),
            patch.object(torch.Tensor, "cpu", side_effect=AssertionError("Engram D2H copy")),
            patch.object(torch.Tensor, "item", side_effect=AssertionError("Engram scalar synchronization")),
            patch.object(torch.Tensor, "tolist", side_effect=AssertionError("Engram host materialization")),
        ):
            prepared = prepare(ids, positions)
        fixed = prepared["engram_lookups"]
        mask = prepared["engram_mask"]
        current_pointers = [mask.data_ptr(), *(value.data_ptr() for value in fixed.values())]
        if pointers is not None:
            assert pointers == current_pointers
        pointers = current_pointers
        for slot, value in enumerate(fixed.values()):
            expected = expected_tables[slot][expected_hashes[:, slot]].flatten(1)
            torch.testing.assert_close(value[: 2 * count].cpu(), expected, rtol=0, atol=0)
            assert not value[2 * count :].cpu().any()
        torch.testing.assert_close(mask[: 2 * count].cpu(), expected_mask)
        assert not mask[2 * count :].cpu().any()
        if count == 1:
            if graph is None:
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
                    consumed = fixed[1][:2] * mask[:2, None]
            graph.replay()
            torch.testing.assert_close(consumed.cpu(), (fixed[1][:2] * mask[:2, None]).cpu(), rtol=0, atol=0)
    context.attn_metadata = None
    with patch.object(model_module, "get_forward_context", return_value=context):
        prepared = prepare(ids, positions)
    assert not prepared["engram_mask"].cpu().any()
    assert all(not value.cpu().any() for value in prepared["engram_lookups"].values())

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import triton

from vllm_ascend.model_executor.warmup.deepseek_v41_triton_warmup import deepseek_v41_triton_warmup
from vllm_ascend.ops.triton.prepare_indexer_indices import prepare_indexer_indices
from vllm_ascend.ops.triton.quantize_indexer_query import quantize_indexer_query
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


def reference(selected, positions, compress_ratio):
    visible = ((positions + 1) // compress_ratio).unsqueeze(-1)
    valid = (selected >= 0) & (selected < visible)
    sentinel = torch.iinfo(torch.int32).max
    selected = torch.where(valid, selected, sentinel).sort(dim=-1).values
    return torch.where(selected == sentinel, -1, selected)


@pytest.mark.parametrize("topk", [1, 7, 8, 33, 128, 512, 2047, 2048])
@pytest.mark.parametrize("tokens", [0, 1, 3, 41, 129])
@pytest.mark.parametrize("compress_ratio", [1, 2])
@torch.inference_mode()
def test_prepare_indexer_indices(topk, tokens, compress_ratio):
    torch.manual_seed(41)
    selected = torch.randint(-3, 1000, (tokens, topk), dtype=torch.int32, device="npu")
    positions = torch.randint(-1, 2000, (tokens,), dtype=torch.int64, device="npu")
    original = selected.clone()
    actual = prepare_indexer_indices(selected, positions, compress_ratio)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual.cpu(), reference(selected.cpu(), positions.cpu(), compress_ratio), rtol=0, atol=0)
    torch.testing.assert_close(selected, original, rtol=0, atol=0)


@pytest.mark.parametrize("compress_ratio", [1, 2])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_prepare_indexer_indices_boundaries(compress_ratio, position_dtype):
    # Preserve distinct INT32 indices above FP32's exact-integer range, ties,
    # negative sentinels, causal boundaries and all-invalid rows.
    row = torch.tensor([0, -1, -3, 9, 9, 10, 11, 2**24 - 1, 2**24, 2**24 + 1, 2**24 + 2, 2**31 - 2, 2**31 - 1])
    selected = row.int().repeat(5, 1).npu()
    positions = torch.tensor([-1, 0, 19, 2**25 + 3, 2**31 - 1], dtype=position_dtype, device="npu")
    actual = prepare_indexer_indices(selected, positions, compress_ratio)
    torch.testing.assert_close(actual.cpu(), reference(selected.cpu(), positions.cpu(), compress_ratio), rtol=0, atol=0)


@torch.inference_mode()
def test_prepare_indexer_indices_full_int32_range():
    torch.manual_seed(42)
    selected = torch.randint(0, 2**31 - 1, (41, 2048), dtype=torch.int32, device="npu")
    positions = torch.full((41,), 2**32, dtype=torch.int64, device="npu")
    actual = prepare_indexer_indices(selected, positions, 2)
    torch.testing.assert_close(actual.cpu(), reference(selected.cpu(), positions.cpu(), 2), rtol=0, atol=0)


@torch.inference_mode()
def test_prepare_indexer_indices_noncontiguous():
    selected = torch.randint(-1, 2000, (41, 256), dtype=torch.int32, device="npu")[:, ::2]
    positions = torch.arange(82, dtype=torch.int64, device="npu")[::2]
    actual = prepare_indexer_indices(selected, positions, 2)
    torch.testing.assert_close(actual.cpu(), reference(selected.cpu(), positions.cpu(), 2), rtol=0, atol=0)


@pytest.mark.parametrize("compress_ratio", [1, 2])
@pytest.mark.parametrize("tokens", [41, 129])
@torch.inference_mode()
def test_prepare_indexer_indices_graph_replay(compress_ratio, tokens):
    selected = torch.randint(-1, 4096, (tokens, 2048), dtype=torch.int32, device="npu")
    positions = torch.full((tokens,), 4095, dtype=torch.int64, device="npu")
    prepare_indexer_indices(selected, positions, compress_ratio)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        actual = prepare_indexer_indices(selected, positions, compress_ratio)
    pointer = actual.data_ptr()
    for last_position in (0, 127, 8191):
        selected.copy_(torch.randint_like(selected, -1, 4096))
        positions.fill_(last_position)
        graph.replay()
        torch.npu.synchronize()
        assert actual.data_ptr() == pointer
        torch.testing.assert_close(
            actual.cpu(), reference(selected.cpu(), positions.cpu(), compress_ratio), rtol=0, atol=0
        )


@torch.inference_mode()
def test_indexer_warmup_prevents_shape_recompilation(monkeypatch):
    config = SimpleNamespace(
        model_type="deepseek_v41_text",
        num_hidden_layers=2,
        compress_ratios=[1, 2],
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2048,
    )
    worker = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config, dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        device=torch.device("npu"),
    )
    deepseek_v41_triton_warmup(worker)
    torch.npu.synchronize()

    def reject_new_compilation(*args, **kwargs):
        raise AssertionError("Indexer token count created a new Triton specialization after startup warmup")

    monkeypatch.setattr(triton.JITFunction, "cache_hook", reject_new_compilation)
    cores = get_vectorcore_num()
    tokens_to_check = {
        1,
        2,
        15,
        16,
        17,
        cores,
        cores + 1,
        2 * cores,
        2 * cores + 1,
        129,
        767,
        768,
        769,
        1023,
        1024,
        4096,
    }
    for tokens in sorted(tokens_to_check):
        for heads in (32, 64):
            query = torch.randn(tokens, heads, 128, dtype=torch.bfloat16, device="npu")
            actual, scale = quantize_indexer_query(query)
            expected_scale = (query.float().abs().amax(-1) / 127.0).half().clamp_min_(2.0**-24)
            expected = (query.float() / expected_scale.float().unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
        selected = torch.randint(-3, 8192, (tokens, config.index_topk), dtype=torch.int32, device="npu")
        positions = torch.randint(-1, 16384, (tokens,), dtype=torch.int64, device="npu")
        for ratio in (1, 2):
            actual = prepare_indexer_indices(selected, positions, ratio)
            torch.testing.assert_close(actual.cpu(), reference(selected.cpu(), positions.cpu(), ratio), rtol=0, atol=0)

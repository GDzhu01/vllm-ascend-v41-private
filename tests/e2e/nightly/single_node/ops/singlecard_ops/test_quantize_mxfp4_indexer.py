# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.dsv41_a5.quantization import (
    _mxfp4_quantize_e8m0_reference,
)
from vllm_ascend.ops.triton.quantize_mxfp4_indexer import (
    quantize_mxfp4_indexer,
)


def assert_matches_reference(query: torch.Tensor) -> None:
    expected = _mxfp4_quantize_e8m0_reference(query)
    actual = quantize_mxfp4_indexer(query)
    for output, golden in zip(actual, expected):
        assert output.is_contiguous()
        torch.testing.assert_close(output.cpu(), golden.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [0, 1, 3, 129, 4080])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_quantize_mxfp4_indexer_random(tokens, dtype):
    torch.manual_seed(41)
    query = torch.randn(tokens, 32, 128, dtype=dtype, device="npu")
    original = query.clone()
    assert_matches_reference(query)
    torch.testing.assert_close(query, original, rtol=0, atol=0)


@torch.inference_mode()
def test_quantize_mxfp4_indexer_rounding_scale_and_signed_zero():
    query = torch.zeros(1, 32, 128, dtype=torch.float32)
    boundaries = torch.tensor(
        [
            0.0,
            -0.0,
            0.25,
            -0.25,
            0.2501,
            -0.2501,
            0.75,
            -0.75,
            1.25,
            -1.25,
            1.75,
            -1.75,
            2.5,
            -2.5,
            3.5,
            -3.5,
            5.0,
            -5.0,
            6.0,
            -6.0,
        ]
    )
    query[0, 0, : boundaries.numel()] = boundaries
    query[0, 0, 31] = 6.0  # Force scale=1 for the first group.
    query[0, 1, :4] = torch.tensor([-0.01, 0.01, -0.25, 0.25])
    query[0, 2] = query[0, 0] * 2.0**-100
    query[0, 3] = query[0, 0] * 2.0**100
    assert_matches_reference(query.npu())


@torch.inference_mode()
def test_quantize_mxfp4_indexer_noncontiguous():
    query = torch.randn(3, 32, 256, dtype=torch.bfloat16, device="npu")[..., ::2]
    assert not query.is_contiguous()
    assert_matches_reference(query)


@pytest.mark.parametrize("tokens", [3, 129])
@torch.inference_mode()
def test_quantize_mxfp4_indexer_graph_replay(tokens):
    query = torch.randn(tokens, 32, 128, dtype=torch.bfloat16, device="npu")
    quantize_mxfp4_indexer(query)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        actual = quantize_mxfp4_indexer(query)
    pointers = tuple(output.data_ptr() for output in actual)
    for magnitude in (2.0, 1.0e-6, 0.0):
        query.copy_(torch.randn_like(query) * magnitude)
        expected = _mxfp4_quantize_e8m0_reference(query)
        graph.replay()
        torch.npu.synchronize()
        assert tuple(output.data_ptr() for output in actual) == pointers
        for output, golden in zip(actual, expected):
            torch.testing.assert_close(output.cpu(), golden.cpu(), rtol=0, atol=0)

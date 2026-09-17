# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.dsv41_a5.rotary import (
    apply_partial_rotary,
    apply_partial_rotary_inplace,
)


@pytest.mark.parametrize(
    "tokens,heads,width,rotary_width", [(1, 64, 512, 64), (6, 64, 512, 64), (129, 1, 512, 64), (4080, 1, 128, 64)]
)
@pytest.mark.parametrize("inverse", [False, True])
@torch.inference_mode()
def test_inplace_partial_rotary_matches_reference(tokens, heads, width, rotary_width, inverse):
    torch.manual_seed(41)
    source = torch.randn(tokens, heads, width, dtype=torch.bfloat16, device="npu")
    cos = torch.randn(tokens, 1, 1, rotary_width, dtype=torch.bfloat16, device="npu")
    sin = torch.randn_like(cos)
    start = width - rotary_width
    expected = apply_partial_rotary(source, cos, sin, start=start, end=width, inverse=inverse)
    actual = source.clone()

    returned = apply_partial_rotary_inplace(actual, cos, sin, start=start, end=width, inverse=inverse)

    assert returned is actual
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@torch.inference_mode()
def test_inplace_partial_rotary_graph_replay():
    source = torch.randn(6, 64, 512, dtype=torch.bfloat16, device="npu")
    cos = torch.randn(6, 1, 1, 64, dtype=torch.bfloat16, device="npu")
    sin = torch.randn_like(cos)
    actual = source.clone()
    apply_partial_rotary_inplace(actual, cos, sin, start=448, end=512)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        apply_partial_rotary_inplace(actual, cos, sin, start=448, end=512)
    expected = apply_partial_rotary(actual, cos, sin, start=448, end=512)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)

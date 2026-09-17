# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.build_window_indices import (
    build_window_indices_triton,
)


def reference(positions, window_size):
    offsets = torch.arange(window_size, device=positions.device)
    lengths = torch.minimum(
        positions + 1,
        positions.new_full(positions.shape, window_size),
    ).to(torch.int32)
    starts = positions + 1 - lengths
    indices = starts[:, None] + offsets[None, :]
    indices = torch.where(offsets[None, :] < lengths[:, None], indices, -1)
    return indices[:, None, :].to(torch.int32), lengths[:, None]


@pytest.mark.parametrize("tokens", [0, 1, 6, 129, 4080])
@pytest.mark.parametrize("window_size", [1, 31, 128, 257])
@torch.inference_mode()
def test_build_window_indices_triton_matches_reference(tokens, window_size):
    positions = torch.arange(tokens, dtype=torch.int64, device="npu") * 7
    expected = reference(positions, window_size)
    actual = build_window_indices_triton(positions, window_size)
    for output, golden in zip(actual, expected):
        torch.testing.assert_close(output.cpu(), golden.cpu(), rtol=0, atol=0)


@torch.inference_mode()
def test_build_window_indices_triton_reuses_output_and_graph():
    positions = torch.arange(129, dtype=torch.int64, device="npu")
    indices = torch.empty(129, 1, 128, dtype=torch.int32, device="npu")
    lengths = torch.empty(129, 1, dtype=torch.int32, device="npu")
    build_window_indices_triton(
        positions,
        128,
        indices_output=indices,
        lengths_output=lengths,
    )
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        actual = build_window_indices_triton(
            positions,
            128,
            indices_output=indices,
            lengths_output=lengths,
        )
    positions.add_(1000)
    expected = reference(positions, 128)
    graph.replay()
    torch.npu.synchronize()
    assert actual[0].data_ptr() == indices.data_ptr()
    assert actual[1].data_ptr() == lengths.data_ptr()
    for output, golden in zip(actual, expected):
        torch.testing.assert_close(output.cpu(), golden.cpu(), rtol=0, atol=0)

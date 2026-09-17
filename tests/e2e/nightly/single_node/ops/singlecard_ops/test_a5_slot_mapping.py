# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.a5_slot_mapping import build_a5_slot_mapping


def reference(slots, positions, query_start_loc, num_actual_reqs, num_actual_tokens, page_size, ratio, skip):
    active = slots.clone()
    valid = active >= 0
    if ratio == 2:
        valid &= (active + 1).remainder(2) == 0
        active = torch.div(active, 2, rounding_mode="floor")
        valid_end = min(int(query_start_loc[num_actual_reqs]), num_actual_tokens)
        valid &= torch.arange(slots.shape[0]) < valid_end
        valid &= positions.remainder(2) == 1
        if skip:
            valid.zero_()
    safe = active.clamp_min(0)
    coordinates = torch.stack((safe // page_size, safe % page_size), dim=-1).int()
    coordinates[~valid] = -1
    return coordinates, torch.where(valid, active, -1).int()


@pytest.mark.parametrize("ratio,page_size", [(1, 128), (2, 64)])
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 17, 257])
@torch.inference_mode()
def test_a5_slot_mapping(ratio, page_size, skip, num_tokens):
    source = torch.arange(num_tokens * 2, dtype=torch.int64)[::2]
    positions = torch.arange(num_tokens * 2, dtype=torch.int64)[::2]
    if ratio == 2:
        source += 1
        positions += 1
    source[::7] = -1
    query_start_loc = torch.tensor([0, max(0, num_tokens - 3)], dtype=torch.int32)
    expected_coordinates, expected_flat = reference(
        source,
        positions,
        query_start_loc,
        1,
        num_tokens - 2,
        page_size,
        ratio,
        skip,
    )
    coordinates = torch.empty((num_tokens, 2), dtype=torch.int32, device="npu")
    flat = torch.empty((num_tokens,), dtype=torch.int32, device="npu")
    actual_coordinates, actual_flat = build_a5_slot_mapping(
        source.npu(),
        positions.npu(),
        query_start_loc.npu(),
        num_tokens,
        1,
        num_tokens - 2,
        page_size,
        ratio,
        skip_update=skip,
        coordinates_output=coordinates,
        flat_output=flat,
    )
    assert actual_coordinates.data_ptr() == coordinates.data_ptr()
    assert actual_flat.data_ptr() == flat.data_ptr()
    torch.testing.assert_close(actual_coordinates.cpu(), expected_coordinates, rtol=0, atol=0)
    torch.testing.assert_close(actual_flat.cpu(), expected_flat, rtol=0, atol=0)


@torch.inference_mode()
def test_a5_slot_mapping_graph_replay():
    num_tokens = 288
    slots = torch.arange(num_tokens, dtype=torch.int64, device="npu")
    positions = torch.arange(num_tokens, dtype=torch.int64, device="npu")
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device="npu")
    coordinates = torch.empty((num_tokens, 2), dtype=torch.int32, device="npu")
    flat = torch.empty((num_tokens,), dtype=torch.int32, device="npu")
    args = (slots, positions, query_start_loc, num_tokens, 1, num_tokens, 128, 1)
    build_a5_slot_mapping(*args, coordinates_output=coordinates, flat_output=flat)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        actual_coordinates, actual_flat = build_a5_slot_mapping(
            *args,
            coordinates_output=coordinates,
            flat_output=flat,
        )
    for invalid in (0, 37, 287):
        slots.copy_(torch.arange(num_tokens, dtype=torch.int64, device="npu"))
        slots[invalid] = -1
        graph.replay()
        torch.npu.synchronize()
        expected_coordinates, expected_flat = reference(
            slots.cpu(),
            positions.cpu(),
            query_start_loc.cpu(),
            1,
            num_tokens,
            128,
            1,
            False,
        )
        torch.testing.assert_close(actual_coordinates.cpu(), expected_coordinates, rtol=0, atol=0)
        torch.testing.assert_close(actual_flat.cpu(), expected_flat, rtol=0, atol=0)

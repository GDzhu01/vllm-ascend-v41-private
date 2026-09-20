# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.c2_ring_metadata import build_c2_ring_metadata


@pytest.mark.parametrize("skip_update", [False, True])
@torch.inference_mode()
def test_c2_ring_metadata_matches_reference(skip_update):
    num_reqs = 5
    num_tokens = 12
    num_actual_reqs = 3
    num_actual_tokens = 7
    query_start_loc = torch.tensor([0, 3, 5, 9, 9, 9], dtype=torch.int32)
    seq_lens = torch.tensor([10, 20, 30, 0, 0], dtype=torch.int32)
    positions = torch.tensor(
        [7, 8, 9, 18, 19, 26, 27, 28, 29, 0, 0, 0],
        dtype=torch.int64,
    )
    block_table = torch.tensor(
        [[7, 8], [11, 12], [19, 20], [0, 0], [0, 0]],
        dtype=torch.int32,
    )
    rope_dim = 64
    source_cos = torch.arange(64 * rope_dim, dtype=torch.float32).view(
        64, 1, 1, rope_dim
    )
    source_sin = -source_cos

    ring = torch.empty((5, num_reqs), dtype=torch.int32, device="npu")
    complete = torch.empty((num_tokens,), dtype=torch.bool, device="npu")
    source_positions = torch.empty((num_tokens,), dtype=torch.int64, device="npu")
    cos = torch.empty(
        (num_tokens, 1, 1, rope_dim), dtype=torch.float32, device="npu"
    )
    sin = torch.empty_like(cos)
    build_c2_ring_metadata(
        query_start_loc.npu(),
        seq_lens.npu(),
        positions.npu(),
        block_table.npu(),
        source_cos.npu(),
        source_sin.npu(),
        num_reqs,
        num_tokens,
        num_actual_reqs,
        num_actual_tokens,
        skip_update=skip_update,
        ring_metadata_output=ring,
        complete_mask_output=complete,
        source_positions_output=source_positions,
        cos_output=cos,
        sin_output=sin,
    )

    starts = query_start_loc[:-1]
    ends = query_start_loc[1:]
    used = (ends.clamp_max(num_actual_tokens) - starts).clamp_min(0)
    used[torch.arange(num_reqs) >= num_actual_reqs] = 0
    if skip_update:
        used.zero_()
    expected_ring = torch.stack(
        (
            (seq_lens - (ends - starts)).clamp_min(0),
            used,
            starts,
            starts,
            torch.where(used > 0, block_table[:, 0], 0),
        )
    )
    valid_end = min(int(query_start_loc[num_actual_reqs]), num_actual_tokens)
    expected_complete = (
        (torch.arange(num_tokens) < valid_end)
        & (positions.remainder(2) == 1)
        & (not skip_update)
    )
    expected_source_positions = torch.where(
        expected_complete,
        positions - 1,
        0,
    )
    expected_cos = source_cos[expected_source_positions]
    expected_sin = source_sin[expected_source_positions]

    torch.testing.assert_close(ring.cpu(), expected_ring, rtol=0, atol=0)
    torch.testing.assert_close(
        complete.cpu(), expected_complete, rtol=0, atol=0
    )
    torch.testing.assert_close(
        source_positions.cpu(), expected_source_positions, rtol=0, atol=0
    )
    torch.testing.assert_close(cos.cpu(), expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(sin.cpu(), expected_sin, rtol=0, atol=0)

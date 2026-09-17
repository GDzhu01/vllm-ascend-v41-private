# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.triton.spec_decode.dspark_swa_indices import (
    build_dspark_swa_indices_triton,
)


def _reference(query_start_loc, seq_lens, num_tokens, index_width, window_size):
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    prefix_lens = seq_lens - query_lens
    start_pos = (prefix_lens - window_size).clamp(min=0)
    visible_lens = seq_lens - start_pos
    cols = torch.arange(index_width, device=seq_lens.device)
    slots = (start_pos[:, None] + cols[None, :]).where(
        cols[None, :] < visible_lens[:, None],
        -1,
    )
    indices = torch.repeat_interleave(
        slots,
        query_lens,
        dim=0,
        output_size=num_tokens,
    ).unsqueeze(1)
    lengths = (
        torch.repeat_interleave(
            visible_lens,
            query_lens,
            dim=0,
            output_size=num_tokens,
        )
        .to(torch.int32)
        .unsqueeze(1)
    )
    return indices.to(torch.int32), lengths


@pytest.mark.parametrize("query_lens", [[1], [6, 6, 6], [1, 3, 6, 2]])
@pytest.mark.parametrize("window_size", [16, 4096])
def test_build_dspark_swa_indices(query_lens, window_size):
    query_lens = torch.tensor(query_lens, dtype=torch.int32, device="npu")
    query_start_loc = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="npu"),
            query_lens.cumsum(dim=0),
        ]
    )
    seq_lens = query_lens + torch.arange(query_lens.shape[0], dtype=torch.int32, device="npu") * (window_size + 7)
    num_tokens = int(query_lens.sum().item())
    index_width = ((window_size + 6 + 127) // 128) * 128

    expected = _reference(
        query_start_loc,
        seq_lens,
        num_tokens,
        index_width,
        window_size,
    )
    actual = build_dspark_swa_indices_triton(
        query_start_loc,
        seq_lens,
        num_tokens,
        index_width,
        window_size,
    )
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])

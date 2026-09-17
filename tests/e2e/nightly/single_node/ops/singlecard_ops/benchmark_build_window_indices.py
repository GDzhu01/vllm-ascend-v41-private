# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone NPU A/B benchmark for causal-window metadata construction."""

from __future__ import annotations

import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.build_window_indices import build_window_indices_triton


def eager_reference(positions, window_size):
    offsets = torch.arange(window_size, device=positions.device)
    lengths = torch.minimum(
        positions + 1,
        positions.new_full(positions.shape, window_size),
    ).to(torch.int32)
    starts = positions + 1 - lengths
    indices = starts[:, None] + offsets[None, :]
    indices = torch.where(offsets[None, :] < lengths[:, None], indices, -1)
    return indices[:, None, :].to(torch.int32).contiguous(), lengths[:, None].contiguous()


def measure(function, positions, iterations):
    for _ in range(3):
        function(positions, 128)
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        function(positions, 128)
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1000 / iterations


def main():
    for tokens, iterations in ((1, 200), (6, 100), (32, 30), (4080, 3)):
        positions = torch.arange(tokens, dtype=torch.int64, device="npu")
        reference_ms = measure(eager_reference, positions, iterations)
        triton_ms = measure(build_window_indices_triton, positions, iterations)
        print(
            f"tokens={tokens} reference_ms={reference_ms:.6f} "
            f"triton_ms={triton_ms:.6f} speedup={reference_ms / triton_ms:.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B benchmark for full-row reconstruction versus partial in-place copy."""

from __future__ import annotations

import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.dsv41_a5.rotary import (
    apply_partial_rotary,
    apply_partial_rotary_inplace,
)


def measure(fn, inputs, iterations):
    for _ in range(3):
        fn(*inputs, start=448, end=512)
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        fn(*inputs, start=448, end=512)
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1000 / iterations


def main():
    for tokens, heads, iterations in ((1, 64, 200), (6, 64, 100), (4080, 1, 10), (4080, 64, 3)):
        x = torch.randn(tokens, heads, 512, dtype=torch.bfloat16, device="npu")
        cos = torch.randn(tokens, 1, 1, 64, dtype=torch.bfloat16, device="npu")
        sin = torch.randn_like(cos)
        inputs = (x, cos, sin)
        reference_ms = measure(apply_partial_rotary, inputs, iterations)
        inplace_ms = measure(apply_partial_rotary_inplace, inputs, iterations)
        print(
            f"tokens={tokens} heads={heads} reference_ms={reference_ms:.6f} "
            f"inplace_ms={inplace_ms:.6f} speedup={reference_ms / inplace_ms:.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()

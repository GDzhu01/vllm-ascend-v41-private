# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B benchmark for fused A5 index-cache quantization and storage."""

from __future__ import annotations

import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.dsv41_a5.writers import write_index_cache


def measure(backend, inputs, iterations):
    for _ in range(3):
        write_index_cache(*inputs, backend=backend)
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        write_index_cache(*inputs, backend=backend)
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1000 / iterations


def main():
    for tokens, iterations in ((1, 200), (6, 100), (32, 30), (4080, 3)):
        values = torch.randn(tokens, 128, dtype=torch.bfloat16, device="npu")
        slots = torch.stack(
            (
                torch.arange(tokens, dtype=torch.int32, device="npu") // 128,
                torch.arange(tokens, dtype=torch.int32, device="npu") % 128,
            ),
            -1,
        )
        pages = max(1, (tokens + 127) // 128)
        inputs = (
            (
                torch.empty(pages, 128, 1, 64, dtype=torch.uint8, device="npu"),
                torch.empty(pages, 128, 1, 4, dtype=torch.uint8, device="npu"),
            ),
            slots,
            values,
        )
        reference_ms = measure("reference", inputs, iterations)
        triton_ms = measure("triton", inputs, iterations)
        print(
            f"tokens={tokens} reference_ms={reference_ms:.6f} "
            f"triton_ms={triton_ms:.6f} speedup={reference_ms / triton_ms:.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()

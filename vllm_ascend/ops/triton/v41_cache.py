# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Masked stores into V4.1's block-strided hybrid cache allocation."""
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@triton.jit
def _scatter_rows(
    cache, slots, values,
    PAGE_SIZE: tl.constexpr, PAGE_STRIDE: tl.constexpr, ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr, VALUE_STRIDE: tl.constexpr, VALUE_COL_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr, BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slots + token)
    col = tl.arange(0, BLOCK)
    value = tl.load(values + token * VALUE_STRIDE + col * VALUE_COL_STRIDE, col < WIDTH, 0)
    page = tl.maximum(slot, 0) // PAGE_SIZE
    row = tl.maximum(slot, 0) % PAGE_SIZE
    tl.store(cache + page * PAGE_STRIDE + row * ROW_STRIDE + col * COL_STRIDE,
             value, (slot >= 0) & (col < WIDTH))


def scatter_cache_rows(cache, slots, values):
    """Skip negative slots without host sync or variable-size indexing."""
    init_device_properties_triton()
    cache = cache.squeeze(-2)
    if values.shape[0] == 0:
        return
    _scatter_rows[(values.shape[0],)](
        cache, slots, values,
        PAGE_SIZE=cache.shape[1], PAGE_STRIDE=cache.stride(0), ROW_STRIDE=cache.stride(1),
        COL_STRIDE=cache.stride(2), VALUE_STRIDE=values.stride(0), VALUE_COL_STRIDE=values.stride(1),
        WIDTH=values.shape[1], BLOCK=triton.next_power_of_2(values.shape[1]), num_warps=4,
    )

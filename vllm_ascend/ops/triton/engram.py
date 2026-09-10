# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-resident Engram hashing and head-sharded HBM lookup.

Uses the slot-history and fixed head ownership approach of Inferact/sra
sra-tracking (Engram #49, #67, #74), with Ascend physical [block, offset] slots.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit(
    do_not_specialize=["num_tokens", "num_actual_tokens", "num_reqs", "num_slots", "vocab_size", "search_steps"],
    do_not_specialize_on_alignment=["ids", "positions", "query_start", "blocks"],
)
def _hash_kernel(
    ids,
    positions,
    token_map,
    query_start,
    blocks,
    cache,
    multipliers,
    primes,
    offsets,
    hashes,
    keep,
    num_tokens,
    num_actual_tokens,
    num_reqs,
    num_slots,
    vocab_size,
    ids_stride: tl.constexpr,
    pos_stride: tl.constexpr,
    block_stride: tl.constexpr,
    block_columns: tl.constexpr,
    cache_block_size: tl.constexpr,
    pad_id: tl.constexpr,
    image_id: tl.constexpr,
    image_pad_id: tl.constexpr,
    LOOKBACK: tl.constexpr,
    HEADS: tl.constexpr,
    LAYERS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    search_steps,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    layer = tl.program_id(1)
    total_query = tl.load(query_start + num_reqs)
    valid = (token < num_tokens) & (token < num_actual_tokens) & (token < total_query)
    lo = tl.full((BLOCK_T,), 0, tl.int32)
    hi = tl.full((BLOCK_T,), num_reqs, tl.int32)
    for _ in range(search_steps):
        mid = (lo + hi) // 2
        end = tl.load(query_start + mid + 1, lo < hi, other=0)
        right = token >= end
        active = lo < hi
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    request = tl.minimum(lo, num_reqs - 1).to(tl.int64)
    chunk_index = tl.load(query_start + request)
    chunk_start = tl.load(positions + chunk_index * pos_stride, valid, other=0).to(tl.int64)
    position = tl.load(positions + token * pos_stride, valid, other=0).to(tl.int64)
    current_id = tl.load(ids + token * ids_stride, valid, other=-1)
    alive = valid & (current_id >= 0) & (current_id < vocab_size)
    alive &= (current_id != image_id) & (current_id != image_pad_id)
    if layer == 0:
        tl.store(keep + token, alive, token < num_tokens)
    head = tl.arange(0, BLOCK_H)
    blocked = tl.full((BLOCK_T,), False, tl.int1)
    rolling = tl.full((BLOCK_T,), 0, tl.int64)
    for shift in tl.static_range(LOOKBACK):
        previous = position - shift
        in_batch = valid & (previous >= chunk_start) & (token >= shift)
        source_index = tl.where(in_batch, token - shift, 0)
        source_id = tl.load(ids + source_index * ids_stride, in_batch, other=-1)
        source_valid = in_batch & (source_id >= 0) & (source_id < vocab_size)
        mapped = tl.load(token_map + tl.where(source_valid, source_id, 0), source_valid, other=-1)
        mapped = tl.where((source_id == image_id) | (source_id == image_pad_id), -1, mapped)
        column = previous // cache_block_size
        from_cache = valid & ~in_batch & (previous >= 0) & (column < block_columns)
        block_index = tl.where(from_cache, request * block_stride + column, 0)
        block = tl.load(blocks + block_index, from_cache, other=-1).to(tl.int64)
        slot = block * cache_block_size + previous % cache_block_size
        cache_valid = from_cache & (slot >= 0) & (slot < num_slots)
        # Ascend may form scalar GM addresses even for masked lanes. Keep
        # inactive addresses inside the allocation, including when expandable
        # segments leave the page preceding the history cache unmapped.
        safe_slot = tl.where(cache_valid, slot, 0)
        cached = tl.load(cache + safe_slot, cache_valid, other=-1)
        source = tl.where(in_batch, mapped, cached).to(tl.int64)
        blocked |= (previous < 0) | (source < 0)
        value = tl.where(blocked, pad_id, source)
        multiplier = tl.load(multipliers + layer * LOOKBACK + shift)
        rolling ^= value * multiplier
        if shift > 0:
            column = (shift - 1) * HEADS + head
            parameter = layer * (LOOKBACK - 1) * HEADS + column
            prime = tl.load(primes + parameter, head < HEADS, other=1)
            offset = tl.load(offsets + parameter, head < HEADS, other=0)
            hashed = rolling[:, None] % prime[None, :] + offset[None, :]
            hashed = tl.where(valid[:, None], hashed, -1)
            output_offset = (token.to(tl.int64) * LAYERS + layer)[:, None] * ((LOOKBACK - 1) * HEADS)
            tl.store(
                hashes + output_offset + column[None, :],
                hashed,
                (token < num_tokens)[:, None] & (head < HEADS)[None, :],
            )


@triton.jit(
    do_not_specialize=["num_tokens", "num_actual_tokens", "num_reqs", "num_slots", "vocab_size"],
    do_not_specialize_on_alignment=["ids", "slots", "query_start"],
)
def _write_history_kernel(
    ids,
    token_map,
    slots,
    cache,
    query_start,
    num_tokens,
    num_actual_tokens,
    num_reqs,
    num_slots,
    vocab_size,
    ids_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    cache_block_size: tl.constexpr,
    image_id: tl.constexpr,
    image_pad_id: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    end = tl.load(query_start + num_reqs)
    valid = (token < num_tokens) & (token < num_actual_tokens) & (token < end)
    block = tl.load(slots + token * slot_stride, valid, other=-1).to(tl.int64)
    offset = tl.load(slots + token * slot_stride + 1, valid, other=-1)
    slot = block * cache_block_size + offset
    valid &= (block >= 0) & (offset >= 0) & (slot < num_slots)
    source = tl.load(ids + token * ids_stride, valid, other=-1)
    source_valid = valid & (source >= 0) & (source < vocab_size)
    mapped = tl.load(token_map + tl.where(source_valid, source, 0), source_valid, other=-1)
    mapped = tl.where((source == image_id) | (source == image_pad_id), -1, mapped)
    tl.store(cache + tl.where(valid, slot, 0), mapped, valid)


def hash_engram(input_ids, positions, metadata, state):
    init_device_properties_triton()
    num_tokens = input_ids.shape[0]
    layers, _, heads = state.primes.shape
    hashes = torch.empty((num_tokens, layers, (state.lookback - 1) * heads), dtype=torch.int64, device=input_ids.device)
    keep = torch.empty(num_tokens, dtype=torch.bool, device=input_ids.device)
    num_reqs = metadata.num_actual_reqs
    num_actual_tokens = min(metadata.num_actual_tokens, metadata.slot_mapping.shape[0])
    if num_tokens == 0 or num_reqs == 0:
        hashes.fill_(-1)
        keep.zero_()
        return hashes, keep
    _hash_kernel[(triton.cdiv(num_tokens, 32), layers)](
        input_ids,
        positions,
        state.token_map,
        metadata.query_start_loc,
        metadata.block_table,
        state.cache,
        state.multipliers,
        state.primes,
        state.offsets,
        hashes,
        keep,
        num_tokens,
        num_actual_tokens,
        num_reqs,
        state.cache.numel(),
        state.token_map.numel(),
        input_ids.stride(0),
        positions.stride(0),
        metadata.block_table.stride(0),
        metadata.block_table.shape[1],
        metadata.storage_block_size,
        state.pad_id,
        state.image_token_id,
        state.image_pad_token_id,
        state.lookback,
        heads,
        layers,
        32,
        triton.next_power_of_2(heads),
        num_reqs.bit_length(),
        num_warps=4,
    )
    # In-batch lookbacks read input_ids. Commit only after hashing so a long
    # prefill cannot overwrite older physical slots before their lookbacks read.
    _write_history_kernel[(triton.cdiv(num_tokens, 256),)](
        input_ids,
        state.token_map,
        metadata.slot_mapping,
        state.cache,
        metadata.query_start_loc,
        num_tokens,
        num_actual_tokens,
        num_reqs,
        state.cache.numel(),
        state.token_map.numel(),
        input_ids.stride(0),
        metadata.slot_mapping.stride(0),
        metadata.storage_block_size,
        state.image_token_id,
        state.image_pad_token_id,
        256,
        num_warps=4,
    )
    return hashes, keep


@triton.jit(
    do_not_specialize=["vocab_start", "vocab_end", "tokens"],
    do_not_specialize_on_alignment=["ids"],
)
def _lookup_heads_kernel(
    weight,
    scales,
    ids,
    output,
    vocab_start,
    vocab_end,
    tokens,
    token_stride: tl.constexpr,
    head_stride: tl.constexpr,
    HEAD_START: tl.constexpr,
    HEAD_COUNT: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    INT8: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    token = (row // PADDED_HEADS).to(tl.int64)
    local_head = row % PADDED_HEADS
    valid = (token < tokens) & (local_head < HEAD_COUNT)
    index = tl.load(ids + token * token_stride + (HEAD_START + local_head) * head_stride, valid, other=-1).to(tl.int64)
    owned = valid & (index >= vocab_start) & (index < vocab_end)
    local = tl.where(owned, index - vocab_start, 0)
    col = tl.arange(0, WIDTH)
    values = tl.load(weight + local[:, None] * WIDTH + col[None, :], owned[:, None], other=0)
    if INT8:
        scale = tl.load(scales + local[:, None] * (WIDTH // 32) + col[None, :] // 32, owned[:, None], other=0)
        values = values.to(tl.float32) * scale
    tl.store(output + row[:, None] * WIDTH + col[None, :], values.to(tl.bfloat16), (token < tokens)[:, None])


def lookup_engram_heads(table, ids):
    init_device_properties_triton()
    output = torch.empty((ids.shape[0], table.padded_heads, table.width), dtype=torch.bfloat16, device=ids.device)
    if output.numel():
        _lookup_heads_kernel[(triton.cdiv(ids.shape[0] * table.padded_heads, 16),)](
            table.weight,
            getattr(table, "weight_scale", table.weight),
            ids,
            output,
            table.start,
            table.end,
            ids.shape[0],
            ids.stride(0),
            ids.stride(1),
            table.head_start,
            table.head_count,
            table.padded_heads,
            table.width,
            table.storage_format == "int8",
            16,
            num_warps=4,
        )
    return output


@triton.jit(do_not_specialize=["elements", "source_tokens", "token_start"])
def _select_rows_kernel(
    source,
    output,
    head_indices,
    elements,
    source_tokens,
    token_start,
    LOCAL_HEADS: tl.constexpr,
    OUTPUT_HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_INDICES: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    tiles = tl.cdiv(elements, BLOCK_R * WIDTH)
    per_core = tl.cdiv(tiles, tl.num_programs(0))
    first = tl.program_id(0) * per_core
    col = tl.arange(0, WIDTH)
    # Compute ownership once per row, then copy its contiguous vector. Keep
    # wide arithmetic only for addresses, which can exceed 2 Gi elements.
    for tile in range(first, tl.minimum(first + per_core, tiles)):
        row = tile * BLOCK_R + tl.arange(0, BLOCK_R)
        token = row // OUTPUT_HEADS
        head = row - token * OUTPUT_HEADS
        if HAS_INDICES:
            head = tl.load(head_indices + head).to(tl.int32)
        rank = head // LOCAL_HEADS
        local_head = head - rank * LOCAL_HEADS
        source_row = (rank.to(tl.int64) * source_tokens + token + token_start) * LOCAL_HEADS + local_head
        valid = row < elements // WIDTH
        value = tl.load(source + source_row[:, None] * WIDTH + col[None, :], valid[:, None], other=0)
        tl.store(output + row.to(tl.int64)[:, None] * WIDTH + col[None, :], value, valid[:, None])


def select_engram_rows(gathered, num_tokens, token_start, output_heads, head_indices=None):
    """Reorder only retained token rows from a rank-major fixed-size gather."""
    init_device_properties_triton()
    _, source_tokens, local_heads, width = gathered.shape
    output = torch.empty((num_tokens, output_heads, width), dtype=gathered.dtype, device=gathered.device)
    if output.numel():
        _select_rows_kernel[(min(triton.cdiv(output.numel(), 16 * width), get_vectorcore_num()),)](
            gathered,
            output,
            head_indices if head_indices is not None else output,
            output.numel(),
            source_tokens,
            token_start,
            local_heads,
            output_heads,
            width,
            head_indices is not None,
            16,
        )
    return output

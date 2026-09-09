# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 DSA metadata and correctness-first eager execution.

The model file owns the network topology and projection modules.  This module
owns the attention execution boundary: it gathers every cache plane's metadata
before running the unfused compressor, indexer and sparse-attention reference
path.  A future fused AscendC implementation can therefore replace the small
operators here without moving cache or scheduler knowledge back into the model.
"""

import os
from pathlib import Path
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
)

from vllm_ascend.core.deepseek_v41 import (
    DeepseekV41CompressorStateSpec,
    DeepseekV41FullSpec,
    DeepseekV41IndexerSpec,
    DeepseekV41SWASpec,
)
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.ops.triton.v41_cache import scatter_cache_rows


@dataclass
class DeepseekV41Metadata(AttentionMetadata):
    """Scheduler and cache-plane contract for one V4.1 cache resource.

    ``seq_lens``/``query_start_loc`` always stay in original-token
    coordinates, matching the common vLLM metadata. The ``cache_*`` fields
    describe the rows visible to the concrete cache plane. Keeping both
    coordinate systems here lets future fused kernels replace the eager path
    without rebuilding scheduling metadata in the model.
    """

    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    compress_ratio: int
    storage_block_size: int
    is_compressor_state: bool
    cache_kind: str = "unknown"
    positions: torch.Tensor | None = None
    cos: Any = None
    sin: Any = None
    num_actual_tokens: int = 0
    num_input_tokens: int = 0
    num_reqs: int = 0
    num_actual_reqs: int = 0
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_prefill_tokens: int = 0
    logical_block_size: int = 0
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    query_lens: torch.Tensor | None = None
    start_pos: torch.Tensor | None = None
    cache_seq_lens: torch.Tensor | None = None
    cache_query_lens: torch.Tensor | None = None
    cache_query_start_loc: torch.Tensor | None = None
    cache_start_pos: torch.Tensor | None = None
    max_query_len: int = 0
    max_seq_len: int = 0
    max_cache_seq_len: int = 0
    num_cache_tokens: int = 0
    attn_state: Any = None
    is_prefilling: torch.Tensor | None = None
    causal: bool | torch.Tensor = True
    ori_win_left: int = 0
    ori_win_right: int = 0


@dataclass(frozen=True)
class DeepseekV41CompressorMetadata:
    """V4-shaped cache/state bundle consumed by the compressor stage."""

    cache: DeepseekV41Metadata
    state: DeepseekV41Metadata | None = None


@dataclass(frozen=True)
class DeepseekV41IndexerMetadata:
    """V4-shaped source cache bundle consumed by the indexer stage."""

    cache: DeepseekV41Metadata


@dataclass(frozen=True)
class DeepseekV41LayerMetadata:
    """All metadata consumed by one V4.1 attention layer invocation."""

    attention: DeepseekV41Metadata | None
    swa: DeepseekV41Metadata
    compressor: DeepseekV41CompressorMetadata | None
    indexer: DeepseekV41IndexerMetadata | None

    @property
    def positions(self) -> torch.Tensor:
        if self.swa.positions is None:
            raise RuntimeError("V4.1 SWA metadata does not contain input positions")
        return self.swa.positions

    def rope(self, layer_name: str, num_tokens: int):
        if self.swa.cos is None or self.swa.sin is None:
            raise RuntimeError("V4.1 SWA metadata does not contain RoPE tensors")
        return self.swa.cos[layer_name][:num_tokens], self.swa.sin[layer_name][:num_tokens]


def compressed_slot_mapping(slot_mapping: torch.Tensor, ratio: int) -> torch.Tensor:
    """Convert original-token physical slots to completed compressed slots.

    Logical block sizes must be divisible by ratio. Negative/padded slots and
    incomplete compression groups never produce a write.
    """
    if ratio not in (1, 2):
        raise ValueError("V4.1 only supports ratio 1 or 2")
    valid = (slot_mapping >= 0) & ((slot_mapping + 1) % ratio == 0)
    return torch.where(valid, slot_mapping // ratio, -1)


def _cache_coordinates(common: Any, ratio: int, compressed: bool):
    """Build original/cache coordinate views without inspecting model state."""
    query_start_loc = common.query_start_loc[: common.num_reqs + 1]
    seq_lens = common.seq_lens[: common.num_reqs]
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    start_pos = seq_lens - query_lens
    plane_ratio = ratio if compressed else 1
    cache_seq_lens = torch.div(seq_lens, plane_ratio, rounding_mode="floor")
    cache_start_pos = torch.div(start_pos, plane_ratio, rounding_mode="floor")
    cache_query_lens = cache_seq_lens - cache_start_pos
    cache_query_start_loc = torch.cat(
        (cache_query_lens.new_zeros(1), cache_query_lens.cumsum(0))
    )
    query_start_loc_cpu = getattr(common, "query_start_loc_cpu", None)
    seq_lens_cpu = getattr(common, "seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = getattr(common, "_seq_lens_cpu", None)
    num_cache_tokens = 0
    max_cache_seq_len = 0
    if query_start_loc_cpu is not None and seq_lens_cpu is not None:
        cpu_query_lens = (
            query_start_loc_cpu[1 : common.num_reqs + 1]
            - query_start_loc_cpu[: common.num_reqs]
        )
        cpu_seq_lens = seq_lens_cpu[: common.num_reqs]
        cpu_start_pos = cpu_seq_lens - cpu_query_lens
        cpu_cache_seq_lens = torch.div(
            cpu_seq_lens, plane_ratio, rounding_mode="floor"
        )
        cpu_cache_start_pos = torch.div(
            cpu_start_pos, plane_ratio, rounding_mode="floor"
        )
        num_cache_tokens = int(
            (cpu_cache_seq_lens - cpu_cache_start_pos).sum().item()
        )
        max_cache_seq_len = int(cpu_cache_seq_lens.max().item()) if common.num_reqs else 0
    elif not compressed:
        # Production always supplies CPU mirrors. This keeps lightweight unit
        # fixtures useful without introducing a device-to-host synchronization.
        num_cache_tokens = int(getattr(common, "num_actual_tokens", 0))
        max_cache_seq_len = int(getattr(common, "max_seq_len", 0))

    return dict(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens_cpu=seq_lens_cpu,
        query_lens=query_lens,
        start_pos=start_pos,
        cache_seq_lens=cache_seq_lens,
        cache_query_lens=cache_query_lens,
        cache_query_start_loc=cache_query_start_loc,
        cache_start_pos=cache_start_pos,
        num_cache_tokens=num_cache_tokens,
        max_cache_seq_len=max_cache_seq_len,
    )


def _request_counts(common: Any, num_reqs: int):
    """Return V4-shaped request counters without synchronizing the NPU."""
    is_prefilling = getattr(common, "is_prefilling", None)
    query_start_loc_cpu = getattr(common, "query_start_loc_cpu", None)
    if (
        is_prefilling is None
        or query_start_loc_cpu is None
        or getattr(is_prefilling, "device", None) is None
        or is_prefilling.device.type != "cpu"
    ):
        return 0, 0, 0, 0
    flags = is_prefilling[:num_reqs].bool()
    query_lens_cpu = (
        query_start_loc_cpu[1 : num_reqs + 1]
        - query_start_loc_cpu[:num_reqs]
    )
    num_prefills = int(flags.sum().item())
    num_decodes = num_reqs - num_prefills
    num_prefill_tokens = int(query_lens_cpu[flags].sum().item())
    num_decode_tokens = int(query_lens_cpu[~flags].sum().item())
    return num_decodes, num_decode_tokens, num_prefills, num_prefill_tokens


def scatter_cache(cache: torch.Tensor, slots: torch.Tensor, values: torch.Tensor) -> None:
    """Write valid rows without a host-side reduction or dynamic indexing.

    ``valid.any()`` on an NPU tensor synchronizes the copy stream and is not
    legal while ACLGraph is being captured.  The Triton kernel masks negative
    slots in device code, so the same operation is graph-safe for both eager
    and capture paths.
    """
    slots = slots[: values.shape[0]].long()
    if values.dtype != cache.dtype:
        values = values.to(cache.dtype)
    # Keep the host/unit-test path independent of the Ascend Triton backend.
    if cache.device.type != "npu":
        cache_view = cache.squeeze(-2)
        valid = slots >= 0
        if valid.any():
            physical = slots[valid]
            pages = torch.div(physical, cache_view.shape[1], rounding_mode="floor")
            rows = physical.remainder(cache_view.shape[1])
            cache_view[pages, rows] = values[valid]
        return
    scatter_cache_rows(cache, slots, values)


def gather_cache_rows(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Read physical rows without flattening a block-strided packed view."""
    cache = cache.squeeze(-2)
    slots = slots.long()
    pages = torch.div(slots, cache.shape[1], rounding_mode="floor")
    rows = slots.remainder(cache.shape[1])
    return cache[pages, rows]


def paged_prefix(cache, block_table, length, block_size):
    """Materialize one request's logical prefix from a paged cache."""
    if length <= 0:
        return cache.new_empty((0, cache.shape[-1]))
    blocks = (length + block_size - 1) // block_size
    page_ids = block_table[:blocks].long()
    return cache.squeeze(-2).index_select(0, page_ids).flatten(0, 1)[:length]


def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    """Return the level-one candidate-position mask used by V4.1.

    A block is scored by its best position.  The newest, partially populated
    block is pinned so recent compressed tokens cannot be dropped merely
    because their block has fewer populated positions.
    """
    width = logits.shape[-1]
    if width == 0:
        return torch.zeros_like(logits, dtype=torch.bool)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(-1)
    num_blocks = scores.shape[-1]
    if not torch.is_tensor(compress_lens):
        compress_lens = torch.tensor(compress_lens, device=logits.device)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last,
        torch.inf,
    )
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def select_index_topk(logits, compress_lens, index_topk):
    """Level-two position TopK, sorted back into chronological order."""
    width = logits.shape[-1]
    if width == 0:
        return torch.empty(
            (*logits.shape[:-1], 0), dtype=torch.int32, device=logits.device
        )
    topk = min(index_topk, width)
    indices = logits.topk(topk, dim=-1, sorted=False).indices.sort(-1).values
    return torch.where(indices < compress_lens, indices, -1).int()


def pad_sparse_indices(indices: torch.Tensor, topk: int) -> torch.Tensor:
    """Convert V4.1's compact [T, K] selection into SMLA [T, 1, topk]."""
    if indices.ndim != 2:
        raise ValueError(f"V4.1 sparse indices must be rank 2, got {indices.shape}")
    if indices.shape[-1] > topk:
        raise ValueError(
            f"V4.1 sparse indices width {indices.shape[-1]} exceeds operator topk {topk}"
        )
    if indices.shape[-1] < topk:
        indices = F.pad(indices, (0, topk - indices.shape[-1]), value=-1)
    return indices.unsqueeze(1).contiguous().int()


def small_op_attention(
    q,
    positions,
    swa_cache,
    swa_metadata,
    *,
    source_cache=None,
    source_metadata=None,
    compress_ratio=0,
    window_size=128,
    index_topk=512,
    compressed_indices=None,
    sinks=None,
    softmax_scale=1.0,
):
    """Unfused eager sparse attention over SWA plus a shared compressed KV.

    ``compressed_indices`` contains the real Indexer result in compressed-KV
    coordinates.  When absent, the bounded newest-row selection remains as a
    diagnostic fallback for local-only tests.
    """
    query_starts = swa_metadata.query_start_loc.tolist()
    seq_lens = swa_metadata.seq_lens.tolist()
    outputs = []
    for req_idx, (q_start, q_end) in enumerate(zip(query_starts[:-1], query_starts[1:])):
        seq_len = int(seq_lens[req_idx])
        local = paged_prefix(
            swa_cache,
            swa_metadata.block_table[req_idx],
            seq_len,
            swa_metadata.storage_block_size,
        )
        compressed = None
        if source_cache is not None:
            compressed_len = int(source_metadata.cache_seq_lens[req_idx])
            compressed = paged_prefix(
                source_cache,
                source_metadata.block_table[req_idx],
                compressed_len,
                source_metadata.storage_block_size,
            )
        for token_idx in range(q_start, q_end):
            position = int(positions[token_idx])
            local_start = max(0, position - window_size + 1)
            keys = local[local_start : position + 1]
            if compressed is not None:
                visible = (position + 1) // compress_ratio
                if compressed_indices is None:
                    selected = compressed[max(0, visible - index_topk) : visible]
                else:
                    indices = compressed_indices[token_idx].long()
                    indices = indices[(indices >= 0) & (indices < visible)]
                    selected = compressed.index_select(0, indices)
                keys = torch.cat((keys, selected))
            logits = torch.einsum("hd,kd->hk", q[token_idx].float(), keys.float())
            logits *= softmax_scale
            if sinks is not None:
                logits = torch.cat((logits, sinks.float().unsqueeze(-1)), -1)
                probs = logits.softmax(-1)[..., :-1]
            else:
                probs = logits.softmax(-1)
            outputs.append(torch.einsum("hk,kd->hd", probs, keys.float()))
    return torch.stack(outputs).to(q.dtype)


class DeepseekV41EagerAttentionImpl:
    """V4-shaped execution boundary backed by correctness-first small ops.

    Projection, compressor and indexer modules remain registered by the model,
    while this object resolves the complete per-layer metadata bundle and owns
    their invocation order.  That is the same separation used by ``dsa_v1``:
    model construction is independent from cache-aware attention execution.
    """

    def __init__(self, prefix, role, topology, long_kv_source_prefix, index_k_source_prefix):
        self.prefix = prefix
        self.layer_name = f"{prefix}.attn"
        self.role = role
        self.topology = topology
        self.swa_prefix = f"{prefix}.swa_cache"
        self.long_kv_source_prefix = long_kv_source_prefix
        self.index_k_source_prefix = index_k_source_prefix
        self.compressor_state_prefix = (
            f"{prefix}.compressor.state_cache"
            if role.is_kv_source and role.compress_ratio == 2
            else None
        )

    def _get_layer_metadata(self, metadata) -> DeepseekV41LayerMetadata:
        try:
            swa = metadata[self.swa_prefix]
            long_kv = (
                metadata[self.long_kv_source_prefix]
                if self.long_kv_source_prefix is not None
                else None
            )
            index_k = (
                metadata[self.index_k_source_prefix]
                if self.index_k_source_prefix is not None
                else None
            )
            compressor_state = (
                metadata[self.compressor_state_prefix]
                if self.compressor_state_prefix is not None
                else None
            )
        except KeyError as exc:
            raise RuntimeError(f"Missing V4.1 cache metadata for {exc.args[0]}") from exc
        return DeepseekV41LayerMetadata(
            attention=long_kv,
            swa=swa,
            compressor=(
                DeepseekV41CompressorMetadata(long_kv, compressor_state)
                if self.role.is_kv_source and long_kv is not None
                else None
            ),
            indexer=(
                DeepseekV41IndexerMetadata(index_k)
                if index_k is not None
                else None
            ),
        )

    @staticmethod
    def _project_q_kv(attn, hidden_states, cos, sin):
        q_a = attn.wq_a(hidden_states)
        qr = attn.q_norm(q_a)
        q = attn.wq_b(qr).unflatten(-1, (attn.n_local_heads, attn.head_dim))
        kv = attn.kv_norm(attn.wkv(hidden_states))
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        kv = kv.view(-1, 1, attn.head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        return q.to(hidden_states.dtype), qr, kv.squeeze(1)

    def _write_compressed_source(self, attn, hidden_states, positions, metadata):
        compressor = attn.compressor
        if compressor is None or metadata.compressor is None or metadata.indexer is None:
            raise RuntimeError("V4.1 KV source is missing compressor or source metadata")
        compressor_metadata = metadata.compressor
        indexer_metadata = metadata.indexer
        ratio = self.role.compress_ratio
        if ratio == 1:
            latent = compressor(hidden_states, 0)
            completed = torch.ones_like(positions, dtype=torch.bool)
        else:
            if compressor_metadata.state is None:
                raise RuntimeError("V4.1 ratio-2 source is missing compressor-state metadata")
            state_cache = compressor.state_cache.kv_cache[0].squeeze(-2)
            kv = compressor.wkv(hidden_states.float())
            score = compressor.wgate(hidden_states.float())
            state_rows = torch.cat((kv, score), -1)
            scatter_cache(
                compressor.state_cache.kv_cache[0],
                compressor_metadata.state.slot_mapping,
                state_rows,
            )
            completed = positions.remainder(ratio) == ratio - 1
            completed_slots = compressor_metadata.state.slot_mapping[: positions.shape[0]].long()
            # Keep a static row count during ACLGraph capture.  Invalid
            # (non-completed) rows use a safe gather slot and are masked before
            # publishing; boolean indexing would lower to dynamic nonzero.
            safe_slots = completed_slots.clamp_min(0)
            current = gather_cache_rows(state_cache, safe_slots)
            previous = gather_cache_rows(state_cache, (safe_slots - 1).clamp_min(0))
            pair = torch.stack((previous, current), 1)
            latent = (
                pair[..., : attn.head_dim]
                * pair[..., attn.head_dim :].softmax(1)
            ).sum(1)
            latent = compressor.norm(latent.to(hidden_states.dtype))
            latent = torch.where(completed.unsqueeze(-1), latent, torch.zeros_like(latent))

        source_positions = torch.where(
            completed, positions + 1 - ratio, torch.zeros_like(positions)
        )
        source_cos, source_sin = get_cos_and_sin_dsa(source_positions)
        source_cos = source_cos[attn.rotary_emb.layername]
        source_sin = source_sin[attn.rotary_emb.layername]
        if attn.indexer is None:
            raise RuntimeError("V4.1 KV source is missing its indexer")
        attn.indexer.update_keys(
            latent,
            torch.where(
                completed,
                indexer_metadata.cache.slot_mapping[: positions.shape[0]].long(),
                torch.full_like(positions, -1),
            ),
            source_cos,
            source_sin,
        )
        latent = latent.view(-1, 1, attn.head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            latent.unsqueeze(1),
            source_cos,
            source_sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        scatter_cache(
            attn.long_kv_cache.kv_cache[0],
            torch.where(
                completed,
                compressor_metadata.cache.slot_mapping[: positions.shape[0]].long(),
                torch.full_like(positions, -1),
            ),
            latent.squeeze(1),
        )

    def _select_sparse_indices(self, attn, hidden_states, qr, positions, cos, sin, metadata):
        if not self.role.has_long_context:
            return None
        shared = attn.shared_state
        if shared is None:
            raise RuntimeError("V4.1 shared attention state is not initialized")
        if not self.role.is_index_source:
            if shared.topk_indices is None:
                raise RuntimeError("V4.1 sparse consumer ran before its index source")
            return shared.topk_indices
        if attn.indexer is None or metadata.indexer is None:
            raise RuntimeError("V4.1 index source is missing indexer metadata")

        context = get_forward_context().no_compile_layers
        source_layer = context[self.index_k_source_prefix]
        selected, candidates = attn.indexer.select(
            hidden_states,
            qr,
            positions,
            cos,
            sin,
            source_layer.kv_cache[0],
            metadata.indexer.cache,
            is_candidate_source=self.role.is_candidate_source,
            uses_candidate_filter=self.role.uses_candidate_filter,
            candidate_topk_blocks=self.topology.candidate_topk_blocks,
            candidate_block_size=self.topology.candidate_block_size,
            candidates=shared.candidates,
        )
        shared.topk_indices = selected
        shared.candidates = candidates
        return selected

    def _attention(self, attn, q, positions, metadata, compressed_indices):
        source_cache = None
        if self.role.has_long_context:
            source_cache = get_forward_context().no_compile_layers[
                self.long_kv_source_prefix
            ].kv_cache[0]
        # A2/A3 SparseFlashMla uses ratio 0 for SWA-only and supports the
        # ratio-1/ratio-2 compressed sparse paths used by this topology.
        if self.role.compress_ratio in (0, 1, 2):
            return self._native_attention(
                attn,
                q,
                metadata,
                source_cache=source_cache,
                compressed_indices=compressed_indices,
            )
        return small_op_attention(
            q,
            positions,
            attn.dsa_attn.swa_cache_layer.kv_cache[0],
            metadata.swa,
            source_cache=source_cache,
            source_metadata=metadata.attention,
            compress_ratio=self.role.compress_ratio,
            window_size=attn.window_size,
            index_topk=self.topology.index_topk,
            compressed_indices=compressed_indices,
            sinks=attn.attn_sink,
            softmax_scale=attn.softmax_scale,
        )

    def _native_attention(
        self,
        attn,
        q,
        metadata,
        *,
        source_cache,
        compressed_indices,
    ):
        """Run SparseFlashMla with the same PA metadata for both operator stages."""
        if attn.head_dim != 512:
            raise ValueError(f"SparseFlashMla requires head_dim 512, got {attn.head_dim}")
        if attn.window_size != 128:
            raise ValueError(
                f"A2/A3 SparseFlashMla requires sliding_window 128, got {attn.window_size}"
            )
        if not 1 <= attn.n_local_heads <= 128 or attn.n_local_heads & (attn.n_local_heads - 1):
            raise ValueError(
                "A2/A3 SparseFlashMla requires the local query-head count to be "
                f"a power of two in [1, 128], got {attn.n_local_heads}"
            )
        has_compressed = self.role.compress_ratio in (1, 2)
        ratio = self.role.compress_ratio if has_compressed else 0
        num_reqs = metadata.swa.num_reqs
        query_start_loc = metadata.swa.query_start_loc[: num_reqs + 1].int()
        seq_lens = metadata.swa.seq_lens[:num_reqs].int()
        ori_block_table = metadata.swa.block_table[:num_reqs].int()
        cmp_block_table = None
        cmp_seq_lens = None
        cmp_residual = None
        cmp_indices = None
        cmp_topk = 0
        if has_compressed:
            if source_cache is None or metadata.attention is None or compressed_indices is None:
                raise RuntimeError("V4.1 compressed attention is missing KV or TopK metadata")
            cmp_block_table = metadata.attention.block_table[:num_reqs].int()
            cmp_seq_lens = metadata.attention.cache_seq_lens[:num_reqs].int()
            cmp_residual = seq_lens.remainder(ratio) if ratio != 1 else None
            cmp_topk = self.topology.index_topk
            if cmp_topk not in (512, 1024):
                raise ValueError(f"SparseFlashMla only supports TopK 512 or 1024, got {cmp_topk}")
            cmp_indices = pad_sparse_indices(compressed_indices, cmp_topk)

        shared = attn.shared_state
        if shared is None:
            raise RuntimeError("V4.1 shared attention state is not initialized")
        op_metadata = shared.smla_metadata.get(ratio)
        if op_metadata is None:
            op_metadata = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
                attn.n_local_heads,
                1,
                attn.head_dim,
                cu_seqlens_q=query_start_loc,
                seqused_ori_kv=seq_lens,
                seqused_cmp_kv=cmp_seq_lens,
                cmp_residual_kv=cmp_residual,
                batch_size=num_reqs,
                max_seqlen_q=metadata.swa.max_query_len,
                max_seqlen_ori_kv=metadata.swa.max_seq_len,
                max_seqlen_cmp_kv=(metadata.attention.max_cache_seq_len if has_compressed else 0),
                ori_topk=0,
                cmp_topk=cmp_topk,
                cmp_ratio=ratio,
                ori_mask_mode=4,
                cmp_mask_mode=3 if has_compressed else 0,
                ori_win_left=attn.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_BBND",
                has_ori_kv=True,
                has_cmp_kv=has_compressed,
            )
            shared.smla_metadata[ratio] = op_metadata
        output, _ = torch.ops._C_ascend.npu_sparse_flash_mla(
            q,
            ori_kv=attn.dsa_attn.swa_cache_layer.kv_cache[0],
            cmp_kv=source_cache,
            cmp_sparse_indices=cmp_indices,
            ori_block_table=ori_block_table,
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=query_start_loc,
            seqused_ori_kv=seq_lens,
            seqused_cmp_kv=cmp_seq_lens,
            cmp_residual_kv=cmp_residual,
            sinks=attn.attn_sink,
            metadata=op_metadata,
            softmax_scale=attn.softmax_scale,
            cmp_ratio=ratio,
            ori_mask_mode=4,
            cmp_mask_mode=3 if has_compressed else 0,
            ori_win_left=attn.window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BBND",
            topk_value_mode=1,
            return_softmax_lse=False,
        )
        return output

    def forward(self, attn, positions, hidden_states):
        forward_context = get_forward_context()
        if forward_context.attn_metadata is None:
            return torch.zeros_like(hidden_states)
        metadata = self._get_layer_metadata(forward_context.attn_metadata)
        positions = metadata.positions[: hidden_states.shape[0]]
        cos, sin = metadata.rope(attn.rotary_emb.layername, hidden_states.shape[0])
        q, qr, kv = self._project_q_kv(attn, hidden_states, cos, sin)
        scatter_cache(
            attn.dsa_attn.swa_cache_layer.kv_cache[0],
            metadata.swa.slot_mapping,
            kv,
        )
        if self.role.is_kv_source:
            self._write_compressed_source(attn, hidden_states, positions, metadata)
        compressed_indices = self._select_sparse_indices(
            attn, hidden_states, qr, positions, cos, sin, metadata
        )
        output = self._attention(attn, q, positions, metadata, compressed_indices)
        dump_root = os.environ.get("VLLM_V41_ATTENTION_DUMP")
        if dump_root and self.role.layer_idx == 24 and hidden_states.shape[0] == 188:
            rank = torch.distributed.get_rank()
            path = Path(dump_root) / f"layer24-rank{rank}.pt"
            if rank % 8 == 0 and not path.exists():
                def cpu_meta(meta):
                    return {field.name: (getattr(meta, field.name).detach().cpu()
                            if isinstance(getattr(meta, field.name), torch.Tensor)
                            else getattr(meta, field.name)) for field in fields(meta)}
                source = forward_context.no_compile_layers[self.long_kv_source_prefix].kv_cache[0]
                payload = {
                    "hidden": hidden_states.detach().cpu(),
                    "q": q.detach().cpu(), "qr": qr.detach().cpu(),
                    "kv": kv.detach().cpu(), "positions": positions.detach().cpu(),
                    "cos": cos.detach().cpu(), "sin": sin.detach().cpu(),
                    "swa_cache": attn.dsa_attn.swa_cache_layer.kv_cache[0].detach().cpu(),
                    "source_cache": source.detach().cpu(),
                    "indices": compressed_indices.detach().cpu(),
                    "sinks": attn.attn_sink.detach().cpu(),
                    "output": output.detach().cpu(),
                    "swa_metadata": cpu_meta(metadata.swa),
                    "source_metadata": cpu_meta(metadata.attention),
                    "scale": attn.softmax_scale,
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, path)
                print(f"V41_ATTENTION_DUMP {path}", flush=True)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            output.unsqueeze(1),
            cos,
            -sin,
            rotary_mode="interleave",
            partial_slice=[attn.nope_head_dim, attn.head_dim],
        )
        projected = torch.empty_like(hidden_states)
        projected = attn.project_output(output, projected)
        return projected


class DeepseekV41MetadataBuilder(AttentionMetadataBuilder[DeepseekV41Metadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._decode_buffers = {}

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        if common_prefix_len:
            raise NotImplementedError("V4.1 prefix caching is not implemented")
        spec = self.kv_cache_spec
        common = common_attn_metadata
        is_compressor_state = isinstance(spec, DeepseekV41CompressorStateSpec)
        ratio = getattr(spec, "compress_ratio", 1)
        if isinstance(spec, DeepseekV41SWASpec):
            cache_kind = "swa"
        elif isinstance(spec, DeepseekV41FullSpec):
            cache_kind = "long_kv"
        elif isinstance(spec, DeepseekV41IndexerSpec):
            cache_kind = "index_k"
        elif is_compressor_state:
            cache_kind = "compressor_state"
        else:
            raise TypeError(f"Unsupported V4.1 cache spec: {type(spec).__name__}")

        # SWA and compressor state are addressed in original-token coordinates.
        # Long KV and index K are addressed in completed compression groups.
        compressed = cache_kind in {"long_kv", "index_k"}
        slots = (
            common.slot_mapping
            if cache_kind in {"swa", "compressor_state"}
            else compressed_slot_mapping(common.slot_mapping, ratio)
        )
        coordinates = _cache_coordinates(common, ratio, compressed)
        # Some Ascend scheduler paths omit the host mirrors.  Materialize
        # them while building metadata (before model/ACLGraph execution), so
        # Engram routing never calls ``.cpu()`` on a graph/replay tensor.
        if coordinates["query_start_loc_cpu"] is None:
            coordinates["query_start_loc_cpu"] = common.query_start_loc.detach().cpu()
        if coordinates["seq_lens_cpu"] is None:
            coordinates["seq_lens_cpu"] = common.seq_lens.detach().cpu()
        positions = getattr(common, "positions", None)
        cos = sin = None
        if cache_kind == "swa" and positions is not None:
            positions = positions[: common.num_input_tokens].long()
            cos, sin = get_cos_and_sin_dsa(positions, use_cache=True)
        num_reqs = int(getattr(common, "num_reqs", common.seq_lens.shape[0]))
        (
            num_decodes,
            num_decode_tokens,
            num_prefills,
            num_prefill_tokens,
        ) = _request_counts(common, num_reqs)
        text_config = self.vllm_config.model_config.hf_text_config
        window_size = int(getattr(text_config, "sliding_window", 0))
        metadata = DeepseekV41Metadata(
            block_table=common.block_table_tensor[:num_reqs],
            slot_mapping=slots,
            compress_ratio=ratio,
            storage_block_size=spec.storage_block_size,
            is_compressor_state=is_compressor_state,
            cache_kind=cache_kind,
            positions=positions,
            cos=cos,
            sin=sin,
            num_actual_tokens=int(getattr(common, "num_actual_tokens", slots.shape[0])),
            num_input_tokens=int(getattr(common, "num_input_tokens", slots.shape[0])),
            num_reqs=num_reqs,
            num_actual_reqs=num_reqs,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            logical_block_size=spec.block_size,
            max_query_len=int(getattr(common, "max_query_len", 0)),
            max_seq_len=int(getattr(common, "max_seq_len", 0)),
            attn_state=getattr(common, "attn_state", None),
            is_prefilling=getattr(common, "is_prefilling", None),
            causal=getattr(common, "causal", True),
            ori_win_left=max(0, window_size - 1),
            ori_win_right=0,
            **coordinates,
        )

        # FULL decode graphs capture addresses of metadata tensors built outside
        # the graph. Keep each batch shape alive and refresh it before replay.
        if metadata.max_query_len == 1:
            key = (metadata.num_input_tokens, metadata.num_reqs)
            buffers = self._decode_buffers.setdefault(key, {})
            for field in fields(metadata):
                value = getattr(metadata, field.name)
                if not isinstance(value, torch.Tensor) or value.device.type == "cpu":
                    continue
                if field.name not in buffers:
                    buffers[field.name] = value.clone()
                else:
                    buffers[field.name].copy_(value)
                setattr(metadata, field.name, buffers[field.name])
        return metadata


class DeepseekV41CacheBackend(AttentionBackend):
    """Cache-only backend: supplies layout and metadata, not an AttentionImpl."""

    @staticmethod
    def get_name():
        return "ASCEND_DSA_V41_CACHE"

    # The V4.1 implementation has a fixed single-token decode contract. The
    # actual attention executes in the model layer, while this backend only
    # supplies metadata; declaring the narrow support lets the outer Ascend
    # ACLGraph capture FULL_DECODE_ONLY without claiming prefill support.
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("V4.1 sparse-attention execution is not implemented yet")

    @staticmethod
    def get_builder_cls():
        return DeepseekV41MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
        return num_blocks, block_size, num_kv_heads, head_size


class DeepseekV41CacheLayer(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(self, vllm_config, prefix, spec):
        super().__init__()
        self.prefix = prefix
        self.spec = spec
        self.kv_cache = [torch.empty(0)]
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate V4.1 cache prefix: {prefix}")
        context[prefix] = self

    def get_kv_cache_spec(self, vllm_config):
        return self.spec

    def get_attn_backend(self):
        return DeepseekV41CacheBackend

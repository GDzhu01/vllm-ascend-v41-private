# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 DSA metadata and fused attention execution.

The model file owns the network topology and projection modules.  This module
owns the attention execution boundary: it gathers every cache plane's metadata
before running the compressor, indexer and sparse-attention operators without
moving cache or scheduler knowledge back into the model.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torch_npu
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
)

from vllm_ascend.attention.dsa_v1 import (
    _dsa_swa_only_cmp_ratio,
    build_dspark_swa_indices,
    dsv4_dsa_overlap_stream,
)
from vllm_ascend.attention.sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata
from vllm_ascend.core.deepseek_v41 import (
    DeepseekV41A5DraftSWASpec,
    DeepseekV41CompressorStateSpec,
    DeepseekV41DraftSWASpec,
    DeepseekV41FullSpec,
    DeepseekV41IndexerSpec,
    DeepseekV41SWASpec,
    is_v41_draft_swa_spec,
    uses_a5_packed_cache,
)
from vllm_ascend.ops.dsv41_a5 import (
    apply_partial_rotary_inplace,
    build_window_indices,
    qsmla,
    write_attention_cache,
)
from vllm_ascend.ops.dsv41_a5.package_loader import import_packaged_a5_module
from vllm_ascend.ops.rope_dsv4 import (
    get_cos_and_sin_dsa,
    get_full_cos_and_sin_dsa_for_layer,
)
from vllm_ascend.ops.triton.a5_slot_mapping import build_a5_slot_mapping
from vllm_ascend.utils import npu_stream_switch
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    DeviceMetadataTask,
    wait_for_device_metadata,
)

V41_METADATA_BUFFER_SIZE = 1024


@eager_break_during_capture
def dsa_v41_forward(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Execute V4.1 attention behind an explicit graph side-effect boundary."""
    forward_context = get_forward_context()
    attn = forward_context.no_compile_layers[layer_name]
    attn.v41_impl.forward(attn, None, hidden_states, output)


def dsa_v41_forward_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="dsa_v41_forward",
    op_func=dsa_v41_forward,
    mutates_args=["output"],
    fake_impl=dsa_v41_forward_fake,
    dispatch_key="PrivateUse1",
)


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read one field from either an HF config object or a raw config dict."""
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


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
    # Persistent linear cache addresses for native A5 cache writers.  Keep
    # ``slot_mapping`` as [T, 2] page/row coordinates for ScatterNd and the
    # index-cache writer; materializing both representations in the metadata
    # builder avoids flattening coordinates once per attention layer.
    flat_slot_mapping: torch.Tensor | None = None
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
    block_table_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    cache_seq_lens: torch.Tensor | None = None
    max_query_len: int = 0
    max_seq_len: int = 0
    max_cache_seq_len: int = 0
    attn_state: Any = None
    is_prefilling: torch.Tensor | None = None
    causal: bool | torch.Tensor = True
    ori_sparse_indices: torch.Tensor | None = None
    ori_topk_length: torch.Tensor | None = None
    ori_mask_mode: int = 4
    ori_win_left: int = 0
    ori_win_right: int = 0
    smla_metadata: torch.Tensor | None = None
    qli_metadata: torch.Tensor | None = None
    cmp_residual: torch.Tensor | None = None
    c2_ring_metadata: torch.Tensor | None = None
    c2_complete_mask: torch.Tensor | None = None
    c2_source_positions: torch.Tensor | None = None
    c2_source_cos: torch.Tensor | None = None
    c2_source_sin: torch.Tensor | None = None
    c2_metadata_group_id: int | None = None
    global_metadata: "DeepseekV41Metadata | None" = None
    cp_token_range: tuple[int, int, int, int] | None = None


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
    cache_query_start_loc = torch.cat((cache_query_lens.new_zeros(1), cache_query_lens.cumsum(0)))
    query_start_loc_cpu = getattr(common, "query_start_loc_cpu", None)
    seq_lens_cpu = getattr(common, "seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = getattr(common, "_seq_lens_cpu", None)
    num_cache_tokens = 0
    max_cache_seq_len = 0
    if query_start_loc_cpu is not None and seq_lens_cpu is not None:
        cpu_query_lens = query_start_loc_cpu[1 : common.num_reqs + 1] - query_start_loc_cpu[: common.num_reqs]
        cpu_seq_lens = seq_lens_cpu[: common.num_reqs]
        cpu_start_pos = cpu_seq_lens - cpu_query_lens
        cpu_cache_seq_lens = torch.div(cpu_seq_lens, plane_ratio, rounding_mode="floor")
        cpu_cache_start_pos = torch.div(cpu_start_pos, plane_ratio, rounding_mode="floor")
        num_cache_tokens = int((cpu_cache_seq_lens - cpu_cache_start_pos).sum().item())
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
    query_lens_cpu = query_start_loc_cpu[1 : num_reqs + 1] - query_start_loc_cpu[:num_reqs]
    num_prefills = int(flags.sum().item())
    num_decodes = num_reqs - num_prefills
    num_prefill_tokens = int(query_lens_cpu[flags].sum().item())
    num_decode_tokens = int(query_lens_cpu[~flags].sum().item())
    return num_decodes, num_decode_tokens, num_prefills, num_prefill_tokens


def scatter_cache_sk(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Store rows using builder-prepared coordinates and CANN ScatterNd.

    V4.1 cache planes can be views into a larger layer-outermost slot, so the
    physical page stride is not necessarily the contiguous stride implied by
    plane shape.  The public torch-npu ScatterNd path preserves that stride and
    treats the builder's ``[-1, -1]`` coordinates as skipped rows.  INT64 is
    required because the packed A5 allocation can exceed the ScatterNd INT32
    internal byte-offset range even though page and row coordinates are small.
    """
    if slot_mapping.ndim != 2 or slot_mapping.shape[-1] != 2:
        raise ValueError(
            f"V4.1 fused cache store requires builder-prepared [T, 2] slot_mapping, got {tuple(slot_mapping.shape)}"
        )
    cache = cache.squeeze(-2)
    indices = slot_mapping[: values.shape[0]].to(torch.int64).contiguous()
    updates = values.to(cache.dtype).contiguous()
    torch_npu.npu_scatter_nd_update_(cache, indices, updates)


def pad_sparse_indices(indices: torch.Tensor, topk: int) -> torch.Tensor:
    """Convert V4.1's compact [T, K] selection into SMLA [T, 1, topk]."""
    if indices.ndim != 2:
        raise ValueError(f"V4.1 sparse indices must be rank 2, got {indices.shape}")
    if indices.shape[-1] > topk:
        raise ValueError(f"V4.1 sparse indices width {indices.shape[-1]} exceeds operator topk {topk}")
    if indices.shape[-1] < topk:
        indices = F.pad(indices, (0, topk - indices.shape[-1]), value=-1)
    return indices.unsqueeze(1).contiguous().int()


class DeepseekV41EagerAttentionImpl:
    """V4-shaped execution boundary backed by fused Ascend operators.

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
            f"{prefix}.compressor.state_cache" if role.is_kv_source and role.compress_ratio == 2 else None
        )

    def _get_layer_metadata(self, metadata) -> DeepseekV41LayerMetadata:
        try:
            swa = metadata[self.swa_prefix]
            long_kv = metadata[self.long_kv_source_prefix] if self.long_kv_source_prefix is not None else None
            index_k = metadata[self.index_k_source_prefix] if self.index_k_source_prefix is not None else None
            compressor_state = (
                metadata[self.compressor_state_prefix] if self.compressor_state_prefix is not None else None
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
            indexer=(DeepseekV41IndexerMetadata(index_k) if index_k is not None else None),
        )

    @staticmethod
    def _project_q(attn, hidden_states, cos, sin):
        qr = attn.q_norm(attn.wq_a(hidden_states))
        q = attn.wq_b(qr).unflatten(-1, (-1, attn.head_dim))
        if attn.uses_a5_packed_cache:
            q = apply_partial_rotary_inplace(
                q,
                cos,
                sin,
                start=attn.nope_head_dim,
                end=attn.head_dim,
            )
        else:
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                q.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[attn.nope_head_dim, attn.head_dim],
            )
        return q.to(hidden_states.dtype), qr

    @staticmethod
    def _project_kv(attn, hidden_states, cos, sin):
        kv = attn.kv_norm(attn.wkv(hidden_states)).view(-1, 1, attn.head_dim)
        if attn.uses_a5_packed_cache:
            kv = apply_partial_rotary_inplace(
                kv,
                cos,
                sin,
                start=attn.nope_head_dim,
                end=attn.head_dim,
            )
        else:
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[attn.nope_head_dim, attn.head_dim],
            )
        return kv.squeeze(1)

    @classmethod
    def _project_q_kv(cls, attn, hidden_states, cos, sin):
        q, qr = cls._project_q(attn, hidden_states, cos, sin)
        return q, qr, cls._project_kv(attn, hidden_states, cos, sin)

    @staticmethod
    def _write_swa_cache(attn, swa_metadata, kv):
        """Store projected SWA KV with the cache-format-specific writer.

        A5 keeps a persistent flat slot mapping so the native WIN writer can
        run directly on whichever NPU stream produced ``kv``.  The coordinate
        mapping remains the ABI for the A3/legacy ScatterNd path.
        """
        cache = attn.dsa_attn.swa_cache_layer.kv_cache[0]
        if attn.uses_a5_packed_cache:
            write_attention_cache(
                cache,
                swa_metadata.flat_slot_mapping[: kv.shape[0]],
                kv,
                kind="win",
            )
            return
        scatter_cache_sk(cache, swa_metadata.slot_mapping, kv)

    def _update_caches(self, attn, hidden_states, metadata):
        if hidden_states.shape[0] == 0:
            return
        positions = metadata.positions[: hidden_states.shape[0]]
        cos, sin = metadata.rope(attn.rotary_emb.layername, hidden_states.shape[0])
        kv = self._project_kv(attn, hidden_states, cos, sin)
        self._write_swa_cache(attn, metadata.swa, kv)
        if self.role.is_kv_source:
            self._write_compressed_source(attn, hidden_states, positions, cos, sin, metadata)

    def _prepare_inputs_and_caches(self, attn, hidden_states, metadata, metadata_by_prefix):
        """Prepare caches before query work; ordinary preprocessing writes them."""
        pass

    def _prepare_queries(self, attn, hidden_states, positions, cos, sin, metadata):
        hidden_states = hidden_states[: metadata.swa.num_actual_tokens]
        v1_impl = attn.dsa_attn.dsa_attn.impl
        use_multistream = v1_impl.multistream_dsv4_dsa_overlap
        if attn.uses_a5_packed_cache:
            # A5's short projection/cache tail benefits only in decode.  For a
            # mixed or pure-prefill batch, per-layer event and stream-switch
            # overhead is larger than the hidden work.  ``num_prefills`` is a
            # CPU-side scheduler counter, so this branch adds no device sync or
            # graph-visible scalar operators.
            use_multistream = use_multistream and metadata.swa.num_prefills == 0
        preprocess = self.multistream_preprocess if use_multistream else self.preprocess
        q, qr = preprocess(attn, hidden_states, cos, sin, metadata.swa)
        if self.role.is_kv_source:
            self._write_compressed_source(attn, hidden_states, positions, cos, sin, metadata)
        return q, qr

    def _project_output(self, attn, output, hidden_states, metadata, *, projected):
        padded = output
        if output.shape[0] != hidden_states.shape[0]:
            padded = output.new_zeros((hidden_states.shape[0], output.shape[1], output.shape[2]))
            padded[: output.shape[0]] = output
        attn.dsa_attn.dsa_attn.impl._forward_o_proj(padded, projected)
        return projected

    def preprocess(self, attn, hidden_states, cos, sin, swa_metadata):
        """Project Q/KV and populate this layer's SWA cache on the current stream."""
        q, qr, kv = self._project_q_kv(attn, hidden_states, cos, sin)
        self._write_swa_cache(attn, swa_metadata, kv)
        return q, qr

    def multistream_preprocess(self, attn, hidden_states, cos, sin, swa_metadata):
        """Overlap Q Vector work with KV Cube work, then reverse their roles.

        Reuse V1's stream and projection wrappers. V4.1 keeps floating-point
        qr for its indexer and has no post-Wq_b Q RMSNorm. Stage events serialize
        the Cube matmuls; the final join makes SWA writes visible to attention.
        """
        main_stream = torch.npu.current_stream()
        aux_stream = dsv4_dsa_overlap_stream()
        v1_impl = attn.dsa_attn.dsa_attn.impl
        a5_packed = attn.uses_a5_packed_cache
        wq_a, wkv, wq_b = v1_impl.cv_wq_a, v1_impl.cv_wkv, v1_impl.cv_wq_b
        share_quant = (
            type(wq_a._quant_method) is type(wkv._quant_method) and wq_a._has_communication == wkv._has_communication
        )

        # Part 1: Q_a matmul (Cube) overlaps independent KV quantization (Vector).
        q_quant, q_scale = wq_a.quantize(hidden_states)
        kv_quant_done = None
        if share_quant:
            kv_quant, kv_scale = q_quant, q_scale
        else:
            q_quant_done = main_stream.record_event()
            with npu_stream_switch(aux_stream, enabled=True):
                aux_stream.wait_event(q_quant_done)
                kv_quant, kv_scale = wkv.quantize(hidden_states)
                kv_quant_done = aux_stream.record_event()
        q_a = wq_a.matmul(q_quant, q_scale, bias=attn.wq_a.bias)

        # Part 2: Q normalization/quantization (Vector) overlaps KV matmul (Cube).
        part2_start = main_stream.record_event()
        if kv_quant_done is not None:
            main_stream.wait_event(kv_quant_done)
        with npu_stream_switch(aux_stream, enabled=True):
            aux_stream.wait_event(part2_start)
            kv = wkv.matmul(kv_quant, kv_scale, bias=attn.wkv.bias)
            kv_matmul_done = aux_stream.record_event()
        qr = attn.q_norm(q_a)
        q_b_quant, q_b_scale = wq_b.quantize(qr)

        # Part 3: Q_b matmul (Cube) overlaps KV norm, RoPE and cache store
        # (Vector).  A3 delays the KV tail until Q normalization/quantization
        # completes so the two Vector chains do not compete.  On A5 that tail
        # is short enough that the extra per-layer event pair costs more than
        # the protected overlap, so let it follow KV matmul directly.
        part3_start = None if a5_packed else main_stream.record_event()
        main_stream.wait_event(kv_matmul_done)
        with npu_stream_switch(aux_stream, enabled=True):
            if part3_start is not None:
                aux_stream.wait_event(part3_start)
            kv = attn.kv_norm(kv).view(-1, 1, attn.head_dim)
            if a5_packed:
                apply_partial_rotary_inplace(
                    kv,
                    cos,
                    sin,
                    start=attn.nope_head_dim,
                    end=attn.head_dim,
                )
            else:
                torch.ops._C_ascend.inplace_partial_rotary_mul(
                    kv.unsqueeze(1),
                    cos,
                    sin,
                    rotary_mode="interleave",
                    partial_slice=[attn.nope_head_dim, attn.head_dim],
                )
            self._write_swa_cache(attn, swa_metadata, kv.squeeze(1))
        q = wq_b.matmul(q_b_quant, q_b_scale, bias=attn.wq_b.bias).unflatten(-1, (attn.n_local_heads, attn.head_dim))
        main_stream.wait_stream(aux_stream)
        if a5_packed:
            apply_partial_rotary_inplace(
                q,
                cos,
                sin,
                start=attn.nope_head_dim,
                end=attn.head_dim,
            )
        else:
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                q.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[attn.nope_head_dim, attn.head_dim],
            )
        return q.to(hidden_states.dtype), qr

    def _write_compressed_source(
        self,
        attn,
        hidden_states,
        positions,
        cos,
        sin,
        metadata,
    ):
        compressor = attn.compressor
        if compressor is None or metadata.compressor is None or metadata.indexer is None:
            raise RuntimeError("V4.1 KV source is missing compressor or source metadata")
        compressor_metadata = metadata.compressor
        indexer_metadata = metadata.indexer
        ratio = self.role.compress_ratio
        if ratio == 1:
            latent = compressor(hidden_states)
            # C1 source positions are the current token positions. Reuse the
            # query RoPE selected by the SWA metadata builder instead of
            # indexing the global table a second time.
            source_cos = cos
            source_sin = sin
            index_slots = indexer_metadata.cache.slot_mapping[: positions.shape[0]]
            long_slots = compressor_metadata.cache.slot_mapping[: positions.shape[0]]
        else:
            if compressor_metadata.state is None:
                raise RuntimeError("V4.1 ratio-2 source is missing compressor-state metadata")
            state_metadata = compressor_metadata.state
            if state_metadata.c2_ring_metadata is None or state_metadata.c2_metadata_group_id is None:
                raise RuntimeError("V4.1 ring compressor metadata is missing")
            wait_for_device_metadata(DeviceMetadataStage.COMPRESSOR, state_metadata.c2_metadata_group_id)
            if attn.uses_a5_packed_cache:
                latent = compressor.pool_hidden(hidden_states, state_metadata)
            else:
                hidden_states_fp32 = hidden_states.float()
                kv = compressor.wkv(hidden_states_fp32)
                score = compressor.wgate(hidden_states_fp32)
                latent = compressor.pool_projected(kv, score, state_metadata)
            source_cos = state_metadata.c2_source_cos
            source_sin = state_metadata.c2_source_sin
            if source_cos is None or source_sin is None:
                fallback_cos, fallback_sin = get_cos_and_sin_dsa(state_metadata.c2_source_positions)
                source_cos = fallback_cos[attn.rotary_emb.layername]
                source_sin = fallback_sin[attn.rotary_emb.layername]
            source_cos = source_cos[: positions.shape[0]]
            source_sin = source_sin[: positions.shape[0]]
            index_slots = indexer_metadata.cache.slot_mapping[: positions.shape[0]]
            long_slots = compressor_metadata.cache.slot_mapping[: positions.shape[0]]

        if attn.indexer is None:
            raise RuntimeError("V4.1 KV source is missing its indexer")
        attn.indexer.update_keys(
            latent,
            index_slots,
            source_cos,
            source_sin,
        )
        latent = latent.view(-1, 1, attn.head_dim)
        if attn.uses_a5_packed_cache:
            flat_long_slots = compressor_metadata.cache.flat_slot_mapping[: positions.shape[0]]
            latent = apply_partial_rotary_inplace(
                latent,
                source_cos,
                source_sin,
                start=attn.nope_head_dim,
                end=attn.head_dim,
            )
            write_attention_cache(
                attn.long_kv_cache.kv_cache[0],
                flat_long_slots,
                latent.squeeze(1),
                kind="cmp",
            )
        else:
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                latent.unsqueeze(1),
                source_cos,
                source_sin,
                rotary_mode="interleave",
                partial_slice=[attn.nope_head_dim, attn.head_dim],
            )
            scatter_cache_sk(
                attn.long_kv_cache.kv_cache[0],
                long_slots,
                latent.squeeze(1),
            )

    def _select_sparse_indices(self, attn, hidden_states, qr, positions, cos, sin, metadata):
        hidden_states = hidden_states[: metadata.swa.num_actual_tokens]
        if not self.role.has_long_context:
            return None
        shared = attn.shared_state
        if shared is None:
            raise RuntimeError("V4.1 shared attention state is not initialized")
        if not self.role.is_index_source:
            return shared.topk_indices[: hidden_states.shape[0]]
        if attn.indexer is None or metadata.indexer is None:
            raise RuntimeError("V4.1 index source is missing indexer metadata")

        context = get_forward_context().no_compile_layers
        source_layer = context[self.index_k_source_prefix]
        candidate_lengths = getattr(shared, "candidate_lengths", None)
        topk_lengths = getattr(shared, "topk_lengths", None)
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
            candidates=shared.candidates[: hidden_states.shape[0]],
            candidate_lengths=(None if candidate_lengths is None else candidate_lengths[: hidden_states.shape[0]]),
            topk_lengths=(None if topk_lengths is None else topk_lengths[: hidden_states.shape[0]]),
            indices_output=shared.topk_indices[: hidden_states.shape[0]],
        )
        if self.role.is_candidate_source:
            shared.candidates[: candidates.shape[0]].copy_(candidates)
        return selected

    def _attention(self, attn, q, metadata, compressed_indices):
        source_cache = None
        if self.role.has_long_context:
            source_cache = get_forward_context().no_compile_layers[self.long_kv_source_prefix].kv_cache[0]
        if attn.uses_a5_packed_cache:
            compressed_lengths = None
            if self.role.has_long_context and attn.shared_state is not None:
                topk_lengths = getattr(attn.shared_state, "topk_lengths", None)
                if topk_lengths is not None:
                    compressed_lengths = topk_lengths[: q.shape[0]]
            return qsmla(
                q,
                attn.dsa_attn.swa_cache_layer.kv_cache[0],
                source_cache,
                metadata,
                compressed_indices,
                compressed_lengths=compressed_lengths,
                window_size=attn.window_size,
                sinks=attn.attn_sink,
                softmax_scale=attn.softmax_scale,
            )
        return self._native_attention(
            attn,
            q,
            metadata,
            source_cache=source_cache,
            compressed_indices=compressed_indices,
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
            raise ValueError(f"A2/A3 SparseFlashMla requires sliding_window 128, got {attn.window_size}")
        num_heads = q.shape[1]
        if not 1 <= num_heads <= 128 or num_heads & (num_heads - 1):
            raise ValueError(
                "A2/A3 SparseFlashMla requires the local query-head count to be "
                f"a power of two in [1, 128], got {num_heads}"
            )
        has_compressed = self.role.compress_ratio in (1, 2)
        ratio = self.role.compress_ratio if has_compressed else 0
        num_reqs = metadata.swa.num_reqs
        query_start_loc = metadata.swa.query_start_loc[: num_reqs + 1]
        seq_lens = metadata.swa.seq_lens[:num_reqs]
        ori_block_table = metadata.swa.block_table[:num_reqs]
        cmp_block_table = None
        cmp_seq_lens = None
        cmp_residual = None
        cmp_indices = None
        cmp_topk = 0
        if has_compressed:
            if source_cache is None or metadata.attention is None or compressed_indices is None:
                raise RuntimeError("V4.1 compressed attention is missing KV or TopK metadata")
            cmp_block_table = metadata.attention.block_table[:num_reqs]
            cmp_seq_lens = metadata.attention.cache_seq_lens[:num_reqs]
            cmp_residual = metadata.attention.cmp_residual
            cmp_topk = self.topology.index_topk
            if cmp_topk not in (512, 1024):
                raise ValueError(f"SparseFlashMla only supports TopK 512 or 1024, got {cmp_topk}")
            cmp_indices = pad_sparse_indices(compressed_indices, cmp_topk)

        operator_metadata = metadata.attention if has_compressed else metadata.swa
        op_metadata = operator_metadata.smla_metadata
        if op_metadata is None:
            raise RuntimeError(f"V4.1 ratio-{ratio} SMLA metadata was not built")
        wait_for_device_metadata(
            DeviceMetadataStage.ATTENTION,
            id(op_metadata),
        )
        output, _ = sparse_flash_mla(
            q,
            ori_kv=attn.dsa_attn.swa_cache_layer.kv_cache[0],
            cmp_kv=source_cache,
            ori_sparse_indices=metadata.swa.ori_sparse_indices,
            ori_topk_length=metadata.swa.ori_topk_length,
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
            cmp_ratio=_dsa_swa_only_cmp_ratio(ratio, attn.dsa_attn.dsa_attn.impl.vllm_config),
            ori_mask_mode=metadata.swa.ori_mask_mode,
            cmp_mask_mode=3 if has_compressed else 0,
            ori_win_left=metadata.swa.ori_win_left,
            ori_win_right=metadata.swa.ori_win_right,
            layout_q="TND",
            layout_kv="PA_BBND",
            topk_value_mode=1,
            return_softmax_lse=False,
        )
        return output

    @staticmethod
    def update_graph_params(*args, **kwargs):
        """V4.1 owns stable metadata buffers; no backend pointer patch is needed."""
        return None

    def forward(self, attn, positions, hidden_states, output: torch.Tensor | None = None):
        # The custom-op caller provides a graph-stable output buffer.  Write
        # O-projection results into it directly instead of materializing a
        # second full hidden-state tensor and copying it at the graph boundary.
        if output is None:
            output = torch.empty_like(hidden_states)
        forward_context = get_forward_context()
        if forward_context.attn_metadata is None:
            output.zero_()
            return output
        metadata = self._get_layer_metadata(forward_context.attn_metadata)
        self._prepare_inputs_and_caches(attn, hidden_states, metadata, forward_context.attn_metadata)
        num_tokens = metadata.swa.num_actual_tokens
        if num_tokens:
            positions = metadata.positions[:num_tokens]
            cos, sin = metadata.rope(attn.rotary_emb.layername, num_tokens)
            q, qr = self._prepare_queries(attn, hidden_states, positions, cos, sin, metadata)
            compressed_indices = self._select_sparse_indices(attn, hidden_states, qr, positions, cos, sin, metadata)
            attention_output = self._attention(attn, q, metadata, compressed_indices)
            if attn.uses_a5_packed_cache:
                attention_output = apply_partial_rotary_inplace(
                    attention_output,
                    cos,
                    sin,
                    start=attn.nope_head_dim,
                    end=attn.head_dim,
                    inverse=True,
                )
            else:
                torch.ops._C_ascend.inplace_partial_rotary_mul(
                    attention_output.unsqueeze(1),
                    cos,
                    -sin,
                    rotary_mode="interleave",
                    partial_slice=[attn.nope_head_dim, attn.head_dim],
                )
        else:
            heads = attn.n_heads if getattr(attn, "enable_dsa_cp", False) else attn.n_local_heads
            attention_output = hidden_states.new_empty((0, heads, attn.head_dim))
        self._project_output(attn, attention_output, hidden_states, metadata, projected=output)
        return output


class DeepseekV41MetadataBuilder(AttentionMetadataBuilder[DeepseekV41Metadata]):
    def __init__(
        self,
        kv_cache_spec,
        layer_names,
        vllm_config,
        device,
        *,
        build_query_metadata=True,
        build_compressor_metadata=True,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        max_tokens = getattr(vllm_config.scheduler_config, "max_num_batched_tokens", 4096)
        max_reqs = getattr(vllm_config.scheduler_config, "max_num_seqs", 256)
        self._max_reqs = max_reqs
        self._supports_device_ops = getattr(device, "type", "cpu") != "cpu"
        # CP uses global cache-write controls and local query controls. These
        # roles are fixed before allocation and graph capture.
        self._build_query_metadata = build_query_metadata
        self._build_compressor_metadata = build_compressor_metadata
        query_metadata_size = V41_METADATA_BUFFER_SIZE if build_query_metadata else 0
        compressor_tokens = max_tokens if build_compressor_metadata else 0
        compressor_reqs = max_reqs if build_compressor_metadata else 0
        uses_a5_draft = isinstance(kv_cache_spec, DeepseekV41A5DraftSWASpec)
        self._uses_a5_packed_cache = uses_a5_packed_cache(vllm_config) or uses_a5_draft
        self._slot_mapping = torch.full((max_tokens,), -1, dtype=torch.int64, device=device)
        self._slot_mapping_2d = torch.full((max_tokens, 2), -1, dtype=torch.int32, device=device)
        self._flat_slot_mapping = torch.full((max_tokens,), -1, dtype=torch.int32, device=device)
        self._seq_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._cache_seq_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._cmp_residual = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._smla_metadata = torch.zeros(query_metadata_size, dtype=torch.int32, device=device)
        self._qli_metadata = torch.zeros(query_metadata_size, dtype=torch.int32, device=device)
        text_config = vllm_config.model_config.hf_text_config
        window_size = int(_config_value(text_config, "sliding_window", 0))
        if (
            self._supports_device_ops
            and self._uses_a5_packed_cache
            and isinstance(kv_cache_spec, DeepseekV41SWASpec)
            and not is_v41_draft_swa_spec(kv_cache_spec)
            and window_size > 0
        ):
            self._a5_causal_swa_indices = torch.empty(
                (max_tokens, 1, window_size),
                dtype=torch.int32,
                device=device,
            )
            self._a5_causal_swa_lengths = torch.empty(
                (max_tokens, 1),
                dtype=torch.int32,
                device=device,
            )
        else:
            self._a5_causal_swa_indices = None
            self._a5_causal_swa_lengths = None
        if self._supports_device_ops and self._uses_a5_packed_cache:
            blocks = int(torch.npu.get_device_properties(device).cube_core_num)
            self._a5_smla_metadata = torch.empty(
                2 * blocks,
                dtype=torch.int32,
                device=device,
            )
            # The packaged AICPU producer explicitly uses only shape[0]. Keep
            # one address-stable [max_tokens, 1] carrier outside model forward.
            self._a5_smla_length_rows = torch.empty(
                (max_tokens, 1),
                dtype=torch.int32,
                device=device,
            )
        else:
            self._a5_smla_metadata = None
            self._a5_smla_length_rows = None
        self._c2_ring_metadata = torch.zeros(5 * compressor_reqs, dtype=torch.int32, device=device)
        self._c2_complete_mask = torch.zeros(compressor_tokens, dtype=torch.bool, device=device)
        self._c2_source_positions = torch.zeros(compressor_tokens, dtype=torch.int64, device=device)
        rope_dim = int(
            _config_value(
                text_config,
                "qk_rope_head_dim",
                _config_value(text_config, "head_dim"),
            )
        )
        c2_rope_rows = (
            compressor_tokens
            if self._supports_device_ops and isinstance(kv_cache_spec, DeepseekV41CompressorStateSpec)
            else 0
        )
        self._c2_source_cos = torch.ones(
            (c2_rope_rows, 1, 1, rope_dim),
            dtype=torch.float32,
            device=device,
        )
        self._c2_source_sin = torch.zeros_like(self._c2_source_cos)
        self._c2_rope_layer_names = tuple(
            name.removesuffix(".compressor.state_cache") + ".attn"
            for name in layer_names
            if name.endswith(".compressor.state_cache")
        )
        self._c2_full_source_rope: tuple[torch.Tensor, torch.Tensor] | None = None
        self._device_metadata_enabled = False
        self._device_metadata_tasks: tuple[DeviceMetadataTask, ...] = ()

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata,
        **kwargs,
    ) -> DeepseekV41Metadata:
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            **kwargs,
        )

    def build_for_drafting(self, common_attn_metadata, draft_index, **kwargs):
        if not is_v41_draft_swa_spec(self.kv_cache_spec):
            raise TypeError("V4.1 drafting requires a draft SWA cache")
        # DSpark issues one eager block per step. Group-local tables and slots
        # remain independent; the builder owns the operator metadata buffers.
        return self.build(0, common_attn_metadata)

    def enable_device_metadata(self) -> None:
        self._device_metadata_enabled = True
        if self._build_compressor_metadata and isinstance(self.kv_cache_spec, DeepseekV41CompressorStateSpec):
            if not self._c2_rope_layer_names:
                raise RuntimeError("V4.1 compressor-state builder has no source RoPE layer")
            source_rope = get_full_cos_and_sin_dsa_for_layer(self._c2_rope_layer_names[0])
            for rope_layer_name in self._c2_rope_layer_names[1:]:
                other_rope = get_full_cos_and_sin_dsa_for_layer(rope_layer_name)
                if any(other.data_ptr() != source.data_ptr() for other, source in zip(other_rope, source_rope)):
                    raise RuntimeError("V4.1 ratio-2 source layers must share one RoPE table")
            self._c2_full_source_rope = source_rope

    def take_device_metadata_tasks(self) -> tuple[DeviceMetadataTask, ...]:
        tasks = self._device_metadata_tasks
        self._device_metadata_tasks = ()
        return tasks

    def _publish_task(
        self,
        shared: dict[str, Any],
        key: str,
        buffer: torch.Tensor,
        stage: DeviceMetadataStage,
        run,
    ) -> torch.Tensor:
        existing = shared.get(key)
        if existing is not None:
            return existing
        shared[key] = buffer
        if self._device_metadata_enabled:
            self._device_metadata_tasks = (
                *self._device_metadata_tasks,
                DeviceMetadataTask(stage, run, id(buffer)),
            )
        else:
            run()
        return buffer

    def _build_batch_metadata(self, common, num_reqs, num_actual_reqs, num_input_tokens):
        self._seq_lens[:num_reqs].copy_(common.seq_lens[:num_reqs])
        if num_actual_reqs < num_reqs:
            self._seq_lens[num_actual_reqs:num_reqs].zero_()
        seq_lens_cpu = getattr(common, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            seq_lens_cpu = getattr(common, "_seq_lens_cpu", None)
        max_seq_len = int(getattr(common, "max_seq_len", 0))
        if seq_lens_cpu is not None:
            max_seq_len = int(seq_lens_cpu[:num_actual_reqs].max().item()) if num_actual_reqs else 0
        num_decodes, num_decode_tokens, num_prefills, num_prefill_tokens = _request_counts(common, num_reqs)
        positions = common.positions
        if positions is not None:
            positions = positions[:num_input_tokens].long()
        return dict(
            query_start_loc=common.query_start_loc[: num_reqs + 1],
            query_start_loc_cpu=getattr(common, "query_start_loc_cpu", None),
            seq_lens=self._seq_lens[:num_reqs],
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            max_cache_seq_len=max_seq_len,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
        )

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build=False,
        **kwargs,
    ):
        if common_prefix_len:
            raise NotImplementedError("V4.1 prefix caching is not implemented")
        self._device_metadata_tasks = ()
        spec = self.kv_cache_spec
        common = common_attn_metadata
        is_compressor_state = isinstance(spec, DeepseekV41CompressorStateSpec)
        ratio = getattr(spec, "compress_ratio", 1)
        if isinstance(
            spec,
            (DeepseekV41SWASpec, DeepseekV41DraftSWASpec, DeepseekV41A5DraftSWASpec),
        ):
            cache_kind = "swa"
        elif isinstance(spec, DeepseekV41FullSpec):
            cache_kind = "long_kv"
        elif isinstance(spec, DeepseekV41IndexerSpec):
            cache_kind = "index_k"
        elif is_compressor_state:
            cache_kind = "compressor_state"
        else:
            raise TypeError(f"Unsupported V4.1 cache spec: {type(spec).__name__}")

        num_reqs = int(getattr(common, "num_reqs", common.seq_lens.shape[0]))
        num_actual_reqs = int(kwargs.get("num_actual_reqs", num_reqs))
        num_actual_reqs = min(num_actual_reqs, num_reqs)
        num_input_tokens = int(getattr(common, "num_input_tokens", common.slot_mapping.shape[0]))
        num_actual_tokens = int(getattr(common, "num_actual_tokens", num_input_tokens))
        shared = kwargs.get("common_v41_metadata")
        if shared is None:
            shared = {}
        batch_shared = kwargs.get("common_v41_batch_metadata")
        if batch_shared is None:
            batch_shared = shared

        # The runner resets both dictionaries on each build. Batch values do
        # not depend on physical block IDs; slot mappings remain group-local.
        batch_metadata = batch_shared.get("batch")
        if batch_metadata is None:
            batch_metadata = self._build_batch_metadata(common, num_reqs, num_actual_reqs, num_input_tokens)
            batch_shared["batch"] = batch_metadata
        coordinates = dict(batch_metadata)
        seq_lens = coordinates["seq_lens"]
        positions = coordinates["positions"]
        full_graph_mode = coordinates.get("full_graph_mode", False)

        # SWA uses original-token coordinates; circular state has no token slots.
        # Long KV and index K are addressed in completed compression groups.
        compressed = cache_kind in {"long_kv", "index_k"}
        if is_compressor_state:
            # State writes use ring ownership metadata; this buffer stays PAD.
            slots = self._slot_mapping[:num_input_tokens]
            flat_slots = None
        else:
            # Scope ``shared`` to one framework KV cache group in the model
            # runner. Long KV and Indexer builders with the same physical
            # layout share persistent [T, 2] and [T] mappings, while every SWA
            # group owns distinct mapping buffers.  The coordinate form remains
            # the ScatterNd/index-writer ABI; native WIN/CMP writers consume the
            # linear form directly without per-layer address arithmetic.
            slot_key = f"slot:coordinates:c{ratio}:b{spec.storage_block_size}"
            flat_slot_key = f"slot:flat:c{ratio}:b{spec.storage_block_size}"
            prepared_slots = shared.get(slot_key)
            prepared_flat_slots = shared.get(flat_slot_key)
            if prepared_slots is None or prepared_flat_slots is None:
                active_slots = common.slot_mapping[:num_input_tokens]
                if self._uses_a5_packed_cache and active_slots.device.type == "npu":
                    prepared_slots, prepared_flat_slots = build_a5_slot_mapping(
                        active_slots,
                        common.positions[:num_input_tokens],
                        common.query_start_loc,
                        num_input_tokens,
                        num_actual_reqs,
                        num_actual_tokens,
                        spec.storage_block_size,
                        ratio if compressed else 1,
                        skip_update=kwargs.get("skip_ring_state_update", False),
                        coordinates_output=self._slot_mapping_2d[:num_input_tokens],
                        flat_output=self._flat_slot_mapping[:num_input_tokens],
                    )
                    shared[slot_key] = prepared_slots
                    shared[flat_slot_key] = prepared_flat_slots
                else:
                    if compressed and ratio != 1:
                        active_slots = compressed_slot_mapping(active_slots, ratio)
                    valid = active_slots >= 0
                    if compressed and ratio == 2:
                        # Match the ring compressor's completion policy and keep
                        # padded graph-capture rows from writing cache.
                        if kwargs.get("skip_ring_state_update", False):
                            valid.zero_()
                        else:
                            valid_end = common.query_start_loc[num_actual_reqs].clamp_max(num_actual_tokens)
                            valid &= torch.arange(num_input_tokens, device=active_slots.device) < valid_end
                            if common.positions is not None:
                                valid &= common.positions[:num_input_tokens].remainder(2) == 1
                    physical = active_slots.clamp_min(0)
                    if prepared_slots is None:
                        self._slot_mapping_2d[:num_input_tokens, 0].copy_(
                            torch.where(
                                valid,
                                torch.div(
                                    physical,
                                    spec.storage_block_size,
                                    rounding_mode="floor",
                                ),
                                -1,
                            )
                        )
                        self._slot_mapping_2d[:num_input_tokens, 1].copy_(
                            torch.where(
                                valid,
                                physical.remainder(spec.storage_block_size),
                                -1,
                            )
                        )
                        prepared_slots = self._slot_mapping_2d[:num_input_tokens]
                        shared[slot_key] = prepared_slots
                    if prepared_flat_slots is None:
                        self._flat_slot_mapping[:num_input_tokens].copy_(torch.where(valid, active_slots, -1))
                        prepared_flat_slots = self._flat_slot_mapping[:num_input_tokens]
                        shared[flat_slot_key] = prepared_flat_slots
            slots = prepared_slots
            flat_slots = prepared_flat_slots
        plane_ratio = ratio if compressed else 1
        coordinates["cache_seq_lens"] = seq_lens
        cmp_residual_buffer = None
        if compressed and ratio == 2:
            compressed_lengths = batch_shared.get("lengths:c2")
            if compressed_lengths is None:
                torch.div(seq_lens, ratio, rounding_mode="floor", out=self._cache_seq_lens[:num_reqs])
                torch.remainder(seq_lens, ratio, out=self._cmp_residual[:num_reqs])
                compressed_lengths = (self._cache_seq_lens[:num_reqs], self._cmp_residual[:num_reqs])
                batch_shared["lengths:c2"] = compressed_lengths
            coordinates["cache_seq_lens"], cmp_residual_buffer = compressed_lengths
        coordinates["max_cache_seq_len"] //= plane_ratio
        cos = sin = None
        if cache_kind == "swa" and positions is not None:
            rope = kwargs.get("rope_views")
            if rope is None:
                rope = batch_shared.get("rope")
            if rope is None:
                rope = get_cos_and_sin_dsa(positions, use_cache=(coordinates["num_prefills"] == 0 or full_graph_mode))
                batch_shared["rope"] = rope
            cos, sin = rope
        text_config = self.vllm_config.model_config.hf_text_config
        window_size = int(_config_value(text_config, "sliding_window", 0))
        n_local_heads = (
            int(_config_value(text_config, "num_attention_heads"))
            // self.vllm_config.parallel_config.tensor_parallel_size
        )
        n_local_heads = int(kwargs.get("num_query_heads", n_local_heads))
        head_dim = int(_config_value(text_config, "head_dim"))
        index_topk = int(_config_value(text_config, "index_topk"))
        ori_sparse_indices = kwargs.get("ori_sparse_indices")
        ori_topk_length = None
        noncausal = not bool(getattr(common, "causal", True))
        if noncausal and not is_v41_draft_swa_spec(spec):
            raise ValueError("V4.1 noncausal attention requires a DSpark draft SWA cache")
        if noncausal and ori_sparse_indices is None:
            ori_sparse_indices, _ = build_dspark_swa_indices(
                common.block_table_tensor[:num_reqs],
                self.vllm_config.speculative_config.num_speculative_tokens,
                window_size,
                spec.storage_block_size,
                common.query_start_loc[: num_reqs + 1],
                seq_lens,
                num_actual_tokens,
                use_logical_indices=True,
            )
        elif (
            not noncausal
            and ori_sparse_indices is None
            and self._a5_causal_swa_indices is not None
            and positions is not None
        ):
            cache_key = f"a5-causal-swa:w{window_size}"
            cached_window = batch_shared.get(cache_key)
            if cached_window is None:
                cached_window = build_window_indices(
                    positions[:num_input_tokens],
                    window_size,
                    indices_output=self._a5_causal_swa_indices[:num_input_tokens],
                    lengths_output=self._a5_causal_swa_lengths[:num_input_tokens],
                )
                batch_shared[cache_key] = cached_window
            ori_sparse_indices, ori_topk_length = cached_window
        if ori_topk_length is None and ori_sparse_indices is not None and noncausal:
            ori_topk_length = (ori_sparse_indices >= 0).sum(dim=-1, dtype=torch.int32)
        ori_mask_mode = 0 if noncausal else 4
        # With explicit DSpark indices, mode 0 makes those indices the complete
        # visibility contract. The packaged SparseFlashMla ABI requires both
        # window attributes to be disabled for every mode other than 4.
        ori_win_left = -1 if noncausal else max(0, window_size - 1)
        ori_win_right = -1 if noncausal else 0
        operator_ratio = 0 if cache_kind == "swa" else ratio
        uses_draft_bf16_swa = isinstance(spec, DeepseekV41DraftSWASpec)
        uses_draft_a5_swa = isinstance(spec, DeepseekV41A5DraftSWASpec)
        smla_metadata = None
        qli_metadata = None

        if (
            self._uses_a5_packed_cache
            and cache_kind == "swa"
            and not uses_draft_bf16_swa
            and self._a5_smla_metadata is not None
            and self._a5_smla_length_rows is not None
        ):

            def build_a5_smla_metadata() -> None:
                if num_actual_tokens == 0:
                    self._a5_smla_metadata.zero_()
                    return
                import_packaged_a5_module("cann_ops_transformer.ops.attention.mixed_quant_sparse_flash_mla_dsl")
                # The SFA call consumes only the real query rows.  Graph padding
                # remains in the address-stable carrier, but must not participate
                # in the AICPU task split or kernels will be scheduled past Q.
                length_rows = self._a5_smla_length_rows[:num_actual_tokens]
                value = torch.ops.cann_ops_transformer.ds41.mixed_quant_sparse_flash_mla_metadata(
                    length_rows,
                    length_rows,
                    num_heads_q=64,
                    num_heads_kv=1,
                    head_dim=512,
                    quant_mode=1,
                    layout_q="TND",
                    layout_kv="PA_BBND",
                    has_win_kv=True,
                    has_cmp_kv=not uses_draft_a5_swa,
                )
                self._a5_smla_metadata.copy_(value)

            smla_metadata = self._publish_task(
                batch_shared,
                "smla:a5",
                self._a5_smla_metadata,
                DeviceMetadataStage.ATTENTION,
                build_a5_smla_metadata,
            )

        # A5 adapters build their packaged metadata at the operator boundary,
        # and the eager golden needs no opaque C++ metadata.  Do not call the
        # legacy A3 extension merely because the cache tensor lives on NPU.
        if (
            self._build_query_metadata
            and self._supports_device_ops
            and (uses_draft_bf16_swa or not self._uses_a5_packed_cache)
            and cache_kind in {"swa", "long_kv"}
        ):
            has_compressed = operator_ratio in (1, 2)
            cmp_seq_lens = coordinates["cache_seq_lens"] if has_compressed else None
            cmp_residual = cmp_residual_buffer

            def build_smla_metadata() -> None:
                # Keep graph event frontiers stable even when this CP rank has no query.
                if num_actual_tokens == 0:
                    self._smla_metadata.zero_()
                    return
                value = sparse_flash_mla_metadata(
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=head_dim,
                    cu_seqlens_q=common.query_start_loc[: num_reqs + 1].int(),
                    seqused_ori_kv=seq_lens,
                    seqused_cmp_kv=cmp_seq_lens,
                    cmp_residual_kv=cmp_residual,
                    batch_size=num_reqs,
                    max_seqlen_q=int(getattr(common, "max_query_len", 0)),
                    max_seqlen_ori_kv=int(getattr(common, "max_seq_len", 0)),
                    max_seqlen_cmp_kv=(coordinates["max_cache_seq_len"] if has_compressed else 0),
                    ori_topk=ori_sparse_indices.shape[-1] if ori_sparse_indices is not None else 0,
                    ori_topk_length=ori_topk_length,
                    cmp_topk=index_topk if has_compressed else 0,
                    cmp_ratio=_dsa_swa_only_cmp_ratio(operator_ratio, self.vllm_config),
                    ori_mask_mode=ori_mask_mode,
                    cmp_mask_mode=3 if has_compressed else 0,
                    ori_win_left=ori_win_left,
                    ori_win_right=ori_win_right,
                    layout_q="TND",
                    layout_kv="PA_BBND",
                    has_ori_kv=True,
                    has_cmp_kv=has_compressed,
                )
                self._smla_metadata.copy_(value)

            smla_metadata = self._publish_task(
                batch_shared,
                f"smla:c{operator_ratio}",
                self._smla_metadata,
                DeviceMetadataStage.ATTENTION,
                build_smla_metadata,
            )

        if (
            self._build_query_metadata
            and self._supports_device_ops
            and not self._uses_a5_packed_cache
            and cache_kind == "index_k"
        ):
            residual = cmp_residual_buffer

            def build_qli_metadata() -> None:
                value = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
                    int(_config_value(text_config, "index_n_heads")),
                    1,
                    int(_config_value(text_config, "index_head_dim")),
                    index_topk,
                    2,
                    cu_seqlens_q=common.query_start_loc[: num_reqs + 1].int(),
                    seqused_k=coordinates["cache_seq_lens"],
                    cmp_residual_k=residual,
                    batch_size=num_reqs,
                    max_seqlen_q=int(getattr(common, "max_query_len", 0)),
                    max_seqlen_k=coordinates["max_cache_seq_len"],
                    layout_q="TND",
                    layout_k="PA_BBND",
                    mask_mode=3,
                    cmp_ratio=ratio,
                )
                self._qli_metadata.copy_(value)

            qli_metadata = self._publish_task(
                batch_shared,
                f"qli:c{ratio}",
                self._qli_metadata,
                DeviceMetadataStage.INDEXER,
                build_qli_metadata,
            )

        c2_ring_metadata = None
        c2_complete_mask = None
        c2_source_positions = None
        c2_source_cos = None
        c2_source_sin = None
        c2_metadata_group_id = None
        if self._build_compressor_metadata and cache_kind == "compressor_state" and positions is not None:
            ring_meta = self._c2_ring_metadata[: 5 * num_reqs].view(5, num_reqs)
            input_positions = positions
            if self._supports_device_ops:
                if self._c2_full_source_rope is None:
                    raise RuntimeError("V4.1 source RoPE buffers were not initialized")
                full_source_cos, full_source_sin = self._c2_full_source_rope
            else:
                full_source_cos = full_source_sin = None

            def build_c2_metadata() -> None:
                starts = common.query_start_loc[:num_reqs].int()
                ends = common.query_start_loc[1 : num_reqs + 1].int()
                query_lens = ends - starts
                live = torch.arange(num_reqs, device=starts.device) < num_actual_reqs
                used = (ends.clamp_max(num_actual_tokens) - starts).clamp_min(0)
                used = torch.where(live, used, 0)
                if kwargs.get("skip_ring_state_update", False):
                    used = torch.zeros_like(used)
                ring_meta[0].copy_((seq_lens - query_lens).clamp_min(0))
                ring_meta[1].copy_(used)
                ring_meta[2].copy_(starts)
                ring_meta[3].copy_(starts)
                ring_meta[4].copy_(torch.where(used > 0, common.block_table_tensor[:num_reqs, 0], 0))
                valid_end = common.query_start_loc[num_actual_reqs].clamp_max(num_actual_tokens)
                valid = torch.arange(num_input_tokens, device=input_positions.device) < valid_end
                complete = (input_positions.remainder(2) == 1) & valid
                if kwargs.get("skip_ring_state_update", False):
                    complete = torch.zeros_like(complete)
                self._c2_complete_mask[:num_input_tokens].copy_(complete)
                self._c2_source_positions[:num_input_tokens].copy_(
                    torch.where(
                        complete,
                        input_positions - 1,
                        torch.zeros_like(input_positions),
                    )
                )
                if full_source_cos is not None and full_source_sin is not None:
                    gather_idx = (
                        self._c2_source_positions[:num_input_tokens]
                        .reshape(-1, 1, 1, 1)
                        .expand(
                            num_input_tokens,
                            1,
                            1,
                            full_source_cos.shape[-1],
                        )
                    )
                    torch.gather(
                        full_source_cos,
                        0,
                        gather_idx,
                        out=self._c2_source_cos[:num_input_tokens],
                    )
                    torch.gather(
                        full_source_sin,
                        0,
                        gather_idx,
                        out=self._c2_source_sin[:num_input_tokens],
                    )

            compressor_group = self._publish_task(
                shared,
                "c2:compressor",
                self._c2_complete_mask,
                DeviceMetadataStage.COMPRESSOR,
                build_c2_metadata,
            )
            if compressor_group is not self._c2_complete_mask:
                raise RuntimeError("V4.1 compressor metadata must have one owner")
            c2_complete_mask = self._c2_complete_mask[:num_input_tokens]
            c2_ring_metadata = ring_meta
            c2_source_positions = self._c2_source_positions[:num_input_tokens]
            if self._supports_device_ops:
                c2_source_cos = self._c2_source_cos[:num_input_tokens]
                c2_source_sin = self._c2_source_sin[:num_input_tokens]
            c2_metadata_group_id = id(self._c2_complete_mask)
        output_block_table = common.block_table_tensor[:num_reqs]
        return DeepseekV41Metadata(
            block_table=output_block_table,
            block_table_cpu=(
                common.block_table_cpu[:num_reqs] if getattr(common, "block_table_cpu", None) is not None else None
            ),
            slot_mapping=slots,
            flat_slot_mapping=flat_slots,
            compress_ratio=ratio,
            storage_block_size=spec.storage_block_size,
            is_compressor_state=is_compressor_state,
            cache_kind=cache_kind,
            cos=cos,
            sin=sin,
            num_actual_tokens=num_actual_tokens,
            num_input_tokens=num_input_tokens,
            num_reqs=num_reqs,
            num_actual_reqs=num_actual_reqs,
            logical_block_size=spec.block_size,
            max_query_len=int(getattr(common, "max_query_len", 0)),
            max_seq_len=int(getattr(common, "max_seq_len", 0)),
            attn_state=getattr(common, "attn_state", None),
            is_prefilling=getattr(common, "is_prefilling", None),
            causal=getattr(common, "causal", True),
            ori_sparse_indices=ori_sparse_indices,
            ori_topk_length=ori_topk_length,
            ori_mask_mode=ori_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            smla_metadata=smla_metadata,
            qli_metadata=qli_metadata,
            cmp_residual=cmp_residual_buffer,
            c2_ring_metadata=c2_ring_metadata,
            c2_complete_mask=c2_complete_mask,
            c2_source_positions=c2_source_positions,
            c2_source_cos=c2_source_cos,
            c2_source_sin=c2_source_sin,
            c2_metadata_group_id=c2_metadata_group_id,
            **coordinates,
        )


class DeepseekV41CacheBackend(AttentionBackend):
    """Cache-only backend: supplies layout and metadata, not an AttentionImpl."""

    @staticmethod
    def get_name():
        return "ASCEND_DSA_V41_CACHE"

    @staticmethod
    def get_impl_cls():
        return DeepseekV41EagerAttentionImpl

    @staticmethod
    def get_builder_cls():
        from vllm_ascend.attention.context_parallel.dsa_v41_cp import get_v41_cp_classes

        return get_v41_cp_classes()[0]

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

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

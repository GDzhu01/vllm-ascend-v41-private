# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 text model and source-shared hybrid-cache graph."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from vllm.distributed import get_pp_group

from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheBackend,
    DeepseekV41CacheLayer,
    DeepseekV41EagerAttentionImpl,
)
from vllm_ascend.core.deepseek_v41 import (
    DeepseekV41FullSpec,
    DeepseekV41SWASpec,
    validate_cache_runtime,
)
from vllm_ascend.models.deepseek_v4.model import (
    AscendDeepseekV4ForCausalLM,
    AscendDeepseekV4SWACache,
    DeepseekV2DecoderLayer,
    DeepseekV4Attention,
    DeepseekV4Model,
)

from .compressor import DeepseekV41Compressor, _read, text_config_of
from .indexer import DeepseekV41Indexer


@dataclass(frozen=True)
class DeepseekV41LayerRole:
    """The attention and future Engram responsibilities of one backbone layer."""

    layer_idx: int
    compress_ratio: int
    kv_source_layer: int | None
    index_source_layer: int | None
    is_kv_source: bool
    is_index_source: bool
    is_candidate_source: bool
    uses_candidate_filter: bool
    engram_slot: int | None

    @property
    def has_long_context(self) -> bool:
        return self.compress_ratio > 0


@dataclass(frozen=True)
class DeepseekV41Topology:
    """Validated, immutable model-wide source/consumer topology."""

    layers: tuple[DeepseekV41LayerRole, ...]
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    candidate_source_layer: int
    candidate_topk_blocks: int
    candidate_block_size: int
    index_topk: int

    def layer(self, layer_idx: int) -> DeepseekV41LayerRole:
        return self.layers[layer_idx]

    def kv_consumers(self, source_layer: int) -> tuple[int, ...]:
        return tuple(role.layer_idx for role in self.layers if role.kv_source_layer == source_layer)

    def index_consumers(self, source_layer: int) -> tuple[int, ...]:
        return tuple(role.layer_idx for role in self.layers if role.index_source_layer == source_layer)


class DeepseekV41SharedAttentionState:
    """Per-forward handoff between index sources and their consumer layers."""

    def __init__(self, topk_indices, candidates):
        self.topk_indices = topk_indices
        self.candidates = candidates

    def reset(self):
        # Source layers overwrite the active rows before any consumer reads
        # them. Keeping the storage intact avoids replay depending on Python
        # state mutation and preserves a fixed address for ACL Graph.
        return None


def _as_int_tuple(config: Any, name: str) -> tuple[int, ...]:
    value = _read(config, name)
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, int) for item in value):
        raise ValueError(f"DeepSeek V4.1 {name} must be a list of integers")
    return tuple(value)


def _latest_source(layer_idx: int, sources: tuple[int, ...]) -> int | None:
    return next((source for source in reversed(sources) if source <= layer_idx), None)


def build_layer_plan(config: Any) -> DeepseekV41Topology:
    """Build and validate the V4.1 layer-sharing graph from a text config.

    ``config`` may be a Transformers config object or the raw ``text_config``
    dictionary.  Extra compression ratios for speculative layers are allowed,
    but only the first ``num_hidden_layers`` entries describe the backbone.
    """

    config = text_config_of(config)
    num_layers = int(_read(config, "num_hidden_layers"))
    ratios = _as_int_tuple(config, "compress_ratios")
    kv_sources = _as_int_tuple(config, "kv_source_layers")
    index_sources = _as_int_tuple(config, "index_source_layers")
    engram_layers = _as_int_tuple(config, "engram_layer_ids")
    candidate_source = int(_read(config, "candidate_source_layer"))
    candidate_topk_blocks = int(_read(config, "candidate_topk_blocks"))
    candidate_block_size = int(_read(config, "candidate_block_size"))
    index_topk = int(_read(config, "index_topk"))

    if num_layers <= 0:
        raise ValueError("DeepSeek V4.1 num_hidden_layers must be positive")
    if len(ratios) < num_layers:
        raise ValueError(
            "DeepSeek V4.1 compress_ratios must cover every backbone layer: "
            f"got {len(ratios)} ratios for {num_layers} layers"
        )
    ratios = ratios[:num_layers]
    if any(ratio not in (0, 1, 2) for ratio in ratios):
        raise ValueError(f"DeepSeek V4.1 backbone only supports compression ratios 0, 1 and 2; got {ratios}")

    for name, sources in (("kv_source_layers", kv_sources), ("index_source_layers", index_sources)):
        if tuple(sorted(set(sources))) != sources:
            raise ValueError(f"DeepSeek V4.1 {name} must be sorted and unique")
        if any(source < 0 or source >= num_layers for source in sources):
            raise ValueError(f"DeepSeek V4.1 {name} contains a layer outside the backbone")
        if any(ratios[source] == 0 for source in sources):
            raise ValueError(f"DeepSeek V4.1 {name} cannot point to a local-only layer")

    if not set(kv_sources).issubset(index_sources):
        raise ValueError("Every DeepSeek V4.1 KV source must also be an index source")
    if candidate_source not in kv_sources:
        raise ValueError("DeepSeek V4.1 candidate_source_layer must be a KV source")
    if candidate_topk_blocks <= 0 or candidate_block_size <= 0 or index_topk <= 0:
        raise ValueError("DeepSeek V4.1 candidate and index TopK values must be positive")
    if len(set(engram_layers)) != len(engram_layers):
        raise ValueError("DeepSeek V4.1 engram_layer_ids must be unique")
    if any(layer < 0 or layer >= num_layers for layer in engram_layers):
        raise ValueError("DeepSeek V4.1 engram_layer_ids contains a layer outside the backbone")

    engram_slots = {layer_idx: slot for slot, layer_idx in enumerate(engram_layers)}
    roles: list[DeepseekV41LayerRole] = []
    for layer_idx, ratio in enumerate(ratios):
        kv_source = _latest_source(layer_idx, kv_sources) if ratio else None
        index_source = _latest_source(layer_idx, index_sources) if ratio else None
        if ratio and (kv_source is None or index_source is None):
            raise ValueError(f"DeepSeek V4.1 layer {layer_idx} has long-context attention but no source layer")
        if kv_source is not None and ratios[kv_source] != ratio:
            raise ValueError(
                f"DeepSeek V4.1 layer {layer_idx} has ratio {ratio}, but its KV source "
                f"layer {kv_source} has ratio {ratios[kv_source]}"
            )

        roles.append(
            DeepseekV41LayerRole(
                layer_idx=layer_idx,
                compress_ratio=ratio,
                kv_source_layer=kv_source,
                index_source_layer=index_source,
                is_kv_source=layer_idx in kv_sources,
                is_index_source=layer_idx in index_sources,
                is_candidate_source=layer_idx == candidate_source,
                # Consumer layers inherit the selection policy of their index
                # source.  For example, layer 26 reuses layer 24 TopK, and that
                # TopK was computed inside layer 20's candidate blocks.
                uses_candidate_filter=index_source is not None and index_source > candidate_source,
                engram_slot=engram_slots.get(layer_idx),
            )
        )

    return DeepseekV41Topology(
        layers=tuple(roles),
        kv_source_layers=kv_sources,
        index_source_layers=index_sources,
        candidate_source_layer=candidate_source,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        index_topk=index_topk,
    )


class AscendDeepseekV41SWACache(AscendDeepseekV4SWACache):
    """V4 execution-compatible SWA plane participating in V4.1 grouping."""

    def get_kv_cache_spec(self, vllm_config):
        spec = super().get_kv_cache_spec(vllm_config)
        return DeepseekV41SWASpec(
            block_size=spec.block_size,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            dtype=spec.dtype,
            sliding_window=spec.sliding_window,
            cache_dtype_str=spec.cache_dtype_str,
            model_version="deepseek_v4",
            alignment=spec.alignment,
        )

    def get_attn_backend(self):
        return DeepseekV41CacheBackend


class DeepseekV41Attention(DeepseekV4Attention):
    """V4 projections plus V4.1 source-owned cache and fused DSA execution."""

    swa_cache_cls = AscendDeepseekV41SWACache

    def __init__(
        self,
        vllm_config,
        config,
        max_position_embeddings=0,
        cache_config=None,
        quant_config=None,
        prefix="",
        topk_indices_buffer=None,
    ):
        config = text_config_of(config)
        validate_cache_runtime(vllm_config)
        layer_idx = int(prefix.split(".")[-2])
        topology = build_layer_plan(config)
        role = topology.layer(layer_idx)
        # Reuse V4's quant-aware projections and stable SWA eager backend.  A
        # zero ratio prevents V4 from creating its incompatible c4/c128 planes.
        original_ratios = config.compress_ratios
        config.compress_ratios = tuple(0 for _ in original_ratios)
        try:
            super().__init__(
                vllm_config=vllm_config,
                config=config,
                max_position_embeddings=max_position_embeddings,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            )
        finally:
            config.compress_ratios = original_ratios
        from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding

        # V4.1 applies YaRN only to layers carrying long-context compressed KV.
        # Pure SWA layers use the unscaled base RoPE even though the allocated
        # lookup table still spans the configured maximum context length.
        self.rotary_emb = ComplexExpRotaryEmbedding(
            vllm_config=vllm_config,
            layername=f"{prefix}.attn",
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            is_neox_style=False,
            scaling_factor=config.rope_parameters["factor"],
            base=(config.compress_rope_theta if role.has_long_context else config.rope_theta),
            beta_fast=config.rope_parameters["beta_fast"],
            beta_slow=config.rope_parameters["beta_slow"],
            original_seq_len=(max_position_embeddings if role.has_long_context else 0),
            rope_groups=["default"],
        )
        block_size = vllm_config.cache_config.block_size
        if block_size <= 0 or block_size % 2:
            raise ValueError("V4.1 logical block_size must be a positive multiple of two")
        owned = []
        if role.is_kv_source:
            owned.extend((f"{prefix}.long_kv_cache", f"{prefix}.indexer.k_cache"))
            if role.compress_ratio == 2:
                owned.append(f"{prefix}.compressor.state_cache")
        duplicates = set(owned) & vllm_config.compilation_config.static_forward_context.keys()
        if duplicates:
            raise ValueError(f"Duplicate V4.1 cache prefixes: {sorted(duplicates)}")
        self.role = role
        self.topology = topology
        self.shared_state = None
        self.prefix = prefix
        width = _read(config, "head_dim")
        self.softmax_scale = width**-0.5
        if role.is_kv_source:
            self.long_kv_cache = DeepseekV41CacheLayer(
                vllm_config,
                f"{prefix}.long_kv_cache",
                DeepseekV41FullSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=width,
                    dtype=torch.bfloat16,
                    compress_ratio=role.compress_ratio,
                ),
            )
        self.compressor = (
            DeepseekV41Compressor(config, role.compress_ratio, vllm_config, f"{prefix}.compressor")
            if role.is_kv_source
            else None
        )
        self.indexer = (
            DeepseekV41Indexer(
                config,
                role.is_kv_source,
                vllm_config,
                f"{prefix}.indexer",
                role.compress_ratio,
                quant_config=quant_config,
            )
            if role.is_index_source
            else None
        )
        root = prefix.rsplit(".layers.", 1)[0]
        source = f"{root}.layers.{role.kv_source_layer}.self_attn"
        self.long_kv_source_prefix = f"{source}.long_kv_cache" if role.has_long_context else None
        self.index_k_source_prefix = f"{source}.indexer.k_cache" if role.has_long_context else None
        self.index_source_layer = role.index_source_layer
        self.v41_impl = DeepseekV41EagerAttentionImpl(
            prefix=prefix,
            role=role,
            topology=topology,
            long_kv_source_prefix=self.long_kv_source_prefix,
            index_k_source_prefix=self.index_k_source_prefix,
        )
        self.v41_layer_name = f"{prefix}.v41_attn"
        context = vllm_config.compilation_config.static_forward_context
        if self.v41_layer_name in context:
            raise ValueError(f"Duplicate V4.1 attention layer: {self.v41_layer_name}")
        context[self.v41_layer_name] = self

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        output = torch.empty_like(hidden_states)
        torch.ops.vllm.dsa_v41_forward(hidden_states, output, self.v41_layer_name)
        return output


class DeepseekV41DecoderLayer(DeepseekV2DecoderLayer):
    """V4.1 block with the checkpoint's delayed mHC coefficient handoff."""

    attention_cls = DeepseekV41Attention

    @staticmethod
    def hc_collapse(x, pre_mix):
        return (pre_mix.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)

    def hc_pre(self, x, hc_fn, hc_scale, hc_base, pre_mix=None):
        return torch.ops._C_ascend.npu_hc_pre_v2(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            pre_mix,
            hc_mult=self.hc_mult,
            hc_sinkhorn_iters=self.hc_sinkhorn_iters,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
        )

    def hc_post(self, x, residual, post, comb):
        return torch.ops._C_ascend.npu_hc_post(
            x.unsqueeze(0),
            residual.unsqueeze(0),
            post.unsqueeze(0),
            comb.unsqueeze(0),
        ).squeeze(0)

    def forward(
        self,
        positions,
        hidden_states,
        pre_mix,
        llama_4_scaling=None,
        input_ids=None,
    ):
        residual = hidden_states
        x, attn_post, attn_comb, attn_pre = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            pre_mix,
        )
        x = self.input_layernorm(x)
        x = self.self_attn(positions, x, llama_4_scaling)
        hidden_states = self.hc_post(x, residual, attn_post, attn_comb)

        residual = hidden_states
        x, ffn_post, ffn_comb, ffn_pre = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            attn_pre,
        )
        x = self.post_attention_layernorm(x)
        x = self.mlp(x, input_ids)
        hidden_states = self.hc_post(x, residual, ffn_post, ffn_comb)
        return hidden_states, ffn_pre


class DeepseekV41Model(DeepseekV4Model):
    """Single V4.1 backbone entry, matching ``deepseek_v4/model.py``."""

    decoder_layer_cls = DeepseekV41DecoderLayer

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # V4.1 collapses with the last block's ffn_pre; it has no hc_head
        # projection in the checkpoint.
        del self.hc_head_fn, self.hc_head_base, self.hc_head_scale, self.hc_norm
        topology = build_layer_plan(self.config)
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        candidate_buffer = torch.full(
            (max_tokens, 1, topology.candidate_topk_blocks),
            -1,
            dtype=torch.int32,
            device=self.topk_indices_buffer.device,
        )
        self.candidate_indices_buffer = candidate_buffer
        self.shared_attention_state = DeepseekV41SharedAttentionState(
            self.topk_indices_buffer,
            candidate_buffer,
        )
        for layer in self.layers:
            if isinstance(layer, DeepseekV41DecoderLayer):
                layer.self_attn.shared_state = self.shared_attention_state

    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None):
        if not get_pp_group().is_first_rank or not get_pp_group().is_last_rank:
            raise NotImplementedError("V4.1 eager milestone currently requires PP=1")
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        self.shared_attention_state.reset()
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        pre_mix = hidden_states.new_zeros(hidden_states.shape[0], self.hc_mult, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
        last_layer = None
        aux_hidden_states = []
        for layer in self.layers:
            last_layer = layer
            # DSpark consumes the residual stream entering its configured
            # target layers. The runner expresses checkpoint IDs as one-based.
            if layer.layer_idx + 1 in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states.mean(dim=1))
            hidden_states, pre_mix = layer(positions, hidden_states, pre_mix, None, input_ids=input_ids)
        assert last_layer is not None
        hidden_states = last_layer.hc_collapse(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class AscendDeepseekV41ForCausalLM(AscendDeepseekV4ForCausalLM):
    model_cls = DeepseekV41Model
    # Engram is intentionally disabled until its tables can be HBM-sharded.
    _DEFERRED_WEIGHT_MARKERS = (".engram.",)
    _DEFERRED_WEIGHT_PREFIXES = ("aligner.", "vision.", "image_", "mtp.")

    @classmethod
    def _is_milestone_weight(cls, name):
        return not name.startswith(cls._DEFERRED_WEIGHT_PREFIXES) and not any(
            marker in name for marker in cls._DEFERRED_WEIGHT_MARKERS
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def milestone_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                if self._is_milestone_weight(name):
                    yield name, tensor

        return super().load_weights(milestone_weights())

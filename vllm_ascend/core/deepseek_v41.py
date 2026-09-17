# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Framework-side V4.1 cache specs and layer-outermost hybrid allocation."""

from dataclasses import dataclass, replace

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.core.kv_cache_utils import may_override_num_blocks
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.circular_buffer import AscendCircularBufferSpec
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec

STATE_RING_ROWS = 32

A5_CMP_LOGICAL_DIM = 512
A5_CMP_QUANT_GROUP_SIZE = 16
A5_CMP_DATA_BYTES = A5_CMP_LOGICAL_DIM // 2
A5_CMP_SCALE_COUNT = A5_CMP_LOGICAL_DIM // A5_CMP_QUANT_GROUP_SIZE
A5_CMP_SCALE_OFFSET_BYTES = A5_CMP_DATA_BYTES
A5_CMP_ROW_BYTES = A5_CMP_DATA_BYTES + A5_CMP_SCALE_COUNT * torch.bfloat16.itemsize

A5_WIN_LOGICAL_DIM = 512
A5_WIN_QUANT_GROUP_SIZE = 32
A5_WIN_DATA_BYTES = A5_WIN_LOGICAL_DIM
A5_WIN_SCALE_COUNT = A5_WIN_LOGICAL_DIM // A5_WIN_QUANT_GROUP_SIZE
A5_WIN_SCALE_OFFSET_BYTES = A5_WIN_DATA_BYTES
A5_WIN_ROW_BYTES = A5_WIN_DATA_BYTES + A5_WIN_SCALE_COUNT * torch.bfloat16.itemsize

A5_INDEX_LOGICAL_DIM = 128
A5_INDEX_QUANT_GROUP_SIZE = 32
A5_INDEX_DATA_BYTES = A5_INDEX_LOGICAL_DIM // 2
A5_INDEX_SCALE_COUNT = A5_INDEX_LOGICAL_DIM // A5_INDEX_QUANT_GROUP_SIZE


@dataclass(frozen=True, kw_only=True)
class DeepseekV41FullSpec(AscendMLAAttentionSpec):
    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, (DeepseekV41FullSpec, DeepseekV41IndexerSpec))
            and s.block_size == self.block_size
            and s.compress_ratio in (1, 2)
            for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41IndexerSpec(AscendMLAAttentionSpec):
    """INT8 index keys followed by FP16 scales inside each shared slot page."""

    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, (DeepseekV41FullSpec, DeepseekV41IndexerSpec))
            and s.block_size == self.block_size
            and s.compress_ratio in (1, 2)
            for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41A5CompressedSpec(DeepseekV41FullSpec):
    """A5 compressed KV row: MXFP4 payload followed by BF16 scales."""

    logical_head_size: int = A5_CMP_LOGICAL_DIM
    quant_group_size: int = A5_CMP_QUANT_GROUP_SIZE
    scale_offset_bytes: int = A5_CMP_SCALE_OFFSET_BYTES

    @property
    def storage_block_size(self) -> int:
        # The packaged QLI/QSMLA ABI now accepts the cache's actual PA page
        # size.  A ratio-2 source therefore stores the 64 completed rows from
        # each 128-token scheduler block in that block's own physical page.
        return self.block_size // self.compress_ratio

    def __post_init__(self):
        if (
            self.dtype != torch.uint8
            or self.block_size != 128
            or self.num_kv_heads != 1
            or self.head_size != A5_CMP_ROW_BYTES
            or self.logical_head_size != A5_CMP_LOGICAL_DIM
            or self.quant_group_size != A5_CMP_QUANT_GROUP_SIZE
            or self.scale_dim != 0
            or self.scale_dtype != torch.bfloat16
            or self.scale_offset_bytes != A5_CMP_SCALE_OFFSET_BYTES
        ):
            raise ValueError("A5 cmp_kv requires one U8[320] row: FP4/16 payload plus 32 embedded BF16 scales")


@dataclass(frozen=True, kw_only=True)
class DeepseekV41A5IndexerSpec(DeepseekV41IndexerSpec):
    """A5 index K: MXFP4 data and E8M0 scales in separate U8 planes."""

    logical_head_size: int = A5_INDEX_LOGICAL_DIM
    quant_group_size: int = A5_INDEX_QUANT_GROUP_SIZE

    @property
    def storage_block_size(self) -> int:
        return self.block_size // self.compress_ratio

    def __post_init__(self):
        if (
            self.dtype != torch.uint8
            or self.block_size != 128
            or self.num_kv_heads != 1
            or self.head_size != A5_INDEX_DATA_BYTES
            or self.logical_head_size != A5_INDEX_LOGICAL_DIM
            or self.quant_group_size != A5_INDEX_QUANT_GROUP_SIZE
            or self.scale_dim != A5_INDEX_SCALE_COUNT
            or self.scale_dtype != torch.uint8
        ):
            raise ValueError("A5 index K requires MXFP4 U8[64] data plus E8M0 U8[4] scales")


@dataclass(frozen=True, kw_only=True)
class DeepseekV41SWASpec(AscendSlidingWindowMLASpec):
    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, DeepseekV41SWASpec) and s.sliding_window == self.sliding_window for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41A5SWASpec(DeepseekV41SWASpec):
    """A5 window KV row: FP8 payload followed by BF16 group scales."""

    logical_head_size: int = A5_WIN_LOGICAL_DIM
    quant_group_size: int = A5_WIN_QUANT_GROUP_SIZE
    scale_dtype: torch.dtype = torch.bfloat16
    scale_offset_bytes: int = A5_WIN_SCALE_OFFSET_BYTES

    def __post_init__(self):
        if (
            self.dtype != torch.uint8
            or self.num_kv_heads != 1
            or self.head_size != A5_WIN_ROW_BYTES
            or self.logical_head_size != A5_WIN_LOGICAL_DIM
            or self.quant_group_size != A5_WIN_QUANT_GROUP_SIZE
            or self.scale_dtype != torch.bfloat16
            or self.scale_offset_bytes != A5_WIN_SCALE_OFFSET_BYTES
        ):
            raise ValueError("A5 win_kv requires one U8[544] row: FP8/32 payload plus 16 embedded BF16 scales")


@dataclass(frozen=True, kw_only=True)
class DeepseekV41DraftSWASpec(AscendSlidingWindowMLASpec):
    """DSpark SWA owned by G12, aliasing target slots at distinct block IDs."""

    def __post_init__(self):
        if self.dtype != torch.bfloat16 or self.num_kv_heads != 1 or self.compress_ratio != 1:
            raise ValueError("Aurora DSpark requires one uncompressed BF16 KV plane")

    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, DeepseekV41DraftSWASpec)
            and s.block_size == self.block_size
            and s.sliding_window == self.sliding_window
            for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41A5DraftSWASpec(AscendSlidingWindowMLASpec):
    """DSpark SWA in the A5 MQSMLA FP8/32 plus BF16-scale row format."""

    logical_head_size: int = A5_WIN_LOGICAL_DIM
    quant_group_size: int = A5_WIN_QUANT_GROUP_SIZE
    scale_dtype: torch.dtype = torch.bfloat16
    scale_offset_bytes: int = A5_WIN_SCALE_OFFSET_BYTES

    def __post_init__(self):
        if (
            self.dtype != torch.uint8
            or self.num_kv_heads != 1
            or self.head_size != A5_WIN_ROW_BYTES
            or self.logical_head_size != A5_WIN_LOGICAL_DIM
            or self.quant_group_size != A5_WIN_QUANT_GROUP_SIZE
            or self.scale_dtype != torch.bfloat16
            or self.scale_offset_bytes != A5_WIN_SCALE_OFFSET_BYTES
            or self.compress_ratio != 1
        ):
            raise ValueError("A5 DSpark win_kv requires one U8[544] row: FP8/32 payload plus 16 embedded BF16 scales")

    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, DeepseekV41A5DraftSWASpec)
            and s.block_size == self.block_size
            and s.sliding_window == self.sliding_window
            for s in specs.values()
        )


def is_v41_draft_swa_spec(spec):
    return isinstance(spec, (DeepseekV41DraftSWASpec, DeepseekV41A5DraftSWASpec))


@dataclass(frozen=True, kw_only=True)
class DeepseekV41CompressorStateSpec(AscendCircularBufferSpec):
    """One private FP32 KV/score ring page for each active request."""

    compress_ratio: int = 1

    def __post_init__(self):
        if self.dtype != torch.float32 or self.block_size != STATE_RING_ROWS or self.compress_ratio != 1:
            raise ValueError("Aurora state requires a 32-row FP32 uncompressed ring")
        if self.num_kv_heads != 1:
            raise ValueError("Aurora state requires one packed KV/score plane")


def is_v41_spec(spec):
    return isinstance(
        spec,
        (
            DeepseekV41FullSpec,
            DeepseekV41IndexerSpec,
            DeepseekV41SWASpec,
            DeepseekV41DraftSWASpec,
            DeepseekV41A5DraftSWASpec,
            DeepseekV41CompressorStateSpec,
        ),
    )


def get_dsv41_config(vllm_config):
    """Read the two V4.1 choices needed before Ascend config initialization."""
    additional = getattr(vllm_config, "additional_config", None) or {}
    raw = additional.get("dsv41_config", {}) if isinstance(additional, dict) else {}
    if isinstance(raw, dict):
        return {
            "cache_format": raw.get("cache_format", "a3_bf16"),
            "compressor": raw.get("compressor", "triton"),
        }
    return {
        "cache_format": getattr(raw, "cache_format", "a3_bf16"),
        "compressor": getattr(raw, "compressor", "triton"),
    }


def uses_a5_packed_cache(vllm_config):
    return get_dsv41_config(vllm_config)["cache_format"] == "a5_packed"


def uses_a5_mqsmla_draft(vllm_config):
    return uses_a5_packed_cache(vllm_config)


def _uniform(members, label):
    if not members:
        raise ValueError(f"V4.1 cache group {label} is empty")
    uniform = UniformTypeKVCacheSpecs.from_specs(members)
    if uniform is None:
        raise ValueError(f"Incompatible V4.1 resource layouts in {label}")
    return uniform


@dataclass(frozen=True)
class CachePlacement:
    name: str
    offset: int
    page_size_bytes: int


@dataclass(frozen=True)
class CacheSlot:
    page_size_bytes: int
    placements: tuple[CachePlacement, ...]


def _layer_number(name):
    try:
        return int(name.rsplit(".layers.", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid V4.1 cache resource name: {name}") from exc


def _cache_plane_sizes(spec):
    rows = spec.storage_block_size * spec.num_kv_heads
    key_bytes = rows * spec.head_size * spec.dtype.itemsize
    if isinstance(spec, DeepseekV41IndexerSpec):
        return key_bytes, rows * spec.scale_dim * spec.scale_dtype.itemsize
    return (key_bytes,)


def _draft_layer_number(name):
    try:
        return int(("." + name).rsplit(".mtp.", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid Aurora DSpark cache resource name: {name}") from exc


def plan_cache_slots(specs):
    """Place source KV/index tuples, state and SWA in four shared layer slots.

    Sizes come from payloads, never previously padded specs. Different groups
    overlay a slot at distinct live block IDs; a source's KV and index share
    the same ID at disjoint offsets within its page.
    """
    if not all(is_v41_spec(spec) for spec in specs.values()):
        raise ValueError(
            "V4.1 requires explicit target or Aurora DSpark cache specs; foreign resources are unsupported"
        )
    full = sorted((n for n, s in specs.items() if isinstance(s, DeepseekV41FullSpec)), key=_layer_number)
    state = sorted((n for n, s in specs.items() if isinstance(s, DeepseekV41CompressorStateSpec)), key=_layer_number)
    swa = sorted((n for n, s in specs.items() if isinstance(s, DeepseekV41SWASpec)), key=_layer_number)
    draft = sorted((n for n, s in specs.items() if is_v41_draft_swa_spec(s)), key=_draft_layer_number)
    if draft and list(map(_draft_layer_number, draft)) != [0, 1, 2]:
        raise ValueError("Aurora DSpark requires exactly three ordered draft layers: mtp.0, mtp.1, mtp.2")
    if list(map(_layer_number, full)) != [2, 8, 14, 20]:
        raise ValueError("V4.1 requires KV source layers 2, 8, 14, 20")
    if list(map(_layer_number, state)) != [2, 8, 14]:
        raise ValueError("V4.1 requires state source layers 2, 8, 14")
    if list(map(_layer_number, swa)) != list(range(40)):
        raise ValueError("V4.1 requires exactly 40 ordered SWA resources")

    slots = []
    for slot_idx, kv_name in enumerate(full):
        prefix, suffix = kv_name.rsplit(".", 1)
        index_name = prefix + ".indexer.k_cache"
        index_spec = specs.get(index_name)
        kv_spec = specs[kv_name]
        ratio = 2 if slot_idx < len(state) else 1
        if (
            suffix != "long_kv_cache"
            or not isinstance(index_spec, DeepseekV41IndexerSpec)
            or kv_spec.compress_ratio != ratio
            or index_spec.compress_ratio != ratio
            or kv_spec.block_size != index_spec.block_size
        ):
            raise ValueError(f"V4.1 source {prefix} has incompatible KV/index specs")
        aliases = ([state[slot_idx]] if slot_idx < len(state) else []) + swa[slot_idx :: len(full)]
        kv_bytes = sum(_cache_plane_sizes(kv_spec))
        index_bytes = sum(_cache_plane_sizes(index_spec))
        capacity = max(kv_bytes + index_bytes, *(sum(_cache_plane_sizes(specs[n])) for n in aliases))
        if slot_idx < len(draft):
            draft_name = draft[slot_idx]
            draft_spec = specs[draft_name]
            swa_spec = specs[swa[slot_idx]]
            if (
                draft_spec.block_size != swa_spec.block_size
                or getattr(draft_spec, "logical_head_size", draft_spec.head_size)
                != getattr(swa_spec, "logical_head_size", swa_spec.head_size)
                or draft_spec.sliding_window != swa_spec.sliding_window
                or sum(_cache_plane_sizes(draft_spec)) > capacity
                or (
                    isinstance(draft_spec, DeepseekV41A5DraftSWASpec) and not isinstance(swa_spec, DeepseekV41A5SWASpec)
                )
            ):
                raise ValueError("Aurora DSpark geometry must match target SWA and fit its existing slot")
            aliases.append(draft_name)
        placements = [
            CachePlacement(kv_name, 0, kv_bytes),
            CachePlacement(index_name, kv_bytes, capacity - kv_bytes),
            *(CachePlacement(name, 0, capacity) for name in aliases),
        ]
        slots.append(CacheSlot(capacity, tuple(placements)))
    names = [p.name for slot in slots for p in slot.placements]
    if len(names) != len(set(names)) or set(names) != set(specs):
        raise ValueError("V4.1 slot placement must cover each resource exactly once")
    return tuple(slots)


def group_cache_specs(specs):
    """Merge full-context resources and pad layer tuples without mutating inputs."""
    if not any(is_v41_spec(s) for s in specs.values()):
        return None
    slots = plan_cache_slots(specs)
    padded = {
        p.name: replace(specs[p.name], page_size_padded=p.page_size_bytes) for slot in slots for p in slot.placements
    }
    full = {n: s for n, s in padded.items() if isinstance(s, (DeepseekV41FullSpec, DeepseekV41IndexerSpec))}
    state = {n: s for n, s in padded.items() if isinstance(s, DeepseekV41CompressorStateSpec)}
    groups = [_uniform(full, "full"), _uniform(state, "state")]
    swa = sorted((n for n, s in padded.items() if isinstance(s, DeepseekV41SWASpec)), key=_layer_number)
    groups.extend(
        _uniform({n: padded[n] for n in swa[start : start + len(slots)]}, f"swa{start}")
        for start in range(0, len(swa), len(slots))
    )
    draft = sorted((n for n, s in padded.items() if is_v41_draft_swa_spec(s)), key=_draft_layer_number)
    if draft:
        groups.append(_uniform({n: padded[n] for n in draft}, "dspark"))
    return groups


def make_cache_groups(grouped_specs):
    return [KVCacheGroupSpec(layer_names=list(s.kv_cache_specs), kv_cache_spec=s) for s in grouped_specs]


def has_v41_groups(groups):
    return any(
        is_v41_spec(s)
        for g in groups
        if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
        for s in g.kv_cache_spec.kv_cache_specs.values()
    )


def cache_slots_from_groups(groups):
    specs = {}
    for group in groups:
        if not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            raise ValueError("V4.1 requires uniform-type cache groups")
        for name in group.layer_names:
            if name in specs:
                raise ValueError(f"V4.1 resource belongs to multiple cache groups: {name}")
            specs[name] = group.kv_cache_spec.kv_cache_specs[name]
    return plan_cache_slots(specs)


def pool_bytes_per_block(groups):
    return sum(slot.page_size_bytes for slot in cache_slots_from_groups(groups))


def request_blocks(vllm_config, groups):
    # Different logical groups consume different IDs in one global block pool.
    return sum(
        max(
            (s.max_memory_usage_bytes(vllm_config) + s.page_size_bytes - 1) // s.page_size_bytes
            for s in g.kv_cache_spec.kv_cache_specs.values()
        )
        for g in groups
    )


def allocate_cache_config(vllm_config, groups, available_memory):
    """Allocate four independent layer slots backed by one global block-ID pool."""
    slots = cache_slots_from_groups(groups)
    capacity = available_memory // sum(slot.page_size_bytes for slot in slots)
    num_blocks = may_override_num_blocks(vllm_config, capacity)
    if num_blocks <= 1 or num_blocks > capacity:
        raise ValueError("Insufficient V4.1 cache memory (including reserved null block), or unsafe block override")
    return num_blocks, [
        KVCacheTensor(
            size=num_blocks * slot.page_size_bytes,
            shared_by=[p.name for p in slot.placements],
            block_stride=slot.page_size_bytes,
        )
        for slot in slots
    ]


def reshape_cache(raw: torch.Tensor, spec, *, num_blocks, offset, block_stride):
    """Create typed per-page views using the containing slot's physical stride."""
    if raw.dtype != torch.uint8 or raw.ndim != 1 or not raw.is_contiguous():
        raise ValueError("V4.1 cache requires contiguous one-dimensional uint8 storage")
    if num_blocks <= 0 or block_stride <= 0 or raw.numel() != num_blocks * block_stride:
        raise ValueError("V4.1 cache backing does not match its declared layout")
    plane_sizes = _cache_plane_sizes(spec)
    if offset < 0 or offset + sum(plane_sizes) > block_stride:
        raise ValueError("V4.1 cache component exceeds its slot page")
    if isinstance(spec, DeepseekV41CompressorStateSpec) and sum(plane_sizes) != block_stride:
        raise ValueError("Aurora circular state must fill its slot with 32 contiguous FP32 rows")

    def view(dtype, width, byte_offset):
        dtype_size = dtype.itemsize
        storage_offset = raw.storage_offset() + byte_offset
        if storage_offset % dtype_size or block_stride % dtype_size or raw.numel() % dtype_size:
            raise ValueError("V4.1 cache offset/stride is not dtype aligned")
        return torch.as_strided(
            raw.view(dtype),
            size=(num_blocks, spec.storage_block_size, spec.num_kv_heads, width),
            stride=(block_stride // dtype_size, spec.num_kv_heads * width, width, 1),
            storage_offset=storage_offset // dtype_size,
        )

    key = view(spec.dtype, spec.head_size, offset)
    if isinstance(spec, DeepseekV41IndexerSpec):
        return key, view(spec.scale_dtype, spec.scale_dim, offset + plane_sizes[0])
    return key


def validate_cache_runtime(vllm_config):
    if vllm_config.use_v2_model_runner:
        raise NotImplementedError("V4.1 cache initialization currently requires model runner V1")
    cudagraph_mode = getattr(
        vllm_config.compilation_config,
        "cudagraph_mode",
        CUDAGraphMode.NONE if vllm_config.model_config.enforce_eager else CUDAGraphMode.FULL,
    )
    if cudagraph_mode not in (
        CUDAGraphMode.NONE,
        CUDAGraphMode.FULL_DECODE_ONLY,
    ):
        raise NotImplementedError("V4.1 currently supports only eager or FULL_DECODE_ONLY graph mode")
    speculative = vllm_config.speculative_config
    if speculative is not None:
        use_dspark = getattr(speculative, "use_dspark", None)
        if not callable(use_dspark) or not use_dspark():
            raise NotImplementedError("Aurora supports only DSpark speculative decoding")
        # Verification writes the anchor and up to S speculative rows. After
        # rejection, the earliest needed residual is the verified anchor.
        # It must survive the final 32-row write: S must be strictly below 32.
        if not 0 < speculative.num_speculative_tokens < STATE_RING_ROWS:
            raise ValueError("Aurora DSpark requires 1..31 speculative tokens to preserve FP32 ring residuals")
        per_batch = getattr(speculative, "num_speculative_tokens_per_batch_size", None) or ()
        if any(not 0 <= count <= speculative.num_speculative_tokens for _, _, count in per_batch):
            raise ValueError("Aurora DSpark per-batch speculation must stay within the configured ring-safe maximum")
    parallel = vllm_config.parallel_config
    if any(
        getattr(parallel, name, 1) != 1
        for name in (
            "pipeline_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        )
    ):
        raise NotImplementedError("V4.1 initial runtime requires PP=DCP=PCP=1")
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        raise ValueError("V4.1 requires the hybrid KV cache manager")
    if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise NotImplementedError("V4.1 cache setup accepts only auto or bfloat16 requests")

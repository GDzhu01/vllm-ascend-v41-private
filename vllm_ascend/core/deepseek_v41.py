# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Framework-side V4.1 cache specs and packed hybrid allocation."""

from collections import defaultdict
from dataclasses import dataclass

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.core.kv_cache_utils import may_override_num_blocks
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec


@dataclass(frozen=True, kw_only=True)
class DeepseekV41FullSpec(AscendMLAAttentionSpec):
    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, (DeepseekV41FullSpec, DeepseekV41IndexerSpec))
            and s.block_size == self.block_size
            and s.compress_ratio == self.compress_ratio
            for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41IndexerSpec(AscendMLAAttentionSpec):
    """Packed INT8 index key followed by one FP16 scale per stored row."""

    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, (DeepseekV41FullSpec, DeepseekV41IndexerSpec))
            and s.block_size == self.block_size
            and s.compress_ratio == self.compress_ratio
            for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41SWASpec(AscendSlidingWindowMLASpec):
    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, DeepseekV41SWASpec) and s.sliding_window == self.sliding_window for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41CompressorStateSpec(AscendSlidingWindowMLASpec):
    """V4-style FP32 KV/score rows, retained by SlidingWindowManager.

    State is not compressed: one row per original token, two vectors per row.
    Only pooling has ratio2; the storage compression ratio remains one.
    """

    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, DeepseekV41CompressorStateSpec) and s.sliding_window == self.sliding_window
            for s in specs.values()
        )


def is_v41_spec(spec):
    return isinstance(
        spec,
        (
            DeepseekV41FullSpec,
            DeepseekV41IndexerSpec,
            DeepseekV41SWASpec,
            DeepseekV41CompressorStateSpec,
        ),
    )


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


def plan_cache_slots(specs):
    """Place source KV/index tuples, state and SWA in four shared layer slots.

    Sizes come from payloads, never previously padded specs. Different groups
    overlay a slot at distinct live block IDs; a source's KV and index share
    the same ID at disjoint offsets within its page.
    """
    target_specs = {name: spec for name, spec in specs.items() if is_v41_spec(spec)}
    draft_specs = {name: spec for name, spec in specs.items() if not is_v41_spec(spec)}
    if not target_specs:
        raise ValueError("V4.1 cache planning requires target cache resources")
    invalid_draft_names = [
        name
        for name, spec in draft_specs.items()
        if not isinstance(spec, AscendSlidingWindowMLASpec)
    ]
    if invalid_draft_names:
        raise ValueError(
            "V4.1 mixed cache supports only dSPark SWA draft resources; "
            f"unsupported resources: {', '.join(sorted(invalid_draft_names))}"
        )
    full = sorted((n for n, s in target_specs.items() if isinstance(s, DeepseekV41FullSpec)), key=_layer_number)
    state = sorted((n for n, s in target_specs.items() if isinstance(s, DeepseekV41CompressorStateSpec)), key=_layer_number)
    swa = sorted((n for n, s in target_specs.items() if isinstance(s, DeepseekV41SWASpec)), key=_layer_number)
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
        index_spec = target_specs.get(index_name)
        kv_spec = target_specs[kv_name]
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
        capacity = max(kv_bytes + index_bytes, *(sum(_cache_plane_sizes(target_specs[n])) for n in aliases))
        placements = [
            CachePlacement(kv_name, 0, kv_bytes),
            CachePlacement(index_name, kv_bytes, capacity - kv_bytes),
            *(CachePlacement(name, 0, capacity) for name in aliases),
        ]
        slots.append(CacheSlot(capacity, tuple(placements)))
    # Draft layers consume the same global scheduler block id concurrently, so
    # they must not alias one another or the target's layer-outermost slots.
    for name, spec in sorted(draft_specs.items()):
        page_size = spec.page_size_bytes
        slots.append(
            CacheSlot(page_size, (CachePlacement(name, 0, page_size),))
        )
    names = [p.name for slot in slots for p in slot.placements]
    if len(names) != len(set(names)) or set(names) != set(specs):
        raise ValueError("V4.1 slot placement must cover each resource exactly once")
    return tuple(slots)


def group_cache_specs(specs):
    """Build the fixed V4.1 ownership graph used by the hybrid manager."""
    if not any(is_v41_spec(s) for s in specs.values()):
        return None
    target_specs = {name: spec for name, spec in specs.items() if is_v41_spec(spec)}
    draft_specs = {name: spec for name, spec in specs.items() if not is_v41_spec(spec)}
    invalid_draft_names = [
        name
        for name, spec in draft_specs.items()
        if not isinstance(spec, AscendSlidingWindowMLASpec)
    ]
    if invalid_draft_names:
        raise ValueError(
            "V4.1 mixed cache supports only dSPark SWA draft resources; "
            f"unsupported resources: {', '.join(sorted(invalid_draft_names))}"
        )

    ratio_groups = {}
    for ratio in (2, 1):
        members = {
            name: spec
            for name, spec in target_specs.items()
            if isinstance(spec, (DeepseekV41FullSpec, DeepseekV41IndexerSpec))
            and spec.compress_ratio == ratio
        }
        ratio_groups[ratio] = _uniform(members, f"ratio{ratio}")

    state = {
        name: spec
        for name, spec in target_specs.items()
        if isinstance(spec, DeepseekV41CompressorStateSpec)
    }
    groups = [ratio_groups[2], ratio_groups[1], _uniform(state, "state")]

    swa = [
        (name, spec)
        for name, spec in target_specs.items()
        if isinstance(spec, DeepseekV41SWASpec)
    ]
    if len(swa) != 40:
        raise ValueError(f"V4.1 requires exactly 40 SWA resources, got {len(swa)}")
    swa_groups = [swa[start : start + 3] for start in range(0, 36, 3)]
    swa_groups.extend((swa[36:38], swa[38:40]))
    groups.extend(
        _uniform(dict(members), f"swa{group_idx}")
        for group_idx, members in enumerate(swa_groups)
    )
    if len(groups) != 17:
        raise AssertionError(f"V4.1 must form 17 cache groups, got {len(groups)}")
    if draft_specs:
        groups.append(_uniform(draft_specs, "dspark"))
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


def pool_bytes_per_block(groups):
    return max(g.kv_cache_spec.page_size_bytes for g in groups)


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
    """Allocate one block-strided backing shared by all scheduler groups."""
    block_stride = pool_bytes_per_block(groups)
    capacity = available_memory // block_stride
    num_blocks = may_override_num_blocks(vllm_config, capacity)
    if num_blocks <= 1 or num_blocks > capacity:
        raise ValueError("Insufficient V4.1 cache memory (including reserved null block), or unsafe block override")

    layers_by_offset = defaultdict(list)
    for group in groups:
        offset = 0
        for name in group.layer_names:
            spec = group.kv_cache_spec.kv_cache_specs[name]
            if offset + spec.page_size_bytes > block_stride:
                raise AssertionError(f"V4.1 resource {name} exceeds packed block stride")
            layers_by_offset[offset].append(name)
            offset += spec.page_size_bytes

    total_size = num_blocks * block_stride
    tensors = [
        KVCacheTensor(
            size=total_size,
            shared_by=layers_by_offset[offset],
            offset=offset,
            block_stride=block_stride,
        )
        for offset in sorted(layers_by_offset)
    ]
    return num_blocks, tensors


def _strided_view(raw, dtype, shape, offset, block_stride):
    dtype_size = torch.empty((), dtype=dtype).element_size()
    if offset % dtype_size or block_stride % dtype_size:
        raise ValueError("V4.1 packed cache offset/stride is not dtype aligned")
    contiguous = torch.empty(shape[1:], device="meta").stride()
    return torch.as_strided(
        raw.view(dtype),
        size=shape,
        stride=(block_stride // dtype_size, *contiguous),
        storage_offset=offset // dtype_size,
    )


def reshape_cache(raw: torch.Tensor, spec, num_blocks=None, offset=0, block_stride=0):
    """Create typed zero-copy views over independent or packed raw storage."""
    if not block_stride:
        block_stride = spec.page_size_bytes
    if num_blocks is None:
        if raw.numel() % block_stride:
            raise ValueError("V4.1 cache allocation is not a whole number of pages")
        num_blocks = raw.numel() // block_stride
    if raw.numel() < num_blocks * block_stride:
        raise ValueError("V4.1 packed backing is smaller than its declared layout")

    shape = (
        num_blocks,
        spec.storage_block_size,
        spec.num_kv_heads,
        spec.head_size,
    )
    if isinstance(spec, DeepseekV41IndexerSpec):
        k = _strided_view(raw, spec.dtype, shape, offset, block_stride)
        k_bytes = spec.storage_block_size * spec.num_kv_heads * spec.head_size
        scale_shape = (
            num_blocks,
            spec.storage_block_size,
            spec.num_kv_heads,
            spec.scale_dim,
        )
        scale = _strided_view(
            raw,
            spec.scale_dtype,
            scale_shape,
            offset + k_bytes,
            block_stride,
        )
        return k, scale
    return _strided_view(raw, spec.dtype, shape, offset, block_stride)


def validate_cache_runtime(vllm_config):
    if vllm_config.use_v2_model_runner:
        raise NotImplementedError("V4.1 cache initialization currently requires model runner V1")
    if (
        not vllm_config.model_config.enforce_eager
        and vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
    ):
        raise NotImplementedError("V4.1 graph runtime requires FULL_DECODE_ONLY")
    if vllm_config.cache_config.enable_prefix_caching:
        raise NotImplementedError("V4.1 prefix state restoration is not implemented")
    speculative = vllm_config.speculative_config
    use_dspark = getattr(speculative, "use_dspark", None)
    if speculative is not None and (not callable(use_dspark) or not use_dspark()):
        raise NotImplementedError("V4.1 currently supports only dSPark speculative decoding")
    if vllm_config.kv_transfer_config is not None:
        raise NotImplementedError("V4.1 KV transfer is not implemented")
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
        raise NotImplementedError("V4.1 initial cache layout requires BF16")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheLayer,
    DeepseekV41MetadataBuilder,
    compressed_slot_mapping,
    gather_cache_rows,
    pad_sparse_indices,
    scatter_cache,
    select_candidate_blocks,
    select_index_topk,
)
from vllm_ascend.core.deepseek_v41 import (
    DeepseekV41IndexerSpec,
    allocate_cache_config,
    group_cache_specs,
    make_cache_groups,
    pool_bytes_per_block,
    reshape_cache,
)
from vllm_ascend.models.deepseek_v41.compressor import DeepseekV41Compressor
from vllm_ascend.models.deepseek_v41.model import build_layer_plan, build_v41_cache_specs


@pytest.fixture
def config():
    # Deliberately small parameter dimensions; source topology matches the backbone.
    return dict(
        num_hidden_layers=40,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        kv_source_layers=[2, 8, 14, 20],
        index_source_layers=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer=20,
        candidate_topk_blocks=16,
        candidate_block_size=8,
        index_topk=8,
        engram_layer_ids=[1, 14],
        sliding_window=128,
        head_dim=8,
        index_head_dim=4,
        hidden_size=16,
        num_attention_heads=4,
        index_n_heads=2,
        q_lora_rank=8,
        o_lora_rank=4,
        o_groups=2,
        rms_norm_eps=1e-6,
    )


@pytest.fixture
def runtime(config):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config, enforce_eager=True),
        cache_config=SimpleNamespace(
            block_size=64, enable_prefix_caching=False, cache_dtype="auto", num_gpu_blocks_override=None
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        use_v2_model_runner=False,
    )


def collect_specs(runtime, prefix="model"):
    return build_v41_cache_specs(runtime.model_config.hf_text_config, runtime, prefix)


def test_owner_counts_nested_config_and_source_resolution(config, runtime):
    topology = build_layer_plan({"text_config": config})
    specs = collect_specs(runtime)
    assert len(specs) == 51
    assert topology.kv_consumers(2) == tuple(range(2, 8))
    assert topology.kv_consumers(20) == tuple(range(20, 40))
    assert topology.layer(26).kv_source_layer == 20
    assert topology.layer(26).index_source_layer == 24
    assert specs["model.layers.20.self_attn.long_kv_cache"].storage_block_size == 64
    assert specs["model.layers.2.self_attn.long_kv_cache"].storage_block_size == 32
    assert "model.layers.20.self_attn.compressor.state_cache" not in specs


@pytest.mark.parametrize("block_size", [0, -2, 3, 63])
def test_invalid_block_sizes(config, runtime, block_size):
    runtime.cache_config.block_size = block_size
    with pytest.raises(ValueError, match="multiple of two"):
        build_v41_cache_specs(config, runtime)


def test_seventeen_groups_and_packed_physical_accounting(config, runtime):
    specs = collect_specs(runtime)
    uniform = group_cache_specs(specs)
    assert [len(g.kv_cache_specs) for g in uniform[:3]] == [6, 2, 3]
    assert [len(g.kv_cache_specs) for g in uniform[3:]] == [3] * 12 + [2] * 2
    assert [g.block_size for g in uniform[:2]] == [64, 64]
    assert uniform[2].block_size == 16
    assert [g.block_size for g in uniform[3:]] == [64] * 14
    groups = make_cache_groups(uniform)
    block_stride = pool_bytes_per_block(groups)
    blocks, tensors = allocate_cache_config(runtime, groups, block_stride * 10 + 1)
    assert blocks == 10
    assert all(t.size == 10 * block_stride for t in tensors)
    assert all(t.block_stride == block_stride for t in tensors)
    assert {name for tensor in tensors for name in tensor.shared_by} == set(specs)

    raw = torch.zeros(blocks * block_stride, dtype=torch.uint8)
    for allocation in tensors:
        for name in allocation.shared_by:
            spec = specs[name]
            cache = reshape_cache(
                raw,
                spec,
                blocks,
                allocation.offset,
                allocation.block_stride,
            )
            views = cache if isinstance(cache, tuple) else (cache,)
            assert views[0].shape == (
                blocks,
                spec.storage_block_size,
                1,
                spec.head_size,
            )
            assert views[0].data_ptr() == raw.data_ptr() + allocation.offset
            assert views[0].stride(0) * views[0].element_size() == block_stride
            if isinstance(spec, DeepseekV41IndexerSpec):
                k, scale = cache
                assert k.dtype == torch.int8
                assert scale.dtype == torch.float16
                k_bytes = spec.storage_block_size * spec.head_size
                assert scale.data_ptr() == k.data_ptr() + k_bytes
                assert scale.stride(0) * scale.element_size() == block_stride


def test_production_layout_matches_design(config, runtime):
    production = dict(config, head_dim=512, index_head_dim=128)
    runtime.cache_config.block_size = 128
    specs = build_v41_cache_specs(production, runtime)
    groups = make_cache_groups(group_cache_specs(specs))
    assert len(groups) == 17
    assert [g.kv_cache_spec.page_size_bytes for g in groups[:3]] == [221568, 147712, 196608]
    assert [g.kv_cache_spec.page_size_bytes for g in groups[3:]] == [393216] * 12 + [262144] * 2
    assert pool_bytes_per_block(groups) == 393216


def test_packed_strided_views_support_row_io_without_cross_resource_corruption(config, runtime):
    specs = collect_specs(runtime)
    groups = make_cache_groups(group_cache_specs(specs))
    stride = pool_bytes_per_block(groups)
    blocks, tensors = allocate_cache_config(runtime, groups, stride * 4)
    descriptors = {
        name: tensor
        for tensor in tensors
        for name in tensor.shared_by
    }
    long_name = "model.layers.2.self_attn.long_kv_cache"
    index_name = "model.layers.2.self_attn.indexer.k_cache"
    raw = torch.zeros(blocks * stride, dtype=torch.uint8)
    long_desc = descriptors[long_name]
    index_desc = descriptors[index_name]
    long_cache = reshape_cache(
        raw, specs[long_name], blocks, long_desc.offset, long_desc.block_stride
    )
    index_k, index_scale = reshape_cache(
        raw, specs[index_name], blocks, index_desc.offset, index_desc.block_stride
    )

    slots = torch.tensor([1 * specs[long_name].storage_block_size + 3])
    value = torch.arange(config["head_dim"], dtype=torch.bfloat16).unsqueeze(0)
    scatter_cache(long_cache, slots, value)
    torch.testing.assert_close(gather_cache_rows(long_cache, slots), value)
    assert not index_k.any()
    assert not index_scale.any()


def test_mixed_layouts_rejected(config, runtime):
    specs = collect_specs(runtime)
    specs["foreign"] = object()
    with pytest.raises(ValueError, match="foreign"):
        group_cache_specs(specs)


def test_unsafe_override_rejected(config, runtime):
    groups = make_cache_groups(group_cache_specs(collect_specs(runtime)))
    runtime.cache_config.num_gpu_blocks_override = 100
    with pytest.raises(ValueError, match="unsafe block override"):
        allocate_cache_config(runtime, groups, 1)


def test_model_registration_and_binding(runtime):
    specs = collect_specs(runtime, "language_model.model")
    context = runtime.compilation_config.static_forward_context
    modules = torch.nn.ModuleDict()
    for index, (name, spec) in enumerate(specs.items()):
        modules[str(index)] = DeepseekV41CacheLayer(runtime, name, spec)
    assert len(context) == 51
    assert all(module.kv_cache[0].numel() == 0 for module in context.values())
    state = context["language_model.model.layers.2.self_attn.compressor.state_cache"]
    assert state is context["language_model.model.layers.2.self_attn.compressor.state_cache"]
    assert state.spec.sliding_window == 2
    assert state.spec.storage_block_size == 16
    owned_names = [name for name, module in modules.named_modules() if hasattr(module, "kv_cache")]
    assert len(owned_names) == 51


@pytest.mark.parametrize("feature", ["prefix", "spec", "pd", "pp", "v2", "graph"])
def test_unsupported_runtime_fails_before_registration(runtime, feature):
    if feature == "prefix":
        runtime.cache_config.enable_prefix_caching = True
    elif feature == "spec":
        runtime.speculative_config = object()
    elif feature == "pd":
        runtime.kv_transfer_config = object()
    elif feature == "pp":
        runtime.parallel_config.pipeline_parallel_size = 2
    elif feature == "v2":
        runtime.use_v2_model_runner = True
    else:
        runtime.model_config.enforce_eager = False
    with pytest.raises(NotImplementedError):
        from vllm_ascend.core.deepseek_v41 import validate_cache_runtime

        validate_cache_runtime(runtime)
    assert not runtime.compilation_config.static_forward_context


def test_compression_slot_mapping():
    slots = torch.tensor([-1, 0, 1, 62, 63, 320, 321, 383])
    assert compressed_slot_mapping(slots, 2).tolist() == [-1, -1, 0, -1, 31, -1, 160, 191]
    assert torch.equal(compressed_slot_mapping(slots, 1), slots)


def test_candidate_blocks_pin_partial_tail_and_drop_unreachable_blocks():
    scores = torch.tensor(
        [[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, -torch.inf, -torch.inf]]
    )
    # With two candidate blocks, the best old block and the partially filled
    # newest block must survive. The unreachable final block must not.
    mask = select_candidate_blocks(
        scores, torch.tensor([[6]]), topk_blocks=2, block_size=2
    )
    assert mask.tolist() == [[True, True, False, False, True, True, False, False]]


def test_index_topk_is_chronological_and_marks_unreachable_slots():
    scores = torch.tensor([[1.0, 7.0, 3.0, -torch.inf, -torch.inf]])
    selected = select_index_topk(scores, torch.tensor([[3]]), index_topk=4)
    assert selected.tolist() == [[0, 1, 2, -1]]


def test_sparse_indices_are_padded_for_native_mla():
    selected = torch.tensor([[2, 7], [1, -1]], dtype=torch.int32)
    padded = pad_sparse_indices(selected, 4)
    assert padded.shape == (2, 1, 4)
    assert padded.tolist() == [[[2, 7, -1, -1]], [[1, -1, -1, -1]]]


def test_state_metadata_keeps_original_token_slots(config, runtime):
    specs = collect_specs(runtime)
    spec = specs["model.layers.2.self_attn.compressor.state_cache"]
    builder = DeepseekV41MetadataBuilder(spec, [], runtime, torch.device("cpu"))
    slots = torch.tensor([7 * 16 + 15, 3 * 16, -1])
    common = SimpleNamespace(
        slot_mapping=slots,
        block_table_tensor=torch.tensor([[7, 3]]),
        query_start_loc=torch.tensor([0, 2]),
        seq_lens=torch.tensor([17]),
    )
    metadata = builder.build(0, common)
    assert metadata.is_compressor_state
    assert metadata.slot_mapping is slots
    assert metadata.compress_ratio == 1
    assert metadata.storage_block_size == 16
    assert metadata.max_query_len == 2
    assert metadata.max_seq_len == 17


def test_actual_attention_parameter_ownership(config, runtime):
    topology = build_layer_plan(config)
    assert not topology.layer(0).has_long_context
    assert topology.layer(2).is_kv_source and topology.layer(2).is_index_source
    assert topology.layer(20).is_kv_source and topology.layer(20).compress_ratio == 1
    assert topology.layer(24).is_index_source and not topology.layer(24).is_kv_source
    assert not topology.layer(26).is_index_source
    assert topology.layer(26).kv_source_layer == 20


@pytest.mark.parametrize("chunks", [(1, 1, 1, 2, 2), (3, 4), (2, 2, 3), (7,)])
@torch.inference_mode()
def test_compressor_chunk_boundary_matches_vector_reference(config, chunks):
    torch.manual_seed(7)
    compressor = DeepseekV41Compressor(config, 2)
    x = torch.randn(7, 16, dtype=torch.bfloat16)
    kv = compressor.wkv(x.float())[:6].reshape(3, 2, 8)
    gate = compressor.wgate(x.float())[:6].reshape(3, 2, 8)
    expected = compressor.norm((kv * gate.softmax(dim=1)).sum(dim=1).to(x.dtype))
    # Each successive group occupies another page. Noncontiguous physical IDs
    # and chunks spanning those pages catch the old circular-buffer addressing.
    state = torch.full((6, 2, 16), float("nan"), dtype=torch.float32)
    block_table = [4, 1, 5, 2]
    actual = []
    start = 0
    for size in chunks:
        actual.append(compressor(x[start : start + size], start, state, block_table))
        start += size
    torch.testing.assert_close(torch.cat(actual), expected)
    torch.testing.assert_close(state[2, 0, :8], compressor.wkv(x[-1:].float())[0])


def test_state_uses_swa_memory_and_block_table_rules(config, runtime):
    from vllm_ascend.core.deepseek_v41 import DeepseekV41CompressorStateSpec
    from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec

    spec = collect_specs(runtime)["model.layers.2.self_attn.compressor.state_cache"]
    assert isinstance(spec, AscendSlidingWindowMLASpec)
    assert isinstance(spec, DeepseekV41CompressorStateSpec)
    assert spec.sliding_window == 2
    assert spec.compress_ratio == 1
    assert spec.storage_block_size == 16
    assert spec.page_size_bytes == 16 * 16 * 4
    assert spec.max_num_blocks_per_req(runtime, 1024) == 64
    runtime.max_in_flight_tokens = 128
    runtime.model_config.max_model_len = 1024
    expected_pages = spec.max_admission_blocks_per_request(128, 1024)
    assert spec.max_memory_usage_bytes(runtime) == expected_pages * spec.page_size_bytes
    assert expected_pages > 1
    assert spec.sliding_window != config["sliding_window"]


@torch.inference_mode()
def test_compressor_rejects_missing_previous_state_page(config):
    compressor = DeepseekV41Compressor(config, 2)
    state = torch.full((3, 2, 16), float("nan"), dtype=torch.float32)
    with pytest.raises(ValueError, match="absent/null"):
        compressor(torch.zeros(1, 16, dtype=torch.bfloat16), 1, state, [0])


@torch.inference_mode()
def test_state_page_reuse_does_not_require_request_reset(config):
    compressor = DeepseekV41Compressor(config, 2)
    state = torch.full((3, 2, 16), float("nan"), dtype=torch.float32)
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    expected = compressor(x, 0, state, [1]).clone()
    state[1].fill_(12345)
    actual = compressor(x, 0, state, [1])
    torch.testing.assert_close(actual, expected)


def test_state_registers_standard_sliding_window_manager(monkeypatch):
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    from vllm_ascend.core.deepseek_v41 import DeepseekV41CompressorStateSpec
    from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs

    registrations = {}

    def record(kvcache_spec_cls, manager_class, uniform_type_base_spec):
        registrations[kvcache_spec_cls] = manager_class

    monkeypatch.setattr(KVCacheSpecRegistry, "register", record)
    register_ascend_kv_cache_specs()
    assert registrations[DeepseekV41CompressorStateSpec] is SlidingWindowManager
    manager = SimpleNamespace(sliding_window=2, extra_retained_tokens=0)
    for computed in (1, 63, 64, 65, 128, 129):
        # At the next query, the immediately previous token is never skipped.
        assert SlidingWindowManager.get_num_skipped_tokens(manager, computed) == computed - 1


@torch.inference_mode()
def test_interleaved_request_state_isolation(config):
    compressor = DeepseekV41Compressor(config, 2)
    state = torch.full((3, 2, 16), float("nan"), dtype=torch.float32)
    first = torch.randn(2, 16, dtype=torch.bfloat16)
    second = torch.randn(2, 16, dtype=torch.bfloat16)
    compressor(first[:1], 0, state, [1])
    saved = state[1, 0].clone()
    compressor(second, 0, state, [2])
    torch.testing.assert_close(state[1, 0], saved)
    actual = compressor(first[1:], 1, state, [1])
    expected = compressor(first, 0, state, [1])
    torch.testing.assert_close(actual, expected)

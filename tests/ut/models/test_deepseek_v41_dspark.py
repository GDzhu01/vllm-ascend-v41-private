# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred torch checks for Aurora DSpark model/cache integration."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.core.deepseek_v41 import (
    DeepseekV41A5DraftSWASpec,
    DeepseekV41DraftSWASpec,
)
from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec, register_ascend_kv_cache_specs
from vllm_ascend.models.deepseek_v4.dspark import DSparkDeepseekV4ForCausalLM
from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4SWACache
from vllm_ascend.models.deepseek_v41 import dspark as deepseek_v41_dspark_module
from vllm_ascend.models.deepseek_v41.dspark import (
    DeepseekV41DSparkAttention,
    DeepseekV41DSparkDecoderLayer,
    DeepseekV41DSparkModel,
    DeepseekV41DSparkSWACache,
    DSparkDeepseekV41ForCausalLM,
)
from vllm_ascend.models.deepseek_v41.model import (
    DeepseekV41DecoderLayer,
    DeepseekV41Model,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def test_v41_dspark_ties_target_embedding_and_lm_head():
    draft = DSparkDeepseekV41ForCausalLM.__new__(DSparkDeepseekV41ForCausalLM)
    seen_names = []

    def load_base(instance, weights):
        seen_names.extend(name for name, _ in weights)
        # Model the inherited V4 name probing that caused the A5 regression.
        instance.has_own_embed_tokens = True
        instance.has_own_lm_head = True
        return {"model.layers.40.main_proj.weight"}

    weights = iter(
        (
            ("embed.weight", torch.empty(0)),
            ("head.weight", torch.empty(0)),
            ("mtp.0.embed.weight", torch.empty(0)),
            ("mtp.2.head.weight", torch.empty(0)),
            ("mtp.0.main_proj.weight", torch.empty(0)),
        )
    )
    draft_parameters = (
        ("model.layers.40.main_proj.weight", torch.empty(0)),
        ("model.embed_tokens.weight", torch.empty(0)),
        ("lm_head.weight", torch.empty(0)),
    )
    with (
        patch.object(DSparkDeepseekV4ForCausalLM, "load_weights", load_base),
        patch.object(
            DSparkDeepseekV41ForCausalLM,
            "named_parameters",
            return_value=iter(draft_parameters),
        ),
    ):
        loaded = draft.load_weights(weights)

    assert seen_names == ["mtp.0.main_proj.weight"]
    assert loaded == {"model.layers.40.main_proj.weight"}
    assert draft.has_own_embed_tokens is False
    assert draft.has_own_lm_head is False


def test_draft_decoder_does_not_materialize_target_engram():
    config = SimpleNamespace(engram_layer_ids=[41])
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
        model_config=SimpleNamespace(hf_config=config),
    )

    def initialize_base(instance, *args, **kwargs):
        torch.nn.Module.__init__(instance)
        instance.layer_idx = 41

    with (
        patch(
            "vllm_ascend.models.deepseek_v41.model.DeepseekV2DecoderLayer.__init__",
            initialize_base,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.model.get_ascend_config",
            return_value=SimpleNamespace(enable_engram=True),
        ),
    ):
        layer = DeepseekV41DecoderLayer(
            vllm_config,
            "model.layers.41",
            config=config,
            is_draft_layer=True,
        )

    assert layer.engram is None


def test_draft_cache_uses_v41_backend_and_explicit_aurora_spec():
    spec = AscendSlidingWindowMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
        cache_dtype_str="bfloat16",
        model_version="deepseek_v4",
    )
    cache = DeepseekV41DSparkSWACache.__new__(DeepseekV41DSparkSWACache)
    with patch.object(AscendDeepseekV4SWACache, "get_kv_cache_spec", return_value=spec):
        draft = cache.get_kv_cache_spec(None)
    from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheBackend

    assert DeepseekV41DSparkSWACache.get_attn_backend(None) is DeepseekV41CacheBackend
    assert type(draft) is DeepseekV41DraftSWASpec
    assert draft.page_size_bytes == 131072
    assert DeepseekV41DSparkDecoderLayer.attention_cls is DeepseekV41DSparkAttention
    assert DeepseekV41DSparkAttention.swa_cache_cls is DeepseekV41DSparkSWACache
    register_ascend_kv_cache_specs()
    assert KVCacheSpecRegistry.get_manager_class(draft) is SlidingWindowManager


def test_draft_cache_uses_its_own_dtype_without_mutating_target_config():
    target_config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="auto"),
        speculative_config=SimpleNamespace(kv_cache_dtype="bfloat16"),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    spec = AscendSlidingWindowMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
        cache_dtype_str="bfloat16",
        model_version="deepseek_v4",
    )
    cache = DeepseekV41DSparkSWACache.__new__(DeepseekV41DSparkSWACache)
    with patch.object(
        AscendDeepseekV4SWACache,
        "get_kv_cache_spec",
        return_value=spec,
    ) as get_base_spec:
        draft = cache.get_kv_cache_spec(target_config)

    passed_config = get_base_spec.call_args.args[-1]
    assert passed_config is not target_config
    assert passed_config.cache_config is not target_config.cache_config
    assert passed_config.cache_config.cache_dtype == "bfloat16"
    assert target_config.cache_config.cache_dtype == "auto"
    assert type(draft) is DeepseekV41DraftSWASpec


def test_draft_cache_uses_mqsmla_packed_row_without_mutating_target_config():
    target_config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="auto"),
        speculative_config=SimpleNamespace(kv_cache_dtype="bfloat16"),
        additional_config={
            "dsv41_config": {
                "cache_format": "a5_packed",
            }
        },
    )
    cache = DeepseekV41DSparkSWACache.__new__(DeepseekV41DSparkSWACache)
    cache.block_size = 128
    cache.window_size = 128

    draft = cache.get_kv_cache_spec(target_config)

    assert type(draft) is DeepseekV41A5DraftSWASpec
    assert draft.dtype == torch.uint8
    assert draft.head_size == 544
    assert draft.logical_head_size == 512
    assert draft.quant_group_size == 32
    assert draft.page_size_bytes == 128 * 544
    assert target_config.cache_config.cache_dtype == "auto"


def test_composite_config_selects_checkpoint_aux_layers():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    text = SimpleNamespace(dspark_target_layer_ids=[37, 38, 39])
    runner.speculative_config = SimpleNamespace(
        use_dspark=lambda: True,
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(text_config=text)),
    )
    with patch.object(GPUModelRunner, "_get_eagle3_aux_layers_from_config", return_value=None):
        assert runner._get_eagle3_aux_layers_from_config() == (38, 39, 40)


def test_target_exports_residual_entering_selected_layers(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v41.model.get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    class Layer:
        def __init__(self, index):
            self.layer_idx = index
            self.engram = None

        def __call__(self, positions, hidden, pre_mix, unused, input_ids):
            return hidden + self.layer_idx + 1, pre_mix

        @staticmethod
        def hc_collapse(hidden, pre_mix):
            return hidden.mean(dim=1)

    model = SimpleNamespace(
        hc_mult=4,
        needs_moe_input_ids=False,
        prepare_engram=lambda input_ids, positions: ({}, torch.empty(0, dtype=torch.bool)),
        aux_hidden_state_layers=(1, 3),
        shared_attention_state=SimpleNamespace(reset=lambda: None),
        layers=[Layer(i) for i in range(3)],
        norm=lambda hidden: hidden,
    )
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    output, aux = DeepseekV41Model.forward(model, torch.arange(3), torch.arange(3), None, inputs_embeds=hidden)
    torch.testing.assert_close(aux[0], hidden)
    torch.testing.assert_close(aux[1], hidden + 3)
    torch.testing.assert_close(output, hidden + 6)


@pytest.mark.parametrize("cp", [False, True])
def test_v41_draft_routes_to_v41_and_disables_post_projection_q_norm(cp):
    from vllm_ascend.models.deepseek_v4.model import DeepseekV4Attention

    ordinary_backend = SimpleNamespace(apply_q_norm=True)
    draft_backend = SimpleNamespace(apply_q_norm=True)

    target_config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="auto"),
        speculative_config=SimpleNamespace(kv_cache_dtype="bfloat16"),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )

    def initialize_base(instance, **kwargs):
        torch.nn.Module.__init__(instance)
        instance.compress_ratio = 0
        instance.scale = 512**-0.5
        instance.dsa_attn = SimpleNamespace(dsa_attn=SimpleNamespace(impl=draft_backend))

    from vllm_ascend.attention.context_parallel.dsa_v41_cp import DeepseekV41CPImpl
    from vllm_ascend.attention.dsa_v41 import DeepseekV41EagerAttentionImpl

    with (
        patch.object(DeepseekV4Attention, "__init__", initialize_base),
        patch("vllm_ascend.attention.context_parallel.dsa_v41_cp.enable_dsa_cp", return_value=cp),
        patch("vllm_ascend.attention.context_parallel.dsa_v41_cp.enable_pcp", return_value=False),
    ):
        draft = DeepseekV41DSparkAttention(vllm_config=target_config, prefix="mtp.0.self_attn")
    assert type(draft.v41_impl) is (DeepseekV41CPImpl if cp else DeepseekV41EagerAttentionImpl)
    assert target_config.compilation_config.static_forward_context[draft.v41_layer_name] is draft
    assert draft.softmax_scale == 512**-0.5
    assert draft.dsa_attn.dsa_attn.impl.apply_q_norm is False
    assert draft.dsa_attn.dsa_attn.impl.vllm_config.cache_config.cache_dtype == "bfloat16"
    assert target_config.cache_config.cache_dtype == "auto"
    assert ordinary_backend.apply_q_norm is True


def test_v41_draft_selects_mqsmla_rollout_without_bf16_config_copy():
    from vllm_ascend.models.deepseek_v4.model import DeepseekV4Attention

    draft_backend = SimpleNamespace(apply_q_norm=True)
    target_config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="auto"),
        speculative_config=SimpleNamespace(kv_cache_dtype="bfloat16"),
        compilation_config=SimpleNamespace(static_forward_context={}),
        additional_config={
            "dsv41_config": {
                "cache_format": "a5_packed",
            }
        },
    )

    def initialize_base(instance, **kwargs):
        torch.nn.Module.__init__(instance)
        instance.compress_ratio = 0
        instance.scale = 512**-0.5
        instance.dsa_attn = SimpleNamespace(dsa_attn=SimpleNamespace(impl=draft_backend))

    with (
        patch.object(DeepseekV4Attention, "__init__", initialize_base),
        patch(
            "vllm_ascend.attention.context_parallel.dsa_v41_cp.enable_dsa_cp",
            return_value=False,
        ),
        patch(
            "vllm_ascend.attention.context_parallel.dsa_v41_cp.enable_pcp",
            return_value=False,
        ),
    ):
        draft = DeepseekV41DSparkAttention(vllm_config=target_config, prefix="mtp.0.self_attn")

    assert draft.uses_a5_packed_cache
    assert draft.dsa_attn.dsa_attn.impl.vllm_config is target_config
    assert target_config.cache_config.cache_dtype == "auto"


def test_v41_dspark_main_projection_uses_checkpoint_fp8_config():
    class DraftLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = SimpleNamespace(gate=SimpleNamespace(tid2eid=None, bias_vl=None))

    quant_config = object()
    text_config = SimpleNamespace(
        hc_mult=4,
        hidden_size=5120,
        dspark_block_size=5,
        dspark_target_layer_ids=[37, 38, 39],
        n_mtp_layers=3,
        num_hidden_layers=40,
        vocab_size=129280,
        rms_norm_eps=1e-6,
    )
    checkpoint_config = SimpleNamespace(
        text_config=text_config,
        quantization_config={"quant_method": "fp8"},
    )
    vllm_config = SimpleNamespace(
        quant_config=quant_config,
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=checkpoint_config,
                hf_text_config=text_config,
            )
        ),
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
    )
    column_parallel_linear = MagicMock(side_effect=lambda *args, **kwargs: torch.nn.Identity())

    with (
        patch("vllm_ascend.models.deepseek_v41.dspark.validate_cache_runtime"),
        patch(
            "vllm_ascend.models.deepseek_v41.dspark.VocabParallelEmbedding",
            side_effect=lambda *args, **kwargs: torch.nn.Identity(),
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.dspark.DeepseekV41DSparkDecoderLayer",
            side_effect=lambda *args, **kwargs: DraftLayer(),
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.dspark.ColumnParallelLinear",
            column_parallel_linear,
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.dspark.RMSNorm",
            side_effect=lambda *args, **kwargs: torch.nn.Identity(),
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.dspark.DSparkMarkovHead",
            side_effect=lambda *args, **kwargs: torch.nn.Identity(),
        ),
        patch(
            "vllm_ascend.models.deepseek_v41.dspark.DSparkConfidenceHead",
            side_effect=lambda *args, **kwargs: torch.nn.Identity(),
        ),
    ):
        DeepseekV41DSparkModel(vllm_config=vllm_config)

    assert column_parallel_linear.call_args.kwargs["quant_config"] is quant_config


def test_v41_draft_sequence_parallel_shards_inputs_and_restores_output(monkeypatch):
    class Layer:
        @staticmethod
        def hc_collapse(hidden, pre_mix):
            return hidden.mean(dim=1)

        def __call__(
            self,
            positions,
            hidden,
            pre_mix,
            llama_4_scaling=None,
            input_ids=None,
        ):
            return hidden, pre_mix

    hidden = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    input_ids = torch.tensor([11, 12, 13, 14])
    padding = torch.tensor([False, True, False, False])
    forward_context = SimpleNamespace(is_padding=padding)
    sharded_hidden = hidden[:2].unsqueeze(1).repeat(1, 4, 1)
    sharded_ids = input_ids[:2]
    gathered = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    sp_shard = MagicMock(side_effect=[sharded_hidden, sharded_ids])
    sp_all_gather = MagicMock(return_value=gathered)
    padding_mask = MagicMock(return_value=torch.tensor([False, True]))
    monkeypatch.setattr(deepseek_v41_dspark_module, "sp_shard", sp_shard)
    monkeypatch.setattr(deepseek_v41_dspark_module, "sp_all_gather", sp_all_gather)
    monkeypatch.setattr(deepseek_v41_dspark_module, "sp_padding_mask", padding_mask)
    monkeypatch.setattr(deepseek_v41_dspark_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(deepseek_v41_dspark_module, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(deepseek_v41_dspark_module.envs, "VLLM_MOE_SKIP_PADDING", True)

    model = SimpleNamespace(
        embed_tokens=MagicMock(return_value=hidden),
        hc_mult=4,
        use_sequence_parallel=True,
        needs_moe_input_ids=False,
        layers={"40": Layer()},
    )

    output = DeepseekV41DSparkModel.forward(model, input_ids, torch.arange(4))

    padding_mask.assert_called_once()
    assert forward_context.is_padding.tolist() == [False, True]
    assert sp_shard.call_args_list[0].args[0].shape == (4, 4, 4)
    assert sp_shard.call_args_list[1].args[0] is input_ids
    sp_all_gather.assert_called_once()
    torch.testing.assert_close(output, gathered[:4])


def test_v41_draft_context_store_uses_physical_pairs_and_preserves_padding():
    from vllm_ascend.models.deepseek_v41.dspark import DeepseekV41DSparkModel

    cache = torch.empty(3, 128, 1, 8)
    attn = SimpleNamespace(dsa_attn=SimpleNamespace(swa_cache_layer=SimpleNamespace(block_size=128, kv_cache=[cache])))
    values = torch.randn(3, 1, 8)
    with patch("vllm_ascend.models.deepseek_v41.dspark.scatter_cache_sk") as store:
        DeepseekV41DSparkModel._store_standard_swa_kv(None, values, torch.tensor([129, -1, 258]), attn)
    actual_cache, slots, updates = store.call_args.args
    assert actual_cache is cache
    assert slots.tolist() == [[1, 1], [-1, -1], [2, 2]]
    torch.testing.assert_close(updates, values.squeeze(1))


def test_v41_mqsmla_context_store_packs_win_rows_before_scatter():
    from vllm_ascend.models.deepseek_v41.dspark import DeepseekV41DSparkModel

    cache = torch.empty(3, 128, 1, 544, dtype=torch.uint8)
    attn = SimpleNamespace(
        dsa_attn=SimpleNamespace(swa_cache_layer=SimpleNamespace(block_size=128, kv_cache=[cache])),
        uses_a5_packed_cache=True,
    )
    values = torch.randn(3, 1, 512)
    with (
        patch("vllm_ascend.models.deepseek_v41.dspark.write_attention_cache") as store,
        patch("vllm_ascend.models.deepseek_v41.dspark.scatter_cache_sk") as legacy_store,
    ):
        DeepseekV41DSparkModel._store_standard_swa_kv(None, values, torch.tensor([129, -1, 258]), attn)

    legacy_store.assert_not_called()
    actual_cache, slots, updates = store.call_args.args
    assert actual_cache is cache
    assert slots.tolist() == [[1, 1], [-1, -1], [2, 2]]
    torch.testing.assert_close(updates, values.squeeze(1))
    assert store.call_args.kwargs == {"kind": "win"}

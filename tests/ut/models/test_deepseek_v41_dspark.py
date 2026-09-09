# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred torch checks for Aurora DSpark model/cache integration."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.core.deepseek_v41 import DeepseekV41DraftSWASpec
from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec, register_ascend_kv_cache_specs
from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4SWACache
from vllm_ascend.models.deepseek_v41.dspark import (
    DeepseekV41DSparkAttention,
    DeepseekV41DSparkDecoderLayer,
    DeepseekV41DSparkSWACache,
)
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def test_draft_cache_keeps_dsa_backend_and_uses_explicit_aurora_spec():
    spec = AscendSlidingWindowMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
        cache_dtype_str="bfloat16",
        model_version="deepseek_v4",
    )
    with patch.object(AscendDeepseekV4SWACache, "get_kv_cache_spec", return_value=spec):
        draft = DeepseekV41DSparkSWACache.get_kv_cache_spec(SimpleNamespace(), None)
    assert type(draft) is DeepseekV41DraftSWASpec
    assert draft.page_size_bytes == 131072
    assert DeepseekV41DSparkDecoderLayer.attention_cls is DeepseekV41DSparkAttention
    assert DeepseekV41DSparkAttention.swa_cache_cls is DeepseekV41DSparkSWACache
    register_ascend_kv_cache_specs()
    assert KVCacheSpecRegistry.get_manager_class(draft) is SlidingWindowManager


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

        def __call__(self, positions, hidden, pre_mix, unused, input_ids):
            return hidden + self.layer_idx + 1, pre_mix

        @staticmethod
        def hc_collapse(hidden, pre_mix):
            return hidden.mean(dim=1)

    model = SimpleNamespace(
        hc_mult=4,
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


def test_v41_draft_disables_only_its_backend_post_projection_q_norm():
    from vllm_ascend.models.deepseek_v4.model import DeepseekV4Attention

    ordinary_backend = SimpleNamespace(apply_q_norm=True)
    draft_backend = SimpleNamespace(apply_q_norm=True)

    def initialize_base(instance, **kwargs):
        torch.nn.Module.__init__(instance)
        instance.compress_ratio = 0
        instance.dsa_attn = SimpleNamespace(dsa_attn=SimpleNamespace(impl=draft_backend))

    with patch.object(DeepseekV4Attention, "__init__", initialize_base):
        draft = DeepseekV41DSparkAttention()
    assert draft.dsa_attn.dsa_attn.impl.apply_q_norm is False
    assert ordinary_backend.apply_q_norm is True

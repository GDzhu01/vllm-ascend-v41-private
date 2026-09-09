# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend multimodal wrapper for DeepSeek V4.1."""

import torch
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_ascend.models.deepseek_v4.vl_model import (
    AscendDeepseekV4ForConditionalGeneration,
)

from .mm_preprocess import (
    DeepseekV41VLDummyInputsBuilder,
    DeepseekV41VLMultiModalProcessor,
    DeepseekV41VLProcessingInfo,
)
from .model import AscendDeepseekV41ForCausalLM


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV41VLMultiModalProcessor,
    info=DeepseekV41VLProcessingInfo,
    dummy_inputs=DeepseekV41VLDummyInputsBuilder,
)
class AscendDeepseekV41ForConditionalGeneration(
    AscendDeepseekV4ForConditionalGeneration,
):
    """Reuse the V4 vision frontend with the V4.1 language backbone."""

    language_model_cls = AscendDeepseekV41ForCausalLM

    def prepare_engram_inputs(self, input_ids, positions, padded_tokens=None):
        return self.language_model.prepare_engram_inputs(
            input_ids,
            positions,
            padded_tokens,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

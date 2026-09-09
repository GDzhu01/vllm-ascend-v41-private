# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""DeepSeek V4.1 multimodal preprocessing adapters."""

from vllm_ascend.deepseek_v41_config import DeepseekV41Config
from vllm_ascend.models.deepseek_v4.mm_preprocess import (
    DeepseekV4VLDummyInputsBuilder,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
)


class DeepseekV41VLProcessingInfo(DeepseekV4VLProcessingInfo):
    """Use the shared V4 processor with the V4.1 config type."""

    def get_hf_config(self) -> DeepseekV41Config:
        return self.ctx.get_hf_config(DeepseekV41Config)


class DeepseekV41VLDummyInputsBuilder(DeepseekV4VLDummyInputsBuilder):
    pass


class DeepseekV41VLMultiModalProcessor(DeepseekV4VLMultiModalProcessor):
    pass

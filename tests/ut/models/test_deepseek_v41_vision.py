from types import SimpleNamespace

from vllm.model_executor.models.interfaces import supports_multimodal
from vllm.multimodal.processing import InputProcessingContext

from vllm_ascend.deepseek_v41_config import DeepseekV41Config
from vllm_ascend.models.deepseek_v41.mm_preprocess import (
    DeepseekV41VLProcessingInfo,
)
from vllm_ascend.models.deepseek_v41.model import AscendDeepseekV41ForCausalLM
from vllm_ascend.models.deepseek_v41.vl_model import (
    AscendDeepseekV41ForConditionalGeneration,
)


def test_v41_vision_wrapper_uses_v41_language_backbone():
    assert supports_multimodal(AscendDeepseekV41ForConditionalGeneration)
    assert (
        AscendDeepseekV41ForConditionalGeneration.language_model_cls
        is AscendDeepseekV41ForCausalLM
    )
    assert "_processor_factory" in AscendDeepseekV41ForConditionalGeneration.__dict__


def test_v41_processing_info_accepts_v41_config():
    config = DeepseekV41Config(
        text_config={},
        vision_config={
            "num_hidden_layers": 1,
            "patch_size": 14,
            "downsample_ratio": 3,
            "max_num_tokens": 1024,
        },
    )
    model_config = SimpleNamespace(hf_config=config)
    ctx = InputProcessingContext(model_config=model_config, tokenizer=None)

    assert DeepseekV41VLProcessingInfo(ctx).get_hf_config() is config

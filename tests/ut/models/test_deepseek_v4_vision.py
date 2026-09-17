from unittest.mock import MagicMock

import torch
from torch import nn
from vllm.model_executor.models.interfaces import supports_eagle3

from vllm_ascend.models.deepseek_v4.vl_model import (
    AscendDeepseekV4ForConditionalGeneration,
)


def test_vision_wrapper_exposes_dspark_aux_hidden_state_interface():
    model = AscendDeepseekV4ForConditionalGeneration.__new__(AscendDeepseekV4ForConditionalGeneration)
    nn.Module.__init__(model)
    language_model = MagicMock()
    model.language_model = language_model

    assert supports_eagle3(model)

    model.set_aux_hidden_state_layers((41, 42, 43))
    language_model.set_aux_hidden_state_layers.assert_called_once_with((41, 42, 43))


def test_text_only_wrapper_skips_checkpoint_vision_weights():
    class LanguageModel(nn.Module):
        def load_weights(self, weights):
            self.loaded = list(weights)
            return {"model.embed_tokens.weight"}

    model = AscendDeepseekV4ForConditionalGeneration.__new__(AscendDeepseekV4ForConditionalGeneration)
    nn.Module.__init__(model)
    model.vision = None
    model.language_model = LanguageModel()
    language_weight = torch.ones(1)

    loaded = model.load_weights(
        iter(
            (
                ("aligner.w1.bias", torch.zeros(1)),
                ("model.embed_tokens.weight", language_weight),
            )
        )
    )

    assert model.language_model.loaded == [("model.embed_tokens.weight", language_weight)]
    assert loaded == {"language_model.model.embed_tokens.weight"}

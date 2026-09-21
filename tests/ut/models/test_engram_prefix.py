# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of runner-to-Engram preparation; model weights are replaced."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_ascend.models.deepseek_v41 import model as model_module
from vllm_ascend.models.deepseek_v41.engram_hash import PagedNgramHistory
from vllm_ascend.models.deepseek_v41.model import AscendDeepseekV41ForCausalLM, DeepseekV41Model
from vllm_ascend.models.deepseek_v41.vl_model import AscendDeepseekV41ForConditionalGeneration
from vllm_ascend.worker import model_runner_v1 as runner_module


@pytest.fixture
def engram_forward(monkeypatch):
    config = SimpleNamespace(
        engram_layer_ids=(0,),
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_vocab_size=17,
        engram_num_embeddings=(256,),
        engram_head_dim=8,
        engram_compressed_vocab_size=2048,
        engram_pad_id=0,
        image_token_id=2046,
        image_pad_token_id=2047,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"token{i}": i for i in range(2048)}, unk_token="token0"))
    )
    history = PagedNgramHistory(config, tokenizer)
    # The lookup leaf returns its row IDs, making every hash directly observable.
    backbone = SimpleNamespace(
        config=config,
        engram_history=history,
        _engram_max_tokens=256,
        _engram_input_buffers=None,
        layers=[
            SimpleNamespace(
                self_attn=SimpleNamespace(dsa_attn=SimpleNamespace(swa_cache_layer=SimpleNamespace(prefix="swa"))),
                engram=SimpleNamespace(embed=lambda ids: ids),
            )
        ],
    )
    backbone.prepare_engram = partial(DeepseekV41Model.prepare_engram, backbone)
    backbone.prepare_engram_inputs = partial(DeepseekV41Model.prepare_engram_inputs, backbone)
    language = SimpleNamespace(model=backbone)
    language.prepare_engram_inputs = partial(AscendDeepseekV41ForCausalLM.prepare_engram_inputs, language)
    model = torch.nn.Module()
    model.language_model = language
    model.prepare_engram_inputs = partial(AscendDeepseekV41ForConditionalGeneration.prepare_engram_inputs, model)
    model.forward = lambda **kwargs: kwargs
    runner = SimpleNamespace(model=model, enable_enpu=False, _update_full_graph_params_if_needed=lambda *args: None)
    context = SimpleNamespace(attn_metadata=None)
    monkeypatch.setattr(model_module, "get_ascend_config", lambda: SimpleNamespace(enable_engram=True))
    monkeypatch.setattr(model_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(runner_module, "get_forward_context", lambda: context)

    def forward(tokens, start, end, pages, *, output_tokens=(), host_outputs=None):
        # Same request state type the runner constructs from SchedulerOutput.
        state = CachedRequestState(
            req_id="req",
            prompt_token_ids=tokens,
            mm_features=[],
            sampling_params=None,
            generator=None,
            block_ids=(pages,),
            num_computed_tokens=start,
            output_token_ids=list(output_tokens if host_outputs is None else host_outputs),
        )
        runner.requests = {"req": state}
        runner.input_batch = SimpleNamespace(req_ids=["req"])
        context.attn_metadata = {
            "swa": SimpleNamespace(
                query_start_loc_cpu=torch.tensor([0, end - start]),
                block_table_cpu=torch.tensor([pages]),
                storage_block_size=128,
            )
        }
        sequence = tokens + list(output_tokens)
        actual = runner_module.NPUModelRunner._model_forward(
            runner,
            end - start,
            input_ids=torch.tensor(sequence[start:end]),
            positions=torch.arange(start, end),
        )["engram_lookups"][0][: end - start].clone()
        # Independent full-sequence run, with no prefix skipping or restoration.
        reference = PagedNgramHistory(config, tokenizer)
        expected, _ = reference.update(
            torch.tensor(sequence[:end]),
            torch.arange(end),
            torch.zeros(end, dtype=torch.long),
            torch.arange((end + 127) // 128).unsqueeze(0),
            128,
        )
        torch.testing.assert_close(actual, expected[start:end].flatten(1))
        return actual

    return history, forward


@pytest.mark.parametrize("query_tokens", [1, 32])
@pytest.mark.parametrize("barrier", [None, 2046, 2047])
@pytest.mark.parametrize("stale_page", [False, True])
def test_transferred_prefix_hashes_match_raw_sequence(engram_forward, query_tokens, barrier, stale_page):
    history, forward = engram_forward
    tokens = list(range(10, 138 + query_tokens))
    if barrier is not None:
        tokens[126] = barrier
    if stale_page:
        forward([321] * 128, 0, 128, [11])
    forward(tokens, 128, len(tokens), [11, 17])
    assert (history.pages[11][:125] == (321 if stale_page else -1)).all()


def test_async_host_placeholders_preserve_computed_history(engram_forward):
    _, forward = engram_forward
    prompt = list(range(10, 136))
    forward(prompt, 0, 126, [11])
    forward(prompt, 126, 128, [11], output_tokens=[201, 202], host_outputs=[-1, -1])
    forward(prompt, 128, 129, [11, 17], output_tokens=[201, 202, 203], host_outputs=[-1, -1, -1])


def test_resumed_prefix_restores_accepted_output_tokens(engram_forward):
    _, forward = engram_forward
    forward(list(range(10, 135)), 128, 132, [23, 45], output_tokens=list(range(201, 208)))

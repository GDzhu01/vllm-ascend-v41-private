# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep CPU futures outside compilation without disabling the backbone."""

import torch
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op


def engram_prefetch_consume(
    hidden_states: torch.Tensor,
    fallback: torch.Tensor,
    output: torch.Tensor,
    model_name: str,
    slot: int,
    prefetch_dependency: torch.Tensor | None = None,
) -> None:
    # Like vLLM's KV-cache dummy dependency, this input keeps the producer
    # in the AOT graph even though its actual result lives in host state.
    del prefetch_dependency
    model = get_forward_context().no_compile_layers[model_name]
    prefetcher = model._engram_prefetcher
    values = (
        prefetcher.consume(slot, hidden_states.shape[0]) if prefetcher is not None and prefetcher.pending else fallback
    )
    output.copy_(values)


def engram_prefetch_consume_fake(
    hidden_states: torch.Tensor,
    fallback: torch.Tensor,
    output: torch.Tensor,
    model_name: str,
    slot: int,
    prefetch_dependency: torch.Tensor | None = None,
) -> None:
    return None


def engram_prefetch_start(hidden_states: torch.Tensor, model_name: str) -> torch.Tensor:
    model = get_forward_context().no_compile_layers[model_name]
    prefetcher = model._engram_prefetcher
    if prefetcher is not None and prefetcher.pending:
        prefetcher.start()
    # Only an ordering token, never read by a device kernel. Unlike consume's
    # output, this zero-length tensor carries no replay-varying data.
    return hidden_states.new_empty(0)


def engram_prefetch_start_fake(hidden_states: torch.Tensor, model_name: str) -> torch.Tensor:
    return hidden_states.new_empty(0)


direct_register_custom_op(
    op_name="engram_prefetch_consume",
    op_func=engram_prefetch_consume,
    mutates_args=["output"],
    fake_impl=engram_prefetch_consume_fake,
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="engram_prefetch_start",
    op_func=engram_prefetch_start,
    mutates_args=[],
    fake_impl=engram_prefetch_start_fake,
    dispatch_key="PrivateUse1",
)

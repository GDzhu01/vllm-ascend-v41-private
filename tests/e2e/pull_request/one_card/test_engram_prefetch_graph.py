# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device regression for eager Engram consumption followed by intact replay.

The transport is stubbed: distributed lookup is covered by the Engram tests.
This test exercises real NPU custom-op dispatch, capture and repeated replay.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41 import engram_prefetch_op as op


def test_prefill_then_full_decode_refreshes_lookup_without_host_work(monkeypatch):
    hidden = torch.ones((4, 32), dtype=torch.bfloat16, device="npu")
    fallback = torch.zeros_like(hidden)
    values = torch.empty_like(hidden)
    prefill_values = torch.full_like(hidden, 5)
    prefetcher = SimpleNamespace(pending=[object()], start=Mock(), consume=Mock(return_value=prefill_values))
    model = SimpleNamespace(_engram_prefetcher=prefetcher)
    context = SimpleNamespace(no_compile_layers={"probe": model})
    monkeypatch.setattr(op, "get_forward_context", lambda: context)

    def backbone():
        dependency = torch.ops.vllm.engram_prefetch_start(hidden, "probe")
        torch.ops.vllm.engram_prefetch_consume(hidden, fallback, values, "probe", 0, dependency)
        return hidden + values

    torch.testing.assert_close(backbone().cpu(), torch.full((4, 32), 6, dtype=torch.bfloat16))
    prefetcher.start.assert_called_once()
    prefetcher.consume.assert_called_once_with(0, 4)

    # The runner drains prefill before decode or graph capture. Existing
    # prefetcher objects must not cause capture to bake in a prefill result.
    prefetcher.pending.clear()
    prefetcher.start.side_effect = AssertionError("CPU worker started during decode")
    prefetcher.consume.side_effect = AssertionError("CPU lookup consumed during decode")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = backbone()
    addresses = (fallback.data_ptr(), values.data_ptr(), output.data_ptr())

    # A full replay must not re-enter the Python custom ops. Each step writes
    # new synchronous lookup results into the captured buffer before replay.
    monkeypatch.setattr(op, "get_forward_context", Mock(side_effect=AssertionError("host op during replay")))
    for value in (2, -3, 7):
        fallback.fill_(value)
        graph.replay()
        torch.npu.synchronize()
        assert (fallback.data_ptr(), values.data_ptr(), output.data_ptr()) == addresses
        torch.testing.assert_close(output.cpu(), torch.full((4, 32), value + 1, dtype=torch.bfloat16))

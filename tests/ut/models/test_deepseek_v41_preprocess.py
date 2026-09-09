# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention import dsa_v41


@pytest.mark.parametrize("share_quant", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 5])
@torch.inference_mode()
def test_preprocess_equivalence_and_stream_dependencies(monkeypatch, share_quant, num_tokens):
    """Check Q/qr/cache parity and the cross-stream producer/consumer ordering."""
    trace = []
    active = "main"

    class Stream:
        def __init__(self, name):
            self.name = name

        def record_event(self):
            event = f"event{len(trace)}"
            trace.append((self.name, "record", event))
            return event

        def wait_event(self, event):
            trace.append((self.name, "wait", event))

        def wait_stream(self, stream):
            trace.append((self.name, "join", stream.name))

    main, aux = Stream("main"), Stream("aux")

    @contextmanager
    def switch(stream, *, enabled):
        nonlocal active
        previous = active
        active = stream.name
        try:
            yield
        finally:
            active = previous

    class Wrapper:
        _has_communication = False

        def __init__(self, name, linear, quant_method):
            self.name, self.linear = name, linear
            self._quant_method = quant_method

        def quantize(self, x):
            trace.append((active, self.name, "quantize"))
            return x, None

        def matmul(self, x, scale, bias=None):
            trace.append((active, self.name, "matmul"))
            assert bias is self.linear.bias
            return self.linear(x)

    def norm(name):
        def apply(x):
            trace.append((active, name, "norm"))
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

        return apply

    def rope(x, cos, sin, **kwargs):
        trace.append((active, "rope", "apply"))
        # Exercise the in-place write on a view, including Q/KV partial slices.
        start, end = kwargs["partial_slice"]
        x[..., start:end].neg_()

    def scatter(cache, slots, values):
        trace.append((active, "cache", "scatter"))
        for slot, value in zip(slots, values):
            if slot[0] >= 0:
                cache[slot[0], slot[1]].copy_(value)

    monkeypatch.setattr(torch.npu, "current_stream", lambda: main)
    monkeypatch.setattr(dsa_v41, "dsv4_dsa_overlap_stream", lambda: aux)
    monkeypatch.setattr(dsa_v41, "npu_stream_switch", switch)
    monkeypatch.setattr(dsa_v41, "scatter_cache_v2", scatter)
    monkeypatch.setattr(torch.ops._C_ascend, "inplace_partial_rotary_mul", rope, raising=False)
    torch.manual_seed(7)
    cache = torch.zeros(2, num_tokens, 4)
    q_a, q_b, kv = torch.nn.Linear(8, 6), torch.nn.Linear(6, 8), torch.nn.Linear(8, 4)
    wrappers = SimpleNamespace(
        cv_wq_a=Wrapper("qa", q_a, object()),
        cv_wq_b=Wrapper("qb", q_b, object()),
        cv_wkv=Wrapper("kv", kv, object() if share_quant else SimpleNamespace()),
    )
    attn = SimpleNamespace(
        wq_a=q_a,
        wq_b=q_b,
        wkv=kv,
        q_norm=norm("q"),
        kv_norm=norm("kv"),
        n_local_heads=2,
        head_dim=4,
        nope_head_dim=2,
        dsa_attn=SimpleNamespace(
            swa_cache_layer=SimpleNamespace(kv_cache=[cache]),
            dsa_attn=SimpleNamespace(impl=wrappers),
        ),
    )
    slots = torch.tensor([[1, i] for i in range(num_tokens)])
    if num_tokens > 1:
        slots[-1] = -1
    metadata = SimpleNamespace(slot_mapping=slots)
    hidden = torch.randn(num_tokens, 8)
    impl = object.__new__(dsa_v41.DeepseekV41EagerAttentionImpl)
    expected_q, expected_qr = impl.preprocess(attn, hidden, None, None, metadata)
    expected_cache = cache.clone()
    cache.zero_()
    trace.clear()
    q, qr = impl.multistream_preprocess(attn, hidden, None, None, metadata)
    torch.testing.assert_close(q, expected_q)
    torch.testing.assert_close(qr, expected_qr)
    torch.testing.assert_close(cache, expected_cache)
    assert qr.is_floating_point()
    assert ("aux", "kv", "quantize") in trace if not share_quant else ("aux", "kv", "quantize") not in trace
    kv_mm = trace.index(("aux", "kv", "matmul"))
    kv_done = trace[kv_mm + 1]
    assert kv_done[:2] == ("aux", "record")
    assert trace.index(("main", "wait", kv_done[2])) < trace.index(("main", "qb", "matmul"))
    part3 = trace[trace.index(("main", "wait", kv_done[2])) - 1]
    assert part3[:2] == ("main", "record")
    assert trace.index(("aux", "wait", part3[2])) < trace.index(("aux", "kv", "norm"))
    assert trace.index(("aux", "cache", "scatter")) < trace.index(("main", "join", "aux"))
    assert trace.index(("main", "join", "aux")) < trace.index(("main", "rope", "apply"))


@pytest.mark.parametrize("enabled", [False, True])
def test_forward_selects_preprocess_from_v1_switch(monkeypatch, enabled):
    impl = object.__new__(dsa_v41.DeepseekV41EagerAttentionImpl)
    impl.role = SimpleNamespace(is_kv_source=False)
    hidden = torch.zeros(1, 8)
    q, qr = torch.zeros(1, 2, 4), torch.zeros(1, 6)
    metadata = SimpleNamespace(positions=torch.zeros(1), swa=object(), rope=lambda *args: (None, None))
    impl._get_layer_metadata = Mock(return_value=metadata)
    impl.preprocess = Mock(return_value=(q, qr))
    impl.multistream_preprocess = Mock(return_value=(q, qr))
    impl._select_sparse_indices = Mock(return_value=None)
    impl._attention = Mock(return_value=q)
    v1_impl = SimpleNamespace(
        multistream_dsv4_dsa_overlap=enabled,
        _forward_o_proj=lambda q, output: output.zero_(),
    )
    attn = SimpleNamespace(
        rotary_emb=SimpleNamespace(layername="layer"),
        dsa_attn=SimpleNamespace(dsa_attn=SimpleNamespace(impl=v1_impl)),
        nope_head_dim=2,
        head_dim=4,
    )
    metadata.rope = lambda *args: (torch.zeros(1), torch.zeros(1))
    monkeypatch.setattr(dsa_v41, "get_forward_context", lambda: SimpleNamespace(attn_metadata={}))
    monkeypatch.setattr(torch.ops._C_ascend, "inplace_partial_rotary_mul", lambda *args, **kwargs: None, raising=False)
    impl.forward(attn, None, hidden)
    selected = impl.multistream_preprocess if enabled else impl.preprocess
    unused = impl.preprocess if enabled else impl.multistream_preprocess
    selected.assert_called_once()
    unused.assert_not_called()

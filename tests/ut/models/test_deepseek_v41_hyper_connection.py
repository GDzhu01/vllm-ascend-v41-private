# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.models.deepseek_v41.model import DeepseekV41DecoderLayer

pytestmark = pytest.mark.usefixtures("mock_npu_rms_norm")


def _layer() -> DeepseekV41DecoderLayer:
    layer = DeepseekV41DecoderLayer.__new__(DeepseekV41DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.hc_mult = 4
    layer.hc_sinkhorn_iters = 3
    layer.norm_eps = 1e-6
    layer.hc_eps = 1e-6
    return layer


def test_v41_hc_pre_dispatches_fused_operator_with_pre_mix():
    layer = _layer()
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    hc_fn = torch.randn(24, 32, dtype=torch.float32)
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(24, dtype=torch.float32)
    pre_mix = torch.randn(2, 4, dtype=torch.float32)
    expected = (
        torch.randn(2, 8, dtype=torch.bfloat16),
        torch.randn(2, 4),
        torch.randn(2, 4, 4),
        torch.randn(2, 4),
    )

    with patch.object(
        torch.ops._C_ascend,
        "npu_hc_pre_v2",
        create=True,
        return_value=expected,
    ) as op:
        actual = layer.hc_pre(x, hc_fn, hc_scale, hc_base, pre_mix)

    assert actual is expected
    op.assert_called_once_with(
        x,
        hc_fn,
        hc_scale,
        hc_base,
        pre_mix,
        hc_mult=4,
        hc_sinkhorn_iters=3,
        norm_eps=1e-6,
        hc_eps=1e-6,
    )


def test_v41_forward_threads_pre_mix_through_fused_hc_pre():
    layer = _layer()
    hidden_states = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    incoming_pre = torch.randn(2, 4, dtype=torch.float32)
    attn_pre = torch.randn(2, 4, dtype=torch.float32)
    ffn_pre = torch.randn(2, 4, dtype=torch.float32)
    post = torch.randn(2, 4, dtype=torch.float32)
    comb = torch.randn(2, 4, 4, dtype=torch.float32)
    collapsed = torch.randn(2, 8, dtype=torch.bfloat16)
    layer.hc_attn_fn = torch.nn.Parameter(torch.empty(24, 32))
    layer.hc_attn_scale = torch.nn.Parameter(torch.empty(3))
    layer.hc_attn_base = torch.nn.Parameter(torch.empty(24))
    layer.hc_ffn_fn = torch.nn.Parameter(torch.empty(24, 32))
    layer.hc_ffn_scale = torch.nn.Parameter(torch.empty(3))
    layer.hc_ffn_base = torch.nn.Parameter(torch.empty(24))
    layer.hc_pre = MagicMock(
        side_effect=[
            (collapsed, post, comb, attn_pre),
            (collapsed, post, comb, ffn_pre),
        ]
    )
    layer.input_layernorm = MagicMock(side_effect=lambda value: value)
    layer.post_attention_layernorm = MagicMock(side_effect=lambda value: value)
    layer.self_attn = MagicMock(side_effect=lambda _positions, value, _scaling: value)
    layer.mlp = MagicMock(side_effect=lambda value, _input_ids: value)
    layer.hc_post = MagicMock(side_effect=lambda _x, residual, _post, _comb: residual)

    output, next_pre = layer.forward(
        torch.arange(2), hidden_states, incoming_pre, input_ids=None
    )

    assert output is hidden_states
    assert next_pre is ffn_pre
    assert layer.hc_pre.call_args_list[0].args[-1] is incoming_pre
    assert layer.hc_pre.call_args_list[1].args[-1] is attn_pre


def test_v41_hc_reference_supports_hidden_size_5120():
    torch.manual_seed(7)
    layer = _layer()
    x = torch.randn(2, 4, 5120, dtype=torch.bfloat16)
    hc_fn = torch.randn(24, 4 * 5120, dtype=torch.float32) / 5120
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(24, dtype=torch.float32)

    pre, post, comb = layer.hc_mixes(x, hc_fn, hc_scale, hc_base)
    y = layer.hc_collapse(x, pre)

    assert y.shape == (2, 5120)
    assert y.dtype == torch.bfloat16
    assert post.shape == (2, 4)
    assert post.dtype == torch.float32
    assert comb.shape == (2, 4, 4)
    assert comb.dtype == torch.float32
    torch.testing.assert_close(comb.sum(-2), torch.ones(2, 4), atol=2e-5, rtol=2e-5)

    restored = layer.hc_post_reference(y, x, post, comb)
    assert restored.shape == x.shape
    assert restored.dtype == x.dtype


def test_v41_hc_post_matches_reference_equation():
    torch.manual_seed(11)
    layer = _layer()
    x = torch.randn(3, 5, dtype=torch.bfloat16)
    residual = torch.randn(3, 4, 5, dtype=torch.bfloat16)
    post = torch.randn(3, 4, dtype=torch.float32)
    comb = torch.randn(3, 4, 4, dtype=torch.float32)

    actual = layer.hc_post_reference(x, residual, post, comb)
    expected = (
        post.unsqueeze(-1) * x.unsqueeze(-2)
        + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=-3)
    ).to(x.dtype)
    torch.testing.assert_close(actual, expected)


def test_v41_hc_post_dispatches_fused_operator_with_batch_dimension():
    layer = _layer()
    x = torch.randn(3, 5, dtype=torch.bfloat16)
    residual = torch.randn(3, 4, 5, dtype=torch.bfloat16)
    post = torch.randn(3, 4, dtype=torch.float32)
    comb = torch.randn(3, 4, 4, dtype=torch.float32)
    expected = torch.randn_like(residual).unsqueeze(0)

    with patch.object(
        torch.ops._C_ascend,
        "npu_hc_post",
        create=True,
        return_value=expected,
    ) as op:
        actual = layer.hc_post(x, residual, post, comb)

    torch.testing.assert_close(actual, expected.squeeze(0))
    op.assert_called_once()
    for actual_arg, expected_arg in zip(
        op.call_args.args,
        (
            x.unsqueeze(0),
            residual.unsqueeze(0),
            post.unsqueeze(0),
            comb.unsqueeze(0),
        ),
    ):
        torch.testing.assert_close(actual_arg, expected_arg)

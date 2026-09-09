# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU coverage for the shared compressor and indexer K normalization."""

import pytest
import torch
import torch_npu

from vllm_ascend.models.deepseek_v41.compressor import DeepseekV41RMSNorm
from vllm_ascend.models.deepseek_v41.engram_gate import engram_gate
from vllm_ascend.models.deepseek_v41.model import DeepseekV41DecoderLayer

NORM_CASES = (
    (128, True, torch.bfloat16),
    (512, True, torch.bfloat16),
    (5120, False, torch.float32),
    (20480, False, torch.float32),
)


def _reference(x, weight, eps):
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return normalized.to(x.dtype) * weight


def _npu_norm(width, has_weight, dtype, eps):
    if has_weight:
        norm = DeepseekV41RMSNorm(width, eps).npu()
        norm.weight.copy_(torch.randn_like(norm.weight))
        return norm, norm.weight.cpu()
    gamma = torch.ones(width, dtype=dtype, device="npu")
    return lambda x: torch_npu.npu_rms_norm(x, gamma, epsilon=eps)[0], gamma.cpu()


@pytest.mark.parametrize("width,has_weight,dtype", NORM_CASES)
@pytest.mark.parametrize("tokens", [0, 1, 32, 4096])
@pytest.mark.parametrize("eps", [1e-6, 1e-3])
@pytest.mark.parametrize("scale", [0.0, 1e-4, 1.0])
@torch.inference_mode()
def test_rmsnorm_matches_reference(width, has_weight, dtype, tokens, eps, scale):
    torch.manual_seed(41)
    x = (torch.randn(tokens, width) * scale).to(dtype)
    norm, weight = _npu_norm(width, has_weight, dtype, eps)
    expected = _reference(x, weight, eps)
    x_npu = x.npu()
    actual = norm(x_npu)
    assert actual.shape == x.shape and actual.dtype == x.dtype
    torch.testing.assert_close(x_npu.cpu(), x, rtol=0, atol=0)
    # The reference rounds the normalized value to BF16 before applying gamma.
    tolerance = dict(rtol=0.016, atol=1e-5) if dtype == torch.bfloat16 else dict(rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.cpu(), expected, **tolerance)


@pytest.mark.parametrize("width,has_weight,dtype", NORM_CASES)
@torch.inference_mode()
def test_rmsnorm_graph_replay_uses_new_input(width, has_weight, dtype):
    torch.manual_seed(42)
    norm, weight = _npu_norm(width, has_weight, dtype, 1e-6)
    x = torch.randn(32, width, dtype=dtype, device="npu")
    norm(x)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        actual = norm(x)
    pointers = (x.data_ptr(), actual.data_ptr())
    for scale in (1.0, 1e-4, 0.0):
        updated = (torch.randn(32, width) * scale).to(dtype)
        x.copy_(updated)
        graph.replay()
        torch.npu.synchronize()
        assert (x.data_ptr(), actual.data_ptr()) == pointers
        expected = _reference(updated, weight, 1e-6)
        tolerance = dict(rtol=0.016, atol=1e-5) if dtype == torch.bfloat16 else dict(rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual.cpu(), expected, **tolerance)


@torch.inference_mode()
def test_weightless_hc_mixes_matches_reference():
    torch.manual_seed(43)
    width, hc_mult = 5120, 4
    layer = DeepseekV41DecoderLayer.__new__(DeepseekV41DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.hc_mult, layer.hc_sinkhorn_iters = hc_mult, 3
    layer.norm_eps, layer.hc_eps = 1e-6, 1e-6
    x = torch.randn(32, hc_mult, width, dtype=torch.bfloat16)
    hc_fn = torch.randn(2 * hc_mult + hc_mult**2, width * hc_mult) / width
    scale, base = torch.randn(3), torch.randn(2 * hc_mult + hc_mult**2)
    flat = x.float().flatten(-2)
    mixes = torch.nn.functional.linear(flat, hc_fn)
    mixes *= torch.rsqrt(flat.square().mean(-1, keepdim=True) + layer.norm_eps)
    pre, post, comb = mixes.split((hc_mult, hc_mult, hc_mult**2), -1)
    pre = torch.sigmoid(pre * scale[0] + base[:hc_mult]) + layer.hc_eps
    post = 2 * torch.sigmoid(post * scale[1] + base[hc_mult : 2 * hc_mult])
    comb = comb.unflatten(-1, (hc_mult, hc_mult)) * scale[2] + base[2 * hc_mult :].view(hc_mult, hc_mult)
    comb = comb.softmax(-1) + layer.hc_eps
    comb /= comb.sum(-2, keepdim=True) + layer.hc_eps
    for _ in range(layer.hc_sinkhorn_iters - 1):
        comb /= comb.sum(-1, keepdim=True) + layer.hc_eps
        comb /= comb.sum(-2, keepdim=True) + layer.hc_eps
    actual = layer.npu().hc_mixes(x.npu(), hc_fn.npu(), scale.npu(), base.npu())
    for result, expected in zip(actual, (pre, post, comb)):
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("scale", [0.0, 1e-4, 1.0])
@torch.inference_mode()
def test_weightless_engram_gate_matches_reference_and_replays(scale):
    torch.manual_seed(44)
    tokens, hc_mult, width = 3, 4, 5120
    hidden = (torch.randn(tokens, hc_mult, width) * scale).bfloat16()
    key = (torch.randn_like(hidden.float()) * scale).bfloat16()
    value = torch.randn(tokens, width).bfloat16()
    channel_weight = torch.randn(hc_mult, width)
    rotation = torch.linalg.qr(torch.randn(32, 32)).Q
    mask = torch.tensor([True, False, True])
    eps = 1e-6

    def reference(h, k):
        original = (h.float().unflatten(-1, (-1, 32)) @ rotation.T).flatten(-2)
        rstd = torch.rsqrt(original.square().mean(-1) + eps)
        rstd *= torch.rsqrt(k.float().square().mean(-1) + eps)
        dot = (original * channel_weight * k.float()).sum(-1) * rstd * width**-0.5
        magnitude = dot.abs().clamp_min(1e-6).sqrt()
        gate = torch.sigmoid(torch.where(torch.signbit(dot), -magnitude, magnitude))
        gate = gate.masked_fill(~mask.unsqueeze(-1), 0)
        return (h.float() + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(h.dtype)

    h_npu, k_npu = hidden.npu(), key.npu()
    v_npu, w_npu, rotation_npu, mask_npu = value.npu(), channel_weight.npu(), rotation.npu(), mask.npu()

    def run():
        return engram_gate(h_npu, k_npu, v_npu, w_npu, rotation_npu, mask_npu, eps)

    torch.testing.assert_close(run().cpu(), reference(hidden, key), rtol=0.016, atol=1e-3)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        actual = run()
    for sign in (1, -1):
        h_npu.copy_(hidden * sign)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), reference(hidden * sign, key), rtol=0.016, atol=1e-3)
        torch.testing.assert_close(actual.cpu()[1], (hidden * sign)[1], rtol=0, atol=0)

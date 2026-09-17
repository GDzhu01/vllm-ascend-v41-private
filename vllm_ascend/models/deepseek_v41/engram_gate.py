# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import torch
from safetensors import safe_open


def load_engram_rotation_block(model_root, hidden_size):
    """Load the A3 Quarot basis, or keep the native A5 basis unchanged."""
    rotation_path = Path(model_root) / "optional/quarot.safetensors"
    if not rotation_path.is_file():
        return torch.eye(32)
    with safe_open(rotation_path, framework="pt") as checkpoint:
        rotation = checkpoint.get_tensor("global_rotation")
    block = rotation[:32, :32].contiguous()
    expected = torch.block_diag(*[block] * (hidden_size // 32))
    if not torch.equal(rotation, expected):
        raise ValueError("Engram gate requires repeated block32 global rotation")
    return block


def engram_gate(
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    channel_weight: torch.Tensor,
    rotation_block: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply gating in the checkpoint's residual/value basis.

    ``hidden`` and ``key`` have shape [tokens, hc_mult, hidden_size].
    A3 supplies its repeated Quarot block; A5 uses the identity block because
    its released Engram projections are already in the native residual basis.
    """
    dim = hidden.shape[-1]
    original = (hidden.float().unflatten(-1, (-1, rotation_block.shape[0])) @ rotation_block.float().T).flatten(-2)
    key = key.float()
    rstd = torch.rsqrt(original.square().mean(-1) + eps)
    rstd *= torch.rsqrt(key.square().mean(-1) + eps)
    dot = (original * channel_weight.float() * key).sum(-1) * rstd * dim**-0.5
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    gate = torch.sigmoid(torch.where(torch.signbit(dot), -magnitude, magnitude))
    gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (hidden.float() + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)

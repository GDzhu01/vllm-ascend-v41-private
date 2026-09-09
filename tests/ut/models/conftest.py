# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu


@pytest.fixture
def mock_npu_rms_norm(monkeypatch):
    """Supply the operator's two outputs for CPU-only model unit tests."""

    def reference(x, gamma, epsilon=1e-6):
        rstd = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + epsilon)
        return (x.float() * rstd * gamma.float()).to(x.dtype), rstd

    monkeypatch.setattr(torch_npu, "npu_rms_norm", reference)

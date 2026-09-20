# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the same raw requests with V4.1's shared backing and index K/scale views."""

import pytest
import torch

from tests.deepseek_v41_cache_utils import allocate_cache_views, make_cache_config
from tests.ut.distributed.ascend_store.test_request_lifecycle import (
    aligned_buffer,
    devices_and_store,  # noqa: F401
    test_raw_sequence_lifecycle,  # noqa: F401
)


@pytest.fixture(params=[None, "dspark"], ids=["flash", "flash-dspark"])
def speculative_method(request):
    return request.param


@pytest.fixture
def cache_layout(speculative_method, monkeypatch):
    plan = make_cache_config(128, draft_layers=3 if speculative_method else 0)
    assert len(plan.kv_cache_groups) == (13 if speculative_method else 12)
    assert len(plan.kv_cache_tensors) == 4

    def allocate():
        original_zeros = torch.zeros

        def aligned_zeros(size, **kwargs):
            if isinstance(size, int) and kwargs.get("dtype") == torch.uint8:
                return aligned_buffer(size)
            return original_zeros(size, **kwargs)

        with monkeypatch.context() as allocation_patch:
            allocation_patch.setattr(torch, "zeros", aligned_zeros)
            _, caches = allocate_cache_views(plan)
        return caches

    return plan, allocate

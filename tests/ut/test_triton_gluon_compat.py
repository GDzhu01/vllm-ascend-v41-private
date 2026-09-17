# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib

import pytest

import vllm_ascend


@pytest.mark.skipif(
    not vllm_ascend._triton_gluon_available,
    reason="installed Triton does not provide Gluon",
)
def test_real_triton_gluon_is_not_shadowed():
    gluon = importlib.import_module("triton.experimental.gluon")

    assert gluon.__spec__ is not None
    assert hasattr(gluon, "__path__")

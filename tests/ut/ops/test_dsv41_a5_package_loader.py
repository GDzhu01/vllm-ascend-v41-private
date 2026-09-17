# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys

from vllm_ascend.ops.dsv41_a5.package_loader import import_packaged_a5_module


def test_targeted_loader_skips_eager_package_initializers(tmp_path, monkeypatch):
    package = tmp_path / "cann_ops_transformer"
    leaf = package / "ops" / "attention" / "wanted"
    leaf.mkdir(parents=True)
    payload = tmp_path / "ops"
    payload.mkdir()
    (payload / "__init__.py").write_text("registered = True\n")
    (package / "__init__.py").write_text("raise AssertionError('root initializer ran')\n")
    (package / "ops" / "__init__.py").write_text("raise AssertionError('ops initializer ran')\n")
    (leaf / "__init__.py").write_text("loaded = True\n")

    for name in tuple(sys.modules):
        if name == "cann_ops_transformer" or name.startswith("cann_ops_transformer."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    loaded = import_packaged_a5_module("cann_ops_transformer.ops.attention.wanted")

    assert loaded.loaded is True
    assert not hasattr(sys.modules["cann_ops_transformer"], "__file__")
    assert not hasattr(sys.modules["cann_ops_transformer.ops"], "__file__")

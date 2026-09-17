# SPDX-License-Identifier: Apache-2.0
from unittest import mock

import torch

from vllm_ascend.attention.sparse_flash_mla import (
    _get_sparse_flash_mla_ops,
    sparse_flash_mla,
    sparse_flash_mla_metadata,
)


def test_adapter_enforces_bf16_paged_layout():
    metadata_op = mock.Mock(return_value=torch.empty(0))
    attention_op = mock.Mock(return_value=torch.empty(0))
    with mock.patch(
        "vllm_ascend.attention.sparse_flash_mla._get_sparse_flash_mla_ops",
        return_value=(attention_op, metadata_op),
    ):
        sparse_flash_mla_metadata(layout_kv="PA_ND")
        sparse_flash_mla(torch.empty(0), layout_kv="PA_ND")

    assert metadata_op.call_args.kwargs["layout_kv"] == "PA_BBND"
    assert attention_op.call_args.kwargs["layout_kv"] == "PA_BBND"


def test_loader_imports_concrete_lazy_operator_module():
    namespace = torch.ops.cann_ops_transformer
    attention_op = mock.Mock()
    metadata_op = mock.Mock()
    _get_sparse_flash_mla_ops.cache_clear()
    with (
        mock.patch("vllm_ascend.attention.sparse_flash_mla.import_module") as imported,
        mock.patch.object(namespace, "sparse_flash_mla", attention_op, create=True),
        mock.patch.object(namespace, "sparse_flash_mla_metadata", metadata_op, create=True),
    ):
        assert _get_sparse_flash_mla_ops() == (attention_op, metadata_op)
    imported.assert_called_once_with("cann_ops_transformer.ops.attention.sparse_flash_mla")
    _get_sparse_flash_mla_ops.cache_clear()

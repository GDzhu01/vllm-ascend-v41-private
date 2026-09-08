# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared DSA token layouts for replicated-cache context parallelism."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from vllm.distributed import get_pcp_group

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.pcp_manager import AscendPCPAttentionContext


def gather_and_restore_hidden_states(hidden_states, hidden_restore_idx, group=None):
    group = get_pcp_group() if group is None else group
    gathered = group.all_gather(hidden_states.contiguous(), dim=0)
    return torch.index_select(gathered, 0, hidden_restore_idx)


def restore_tp_heads(output, tp_group):
    """Exchange [local tokens, all heads] for [all tokens, local heads]."""
    if tp_group.world_size == 1:
        return output
    tokens, heads, width = output.shape
    local_heads = heads // tp_group.world_size
    send = (
        output.view(tokens, tp_group.world_size, local_heads, width)
        .permute(1, 0, 2, 3)
        .contiguous()
        .view(-1, local_heads, width)
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=tp_group.device_group)
    return recv


class PCPMetadataMixin:
    """Canonical cache-update view and rank-local slot view of a PCP batch."""

    @staticmethod
    def _build_global_common_attn_metadata(
        pcp_context: AscendPCPAttentionContext,
        cache_group_idx: int,
        local_common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendCommonAttentionMetadata:
        global_batch = pcp_context.global_batch
        num_reqs = global_batch.num_reqs_after_padding
        return AscendCommonAttentionMetadata(
            query_start_loc=global_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(global_batch.query_start_loc_np),
            seq_lens=global_batch.seq_lens[:num_reqs],
            seq_lens_cpu=torch.from_numpy(global_batch.seq_lens_np)[:num_reqs],
            seq_lens_cpu_upper_bound=global_batch.seq_lens_cpu_upper_bound[:num_reqs],
            num_computed_tokens_cpu=torch.from_numpy(global_batch.num_computed_tokens_np),
            num_reqs=num_reqs,
            num_actual_tokens=global_batch.num_tokens,
            max_query_len=int(global_batch.num_scheduled_tokens.max()),
            max_seq_len=local_common_attn_metadata.max_seq_len,
            block_table_tensor=pcp_context.global_block_tables[cache_group_idx],
            slot_mapping=pcp_context.global_slot_mappings[cache_group_idx],
            causal=local_common_attn_metadata.causal,
            dcp_local_seq_lens=global_batch.dcp_local_seq_lens,
            positions=global_batch.positions,
            attn_state=global_batch.attn_state,
            num_input_tokens=global_batch.num_tokens_after_padding,
            is_prefilling=torch.from_numpy(global_batch.is_prefilling_np),
        )

    def _build_local_common_attn_metadata(
        self,
        pcp_context: AscendPCPAttentionContext,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendCommonAttentionMetadata:
        num_local_padded_tokens = common_attn_metadata.num_input_tokens
        gathered_slot_mapping = common_attn_metadata.slot_mapping
        if pcp_context.global_batch.is_dummy:
            gathered_slot_mapping.fill_(-1)
        local_slot_mapping = gathered_slot_mapping.view(
            self._pcp_world_size,
            num_local_padded_tokens,
        )[self._pcp_rank]
        return common_attn_metadata.replace(
            slot_mapping=local_slot_mapping,
        )

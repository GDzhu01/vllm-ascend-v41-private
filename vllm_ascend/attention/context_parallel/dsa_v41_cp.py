# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 replicated-cache TP-token DSA CP adapter."""

from dataclasses import replace

from vllm.distributed import get_tp_group

from vllm_ascend.attention.context_parallel.dsa_common import restore_tp_heads
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder
from vllm_ascend.attention.dsa_v41 import DeepseekV41EagerAttentionImpl, DeepseekV41MetadataBuilder
from vllm_ascend.attention.utils import enable_pcp
from vllm_ascend.utils import enable_dsa_cp


def get_v41_cp_classes():
    if enable_pcp():
        raise NotImplementedError("V4.1 PCP is not supported")
    if enable_dsa_cp():
        return DeepseekV41CPMetadataBuilder, DeepseekV41CPImpl
    return DeepseekV41MetadataBuilder, DeepseekV41EagerAttentionImpl


class DeepseekV41CPMetadataBuilder(DeepseekV41MetadataBuilder):
    # Reuse Legacy DSACP's request intersection and causal-prefix calculation.
    _local_token_range = staticmethod(AscendDSACPMetadataBuilder._local_token_range)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs):
        common = common_attn_metadata
        global_metadata = super().build(common_prefix_len, common, fast_build)
        start, end, per_rank, padded, qsl, seq_lens = AscendDSACPMetadataBuilder._build_local_token_metadata(
            self,
            common.num_reqs,
            common.num_input_tokens,
            common.query_start_loc_cpu,
            common.seq_lens_cpu,
        )
        actual_end = min(end, common.num_actual_tokens)
        actual_start = min(start, actual_end)
        # Padding participates in the output exchange, not in cache reads.
        qsl = qsl.clamp_max(actual_end - actual_start)
        local_common = common.replace(
            query_start_loc=qsl.to(common.query_start_loc.device),
            query_start_loc_cpu=qsl,
            seq_lens=seq_lens.to(common.seq_lens.device),
            seq_lens_cpu=seq_lens,
            num_actual_tokens=actual_end - actual_start,
            num_input_tokens=actual_end - actual_start,
            positions=common.positions[actual_start:actual_end],
            slot_mapping=common.slot_mapping[actual_start:actual_end],
            max_query_len=int((qsl[1:] - qsl[:-1]).max()) if common.num_reqs else 0,
            max_seq_len=int(seq_lens.max()) if common.num_reqs else 0,
        )
        local = super().build(common_prefix_len, local_common, fast_build)
        return replace(local, global_metadata=global_metadata, cp_token_range=(start, end, per_rank, padded))


class DeepseekV41CPImpl(DeepseekV41EagerAttentionImpl):
    def _global_layer_metadata(self, metadata_by_prefix):
        global_by_prefix = {}
        for prefix, metadata in metadata_by_prefix.items():
            if metadata.global_metadata is None:
                raise ValueError(f"V4.1 CP is missing global cache metadata for {prefix}")
            global_by_prefix[prefix] = metadata.global_metadata
        return self._get_layer_metadata(global_by_prefix)

    def _prepare_inputs_and_caches(self, attn, hidden_states, metadata, metadata_by_prefix):
        global_metadata = self._global_layer_metadata(metadata_by_prefix)
        self._update_caches(attn, hidden_states[: global_metadata.swa.num_actual_tokens], global_metadata)
        start, _, _, _ = metadata.swa.cp_token_range
        return hidden_states[start : start + metadata.swa.num_actual_tokens]

    def _project_output(self, attn, output, hidden_states, metadata):
        _, _, per_rank, _ = metadata.swa.cp_token_range
        padded = output.new_zeros((per_rank, output.shape[1], output.shape[2]))
        padded[: output.shape[0]] = output
        exchanged = restore_tp_heads(padded, get_tp_group())
        # The inherited V4 module owns quantized weights and TP projection logic.
        projected = attn.dsa_attn.dsa_attn.impl._forward_o_proj(exchanged)
        return projected[: hidden_states.shape[0]]

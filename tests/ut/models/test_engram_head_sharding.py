# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_ascend.models.deepseek_v41.engram_hbm import (
    EngramQueryGroup,
    NodeShardedEngram,
    dequantize_engram_rows,
    quantize_engram_rows,
)

HEAD_SIZES = (7, 11, 13, 17, 19, 23, 29)


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60))
    tp = dp = None
    for ranks in ([0, 1], [2, 3]):
        group = dist.new_group(ranks)
        if rank in ranks:
            tp = group
    for ranks in ([0, 2], [1, 3]):
        group = dist.new_group(ranks)
        if rank in ranks:
            dp = group
    q = EngramQueryGroup(dist.group.WORLD, dist.group.WORLD, tp, rank // 2 * 2, dp)
    references, tables = [], []
    for storage in ("bf16", "int8"):
        table = NodeShardedEngram(sum(HEAD_SIZES), 256, q, "cpu", storage, HEAD_SIZES)
        full = (torch.arange(sum(HEAD_SIZES) * 256).reshape(-1, 256).float() % 79 - 39).bfloat16()
        table.set_rows(0, full[table.start : table.end])
        if storage == "int8":
            full = dequantize_engram_rows(*quantize_engram_rows(full))
        tables.append(table)
        references.append(full)
    for counts in ((2, 5), (7, 1), (0, 3), (0, 0)):
        count = counts[q.dp_rank]
        hashes = torch.empty((count, len(tables), len(HEAD_SIZES)), dtype=torch.int64)
        start = 0
        for head, size in enumerate(HEAD_SIZES):
            for layer in range(len(tables)):
                hashes[:, layer, head] = start + (torch.arange(count) * 3 + head + q.dp_rank + layer) % size
            start += size
        if count:
            hashes[-1, 0, 0] = -1
        outputs = tables[0].route_heads(tables, hashes, max(counts))
        for layer, output in enumerate(outputs):
            ids = hashes[:, layer]
            expected = references[layer][ids.clamp_min(0)].masked_fill((ids < 0).unsqueeze(-1), 0)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    dist.destroy_process_group()


def test_head_sharding_variable_prefill_idle_and_uneven_heads(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'heads'}",), nprocs=4, join=True)


def test_head_shards_cover_buckets_once():
    ranges = []
    for rank in range(4):
        q = SimpleNamespace(size=4, head_shard_rank=rank)
        table = NodeShardedEngram(sum(HEAD_SIZES) + 3, 32, q, device="cpu", head_sizes=HEAD_SIZES)
        ranges.append((table.start, table.end))
        assert table.head_count > 0
        assert table.weight.shape[0] == sum(HEAD_SIZES[table.head_start : table.head_start + table.head_count])
    assert ranges[0][0] == 0
    assert ranges[-1][1] == sum(HEAD_SIZES)
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))


def test_reject_more_shards_than_heads():
    with pytest.raises(ValueError, match="at least one complete hash head"):
        NodeShardedEngram(200, 32, SimpleNamespace(size=8, head_shard_rank=0), device="cpu", head_sizes=HEAD_SIZES)

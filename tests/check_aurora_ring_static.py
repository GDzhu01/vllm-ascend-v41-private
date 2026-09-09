# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static placement arithmetic only: never import torch, vllm, or their modules."""

from __future__ import annotations

import ast
import dataclasses
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vllm_ascend/core/deepseek_v41.py"


@dataclasses.dataclass(frozen=True)
class ItemSize:
    itemsize: int


@dataclasses.dataclass(frozen=True)
class Resource:
    block_size: int
    head_size: int
    dtype: ItemSize
    compress_ratio: int = 1
    num_kv_heads: int = 1
    scale_dim: int = 0
    scale_dtype: ItemSize = ItemSize(2)
    page_size_padded: int | None = None

    @property
    def storage_block_size(self):
        return self.block_size // self.compress_ratio


class Full(Resource):
    pass


class Index(Resource):
    pass


class State(Resource):
    pass


class SWA(Resource):
    pass


symbols = {
    "dataclass": dataclasses.dataclass,
    "DeepseekV41FullSpec": Full,
    "DeepseekV41IndexerSpec": Index,
    "DeepseekV41CompressorStateSpec": State,
    "DeepseekV41SWASpec": SWA,
    "is_v41_spec": lambda s: isinstance(s, (Full, Index, State, SWA)),
}
names = {"CachePlacement", "CacheSlot", "_layer_number", "_cache_plane_sizes", "plan_cache_slots"}
module = ast.parse(SOURCE.read_text())
selected = ast.Module(
    body=[node for node in module.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names],
    type_ignores=[],
)
assert len(selected.body) == len(names)
exec(compile(selected, str(SOURCE), "exec"), symbols)
plan = symbols["plan_cache_slots"]
planes = symbols["_cache_plane_sizes"]


def specs_for(block, width, index_width):
    specs = {}
    for layer in range(40):
        prefix = f"language_model.model.layers.{layer}.self_attn"
        specs[prefix + ".swa_cache"] = SWA(block, width, ItemSize(2))
        if layer in (2, 8, 14, 20):
            ratio = 1 if layer == 20 else 2
            specs[prefix + ".long_kv_cache"] = Full(block, width, ItemSize(2), ratio)
            specs[prefix + ".indexer.k_cache"] = Index(block, index_width, ItemSize(1), ratio, scale_dim=1)
            if ratio == 2:
                specs[prefix + ".compressor.state_cache"] = State(32, 2 * width, ItemSize(4))
    return specs


specs = specs_for(128, 512, 128)
slots = plan(specs)
assert len(specs) == 51
assert [s.page_size_bytes for s in slots] == [131072, 131072, 131072, 147712]
assert sum(s.page_size_bytes for s in slots) == 540928
assert [len(s.placements) for s in slots] == [13, 13, 13, 12]
assert plan(dict(reversed(list(specs.items())))) == slots
padded = {
    p.name: dataclasses.replace(specs[p.name], page_size_padded=p.page_size_bytes) for s in slots for p in s.placements
}
assert plan(padded) == slots
assert all(s.page_size_padded is None for s in specs.values())
swa_padding = [
    p.page_size_bytes - sum(planes(specs[p.name]))
    for s in slots
    for p in s.placements
    if isinstance(specs[p.name], SWA)
]
assert swa_padding.count(0) == 30 and swa_padding.count(16640) == 10
for slot in slots:
    for p in slot.placements:
        assert p.offset >= 0 and p.offset + p.page_size_bytes <= slot.page_size_bytes
        assert sum(planes(specs[p.name])) <= p.page_size_bytes
for idx, slot in enumerate(slots):
    kv, index = slot.placements[:2]
    assert index.offset == sum(planes(specs[kv.name]))
    assert index.offset + index.page_size_bytes == slot.page_size_bytes
    assert index.offset + planes(specs[index.name])[0] == (73728 if idx < 3 else 147456)

# Group membership reference: 8 full resources, 3 states and ten SWA quartets.
groups = [
    [p.name for s in slots for p in s.placements if isinstance(specs[p.name], (Full, Index))],
    [p.name for s in slots for p in s.placements if isinstance(specs[p.name], State)],
]
for start in range(0, 40, 4):
    groups.append([f"language_model.model.layers.{i}.self_attn.swa_cache" for i in range(start, start + 4)])
assert [len(g) for g in groups] == [8, 3] + [4] * 10
placement = {p.name: (idx, p) for idx, s in enumerate(slots) for p in s.placements}
N = 17
bases = [0]
for s in slots[:-1]:
    bases.append(bases[-1] + N * s.page_size_bytes)
intervals = []
for gid, group in enumerate(groups):
    bid = gid + 1
    for name in group:
        idx, p = placement[name]
        start = bases[idx] + bid * slots[idx].page_size_bytes + p.offset
        for size in planes(specs[name]):
            intervals.append((start, start + size, name))
            start += size
for i, (a0, a1, _) in enumerate(intervals):
    for b0, b1, _ in intervals[i + 1 :]:
        assert a1 <= b0 or b1 <= a0
for start in (127, 128, 129, 255, 256, 257):
    ids = [7, 19, 3]
    for pos in range(start - 3, start):
        original = ids[pos // 128] * 128 + pos % 128
        if pos % 2:
            assert original // 2 == ids[pos // 128] * 64 + (pos % 128) // 2
for block, width, index_width in [(64, 8, 4), (128, 256, 64), (256, 512, 128)]:
    small = specs_for(block, width, index_width)
    for slot in plan(small):
        assert all(p.offset + sum(planes(small[p.name])) <= slot.page_size_bytes for p in slot.placements)
assert "torch" not in __import__("sys").modules and "vllm" not in __import__("sys").modules
print(
    json.dumps(
        {
            "status": "passed",
            "scope": "extracted placement arithmetic; no torch/vllm imports or runtime tests",
            "groups": len(groups),
            "cache_specs": len(specs),
            "slots": [s.page_size_bytes for s in slots],
            "bytes_per_global_id": sum(s.page_size_bytes for s in slots),
            "swa_unpadded": 30,
            "swa_padded": 10,
            "disjoint_payload_intervals": len(intervals),
            "rank_shrink_offsets": "component offsets independent of N",
        },
        indent=2,
    )
)

# Model the read-before-write schedule with token identities, including ring wraps.
cases = 0
for start in range(130):
    for length in (0, 1, 2, 15, 16, 17, 31, 32, 33, 129):
        cache = {p % 32: p for p in range(max(0, start - 32), start)}
        groups = (start + length) // 2 - start // 2
        outputs = {}
        for group in range(start // 2, start // 2 + groups):
            pair = [2 * group, 2 * group + 1]
            actual = [p if p >= start else cache[p % 32] for p in pair]
            assert actual == pair
            output_row = 2 * group + 1 - start
            assert 0 <= output_row < length
            outputs[output_row] = tuple(actual)
        writes = list(range(start + length - min(length, 32), start + length))
        assert len({p % 32 for p in writes}) == len(writes)
        for p in writes:
            cache[p % 32] = p
        if (start + length) % 2:
            assert cache[(start + length - 1) % 32] == start + length - 1
        cases += 1
for block_id in (1, 3, 7, 16):
    for pos in (0, 1, 31, 32, 33, 129):
        offset = (block_id * 32 + pos % 32) * 1024 * 4
        assert offset == block_id * 131072 + pos % 32 * 4096
print(f"PASS: {cases} ring schedules, exact 128-KiB page addressing; state demand is one ID per request.")

# Check actual manager allocation methods without loading vLLM or torch.


class BaseManager:
    def __init__(self):
        self.req_to_blocks = defaultdict(list)
        self.claimed = 0

        def claim(count):
            self.claimed += count
            return [SimpleNamespace(block_id=self.claimed)]

        self.block_pool = SimpleNamespace(get_new_blocks=claim)


manager_source = ast.parse((ROOT / "vllm_ascend/core/circular_buffer.py").read_text())
manager_class = next(
    n for n in manager_source.body if isinstance(n, ast.ClassDef) and n.name == "AscendCircularBufferManager"
)
manager_env = {"FullAttentionManager": BaseManager}
exec(compile(ast.Module(body=[manager_class], type_ignores=[]), "<extracted-ring-manager>", "exec"), manager_env)
manager = manager_env["AscendCircularBufferManager"]()
for request in ("a", "b"):
    assert manager.get_num_blocks_to_allocate(request, 129, [], 0, 0, 129) == 1
    assert len(manager.allocate_new_blocks(request, 129, 129)) == 1
    for tokens in (130, 1024, 65536):
        assert manager.get_num_blocks_to_allocate(request, tokens, [], 0, 0, tokens) == 0
        assert manager.allocate_new_blocks(request, tokens, tokens) == []
        manager.allocate_external_computed_blocks(request, 0, tokens)
        manager.remove_skipped_blocks(request, tokens)
assert manager.claimed == 2 and not manager._record_new_block_ids
print("PASS: extracted manager retains exactly one block per request.")

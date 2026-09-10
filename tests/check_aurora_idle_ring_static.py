# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise idle dispatch and ring initialization without importing torch/vllm.

The byte backings model G0 and G1 aliasing one physical slot. This verifies
Python dispatch/initialization only, not graph replay or kernel correctness.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def method(path, class_name, method_name):
    module = ast.parse((ROOT / path).read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)


dispatch = method("vllm_ascend/worker/worker.py", "NPUWorker", "execute_dummy_batch")
dummy = method("vllm_ascend/worker/model_runner_v1.py", "NPUModelRunner", "_dummy_run")
defaults = dict(zip([a.arg for a in dummy.args.args][-len(dummy.args.defaults) :], dummy.args.defaults))
assert ast.literal_eval(defaults["skip_gdn_state_update"]) is False
assert ast.literal_eval(defaults["skip_ring_state_update"]) is False
ring_init = next(
    node
    for node in ast.walk(dummy)
    if isinstance(node, ast.For)
    and ast.unparse(node.target) == "(gid, group)"
    and "skip_ring_state_update" in ast.unparse(node)
    and "context[name].kv_cache" in ast.unparse(node)
)
ring_code = compile(ast.Module(body=[ring_init], type_ignores=[]), "<actual dummy ring initialization>", "exec")
namespace = {}
exec(compile(ast.Module(body=[dispatch], type_ignores=[]), "<actual idle dispatch>", "exec"), namespace)
should_build = method("vllm_ascend/worker/model_runner_v1.py", "NPUModelRunner", "_should_build_dummy_attn_metadata")
mode_env = {"CUDAGraphMode": type("Modes", (), {"NONE": 0, "FULL": 1})}
metadata_module = ast.Module(
    body=ast.parse("from __future__ import annotations").body + [should_build], type_ignores=[]
)
exec(compile(metadata_module, "<actual dummy metadata dispatch>", "exec"), mode_env)
assert not mode_env["_should_build_dummy_attn_metadata"](None, False, False, 0)
assert mode_env["_should_build_dummy_attn_metadata"](None, False, False, 1)

PAGE = 131072


class Table:
    def __getitem__(self, index):
        return self

    def __setitem__(self, index, value):
        pass

    def fill(self, value):
        pass


class RingView:
    def __init__(self, backing):
        self.backing = backing

    def __getitem__(self, pages):
        start, stop = pages.start * PAGE, pages.stop * PAGE

        def zero():
            self.backing[start:stop] = bytes(stop - start)

        return SimpleNamespace(zero_=zero)


def initialize(skip, skip_gdn=False):
    # G0 owns global ID 1. Dummy G1 must not clear the same slot bytes.
    backings = [bytearray(b"\x80\x3f" * (3 * PAGE // 2)) for _ in range(3)]
    names = [f"source{i}.compressor.state_cache" for i in (2, 8, 14)]
    context = {name: SimpleNamespace(kv_cache=[RingView(raw)]) for name, raw in zip(names, backings)}
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            num_blocks=3,
            kv_cache_groups=[SimpleNamespace(kv_cache_spec="ring", layer_names=names)],
        ),
        input_batch=SimpleNamespace(block_table=[SimpleNamespace(block_table=SimpleNamespace(np=Table()))]),
        compilation_config=SimpleNamespace(static_forward_context=context),
    )
    exec(
        ring_code,
        {
            "self": runner,
            "skip_ring_state_update": skip,
            "skip_gdn_state_update": skip_gdn,
            "is_circular_spec": lambda spec: spec == "ring",
            "num_reqs": 1,
            "num_reqs_padded": 1,
            "np": SimpleNamespace(arange=lambda start, end: list(range(start, end))),
        },
    )
    return [bytes(raw[PAGE : 2 * PAGE]) for raw in backings]


expected = [b"\x80\x3f" * (PAGE // 2)] * 3
for skip_gdn in (False, True):
    assert initialize(False, skip_gdn) == [bytes(PAGE)] * 3
    assert initialize(True, skip_gdn) == expected

# Verify the actual keyword expressions keep ring suppression independent
# through both runner layers before the Aurora builder sees it.
build_metadata = method("vllm_ascend/worker/model_runner_v1.py", "NPUModelRunner", "_build_attention_metadata")
for function in (dummy, build_metadata):
    keyword = next(n for n in ast.walk(function) if isinstance(n, ast.keyword) and n.arg == "skip_ring_state_update")
    expression = compile(ast.Expression(body=keyword.value), "<ring metadata flag>", "eval")
    for skip_ring in (False, True):
        for skip_gdn in (False, True):
            assert (
                eval(expression, {"skip_ring_state_update": skip_ring, "skip_gdn_state_update": skip_gdn}) == skip_ring
            )
for use_v2 in (False, True):
    for query_len in (1, 8):
        calls = []

        def run(count, *, _calls=calls, _use_v2=use_v2, **kwargs):
            _calls.append((count, kwargs))
            if not _use_v2:
                assert initialize(kwargs.get("skip_ring_state_update", False)) == expected

        worker = SimpleNamespace(
            use_v2_model_runner=use_v2,
            log_memory_stats=lambda: None,
            model_runner=SimpleNamespace(uniform_decode_query_len=query_len, _dummy_run=run),
        )
        namespace["execute_dummy_batch"](worker)
        assert len(calls) == 1 and calls[0][0] == query_len
        assert calls[0][1]["uniform_decode"] is True
        assert "skip_gdn_state_update" not in calls[0][1]
        if use_v2:
            assert "skip_ring_state_update" not in calls[0][1]
        else:
            assert calls[0][1]["skip_ring_state_update"] is True

print("PASS: unguarded dummy initialization overwrites all three G0 C2 slots at ID 1.")
print("PASS: V1 idle dispatch preserves those pages for query lengths 1/8; V2 arguments unchanged.")
print("PASS: ring initialization and metadata suppression are independent of the GDN flag.")
print("Scope: actual Python dispatch/initialization with byte-array backings; no tensor or NPU execution.")

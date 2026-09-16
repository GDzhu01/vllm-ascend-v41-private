# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import sys
import threading
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def load(name):
    source = Path(__file__).resolve().parents[3] / f"vllm_ascend/models/deepseek_v41/{name}.py"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    hbm, pipeline = load("engram_hbm"), load("engram_prefetch")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60))
    for ranks in ([0, 1], [2, 3]):
        group = dist.new_group(ranks)
        if rank in ranks:
            tp = group
    query = hbm.EngramQueryGroup(dist.group.WORLD, dist.group.WORLD, tp, rank // 2 * 2)
    tables = [
        hbm.NodeShardedEngram(257, 32, query, storage_format=storage, cpu_offload=True) for storage in ("int8", "fp8")
    ]
    codes = ((torch.arange(257)[:, None] * 13 + torch.arange(32) * 7) % 31 - 15).float()
    scales = 2.0 ** (torch.arange(257)[:, None] % 3 - 2).float()
    reference = (codes * scales).bfloat16()
    for table in tables:
        table.weight.data.copy_(codes[table.start : table.end])
        table.weight_scale.copy_(scales[table.start : table.end])
    prefetch = pipeline.EngramPrefetcher("cpu")
    submit = prefetch.executor.submit

    def checked_submit(*args, **kwargs):
        # No CPU lookup is launched while another table is still routing IDs.
        assert len(prefetch.pending) == len(tables)
        return submit(*args, **kwargs)

    prefetch.executor.submit = checked_submit
    caller = threading.get_ident()
    all_to_all = dist.all_to_all_single

    def checked_collective(*args, **kwargs):
        assert threading.get_ident() == caller
        return all_to_all(*args, **kwargs)

    dist.all_to_all_single = checked_collective
    started, release = threading.Event(), threading.Event()
    lookup = tables[1].lookup_local

    def delayed_lookup(ids):
        assert threading.get_ident() != caller
        started.set()
        assert release.wait(timeout=30)
        return lookup(ids)

    tables[1].lookup_local = delayed_lookup
    receive = prefetch._receive

    def checked_receive(item, values, padded_tokens, slot):
        if item.table is tables[0]:
            assert prefetch.pending[1].future is None
        return receive(item, values, padded_tokens, slot)

    prefetch._receive = checked_receive
    try:
        for a, b in ((0, 0), (1, 0), (0, 17), (3, 7), (129, 1), (0, 0)):
            started.clear()
            release.clear()
            n = (a, b)[rank // 2]
            ids = (torch.arange(n * 3).reshape(n, 3) * 101) % 257
            prefetch.begin(tables, [ids, ids])
            assert not started.is_set()
            assert prefetch.pending[0].future is None
            assert prefetch.pending[1].future is None
            prefetch.start()
            assert prefetch.pending[0].future is not None
            first = prefetch.consume(0, n + 1)
            # The second lookup must run during the intervening backbone,
            # before its own consume, without blocking the first response.
            assert started.wait(timeout=30)
            assert not prefetch.pending[1].future.done()
            release.set()
            second = prefetch.consume(1, n + 1)
            expected = torch.cat((reference[ids].reshape(n, 96), torch.zeros(1, 96))).bfloat16()
            assert torch.equal(first.view(torch.int16), expected.view(torch.int16))
            assert torch.equal(second.view(torch.int16), expected.view(torch.int16))
            prefetch.drain()
        invalid = torch.tensor([[257 if rank // 2 else 0]])
        with pytest.raises(IndexError, match="outside table"):
            prefetch.begin(tables, [invalid, invalid])
        assert not prefetch.pending
    finally:
        release.set()
        prefetch.close()
        dist.all_to_all_single = all_to_all
        dist.destroy_process_group()


def test_prefetch_idle_padding_and_collective_thread(tmp_path):
    mp.spawn(worker, args=(f"file://{tmp_path / 'prefetch'}",), nprocs=4, join=True)


def test_lookup_waits_for_id_transfer_before_reading():
    from types import SimpleNamespace

    pipeline = load("engram_prefetch")
    prefetch = pipeline.EngramPrefetcher("cpu")
    host_ids = torch.full((3,), -1, dtype=torch.int64)
    entered, release = threading.Event(), threading.Event()

    class Transfer:
        def synchronize(self):
            entered.set()
            assert release.wait(timeout=10)
            host_ids.copy_(torch.tensor([101, 103, 107]))

    seen = []

    def lookup(ids):
        seen.append(ids.clone())
        return ids

    table = SimpleNamespace(start=100, lookup_local=lookup)
    try:
        future = prefetch.executor.submit(prefetch._lookup, table, host_ids, Transfer())
        assert entered.wait(timeout=10)
        assert not future.done()
        assert not seen
        release.set()
        assert torch.equal(future.result(timeout=10), torch.tensor([1, 3, 7]))
    finally:
        release.set()
        prefetch.close()


def test_drain_waits_for_deferred_id_copy_without_starting_lookup():
    from concurrent.futures import Future
    from types import SimpleNamespace

    pipeline = load("engram_prefetch")
    prefetch = pipeline.EngramPrefetcher("cpu")
    events = []
    failed = Future()
    failed.set_exception(RuntimeError("lookup failed"))
    prefetch.pending = [
        SimpleNamespace(future=failed, ready=None),
        SimpleNamespace(future=None, ready=SimpleNamespace(synchronize=lambda: events.append("copy_done"))),
    ]
    try:
        with pytest.raises(RuntimeError, match="lookup failed"):
            prefetch.drain()
        assert events == ["copy_done"]
        assert not prefetch.pending
    finally:
        prefetch.close()


def test_prefetch_future_stays_outside_fullgraph_backbone(monkeypatch):
    from concurrent.futures import Future
    from types import SimpleNamespace

    from vllm_ascend.models.deepseek_v41 import engram_prefetch_op as op
    from vllm_ascend.models.deepseek_v41.model import AscendDeepseekV41ForCausalLM

    pending = Future()
    pending.set_result(torch.full((3,), 2.0))
    calls = []

    def consume(slot, tokens):
        assert not torch.compiler.is_compiling()
        calls.append((slot, tokens))
        return pending.result()

    prefetcher = SimpleNamespace(consume=consume, pending=[pending])
    context = SimpleNamespace(no_compile_layers={"test": SimpleNamespace(_engram_prefetcher=prefetcher)})
    monkeypatch.setattr(op, "get_forward_context", lambda: context)
    library = torch.library.Library("vllm", "IMPL", "CPU")
    library.impl("engram_prefetch_consume", op.engram_prefetch_consume)

    class Backbone:
        def forward(self, input_ids, *args, engram_prefetch=None, **kwargs):
            assert engram_prefetch is None
            output = torch.empty_like(input_ids)
            torch.ops.vllm.engram_prefetch_consume(input_ids, input_ids, output, "test", 0)
            return input_ids + output

        @torch.compile(backend="eager", fullgraph=True)
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    backbone = Backbone()
    inputs = torch.ones(3)
    shell = SimpleNamespace(model=backbone)
    actual = AscendDeepseekV41ForCausalLM.forward(shell, inputs, inputs, engram_prefetch=pending)
    assert torch.equal(actual, inputs + 2)
    assert calls == [(0, 3)]
    # Subsequent synchronous decode still uses the compiled call.
    prefetcher.pending.clear()
    actual = AscendDeepseekV41ForCausalLM.forward(shell, inputs, inputs)
    assert torch.equal(actual, inputs + 1)
    assert calls == [(0, 3)]


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_prefetch_start_survives_aot_compilation(monkeypatch, backend):
    from types import SimpleNamespace

    from vllm_ascend.models.deepseek_v41 import engram_prefetch_op as op

    calls = []

    def consume(slot, tokens):
        assert calls[-1] == "start"
        calls.append("consume")
        return torch.full((tokens,), 2.0)

    prefetcher = SimpleNamespace(pending=[object()], start=lambda: calls.append("start"), consume=consume)
    context = SimpleNamespace(no_compile_layers={"start_test": SimpleNamespace(_engram_prefetcher=prefetcher)})
    monkeypatch.setattr(op, "get_forward_context", lambda: context)
    library = torch.library.Library("vllm", "IMPL", "CPU")
    library.impl("engram_prefetch_start", op.engram_prefetch_start)
    library.impl("engram_prefetch_consume", op.engram_prefetch_consume)

    @torch.compile(backend=backend, fullgraph=True)
    def backbone(hidden):
        dependency = torch.ops.vllm.engram_prefetch_start(hidden, "start_test")
        hidden = hidden + 1
        output = torch.empty_like(hidden)
        torch.ops.vllm.engram_prefetch_consume(hidden, hidden, output, "start_test", 0, dependency)
        return hidden + output

    for _ in range(2):
        assert torch.equal(backbone(torch.ones(3)), torch.full((3,), 4.0))
    assert calls == ["start", "consume", "start", "consume"]
    prefetcher.pending.clear()
    assert torch.equal(backbone(torch.ones(3)), torch.full((3,), 4.0))
    assert calls == ["start", "consume", "start", "consume"]


def test_prefetch_preparation_hashes_once_without_wait(monkeypatch):
    from types import SimpleNamespace

    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.models.deepseek_v41 import model as implementation

    events = []
    hashes = torch.arange(5 * 2 * 3).reshape(5, 2, 3)
    mask = torch.tensor([True, True, False, True, False])
    copy = torch.Tensor.copy_

    def record_mask_copy(destination, source, *args, **kwargs):
        if source is mask:
            events.append("mask_copy")
        return copy(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", record_mask_copy)

    def update(*args):
        events.append("hash")
        return hashes, mask

    def begin(tables, ids):
        events.append("begin")
        assert len(tables) == 2
        assert torch.equal(ids[0], hashes[:, 0])
        assert torch.equal(ids[1], hashes[:, 1])

    metadata = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 3, 5]),
        block_table_cpu=torch.zeros(2, 1, dtype=torch.int32),
        storage_block_size=16,
    )
    layers = [SimpleNamespace(engram=SimpleNamespace(embed=object())) for _ in range(15)]
    layers[0].self_attn = SimpleNamespace(dsa_attn=SimpleNamespace(swa_cache_layer=SimpleNamespace(prefix="swa")))
    graph_mask = torch.ones(8, dtype=torch.bool)
    shell = SimpleNamespace(
        config=SimpleNamespace(engram_layer_ids=[1, 14], engram_n_heads=1, engram_max_ngram_size=4),
        layers=layers,
        engram_history=SimpleNamespace(update=update),
        _engram_prefetcher=SimpleNamespace(begin=begin, start=lambda: events.append("start")),
        prepare_engram_graph_inputs=lambda *args: {
            "engram_lookups": {1: torch.empty(8, 96), 14: torch.empty(8, 96)},
            "engram_mask": graph_mask,
        },
    )
    shell._prepare_engram_hashes = implementation.DeepseekV41Model._prepare_engram_hashes.__get__(shell)
    monkeypatch.setattr(
        implementation, "get_ascend_config", lambda: SimpleNamespace(enable_engram=True, enable_engram_trace=False)
    )
    monkeypatch.setattr(implementation, "get_forward_context", lambda: SimpleNamespace(attn_metadata={"swa": metadata}))
    inputs = implementation.DeepseekV41Model.prepare_engram_prefetch_inputs(shell, torch.arange(5), torch.arange(5), 8)
    assert events == ["hash", "mask_copy", "begin", "start"]
    assert set(inputs["engram_lookups"]) == {1, 14}
    assert torch.equal(inputs["engram_mask"], torch.cat((mask, torch.zeros(3, dtype=torch.bool))))
    assert inputs["engram_mask"].data_ptr() == graph_mask.data_ptr()
    mask.logical_not_()
    replay = implementation.DeepseekV41Model.prepare_engram_prefetch_inputs(shell, torch.arange(5), torch.arange(5), 8)
    assert replay["engram_mask"].data_ptr() == inputs["engram_mask"].data_ptr()
    assert torch.equal(replay["engram_mask"][:5], mask)
    assert not replay["engram_mask"][5:].any()


def test_model_consumes_later_lookup_at_layer14(monkeypatch):
    from types import SimpleNamespace

    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.models.deepseek_v41 import model as implementation

    events = []

    class Layer(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.layer_idx = index
            self.engram = None
            if index in (1, 14):
                self.engram = torch.nn.Module()
                self.engram.wkv = torch.nn.Linear(96, 64, bias=False, dtype=torch.bfloat16)
                self.engram.q_weight = torch.nn.Parameter(torch.ones(1, 32))
                self.engram.k_weight = torch.nn.Parameter(torch.ones(1, 32))

        def forward(self, positions, hidden, pre_mix, unused, input_ids=None):
            events.append(self.layer_idx)
            assert hidden.shape == (5, 1, 32)
            return hidden + 0.125, pre_mix

        def hc_collapse(self, hidden, pre_mix):
            return hidden[:, 0]

    torch.manual_seed(7)
    shell = implementation.DeepseekV41Model.__new__(implementation.DeepseekV41Model)
    torch.nn.Module.__init__(shell)
    shell.config = SimpleNamespace(hc_mult=1, hidden_size=32, engram_layer_ids=[1, 14], rms_norm_eps=1e-6)
    shell.hc_mult = 1
    shell.layers = torch.nn.ModuleList([Layer(i) for i in range(15)])
    shell.shared_attention_state = SimpleNamespace(reset=lambda: None)
    shell.engram_rotation = torch.eye(32)
    shell.aux_hidden_state_layers = []
    shell.needs_moe_input_ids = False
    shell.norm = torch.nn.Identity()
    shell._engram_prefetch_name = "test"
    shell._engram_compiled_prefetch = False
    monkeypatch.setattr(implementation, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    inputs = torch.randn(5, 32).bfloat16()
    values = {index: torch.randn(5, 96).bfloat16() for index in (1, 14)}
    mask = torch.tensor([True, True, False, True, False])
    kwargs = dict(
        input_ids=torch.arange(5),
        positions=torch.arange(5),
        intermediate_tensors=None,
        inputs_embeds=inputs,
        engram_mask=mask,
    )
    expected = shell.forward(**kwargs, engram_lookups=values)
    events.clear()

    def consume(slot, tokens):
        assert tokens == 5
        layer = (1, 14)[slot]
        assert events == list(range(layer))
        return values[layer]

    def consume_op(hidden, fallback, output, name, slot, prefetch_dependency=None):
        output.copy_(consume(slot, hidden.shape[0]))

    monkeypatch.setattr(torch.ops.vllm, "engram_prefetch_consume", consume_op)
    shell._engram_compiled_prefetch = True
    actual = shell.forward(**kwargs, engram_lookups=values.copy(), engram_prefetch=True)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("capture_first", [False, True])
def test_graph_inputs_preserve_flattened_width_and_replay_storage(monkeypatch, capture_first):
    from types import SimpleNamespace

    from vllm_ascend.models.deepseek_v41 import model as implementation

    DeepseekV41Model = implementation.DeepseekV41Model
    capacity, width = 16, 32
    columns = 3 * 8
    layers = [SimpleNamespace(engram=SimpleNamespace(embed=SimpleNamespace(width=width))) for _ in range(15)]
    values = {layer: torch.ones(2, columns * width, dtype=torch.bfloat16) for layer in (1, 14)}
    shell = SimpleNamespace(
        config=SimpleNamespace(engram_layer_ids=[1, 14], engram_max_ngram_size=4, engram_n_heads=8),
        layers=layers,
        engram_rotation=torch.eye(32),
        _engram_max_tokens=capacity,
        _engram_input_buffers=None,
        prepare_engram=lambda *args: (values, torch.ones(2, dtype=torch.bool)),
    )
    # The graph-input path reads the Engram switch; a bare shell has no
    # initialized Ascend config.
    monkeypatch.setattr(
        implementation, "get_ascend_config", lambda: SimpleNamespace(enable_engram=True, enable_engram_trace=False)
    )
    # The synchronous refresh materializes the persistent buffers through the
    # capture path before it refreshes them.
    shell.prepare_engram_graph_inputs = DeepseekV41Model.prepare_engram_graph_inputs.__get__(shell)
    prepare = DeepseekV41Model.prepare_engram_inputs.__get__(shell)
    capture = shell.prepare_engram_graph_inputs
    if not capture_first:
        prepare(torch.arange(2), torch.arange(2), 2)
    captured = capture(4)
    for layer in (1, 14):
        assert captured["engram_lookups"][layer].shape == (capacity, columns * width)
    assert captured["engram_mask"].shape == (capacity,)
    replay = prepare(torch.arange(2), torch.arange(2), 2)
    for layer in (1, 14):
        assert captured["engram_lookups"][layer].data_ptr() == replay["engram_lookups"][layer].data_ptr()
        torch.testing.assert_close(captured["engram_lookups"][layer][:2], values[layer])
        assert not captured["engram_lookups"][layer][2:].any()
    assert captured["engram_mask"].data_ptr() == replay["engram_mask"].data_ptr()
    assert captured["engram_mask"][:2].all()
    assert not captured["engram_mask"][2:].any()


def test_runner_capture_through_vl_wrapper(monkeypatch):
    from types import SimpleNamespace

    from vllm_ascend.models.deepseek_v41.vl_model import AscendDeepseekV41ForConditionalGeneration
    from vllm_ascend.worker import model_runner_v1 as implementation

    lookups = {1: torch.ones(16, 768, dtype=torch.bfloat16)}
    mask = torch.ones(16, dtype=torch.bool)

    class LanguageModel(torch.nn.Module):
        def prepare_engram_inputs(self, *args):
            raise AssertionError("CPU routing during capture")

        def prepare_engram_prefetch_inputs(self, *args):
            raise AssertionError("CPU prefetch during capture")

        def prepare_engram_graph_inputs(self, padded_tokens):
            assert padded_tokens == 4
            return {"engram_lookups": lookups, "engram_mask": mask}

        def forward(self, *args, engram_lookups=None, engram_mask=None):
            assert engram_lookups is lookups
            assert engram_mask is mask
            return 42

    model = AscendDeepseekV41ForConditionalGeneration.__new__(AscendDeepseekV41ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.language_model = LanguageModel()
    monkeypatch.setattr(
        implementation,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=implementation.CUDAGraphMode.NONE),
    )
    monkeypatch.setattr(
        implementation,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_engram_prefetch=True, enable_engram_trace=False),
    )
    runner = SimpleNamespace(
        model=model,
        enable_enpu=False,
        _engram_capture_active=True,
        _update_full_graph_params_if_needed=lambda *args: None,
    )
    assert implementation.NPUModelRunner._model_forward(runner, 4) == 42


@pytest.mark.parametrize("mode_name", ["NONE", "FULL", "FULL_DECODE_ONLY.decode", "FULL_DECODE_ONLY.mixed"])
@pytest.mark.parametrize("enable_prefetch", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("capture_active", [False, True])
def test_runner_prefetch_selection_and_cleanup(monkeypatch, mode_name, enable_prefetch, fail, capture_active):
    from types import SimpleNamespace

    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.worker import model_runner_v1 as implementation

    events = []
    pending = SimpleNamespace(drain=lambda: events.append("drain"))

    class Model:
        do_not_compile = False

        def prepare_engram_inputs(self, *args):
            events.append("sync")
            return {}

        def prepare_engram_prefetch_inputs(self, *args):
            events.append("prefetch")
            return {"engram_prefetch": pending}

        def prepare_engram_graph_inputs(self, *args):
            events.append("capture")
            return {}

        def __call__(self, **kwargs):
            assert not self.do_not_compile
            events.append("model")
            if fail:
                raise RuntimeError("backbone failure")
            return 42

    if "." in mode_name:
        config_name, phase = mode_name.split(".")
        mode = getattr(getattr(implementation.CUDAGraphMode, config_name), f"{phase}_mode")()
    else:
        mode = getattr(implementation.CUDAGraphMode, mode_name)
    assert mode.is_valid_runtime_mode()
    monkeypatch.setattr(implementation, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=mode))
    monkeypatch.setattr(
        implementation,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_engram_prefetch=enable_prefetch, enable_engram_trace=False),
    )
    runner = SimpleNamespace(model=Model(), enable_enpu=False, _update_full_graph_params_if_needed=lambda *args: None)
    runner._engram_capture_active = capture_active
    runner.get_model = lambda: runner.model
    if fail:
        with pytest.raises(RuntimeError, match="backbone failure"):
            implementation.NPUModelRunner._model_forward(runner, 4096, torch.ones(4096), torch.arange(4096))
    else:
        assert implementation.NPUModelRunner._model_forward(runner, 4096, torch.ones(4096), torch.arange(4096)) == 42
    if mode_name == "FULL_DECODE_ONLY.mixed":
        assert mode == implementation.CUDAGraphMode.NONE
    uses_prefetch = enable_prefetch and mode_name in ("NONE", "FULL_DECODE_ONLY.mixed")
    expected = ["prefetch", "model", "drain"] if uses_prefetch else ["sync", "model"]
    if capture_active:
        expected = ["capture", "model"]
    assert events == expected
    assert not runner.model.do_not_compile


@pytest.mark.parametrize("tokens, start", [(0, 0), (24, 0), (496, 0), (496, 2048), (512, 4096)])
def test_runner_prefetch_protocol_ignores_local_positions(monkeypatch, tokens, start):
    from types import SimpleNamespace

    from vllm_ascend.worker import model_runner_v1 as implementation

    events = []
    pending = SimpleNamespace(drain=lambda: events.append("drain"))

    class Model:
        do_not_compile = False

        def prepare_engram_inputs(self, *args):
            raise AssertionError("idle and active replicas must choose the same protocol")

        def prepare_engram_prefetch_inputs(self, *args):
            events.append("prefetch")
            return {"engram_prefetch": pending}

        def __call__(self, **kwargs):
            assert not self.do_not_compile
            events.append("model")
            return 42

    def scalar_read(*args, **kwargs):
        raise AssertionError("routing must not synchronize device positions")

    monkeypatch.setattr(torch.Tensor, "__int__", scalar_read)
    monkeypatch.setattr(torch.Tensor, "item", scalar_read)
    monkeypatch.setattr(implementation.dist, "all_reduce", scalar_read)
    monkeypatch.setattr(
        implementation,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=implementation.CUDAGraphMode.NONE),
    )
    monkeypatch.setattr(
        implementation,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_engram_prefetch=True, enable_engram_trace=False),
    )
    runner = SimpleNamespace(
        model=Model(),
        enable_enpu=False,
        _engram_capture_active=False,
        _update_full_graph_params_if_needed=lambda *args: None,
    )
    runner.get_model = lambda: runner.model
    assert (
        implementation.NPUModelRunner._model_forward(
            runner, tokens, torch.ones(tokens), torch.arange(start, start + tokens)
        )
        == 42
    )
    assert events == ["prefetch", "model", "drain"]


def test_start_is_idempotent_after_early_submit():
    from types import SimpleNamespace

    pipeline = load("engram_prefetch")
    prefetch = pipeline.EngramPrefetcher("cpu")
    submitted = []
    future = SimpleNamespace()
    future.done = lambda: False
    prefetch.executor.submit = lambda *args, **kwargs: submitted.append((args, kwargs)) or future
    prefetch.pending = [SimpleNamespace(table=object(), future=None, host_ids=None, ready=None)]
    try:
        prefetch.start()
        prefetch.start()
        assert len(submitted) == 1
        assert prefetch.pending[0].future is future
    finally:
        prefetch.pending.clear()
        prefetch.close()


def test_consume_rejects_stale_generation():
    from types import SimpleNamespace

    pipeline = load("engram_prefetch")
    prefetch = pipeline.EngramPrefetcher("cpu")
    item = SimpleNamespace(generation=1, ids=torch.empty(0, 1), future=None)
    prefetch.pending = [item]
    prefetch._generation = 2
    try:
        with pytest.raises(RuntimeError, match="generation is stale"):
            prefetch.consume(0, 0)
    finally:
        prefetch.pending.clear()
        prefetch.close()


def test_stage_trace_is_opt_in_and_reports_per_slot(monkeypatch):
    from types import SimpleNamespace

    pipeline = load("engram_prefetch")
    # vLLM's logger does not propagate, so capture the call directly.
    messages = []
    monkeypatch.setattr(pipeline.logger, "info", lambda msg, *args: messages.append(msg % args))
    assert pipeline.EngramPrefetcher("cpu").trace_enabled is False
    assert pipeline.EngramPrefetcher("cpu", trace=True).trace_enabled is True

    prefetch = pipeline.EngramPrefetcher("cpu", trace=True)
    # Hand-written stage order: both layers route first, then each lookup and
    # response is measured against its own slot.
    prefetch.trace = [
        ("begin", -1, 0, 0),
        ("lookup_wait", 0, 1_000_000, 0),
        ("lookup_start", 0, 1_200_000, 8_192),
        ("lookup_done", 0, 3_200_000, 0),
        ("consume_wait", 0, 3_200_000, 0),
        ("response_start", 0, 3_300_000, 0),
        ("response_done", 0, 3_900_000, 0),
        ("consume_done", 0, 3_900_000, 0),
        ("lookup_wait", 1, 3_900_000, 0),
        ("lookup_start", 1, 4_000_000, 4_096),
        ("lookup_done", 1, 6_000_000, 0),
        ("consume_wait", 1, 6_000_000, 0),
        ("response_start", 1, 6_100_000, 0),
        ("response_done", 1, 6_700_000, 0),
        ("consume_done", 1, 6_700_000, 0),
    ]
    prefetch.pending = [SimpleNamespace(ids=torch.empty(24576, 1), future=None, ready=None)]
    prefetch.drain()
    prefetch.close()
    assert "trace_enabled=True" in messages[0]
    message = messages[-1]
    assert "tokens=24576" in message
    assert "slot0(rows=8192)=lookup 2.00ms/response 0.60ms/blocked 0.70ms" in message
    assert "slot1(rows=4096)=lookup 2.00ms/response 0.60ms/blocked 0.70ms" in message
    # 1.0ms of ID routing/transfer precedes the first CPU lookup.
    assert "route=1.00ms" in message


def test_shared_runner_dispatch_selects_paths():
    """Contract shared with other runners (for example a v2 capture manager)."""
    from types import SimpleNamespace

    from vllm.config import CUDAGraphMode

    from vllm_ascend.models.deepseek_v41 import engram_runner

    calls = []

    class Model:
        def prepare_engram_inputs(self, input_ids, positions, padded_tokens=None):
            calls.append("sync")
            return {"engram_lookups": {}, "engram_mask": None}

        def prepare_engram_prefetch_inputs(self, input_ids, positions, padded_tokens=None):
            calls.append("prefetch")
            prefetcher = SimpleNamespace()
            return {"engram_lookups": {}, "engram_mask": None, "engram_prefetch": prefetcher}

        def prepare_engram_graph_inputs(self, padded_tokens=None):
            calls.append("capture")
            return {"engram_lookups": {}, "engram_mask": None}

    model = Model()
    none_mode = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE, capturing=False)
    full_mode = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL, capturing=False)

    on = engram_runner.prepare_engram_for_forward(
        model, none_mode, torch.ones(4), torch.arange(4), 4, prefetch_flag=True
    )
    assert on.selected == "prepare_engram_prefetch_inputs" and on.prefetcher is not None

    off = engram_runner.prepare_engram_for_forward(
        model, none_mode, torch.ones(4), torch.arange(4), 4, prefetch_flag=False
    )
    assert off.selected == "prepare_engram_inputs" and off.prefetcher is None

    decode = engram_runner.prepare_engram_for_forward(
        model, full_mode, torch.ones(4), torch.arange(4), 4, prefetch_flag=True
    )
    assert decode.selected == "prepare_engram_inputs"

    capture = engram_runner.prepare_engram_for_forward(
        model, none_mode, torch.ones(4), torch.arange(4), 4, capture_active=True
    )
    assert capture.selected == "prepare_engram_graph_inputs" and not capture.inputs.get("engram_prefetch")

    plain = engram_runner.prepare_engram_for_forward(SimpleNamespace(), none_mode, None, torch.arange(4), 4)
    assert plain.selected == "none" and plain.inputs == {}
    assert calls == ["prefetch", "sync", "sync", "capture"]


def test_missing_engram_inputs_fail_fast_only_during_capture():
    """An unported runner must not capture host routing/collectives."""
    from types import SimpleNamespace

    from vllm_ascend.models.deepseek_v41 import engram_runner

    eager = SimpleNamespace(capturing=False)
    # Eager: keep the inline fallback, only warn.
    engram_runner.check_engram_inputs_prepared(eager, engram_enabled=False)
    engram_runner.check_engram_inputs_prepared(eager, engram_enabled=True)

    capturing = SimpleNamespace(capturing=True)
    engram_runner.check_engram_inputs_prepared(capturing, engram_enabled=False)
    with pytest.raises(RuntimeError, match="prepare_engram_for_forward"):
        engram_runner.check_engram_inputs_prepared(capturing, engram_enabled=True)

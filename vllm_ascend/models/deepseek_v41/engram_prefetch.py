# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram CPU lookup pipeline.

The worker/collective protocol is intentionally eager/prefill-only.
FULL_DECODE_ONLY replay consumes persistent device buffers and must not call
this module from a graph replay; decode keeps the graph intact because its
lookup batches are small.
"""

from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from os import environ
from time import perf_counter_ns
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from vllm.logger import logger

if TYPE_CHECKING:
    from .engram_hbm import NodeShardedEngram


@dataclass
class PendingLookup:
    table: "NodeShardedEngram"
    ids: torch.Tensor
    order: torch.Tensor
    send: list[int]
    recv: list[int]
    active: bool
    future: Future[torch.Tensor] | None
    host_ids: torch.Tensor | None = None
    ready: object | None = None
    generation: int = 0


class EngramPrefetcher:
    """Two CPU lookup workers per model rank; at most one batch in flight.

    Request/response collectives are submitted by the model thread in layer
    order, including on idle DP replicas. The CPU worker never calls HCCL.
    """

    def __init__(self, device, trace: bool | None = None):
        self.device = torch.device(device)
        # Per-stage P/D timings stay opt-in; the counters only observe
        # prefetch, so decode replay keeps the graph untouched.
        self.trace_enabled = environ.get("ENGRAM_PREFETCH_TRACE") == "1" if trace is None else trace
        self._trace_reported = False
        self.trace: list[tuple[str, int, int, int]] = []
        # Keep HCCL request/response collectives on the model's current
        # stream.  Ascend HCCL does not guarantee ordering across an auxiliary
        # stream; CPU lookup remains asynchronous and is the dominant overlap
        # opportunity.
        # Route both tables before the backbone. Submit the second lookup
        # after the first response so it can overlap the intervening layers.
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="engram-lookup")
        self.pending = []
        self._generation = 0

    def _trace(self, name: str, slot: int = -1, rows: int = 0) -> None:
        if self.trace_enabled:
            self.trace.append((name, slot, perf_counter_ns(), rows))

    @torch.inference_mode()
    def _lookup(self, table, ids, ready=None, slot=-1):
        self._trace("lookup_wait", slot)
        if ready is not None:
            ready.synchronize()
        # Lookup waits for any prior H2D use before reusing a pinned buffer.
        # New H2D copies and collectives are submitted by ``consume``.
        # Row count of the routed shard: this is what the CPU operator sees
        # and what the dedicated-pool threshold is applied to.
        self._trace("lookup_start", slot, ids.numel())
        rows = ids - table.start
        result = table.lookup_local(rows)
        self._trace("lookup_done", slot)
        return result

    @torch.inference_mode()
    def begin(self, tables, ids_list):
        if self.pending:
            raise RuntimeError("Previous Engram batch has not been drained")
        if not tables or len(tables) != len(ids_list):
            raise ValueError("Engram prefetch requires matching tables and IDs")
        query = tables[0].query_group
        if any(table.weight.device.type != "cpu" or table.query_group is not query for table in tables):
            raise ValueError("Engram prefetch requires CPU tables in one query group")
        self._generation += 1
        self.trace.clear()
        self._trace("begin")
        generation = self._generation
        routing = [table._metadata(ids) for table, ids in zip(tables, ids_list)]
        metadata = torch.cat([item[2] for item in routing])
        if query.metadata_on_device:
            gathered = tables[0]._gather_metadata_device(metadata, self.device, query.group)
        else:
            gathered = [torch.empty_like(metadata) for _ in range(query.size)]
            dist.all_gather(gathered, metadata, group=query.cpu_group)
        counts = torch.stack(gathered).reshape(query.size, len(tables), query.size + 1)
        if bool(counts[:, :, -1].any()):
            raise IndexError("Engram hash ID outside table")
        try:
            # Request-side fusion is on by default; the env knob is the
            # measurement/rollback path (7 -> 6 collectives per forward).
            fused_requests = environ.get("ENGRAM_PREFETCH_FUSE_REQUEST_A2A", "1") != "0"
            sends = [meta[:-1].tolist() for *_rest, meta in routing]
            receives = [counts[:, index, query.rank].tolist() for index in range(len(tables))]
            payloads = [
                flat[order].to(self.device) for _table, _ids, (flat, order, _meta) in zip(tables, ids_list, routing)
            ]
            # Every layer's request payload is known before the loop, so one
            # all-to-all replaces the per-layer ones.  Split sizes are per
            # destination, so the payload interleaves each destination's layer
            # chunks; the receiver slices every incoming segment back per
            # layer.  Response collectives keep their per-layer order.
            combined_send = [sum(item[index] for item in sends) for index in range(query.size)]
            combined_recv = [sum(item[index] for item in receives) for index in range(query.size)]
            combined = None
            if fused_requests:
                cursor = [0] * len(tables)
                pieces = []
                for rank in range(query.size):
                    for index in range(len(tables)):
                        width = sends[index][rank]
                        pieces.append(payloads[index][cursor[index] : cursor[index] + width])
                        cursor[index] += width
                combined = torch.empty(sum(combined_recv), dtype=torch.int64, device=self.device)
                if any(bool(counts[:, index, :-1].any()) for index in range(len(tables))):
                    dist.all_to_all_single(combined, torch.cat(pieces), combined_recv, combined_send, group=query.group)
                starts = [0]
                for value in combined_recv:
                    starts.append(starts[-1] + value)
            for index, (table, ids, (flat, order, meta)) in enumerate(zip(tables, ids_list, routing)):
                send = sends[index]
                recv = receives[index]
                active = bool(counts[:, index, :-1].any())
                if fused_requests:
                    picks = []
                    for rank in range(query.size):
                        begin = starts[rank] + sum(receives[prev][rank] for prev in range(index))
                        picks.append(torch.arange(begin, begin + receives[index][rank], device=self.device))
                    incoming = combined.index_select(0, torch.cat(picks))
                else:
                    incoming = torch.empty(sum(recv), dtype=torch.int64, device=self.device)
                    if active:
                        dist.all_to_all_single(incoming, flat[order].to(self.device), recv, send, group=query.group)
                ready = None
                if self.device.type == "npu":
                    # Queue ID transfer without stopping backbone submission.
                    # The worker waits for this copy only, not subsequent NPU
                    # computation. Its future owns the pinned destination.
                    host_ids = torch.empty(incoming.shape, dtype=torch.int64, pin_memory=True)
                    host_ids.copy_(incoming, non_blocking=True)
                    ready = torch.npu.Event()
                    ready.record(torch.npu.current_stream(self.device))
                else:
                    host_ids = incoming.cpu()
                self.pending.append(
                    PendingLookup(table, ids, order, send, recv, active, None, host_ids, ready, generation)
                )
        except BaseException:
            self.drain()
            raise

    @torch.inference_mode()
    def _receive(self, item, local_values, padded_tokens, slot):
        self._trace("response_start", slot)
        table, query = item.table, item.table.query_group
        returned = torch.empty((sum(item.send), table.width), dtype=torch.bfloat16, device=self.device)
        if item.active:
            values = local_values.to(self.device, non_blocking=table.offload_pinned)
            dist.all_to_all_single(returned, values, item.send, item.recv, group=query.group)
            table._record_offload_use(local_values.data_ptr(), self.device)
        # Padding is local to a TP group; idle DP ranks still served their shards.
        result = torch.zeros((padded_tokens, item.ids.shape[-1], table.width), dtype=torch.bfloat16, device=self.device)
        if query.is_source:
            result.view(-1, table.width)[item.order.to(self.device)] = returned
        if result.numel():
            dist.broadcast(result, src=query.tp_source, group=query.tp_group)
        self._trace("response_done", slot)
        return result.flatten(1)

    def consume(self, index, padded_tokens):
        """Wait for this layer only, then submit its response on the model stream."""
        item = self.pending[index]
        self._trace("consume_wait", index)
        if item.generation != self._generation:
            raise RuntimeError("Engram prefetch generation is stale")
        if padded_tokens < item.ids.shape[0]:
            raise ValueError("Engram output capacity is smaller than the query")
        if item.future is None:
            # Preserve correctness for callers that consume without start.
            # Compiled callers retain start through an explicit dependency;
            # reaching this fallback does not prove prefetch ran earlier.
            item.future = self.executor.submit(self._lookup, item.table, item.host_ids, item.ready, index)
        local_values = item.future.result()
        result = self._receive(item, local_values, padded_tokens, index)
        self._trace("consume_done", index)
        item.host_ids = None
        item.ready = None
        if index + 1 < len(self.pending):
            next_item = self.pending[index + 1]
            if next_item.future is None:
                next_item.future = self.executor.submit(
                    self._lookup, next_item.table, next_item.host_ids, next_item.ready, index + 1
                )
        return result

    def start(self):
        """Submit the first lookup at backbone entry, after ID routing."""
        if not self.pending:
            return
        first = self.pending[0]
        if first.future is None:
            first.future = self.executor.submit(self._lookup, first.table, first.host_ids, first.ready, 0)

    def _report(self, tokens):
        """Log one compact per-forward stage line for the prefetch path."""
        pairs = {
            "lookup_done": "lookup_start",
            "response_done": "response_start",
            "consume_done": "consume_wait",
        }
        opened: dict[tuple[str, int], int] = {}
        stages: dict[tuple[str, int], float] = {}
        rows: dict[int, int] = {}
        for name, slot, stamp, extra in self.trace:
            if pairs.get(name) is not None:
                start = opened.pop((pairs[name], slot), None)
                if start is not None:
                    key = (pairs[name], slot)
                    stages[key] = stages.get(key, 0.0) + (stamp - start) / 1e6
            else:
                opened[(name, slot)] = stamp
                if name == "lookup_start":
                    rows[slot] = extra
        route = self.trace[0][2]
        parts = []
        for slot in sorted({slot for _, slot in stages}):
            parts.append(
                f"slot{slot}(rows={rows.get(slot, 0)})=lookup {stages.get(('lookup_start', slot), 0.0):.2f}ms/"
                f"response {stages.get(('response_start', slot), 0.0):.2f}ms/"
                f"blocked {stages.get(('consume_wait', slot), 0.0):.2f}ms"
            )
        logger.info(
            "engram prefetch tokens=%d total=%.2fms route=%.2fms %s",
            tokens,
            (self.trace[-1][2] - route) / 1e6,
            (opened.get(("lookup_wait", 0), route) - route) / 1e6,
            " ".join(parts),
        )

    def drain(self):
        """Join CPU jobs even when model execution raises; never issue collectives."""
        pending, self.pending = self.pending, []
        # Join all jobs before propagating the first exception.
        futures = [item.future for item in pending if item.future is not None]
        # A deferred ID copy may still be in flight even though no worker has
        # consumed it.  Synchronize it before releasing the request state.
        for item in pending:
            if item.future is None and item.ready is not None:
                item.ready.synchronize()
        wait(futures)
        for future in futures:
            future.result()
        if pending and not self._trace_reported:
            self._trace_reported = True
            # One line per prefetcher instance: proves the overlap path ran and
            # reports why a stage line is missing when tracing is off.
            logger.info(
                "engram prefetch drain: trace_enabled=%s trace_events=%d pending=%d",
                self.trace_enabled,
                len(self.trace),
                len(pending),
            )
        # Keep-priority dummy batches have no tokens; their spans cover idle
        # time and would hide the real per-forward stage numbers.
        if self.trace_enabled and self.trace and pending and pending[0].ids.numel():
            self._report(pending[0].ids.shape[0])

    def close(self):
        try:
            self.drain()
        finally:
            self.executor.shutdown(wait=True)


class PinnedMaskStaging:
    """Stage the Engram token mask through pinned memory.

    The captured device mask is refreshed every forward.  ``copy_`` from a
    pageable CPU tensor blocks the host (and synchronizes the device stream),
    so the mask goes through a persistent pinned buffer and an async H2D.
    That copy runs on its own stream: issued on the model stream it would sit
    behind the previous step's kernels, and waiting for it would serialize the
    host behind the device.  The model stream only waits on the copy event
    instead, and the pinned source is reused once that copy landed.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self._buffer: torch.Tensor | None = None
        self._stream = None
        self._event = None

    def upload(self, mask: torch.Tensor, padded_mask: torch.Tensor) -> None:
        count = int(mask.numel())
        if self._buffer is None or self._buffer.numel() < count:
            self._buffer = torch.empty(count, dtype=torch.bool, pin_memory=True)
        pinned = self._buffer[:count]
        if self.device.type == "npu":
            if self._stream is None:
                self._stream = torch.npu.Stream(device=self.device)
            if self._event is not None and not self._event.query():
                # The previous copy may still be reading this single buffer.
                self._event.synchronize()
            pinned.copy_(mask)
            with torch.npu.stream(self._stream):
                padded_mask.zero_()
                padded_mask[:count].copy_(pinned, non_blocking=True)
            self._event = torch.npu.Event()
            self._event.record(self._stream)
            torch.npu.current_stream(self.device).wait_event(self._event)
            return
        pinned.copy_(mask)
        padded_mask.zero_()
        padded_mask[:count].copy_(pinned)

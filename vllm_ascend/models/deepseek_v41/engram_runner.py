# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner-side Engram dispatch shared by model runner versions.

Variable Engram routing must run outside graph capture/replay and refresh the
persistent device buffers the captured backbone consumes.  Model runner v1
calls :func:`prepare_engram_for_forward` from ``_model_forward``; any other
runner (a v2 runner or a draft/DSpark graph) can call the same helper from its
own prepare-inputs hook and then drain the returned prefetcher once the forward
finished.  Keeping the dispatch here avoids a second copy of the
eager/capture/runtime-mode rules.
"""

from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.config import CUDAGraphMode
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config


@dataclass
class EngramPrep:
    """Result of one Engram preparation step."""

    inputs: dict[str, Any] = field(default_factory=dict)
    prefetcher: Any | None = None
    prefetch_flag: bool = False
    mode: Any = None
    capturing: bool = False
    capture_active: bool = False
    selected: str = "none"


def is_capturing(forward_context: Any) -> bool:
    """Capture state for the current forward.

    ``forward_context.capturing`` is the vLLM signal; the NPU stream query
    covers runners that capture outside that flag (for example a manager that
    only marks the stream).
    """

    capturing = bool(getattr(forward_context, "capturing", False))
    if hasattr(torch, "npu"):
        with suppress(RuntimeError):
            capturing = capturing or bool(torch.npu.is_current_stream_capturing())
    return capturing


def check_engram_inputs_prepared(forward_context: Any, engram_enabled: bool) -> None:
    """Guard the inline Engram fallback when a runner skipped the hook.

    The captured backbone consumes only the persistent buffers, so a runner
    that forgets to prepare them would otherwise capture host routing and HCCL
    collectives (or, on replay, silently reuse stale rows).  Failing fast names
    the missing call; eager forwards keep the old inline path with a warning so
    an unported runner still runs, just slowly.
    """

    if not engram_enabled:
        return
    if is_capturing(forward_context):
        raise RuntimeError(
            "Engram inputs are missing during graph capture: the active model runner must call "
            "vllm_ascend.models.deepseek_v41.engram_runner.prepare_engram_for_forward before the "
            "model forward and drain the returned prefetcher afterwards"
        )
    logger.warning_once(
        "Engram inputs were not prepared by the model runner; falling back to the inline eager "
        "path (routing, CPU lookup and HCCL collectives inside the forward). Eager execution "
        "stays correct but loses the prefetch overlap."
    )


def prepare_engram_for_forward(
    model: Any,
    forward_context: Any,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    num_tokens_padded: int,
    capture_active: bool = False,
    prefetch_flag: bool | None = None,
) -> EngramPrep:
    """Select and run the Engram preparation for one forward.

    ``FULL_DECODE_ONLY`` uses ``NONE`` for prefill/mixed batches and ``FULL``
    for captured decode, so the mode coordinated across DP (including idle
    replicas) decides between the overlap path and the synchronous refresh.
    Capture itself only re-registers the persistent buffers.
    """

    prepare = getattr(model, "prepare_engram_inputs", None)
    result = EngramPrep(
        mode=getattr(forward_context, "cudagraph_runtime_mode", None),
        capture_active=capture_active,
    )

    if prepare is None:
        return result
    result.capturing = is_capturing(forward_context)
    if result.capturing or capture_active:
        graph_prepare = getattr(model, "prepare_engram_graph_inputs", None)
        if graph_prepare is not None:
            result.inputs = graph_prepare(num_tokens_padded)
            result.selected = "prepare_engram_graph_inputs"
        return result
    # Read lazily so capture-only and non-Engram models stay usable without an
    # initialized Ascend config (tests, standalone runners).
    result.prefetch_flag = get_ascend_config().enable_engram_prefetch if prefetch_flag is None else prefetch_flag
    if result.prefetch_flag and result.mode == CUDAGraphMode.NONE:
        prepare = getattr(model, "prepare_engram_prefetch_inputs", prepare)
    result.inputs = prepare(input_ids, positions, num_tokens_padded)
    result.prefetcher = result.inputs.get("engram_prefetch")
    result.selected = getattr(prepare, "__name__", str(prepare))
    return result

"""Fail-loud exception hierarchy, boundary invariant guards, structured logging, and crash reporting.

Every silent fallback is prohibited across ``swarm``. Invalid configurations, missing backends,
non-finite numbers, invalid probability simplices, cyclic intra-slice graphs, and broken topology
or checkpoint invariants immediately raise a typed :class:`SwarmError` subclass carrying structured
diagnostic context. Top-level CLI commands and worker processes wrap execution with
:func:`fail_loud_entrypoint` to emit atomic ``error_report_<UTC>_pid<PID>.json`` diagnostics.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import wraps
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from typing import Any, ParamSpec, TypeVar

import numpy as np
import torch

P = ParamSpec("P")
R = TypeVar("R")

LOGGER_ROOT_NAME = "swarm"


class SwarmError(Exception):
    """Base class for all fail-loud domain errors in ``swarm``."""

    def __init__(self, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        self.message = message
        self.context: dict[str, Any] = dict(context) if context is not None else {}
        formatted = message
        if self.context:
            formatted = f"{message} | context={ _json_safe(self.context) }"
        super().__init__(formatted)


class ConfigValidationError(SwarmError):
    """Raised when configuration fields, types, paths, or seed splits fail strict validation."""


class BackendUnavailableError(SwarmError):
    """Raised when a requested simulator or compute backend/precision is unavailable on host."""


class SimulationLifecycleError(SwarmError):
    """Raised when SUMO/libsumo/traci process or singleton lifecycle invariants are violated."""


class TopologyIntegrityError(SwarmError):
    """Raised when road network corridors, turn options, or junction foe tables are invalid."""


class RoutingIntegrityError(SwarmError):
    """Raised when an onboard or initial OD route violates connectivity or edge invariants."""


class CheckpointIntegrityError(SwarmError):
    """Raised when a neural DBN checkpoint is missing, mismatched, or uncalibrated."""


class NumericalInvariantError(SwarmError):
    """Raised when finite-value, tensor-shape, probability, or PSD covariance guards fail."""


class CausalityFirewallError(SwarmError):
    """Raised when runtime modules attempt to access ground-truth or future telemetry."""


class DAGInvariantError(SwarmError):
    """Raised when an intra-slice DBN coordination graph violates strict acyclicity or ordering."""


# Explicit alias so both `#18` (`DAGViolationError`) and `#20` (`DAGInvariantError`) resolve identically.
DAGViolationError = DAGInvariantError


class EpisodeExecutionError(SwarmError):
    """Raised when an evaluation, rollout collection, or training worker episode fails."""


@dataclass
class CrashContext:
    """Process-local execution metadata attached to structured crash reports."""

    command: str = "unknown"
    scenario_id: str | None = None
    policy_name: str | None = None
    seed: int | None = None
    sim_time: float | None = None
    step_idx: int | None = None
    active_veh_count: int | None = None
    checkpoint_sha256: str | None = None
    state_digest: dict[str, Any] = field(default_factory=dict)


_KNOWN_CRASH_FIELDS: frozenset[str] = frozenset(
    {
        "command",
        "scenario_id",
        "policy_name",
        "seed",
        "sim_time",
        "step_idx",
        "active_veh_count",
        "checkpoint_sha256",
        "state_digest",
    }
)

_CRASH_CONTEXT_VAR: ContextVar[CrashContext | None] = ContextVar("swarm_crash_context", default=None)


def get_crash_context() -> CrashContext:
    """Return the active process/task :class:`CrashContext`, initializing if needed."""
    ctx = _CRASH_CONTEXT_VAR.get()
    if ctx is None:
        ctx = CrashContext()
        _CRASH_CONTEXT_VAR.set(ctx)
    return ctx


def update_crash_context(**kwargs: Any) -> CrashContext:
    """Update fields on the active :class:`CrashContext`; extra keys merge into ``state_digest``."""
    ctx = get_crash_context()
    for key, value in kwargs.items():
        if key == "state_digest":
            if isinstance(value, Mapping):
                ctx.state_digest.update(dict(value))
            else:
                ctx.state_digest["digest"] = _json_safe(value)
        elif key in _KNOWN_CRASH_FIELDS:
            setattr(ctx, key, value)
        else:
            ctx.state_digest[key] = _json_safe(value)
    return ctx


def reset_crash_context() -> CrashContext:
    """Reset the active :class:`CrashContext` to clean defaults."""
    fresh = CrashContext()
    _CRASH_CONTEXT_VAR.set(fresh)
    return fresh


class _JsonLinesFormatter(logging.Formatter):
    """Structured JSONL log formatter for machine-auditable run logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp_utc": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "pid": record.process,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def configure_logging(level: str = "INFO", log_jsonl_path: Path | str | None = None) -> logging.Logger:
    """Configure the hierarchical ``swarm`` root logger with console and optional JSONL output."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ConfigValidationError(
            f"Invalid log level {level!r}; expected DEBUG, INFO, WARNING, ERROR, or CRITICAL.",
            context={"level": level},
        )

    logger = logging.getLogger(LOGGER_ROOT_NAME)
    logger.setLevel(numeric_level)
    logger.propagate = False

    # Clear prior handlers to keep repeated CLI/test invocations idempotent.
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        existing.close()

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | pid=%(process)d | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(stream_handler)

    if log_jsonl_path is not None:
        jsonl_path = Path(log_jsonl_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(jsonl_path, mode="a", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(_JsonLinesFormatter())
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger guaranteed to live under the ``swarm`` hierarchy."""
    cleaned = name.strip(".")
    if not cleaned or cleaned == LOGGER_ROOT_NAME:
        return logging.getLogger(LOGGER_ROOT_NAME)
    if cleaned.startswith(f"{LOGGER_ROOT_NAME}."):
        return logging.getLogger(cleaned)
    return logging.getLogger(f"{LOGGER_ROOT_NAME}.{cleaned}")


def _json_safe(value: Any) -> Any:
    """Convert arbitrary nested objects into deterministic JSON-serializable primitives."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        if value.size <= 32:
            return [_json_safe(v) for v in value.reshape(-1).tolist()]
        return {
            "ndarray_shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite_count": int(np.isfinite(value).sum()) if np.issubdtype(value.dtype, np.number) else None,
        }
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        if detached.numel() <= 32:
            return [_json_safe(v) for v in detached.reshape(-1).tolist()]
        is_num = detached.is_floating_point() or detached.is_complex()
        return {
            "tensor_shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "device": str(value.device),
            "finite_count": int(torch.isfinite(detached).sum().item()) if is_num else int(detached.numel()),
        }
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        seq = list(value)
        if len(seq) <= 64:
            return [_json_safe(v) for v in seq]
        return {"sequence_len": len(seq), "head": [_json_safe(v) for v in seq[:16]]}
    return repr(value)


def require_finite(
    x: np.ndarray | torch.Tensor | float | int,
    name: str,
    *,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Raise :class:`NumericalInvariantError` if ``x`` is boolean or contains any NaN/Inf."""
    if isinstance(x, bool):
        raise NumericalInvariantError(
            f"Expected numeric scalar/array/tensor for {name!r}, got bool.",
            context={"name": name, **(dict(context) if context else {})},
        )
    if isinstance(x, (int, float, np.number)):
        scalar = float(x)
        if not math.isfinite(scalar):
            raise NumericalInvariantError(
                f"Non-finite scalar in {name!r}: {scalar!r}.",
                context={"name": name, "value": scalar, **(dict(context) if context else {})},
            )
        return

    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return
        finite_mask = torch.isfinite(x)
        if not bool(finite_mask.all().item()):
            detached = x.detach().cpu().float()
            nan_count = int(torch.isnan(detached).sum().item())
            inf_count = int(torch.isinf(detached).sum().item())
            finite_vals = detached[torch.isfinite(detached)]
            diag: dict[str, Any] = {
                "name": name,
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "device": str(x.device),
                "nan_count": nan_count,
                "inf_count": inf_count,
                "finite_min": float(finite_vals.min().item()) if finite_vals.numel() > 0 else None,
                "finite_max": float(finite_vals.max().item()) if finite_vals.numel() > 0 else None,
            }
            if context:
                diag.update(dict(context))
            raise NumericalInvariantError(f"Non-finite entries detected in tensor {name!r}.", context=diag)
        return

    if isinstance(x, np.ndarray):
        if x.size == 0:
            return
        if not np.issubdtype(x.dtype, np.number):
            raise NumericalInvariantError(
                f"Expected numeric ndarray for {name!r}, got dtype={x.dtype}.",
                context={"name": name, "dtype": str(x.dtype), **(dict(context) if context else {})},
            )
        finite_mask = np.isfinite(x)
        if not bool(finite_mask.all()):
            nan_count = int(np.isnan(x).sum())
            inf_count = int(np.isinf(x).sum())
            finite_vals = x[finite_mask]
            diag = {
                "name": name,
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "nan_count": nan_count,
                "inf_count": inf_count,
                "finite_min": float(finite_vals.min()) if finite_vals.size > 0 else None,
                "finite_max": float(finite_vals.max()) if finite_vals.size > 0 else None,
            }
            if context:
                diag.update(dict(context))
            raise NumericalInvariantError(f"Non-finite entries detected in ndarray {name!r}.", context=diag)
        return

    raise NumericalInvariantError(
        f"Unsupported type {type(x).__name__} passed to require_finite for {name!r}.",
        context={"name": name, "type": type(x).__name__, **(dict(context) if context else {})},
    )


def require_shape(
    x: np.ndarray | torch.Tensor,
    expected: tuple[int | None, ...],
    name: str,
    *,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Validate rank and non-None dimension sizes of ``x``, raising :class:`NumericalInvariantError`."""
    if not isinstance(x, (np.ndarray, torch.Tensor)):
        raise NumericalInvariantError(
            f"Expected ndarray or Tensor for {name!r}, got {type(x).__name__}.",
            context={"name": name, "expected": list(expected), **(dict(context) if context else {})},
        )
    actual = tuple(int(d) for d in x.shape)
    if len(actual) != len(expected):
        raise NumericalInvariantError(
            f"Rank mismatch for {name!r}: actual shape {actual} vs expected {expected}.",
            context={
                "name": name,
                "actual_shape": list(actual),
                "expected_shape": list(expected),
                **(dict(context) if context else {}),
            },
        )
    for axis, (act_dim, exp_dim) in enumerate(zip(actual, expected, strict=True)):
        if exp_dim is not None and act_dim != exp_dim:
            raise NumericalInvariantError(
                f"Shape mismatch for {name!r} at axis {axis}: actual {actual} vs expected {expected}.",
                context={
                    "name": name,
                    "axis": axis,
                    "actual_shape": list(actual),
                    "expected_shape": list(expected),
                    **(dict(context) if context else {}),
                },
            )


def require_probability_range(
    x: np.ndarray | torch.Tensor | float,
    name: str,
    *,
    tol: float = 1e-5,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Require ``x`` to be finite and within ``[-tol, 1 + tol]``."""
    require_finite(x, name, context=context)
    if isinstance(x, (int, float, np.number)):
        val = float(x)
        if val < -tol or val > 1.0 + tol:
            raise NumericalInvariantError(
                f"Probability scalar {name!r} out of [0, 1] bounds: {val}.",
                context={"name": name, "value": val, "tol": tol, **(dict(context) if context else {})},
            )
        return
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return
        min_v = float(x.min().item())
        max_v = float(x.max().item())
    else:
        if x.size == 0:
            return
        min_v = float(x.min())
        max_v = float(x.max())

    if min_v < -tol or max_v > 1.0 + tol:
        raise NumericalInvariantError(
            f"Probability values in {name!r} out of [0, 1] bounds: min={min_v}, max={max_v}.",
            context={
                "name": name,
                "min": min_v,
                "max": max_v,
                "tol": tol,
                **(dict(context) if context else {}),
            },
        )


def require_probability_simplex(
    x: np.ndarray | torch.Tensor | Sequence[float],
    name: str,
    *,
    axis: int = -1,
    atol: float = 1e-4,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Require ``x`` to be a valid non-negative probability distribution summing to 1 along ``axis``."""
    arr: np.ndarray | torch.Tensor
    if isinstance(x, (torch.Tensor, np.ndarray)):
        arr = x
    else:
        arr = np.asarray(list(x), dtype=np.float64)

    require_finite(arr, name, context=context)

    if isinstance(arr, torch.Tensor):
        if arr.ndim == 0 or arr.shape[axis] == 0:
            raise NumericalInvariantError(
                f"Empty or 0-D simplex tensor for {name!r}: shape={tuple(arr.shape)}.",
                context={"name": name, "shape": list(arr.shape), **(dict(context) if context else {})},
            )
        min_v = float(arr.min().item())
        max_v = float(arr.max().item())
        sums = arr.float().sum(dim=axis)
        max_sum_err = float((sums - 1.0).abs().max().item()) if sums.numel() > 0 else 0.0
    else:
        if arr.ndim == 0 or arr.shape[axis] == 0:
            raise NumericalInvariantError(
                f"Empty or 0-D simplex array for {name!r}: shape={arr.shape}.",
                context={"name": name, "shape": list(arr.shape), **(dict(context) if context else {})},
            )
        min_v = float(arr.min()) if arr.size > 0 else 0.0
        max_v = float(arr.max()) if arr.size > 0 else 0.0
        sums = arr.sum(axis=axis)
        max_sum_err = float(np.max(np.abs(sums - 1.0))) if np.size(sums) > 0 else 0.0

    if min_v < -atol or max_v > 1.0 + atol or max_sum_err > atol:
        raise NumericalInvariantError(
            f"Probability simplex violation in {name!r}: min={min_v}, max={max_v}, max_sum_err={max_sum_err}.",
            context={
                "name": name,
                "min": min_v,
                "max": max_v,
                "max_sum_err": max_sum_err,
                "atol": atol,
                **(dict(context) if context else {}),
            },
        )


def require_psd_covariance(
    cov_2x2: np.ndarray | torch.Tensor,
    name: str,
    *,
    min_eig: float = -1e-7,
    sym_atol: float = 1e-5,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Validate that ``cov_2x2`` is finite, symmetric ``(..., 2, 2)``, and positive semi-definite."""
    require_finite(cov_2x2, name, context=context)
    if cov_2x2.ndim < 2 or cov_2x2.shape[-2:] != (2, 2):
        raise NumericalInvariantError(
            f"Expected (..., 2, 2) covariance for {name!r}, got shape {tuple(cov_2x2.shape)}.",
            context={"name": name, "shape": list(cov_2x2.shape), **(dict(context) if context else {})},
        )

    arr = cov_2x2.detach().cpu().numpy() if isinstance(cov_2x2, torch.Tensor) else np.asarray(cov_2x2, dtype=np.float64)
    if arr.size == 0:
        return
    asym = float(np.max(np.abs(arr - np.swapaxes(arr, -1, -2))))
    if asym > sym_atol:
        raise NumericalInvariantError(
            f"Covariance {name!r} violates symmetry: max |C - C^T| = {asym}.",
            context={"name": name, "max_asymmetry": asym, "sym_atol": sym_atol, **(dict(context) if context else {})},
        )
    sym_arr = 0.5 * (arr + np.swapaxes(arr, -1, -2))
    eigvals = np.linalg.eigvalsh(sym_arr)
    smallest_eig = float(np.min(eigvals))
    if smallest_eig < min_eig:
        raise NumericalInvariantError(
            f"Covariance {name!r} is not PSD: min_eigenvalue={smallest_eig} < {min_eig}.",
            context={
                "name": name,
                "min_eigenvalue": smallest_eig,
                "min_eig_threshold": min_eig,
                **(dict(context) if context else {}),
            },
        )


def _extract_edge_endpoints(edge: Any) -> tuple[str, str]:
    """Extract ``(src_id, dst_id)`` from either a ``DBNEdge``-like object or a 2-tuple."""
    if hasattr(edge, "src_id") and hasattr(edge, "dst_id"):
        return str(edge.src_id), str(edge.dst_id)
    if isinstance(edge, Sequence) and not isinstance(edge, (str, bytes)) and len(edge) >= 2:
        return str(edge[0]), str(edge[1])
    raise DAGInvariantError(
        f"Unsupported edge representation {edge!r}; expected DBNEdge or (src_id, dst_id).",
        context={"edge_repr": repr(edge)},
    )


def require_dag(
    veh_ids: Sequence[str],
    edges: Sequence[Any],
    precedence_scores: Mapping[str, float] | None = None,
    *,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Enforce strict acyclicity (Kahn's algorithm) and optional total precedence ordering on ``edges``."""
    node_set = set(veh_ids)
    if len(node_set) != len(veh_ids):
        raise DAGInvariantError(
            "Duplicate vehicle IDs passed to require_dag.",
            context={"veh_count": len(veh_ids), "unique_count": len(node_set), **(dict(context) if context else {})},
        )

    if precedence_scores is not None:
        for vid in veh_ids:
            if vid not in precedence_scores:
                raise DAGInvariantError(
                    f"Missing precedence score for vehicle {vid!r}.",
                    context={"veh_id": vid, **(dict(context) if context else {})},
                )
            require_finite(precedence_scores[vid], f"precedence_scores[{vid}]", context=context)

    adj: dict[str, list[str]] = {vid: [] for vid in veh_ids}
    indegree: dict[str, int] = {vid: 0 for vid in veh_ids}

    for edge in edges:
        src, dst = _extract_edge_endpoints(edge)
        if src not in node_set or dst not in node_set:
            raise DAGInvariantError(
                f"Edge ({src!r} -> {dst!r}) references vehicle outside active veh_ids.",
                context={"src": src, "dst": dst, **(dict(context) if context else {})},
            )
        if src == dst:
            raise DAGInvariantError(
                f"Self-loop ({src!r} -> {dst!r}) is forbidden in intra-slice DBN graph.",
                context={"src": src, **(dict(context) if context else {})},
            )
        if precedence_scores is not None:
            r_src = float(precedence_scores[src])
            r_dst = float(precedence_scores[dst])
            strictly_precedes = (r_src > r_dst) or (r_src == r_dst and src < dst)
            if not strictly_precedes:
                raise DAGInvariantError(
                    f"Precedence violation on edge ({src!r} -> {dst!r}): r_src={r_src} vs r_dst={r_dst}.",
                    context={
                        "src": src,
                        "dst": dst,
                        "r_src": r_src,
                        "r_dst": r_dst,
                        **(dict(context) if context else {}),
                    },
                )
        adj[src].append(dst)
        indegree[dst] += 1

    queue: deque[str] = deque(vid for vid, deg in indegree.items() if deg == 0)
    visited = 0
    while queue:
        u = queue.popleft()
        visited += 1
        for v in adj[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    if visited != len(node_set):
        cycle_nodes = sorted(vid for vid, deg in indegree.items() if deg > 0)
        raise DAGInvariantError(
            f"Directed cycle detected in intra-slice DBN graph involving {len(cycle_nodes)} vehicles.",
            context={"cycle_nodes": cycle_nodes[:32], **(dict(context) if context else {})},
        )


def _probe_git_metadata(cwd: Path | None = None) -> dict[str, Any]:
    """Query git commit SHA and working-tree dirty flag without swallowing exceptions."""
    work_dir = str(cwd) if cwd is not None else os.getcwd()
    head_proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    status_proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    commit_sha = head_proc.stdout.strip() if head_proc.returncode == 0 else None
    is_dirty = bool(status_proc.stdout.strip()) if status_proc.returncode == 0 else None
    return {"commit_sha": commit_sha, "is_dirty": is_dirty}


def _installed_version(dist_name: str) -> str | None:
    """Return installed package distribution version if present, else ``None``."""
    spec_found = importlib.util.find_spec(dist_name.replace("-", "_")) is not None
    for candidate in (dist_name, dist_name.replace("_", "-")):
        dists = [d for d in importlib.metadata.distributions() if (d.metadata["Name"] or "").lower() == candidate.lower()]
        if dists:
            return dists[0].version
    return "present_unknown_version" if spec_found else None


def write_error_report(
    exc: BaseException,
    report_dir: Path | str = Path("."),
    config_snapshot: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write ``error_report_<UTC>_pid<PID>.json`` with full crash diagnostics."""
    target_dir = Path(report_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc)
    stamp = now_utc.strftime("%Y%m%dT%H%M%S%fZ")
    pid = os.getpid()
    report_path = target_dir / f"error_report_{stamp}_pid{pid}.json"
    tmp_path = target_dir / f".error_report_{stamp}_pid{pid}.json.tmp"

    ctx = get_crash_context()
    payload: dict[str, Any] = {
        "schema_version": 2,
        "timestamp_utc": now_utc.isoformat(),
        "pid": pid,
        "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "exception_message": str(exc),
        "swarm_error_context": _json_safe(exc.context) if isinstance(exc, SwarmError) else None,
        "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
        "crash_context": _json_safe(asdict(ctx)),
        "git": _probe_git_metadata(),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "eclipse_sumo_version": _installed_version("eclipse-sumo"),
            "libsumo_version": _installed_version("libsumo"),
            "scipy_version": _installed_version("scipy"),
        },
        "config_snapshot": _json_safe(config_snapshot) if config_snapshot is not None else None,
    }

    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp_path.write_text(serialized, encoding="utf-8")
    os.replace(tmp_path, report_path)
    return report_path


def fail_loud_entrypoint(
    report_dir_resolver: Callable[..., Path | str] | Path | str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrap a CLI or worker entrypoint to dump an atomic crash report and re-raise on failure.

    Can be applied either as ``@fail_loud_entrypoint()`` / ``@fail_loud_entrypoint(Path(...))``
    or with a callable ``report_dir_resolver(*args, **kwargs) -> Path``.
    """
    # Support bare `@fail_loud_entrypoint` without parentheses when passed the target function directly
    # ONLY if report_dir_resolver is not intended as a resolver. To keep typing unambiguous, inspect
    # whether report_dir_resolver is a Path/str/None vs a function; for resolver functions, return
    # the standard decorator closure.
    def _decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def _wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except BaseException as exc:
                if isinstance(exc, SystemExit) and (exc.code is None or exc.code == 0):
                    raise
                if callable(report_dir_resolver):
                    target_dir = Path(report_dir_resolver(*args, **kwargs))
                elif report_dir_resolver is not None:
                    target_dir = Path(report_dir_resolver)
                else:
                    target_dir = Path.cwd()
                report_path = write_error_report(exc, report_dir=target_dir)
                get_logger("crash").critical(
                    "Unhandled %s in %s; crash report written to %s",
                    type(exc).__name__,
                    func.__qualname__,
                    report_path,
                )
                raise

        return _wrapped

    return _decorator


__all__ = [
    "BackendUnavailableError",
    "CausalityFirewallError",
    "CheckpointIntegrityError",
    "ConfigValidationError",
    "CrashContext",
    "DAGInvariantError",
    "DAGViolationError",
    "EpisodeExecutionError",
    "NumericalInvariantError",
    "RoutingIntegrityError",
    "SimulationLifecycleError",
    "SwarmError",
    "TopologyIntegrityError",
    "configure_logging",
    "fail_loud_entrypoint",
    "get_crash_context",
    "get_logger",
    "require_dag",
    "require_finite",
    "require_probability_range",
    "require_probability_simplex",
    "require_psd_covariance",
    "require_shape",
    "reset_crash_context",
    "update_crash_context",
    "write_error_report",
]

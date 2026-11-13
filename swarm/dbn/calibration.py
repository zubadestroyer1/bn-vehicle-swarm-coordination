"""Post-hoc Validation Temperature Scaling & Predictive Calibration Metrics.

Boundary Lock B (``#19`` §2):
- Temperature calibration executes strictly in ``model.eval()`` mode on held-out validation logits.
- Fits ``(T_cal_edge, T_cal_delay, T_cal_link)`` via bounded L-BFGS-B on ``[0.1, 10.0]`` minimizing
  binary cross-entropy negative log-likelihood.
- Computes publication-grade predictive structure/delay/spillback metrics: ROC-AUC, PR-AUC,
  Brier score, Log-Loss, 10-bin ECE, MCE, reliability diagram bins, and ROC/PR curves.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit
import torch
from torch import Tensor, nn

from ..logging_utils import (
    CheckpointIntegrityError,
    ConfigValidationError,
    NumericalInvariantError,
    require_finite,
    require_probability_range,
)

TEMP_MIN: float = 0.1
TEMP_MAX: float = 10.0


def _to_1d_float64(arr: np.ndarray | Tensor | list[float], name: str) -> np.ndarray:
    if isinstance(arr, Tensor):
        out = arr.detach().cpu().to(torch.float64).numpy().reshape(-1)
    else:
        out = np.asarray(arr, dtype=np.float64).reshape(-1)
    require_finite(out, name)
    return out


def _validate_binary_targets(targets: np.ndarray, name: str) -> np.ndarray:
    require_finite(targets, name)
    if targets.size == 0:
        raise ConfigValidationError(f"{name} cannot be empty.", context={"name": name})
    is_binary = np.all((np.abs(targets) <= 1e-6) | (np.abs(targets - 1.0) <= 1e-6))
    if not bool(is_binary):
        raise NumericalInvariantError(
            f"{name} must contain binary targets in {{0, 1}}.",
            context={"name": name, "min": float(targets.min()), "max": float(targets.max())},
        )
    return (targets >= 0.5).astype(np.float64)


@dataclass(frozen=True)
class TemperatureScales:
    """Post-hoc logit temperature parameters ``(T_cal_edge, T_cal_delay, T_cal_link)``."""

    t_edge: float = 1.0
    t_delay: float = 1.0
    t_link: float = 1.0
    calibrated: bool = False

    def __post_init__(self) -> None:
        for field_name, val in (
            ("t_edge", self.t_edge),
            ("t_delay", self.t_delay),
            ("t_link", self.t_link),
        ):
            require_finite(val, f"TemperatureScales.{field_name}")
            if not (TEMP_MIN - 1e-7 <= float(val) <= TEMP_MAX + 1e-7):
                raise CheckpointIntegrityError(
                    f"TemperatureScales.{field_name}={val} outside valid bounds [{TEMP_MIN}, {TEMP_MAX}].",
                    context={"field": field_name, "value": val},
                )

    @property
    def T_cal_edge(self) -> float:
        return self.t_edge

    @property
    def T_cal_delay(self) -> float:
        return self.t_delay

    @property
    def T_cal_link(self) -> float:
        return self.t_link

    @classmethod
    def uncalibrated_unit(cls) -> TemperatureScales:
        """Explicit uncalibrated unit temperatures ``(1.0, 1.0, 1.0, calibrated=False)``."""
        return cls(t_edge=1.0, t_delay=1.0, t_link=1.0, calibrated=False)

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "t_edge": float(self.t_edge),
            "t_delay": float(self.t_delay),
            "t_link": float(self.t_link),
            "calibrated": bool(self.calibrated),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TemperatureScales:
        required = {"t_edge", "t_delay", "t_link", "calibrated"}
        missing = sorted(required - set(data.keys()))
        if missing:
            raise CheckpointIntegrityError(
                f"TemperatureScales metadata missing required keys: {missing}.",
                context={"missing": missing},
            )
        if not isinstance(data["calibrated"], bool):
            raise CheckpointIntegrityError("TemperatureScales.calibrated must be a boolean.")
        return cls(
            t_edge=float(data["t_edge"]),
            t_delay=float(data["t_delay"]),
            t_link=float(data["t_link"]),
            calibrated=bool(data["calibrated"]),
        )


def fit_binary_temperature(
    logits: np.ndarray | Tensor,
    targets: np.ndarray | Tensor,
    *,
    name: str = "head",
) -> float:
    """Fit scalar temperature ``T in [0.1, 10.0]`` minimizing validation BCE NLL via L-BFGS-B."""
    z = _to_1d_float64(logits, f"{name}.logits")
    y = _validate_binary_targets(_to_1d_float64(targets, f"{name}.targets"), f"{name}.targets")
    if z.shape != y.shape:
        raise NumericalInvariantError(
            f"Shape mismatch in fit_binary_temperature({name}): logits {z.shape} vs targets {y.shape}.",
            context={"name": name, "logits_shape": list(z.shape), "targets_shape": list(y.shape)},
        )

    def _nll_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        temp = float(params[0])
        scaled = z / temp
        # Stable BCE NLL: -y * log(sigmoid(s)) - (1 - y) * log(sigmoid(-s))
        loss_vec = -(y * log_expit(scaled) + (1.0 - y) * log_expit(-scaled))
        nll = float(np.mean(loss_vec))
        probs = expit(scaled)
        grad_t = float(-np.mean(z * (probs - y)) / (temp * temp))
        return nll, np.array([grad_t], dtype=np.float64)

    res = minimize(
        _nll_and_grad,
        x0=np.array([1.0], dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[(TEMP_MIN, TEMP_MAX)],
        options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 100},
    )
    fitted_t = float(np.clip(res.x[0], TEMP_MIN, TEMP_MAX))
    require_finite(fitted_t, f"fit_binary_temperature.{name}")
    return fitted_t


def calibrate_temperatures(
    *,
    edge_logits: np.ndarray | Tensor,
    edge_targets: np.ndarray | Tensor,
    delay_logits: np.ndarray | Tensor,
    delay_targets: np.ndarray | Tensor,
    link_logits: np.ndarray | Tensor,
    link_targets: np.ndarray | Tensor,
    model: nn.Module | None = None,
) -> TemperatureScales:
    """Fit ``(T_cal_edge, T_cal_delay, T_cal_link)`` on validation logits (Boundary Lock B)."""
    if model is not None and model.training:
        raise ConfigValidationError(
            "Boundary Lock B violation: calibrate_temperatures called while model.training is True. "
            "Switch to model.eval() before collecting validation logits and calibrating temperatures."
        )

    t_edge = fit_binary_temperature(edge_logits, edge_targets, name="edge")
    t_delay = fit_binary_temperature(delay_logits, delay_targets, name="delay")
    t_link = fit_binary_temperature(link_logits, link_targets, name="link")
    return TemperatureScales(t_edge=t_edge, t_delay=t_delay, t_link=t_link, calibrated=True)


# Alias for external callers
calibrate_temperature = calibrate_temperatures


def _compute_roc_and_pr(
    probs: np.ndarray,
    targets: np.ndarray,
    *,
    max_curve_points: int = 128,
) -> tuple[float | None, float | None, dict[str, list[float]], dict[str, list[float]]]:
    """Compute exact ROC-AUC, PR-AUC (average precision), and downsampled ROC/PR curve points."""
    n_total = int(targets.size)
    n_pos = int(np.sum(targets > 0.5))
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, None, {"fpr": [], "tpr": []}, {"recall": [], "precision": []}

    # Sort descending by predicted probability; break ties deterministically
    order = np.argsort(-probs, kind="mergesort")
    p_sorted = probs[order]
    y_sorted = targets[order]

    # Distinct score threshold indices (last index of each tied score group)
    distinct_mask = np.r_[np.diff(p_sorted) != 0.0, True]
    thresh_idx = np.nonzero(distinct_mask)[0]

    tps = np.cumsum(y_sorted)[thresh_idx]
    fps = (1.0 + thresh_idx) - tps

    tps_ext = np.r_[0.0, tps]
    fps_ext = np.r_[0.0, fps]
    fpr = fps_ext / float(n_neg)
    tpr = tps_ext / float(n_pos)
    roc_auc = float(np.trapezoid(tpr, fpr))

    precision = tps / np.maximum(tps + fps, 1e-12)
    recall = tps / float(n_pos)
    recall_ext = np.r_[0.0, recall]
    precision_ext = np.r_[1.0, precision]
    # Exact step-wise Average Precision: sum_k (R_k - R_{k-1}) * P_k
    pr_auc = float(np.sum((recall_ext[1:] - recall_ext[:-1]) * precision_ext[1:]))

    if fpr.size > max_curve_points:
        sel = np.unique(np.linspace(0, fpr.size - 1, max_curve_points, dtype=int))
        fpr_out = fpr[sel].tolist()
        tpr_out = tpr[sel].tolist()
    else:
        fpr_out = fpr.tolist()
        tpr_out = tpr.tolist()

    if recall_ext.size > max_curve_points:
        sel_pr = np.unique(np.linspace(0, recall_ext.size - 1, max_curve_points, dtype=int))
        rec_out = recall_ext[sel_pr].tolist()
        prec_out = precision_ext[sel_pr].tolist()
    else:
        rec_out = recall_ext.tolist()
        prec_out = precision_ext.tolist()

    return roc_auc, pr_auc, {"fpr": fpr_out, "tpr": tpr_out}, {"recall": rec_out, "precision": prec_out}


def compute_predictive_metrics(
    probs: np.ndarray | Tensor,
    targets: np.ndarray | Tensor,
    *,
    n_bins: int = 10,
    name: str = "predictions",
) -> dict[str, Any]:
    """Compute ROC-AUC, PR-AUC, Brier, Log-Loss, 10-bin ECE, MCE, and reliability diagram bins."""
    if n_bins < 2:
        raise ConfigValidationError(f"n_bins must be >= 2, got {n_bins}.")
    p = _to_1d_float64(probs, f"{name}.probs")
    require_probability_range(p, f"{name}.probs")
    p = np.clip(p, 0.0, 1.0)
    y = _validate_binary_targets(_to_1d_float64(targets, f"{name}.targets"), f"{name}.targets")

    if p.shape != y.shape:
        raise NumericalInvariantError(
            f"Shape mismatch in compute_predictive_metrics({name}): probs {p.shape} vs targets {y.shape}.",
            context={"name": name, "probs_shape": list(p.shape), "targets_shape": list(y.shape)},
        )

    n_samples = int(p.size)
    brier = float(np.mean((p - y) ** 2))
    p_safe = np.clip(p, 1e-7, 1.0 - 1e-7)
    log_loss = float(-np.mean(y * np.log(p_safe) + (1.0 - y) * np.log1p(-p_safe)))
    accuracy = float(np.mean((p >= 0.5).astype(np.float64) == y))
    pos_rate = float(np.mean(y))

    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    bin_ids = np.minimum(np.searchsorted(edges[1:-1], p, side="right"), n_bins - 1)

    ece = 0.0
    mce = 0.0
    reliability_bins: list[dict[str, float | int]] = []
    for b in range(n_bins):
        mask = bin_ids == b
        cnt = int(np.sum(mask))
        low = float(edges[b])
        high = float(edges[b + 1])
        if cnt > 0:
            mean_p = float(np.mean(p[mask]))
            emp_y = float(np.mean(y[mask]))
            gap = abs(mean_p - emp_y)
            ece += (float(cnt) / float(n_samples)) * gap
            mce = max(mce, gap)
        else:
            mean_p = 0.5 * (low + high)
            emp_y = 0.0
            gap = 0.0
        reliability_bins.append(
            {
                "bin_index": b,
                "bin_lower": low,
                "bin_upper": high,
                "count": cnt,
                "mean_pred": mean_p,
                "empirical_pos_rate": emp_y,
                "abs_gap": gap,
            }
        )

    roc_auc, pr_auc, roc_curve, pr_curve = _compute_roc_and_pr(p, y)

    return {
        "n_samples": n_samples,
        "positive_rate": pos_rate,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "log_loss": log_loss,
        "ece": float(ece),
        "mce": float(mce),
        "accuracy": accuracy,
        "reliability_bins": reliability_bins,
        "roc_curve": roc_curve,
        "pr_curve": pr_curve,
    }


__all__ = [
    "TEMP_MAX",
    "TEMP_MIN",
    "TemperatureScales",
    "calibrate_temperature",
    "calibrate_temperatures",
    "compute_predictive_metrics",
    "fit_binary_temperature",
]

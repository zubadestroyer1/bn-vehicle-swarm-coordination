"""Publication-grade paired bootstrap 95% CIs and Holm-Bonferroni Wilcoxon hypothesis testing.

Implements exact paired seed-level comparisons across scenarios, controllers, and surgical
ablations without normal-distribution small-sample approximations:
- 10,000-resample paired percentile bootstrap 95% confidence intervals on means and mean differences.
- Exact two-sided Wilcoxon signed-rank p-values (handling zero-difference ties deterministically).
- Step-down Holm-Bonferroni family-wise error rate adjustment across all baselines per scenario/metric.
- Explicit Win / Tie / Loss (``W/T/L``) seed counts respecting metric directionality.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
from scipy.stats import wilcoxon

from ..contracts import EpisodeMetrics
from ..logging_utils import ConfigValidationError, NumericalInvariantError, require_finite

# True => smaller value is better (e.g., delay, stops, spillback, CO2, braking, collisions).
# False => larger value is better (e.g., throughput, completion rate, V2X efficiency).
METRIC_LOWER_IS_BETTER: Final[dict[str, bool]] = {
    "cohort_mean_total_delay_s": True,
    "cohort_p95_total_delay_s": True,
    "cohort_mean_completed_delay_s": True,
    "cohort_completion_rate": False,
    "throughput_veh_per_hr": False,
    "mean_stops_per_veh": True,
    "spillback_corridor_seconds": True,
    "mean_co2_g_per_veh": True,
    "reroutes_per_veh": True,
    "collision_count": True,
    "teleport_count": True,
    "emergency_stop_count": True,
    "emergency_brake_count": True,
    "emergency_brake_rate_per_veh_km": True,
    "ttc_near_miss_count": True,
    "min_ttc_s": False,
    "safety_filter_clamp_count": True,
    "safety_filter_clamp_rate": True,
    "zero_speed_deadzone_count": True,
    "v2x_delivery_efficiency": False,
    "mean_dbn_update_latency_ms": True,
    "mean_dbn_inference_latency_ms": True,
    "mean_control_latency_ms": True,
    "edge_churn_hamming_rate": True,
}

DEFAULT_SUMMARY_METRICS: Final[tuple[str, ...]] = (
    "cohort_mean_total_delay_s",
    "cohort_p95_total_delay_s",
    "cohort_mean_completed_delay_s",
    "cohort_completion_rate",
    "throughput_veh_per_hr",
    "mean_stops_per_veh",
    "spillback_corridor_seconds",
    "mean_co2_g_per_veh",
    "emergency_brake_rate_per_veh_km",
    "ttc_near_miss_count",
    "collision_count",
    "teleport_count",
    "safety_filter_clamp_rate",
    "v2x_delivery_efficiency",
    "edge_churn_hamming_rate",
    "mean_dbn_update_latency_ms",
)


def _as_row_dict(item: EpisodeMetrics | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(item, EpisodeMetrics):
        return item.to_dict()
    if isinstance(item, Mapping):
        return dict(item)
    raise ConfigValidationError(f"Expected EpisodeMetrics or Mapping row, got {type(item).__name__}.")


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` via vectorized percentile bootstrap."""
    if n_resamples < 100:
        raise ConfigValidationError(f"n_resamples must be >= 100, got {n_resamples}.")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ConfigValidationError(f"alpha must lie in (0, 1), got {alpha}.")

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise NumericalInvariantError(f"bootstrap_mean_ci requires non-empty 1D array, got shape={arr.shape}.")
    require_finite(arr, "bootstrap_mean_ci.values")

    mean_val = float(np.mean(arr))
    n = arr.size
    if n == 1:
        return mean_val, mean_val, mean_val

    gen = rng if rng is not None else np.random.default_rng(202611)
    sample_idx = gen.integers(0, n, size=(n_resamples, n))
    boot_means = arr[sample_idx].mean(axis=1)
    low_q = 100.0 * (0.5 * alpha)
    high_q = 100.0 * (1.0 - 0.5 * alpha)
    ci_low, ci_high = np.percentile(boot_means, [low_q, high_q])
    return mean_val, float(ci_low), float(ci_high)


def exact_wilcoxon_two_sided(diffs: Sequence[float] | np.ndarray, *, zero_tol: float = 1e-12) -> tuple[float, float]:
    """Compute two-sided Wilcoxon signed-rank ``(statistic, p_value)`` cleanly handling all-zero ties."""
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise NumericalInvariantError(f"Wilcoxon requires non-empty 1D differences, got shape={arr.shape}.")
    require_finite(arr, "wilcoxon.diffs")

    nonzero = arr[np.abs(arr) > zero_tol]
    if nonzero.size == 0:
        return 0.0, 1.0
    if nonzero.size == 1:
        # A single non-zero paired observation cannot reject two-sided null at p < 1.0.
        return 1.0, 1.0

    res = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", mode="auto")
    stat = float(res.statistic)
    pval = float(res.pvalue)
    require_finite(pval, "wilcoxon.pvalue")
    return stat, min(1.0, max(0.0, pval))


def holm_bonferroni_adjust(raw_p_values: Sequence[float]) -> list[float]:
    """Apply step-down Holm-Bonferroni multiple-comparison adjustment to ``raw_p_values``."""
    m = len(raw_p_values)
    if m == 0:
        return []
    arr = np.asarray(raw_p_values, dtype=np.float64)
    require_finite(arr, "holm_bonferroni.raw_p_values")
    if np.any((arr < 0.0) | (arr > 1.0 + 1e-9)):
        raise NumericalInvariantError(
            "Raw p-values passed to holm_bonferroni_adjust must lie in [0, 1].",
            context={"p_values": arr.tolist()},
        )

    order = np.argsort(arr, kind="mergesort")
    adjusted_sorted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank_idx, orig_idx in enumerate(order):
        multiplier = m - rank_idx
        candidate = min(1.0, float(arr[orig_idx]) * multiplier)
        running_max = max(running_max, candidate)
        adjusted_sorted[rank_idx] = running_max

    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return [float(p) for p in adjusted]


def significance_stars(p_value: float) -> str:
    """Standard publication significance marker from adjusted p-value."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def compute_paired_statistics(
    episodes: Sequence[EpisodeMetrics | Mapping[str, Any]],
    *,
    reference_policy: str = "neural_dbn",
    metrics: Sequence[str] = DEFAULT_SUMMARY_METRICS,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    rng_seed: int = 202611,
) -> dict[str, Any]:
    """Compute per-scenario summaries, paired bootstrap 95% CIs, and Holm-Bonferroni Wilcoxon tests."""
    if not episodes:
        raise ConfigValidationError("compute_paired_statistics received zero episode rows.")
    if not metrics:
        raise ConfigValidationError("metrics list cannot be empty.")

    rows = [_as_row_dict(ep) for ep in episodes]
    rng = np.random.default_rng(rng_seed)

    # Group strictly by scenario_id -> policy_name -> seed -> row
    grouped: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for row in rows:
        for req_key in ("scenario_id", "policy_name", "seed"):
            if req_key not in row:
                raise ConfigValidationError(f"Episode row missing required key {req_key!r}.", context={"row": row})
        scen = str(row["scenario_id"])
        pol = str(row["policy_name"])
        seed = int(row["seed"])
        by_pol = grouped.setdefault(scen, {}).setdefault(pol, {})
        if seed in by_pol:
            raise ConfigValidationError(
                f"Duplicate episode row for scenario={scen!r}, policy={pol!r}, seed={seed}.",
                context={"scenario_id": scen, "policy_name": pol, "seed": seed},
            )
        by_pol[seed] = row

    scenarios_out: dict[str, Any] = {}
    for scen_id in sorted(grouped):
        pol_map = grouped[scen_id]
        policy_summaries: dict[str, Any] = {}

        for pol_name in sorted(pol_map):
            seed_map = pol_map[pol_name]
            ordered_seeds = sorted(seed_map)
            metric_stats: dict[str, Any] = {}
            for m_name in metrics:
                vals = np.asarray([float(seed_map[s][m_name]) for s in ordered_seeds], dtype=np.float64)
                require_finite(vals, f"{scen_id}.{pol_name}.{m_name}")
                mean_v, ci_lo, ci_hi = bootstrap_mean_ci(vals, n_resamples=n_resamples, alpha=alpha, rng=rng)
                std_v = float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0
                q25, med_v, q75, p95 = np.percentile(vals, [25.0, 50.0, 75.0, 95.0])
                metric_stats[m_name] = {
                    "n_seeds": int(vals.size),
                    "seeds": ordered_seeds,
                    "mean": mean_v,
                    "std": std_v,
                    "median": float(med_v),
                    "iqr": float(q75 - q25),
                    "p95": float(p95),
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                }
            policy_summaries[pol_name] = metric_stats

        paired_vs_baselines: dict[str, dict[str, Any]] = {}
        if reference_policy in pol_map:
            ref_seeds_map = pol_map[reference_policy]
            ref_seeds = sorted(ref_seeds_map)
            baseline_names = [p for p in sorted(pol_map) if p != reference_policy]

            # Verify 1-to-1 paired seeds across all compared policies in this scenario
            for base_pol in baseline_names:
                base_seeds = sorted(pol_map[base_pol])
                if base_seeds != ref_seeds:
                    raise ConfigValidationError(
                        f"Unpaired seed mismatch in scenario {scen_id!r}: {reference_policy}={ref_seeds} "
                        f"vs {base_pol}={base_seeds}.",
                        context={
                            "scenario_id": scen_id,
                            "reference_policy": reference_policy,
                            "baseline_policy": base_pol,
                            "ref_seeds": ref_seeds,
                            "base_seeds": base_seeds,
                        },
                    )
                paired_vs_baselines[base_pol] = {}

            for m_name in metrics:
                lower_better = METRIC_LOWER_IS_BETTER.get(m_name, True)
                ref_vals = np.asarray([float(ref_seeds_map[s][m_name]) for s in ref_seeds], dtype=np.float64)
                raw_pvals: list[float] = []
                temp_entries: list[tuple[str, dict[str, Any]]] = []

                for base_pol in baseline_names:
                    base_vals = np.asarray(
                        [float(pol_map[base_pol][s][m_name]) for s in ref_seeds], dtype=np.float64
                    )
                    diffs = ref_vals - base_vals  # reference - baseline
                    mean_d, d_lo, d_hi = bootstrap_mean_ci(diffs, n_resamples=n_resamples, alpha=alpha, rng=rng)
                    std_d = float(np.std(diffs, ddof=1)) if diffs.size >= 2 else 0.0
                    med_d = float(np.median(diffs))
                    w_stat, p_raw = exact_wilcoxon_two_sided(diffs)

                    tie_tol = 1e-9
                    ties = int(np.sum(np.abs(diffs) <= tie_tol))
                    if lower_better:
                        wins = int(np.sum(diffs < -tie_tol))
                        losses = int(np.sum(diffs > tie_tol))
                    else:
                        wins = int(np.sum(diffs > tie_tol))
                        losses = int(np.sum(diffs < -tie_tol))

                    base_mean = float(np.mean(base_vals))
                    rel_pct = float(100.0 * mean_d / abs(base_mean)) if abs(base_mean) > 1e-9 else None

                    entry = {
                        "reference_policy": reference_policy,
                        "baseline_policy": base_pol,
                        "lower_is_better": lower_better,
                        "n_paired_seeds": len(ref_seeds),
                        "mean_diff": mean_d,
                        "std_diff": std_d,
                        "median_diff": med_d,
                        "relative_change_pct": rel_pct,
                        "ci95_low": d_lo,
                        "ci95_high": d_hi,
                        "wilcoxon_statistic": w_stat,
                        "p_value_raw": p_raw,
                        "wins": wins,
                        "ties": ties,
                        "losses": losses,
                        "wtl_summary": f"{wins}/{ties}/{losses}",
                    }
                    raw_pvals.append(p_raw)
                    temp_entries.append((base_pol, entry))

                adj_pvals = holm_bonferroni_adjust(raw_pvals)
                for (base_pol, entry), p_adj in zip(temp_entries, adj_pvals, strict=True):
                    entry["p_value_holm"] = p_adj
                    entry["significant_holm_005"] = bool(p_adj < alpha)
                    entry["significance_stars"] = significance_stars(p_adj)
                    paired_vs_baselines[base_pol][m_name] = entry

        scenarios_out[scen_id] = {
            "policies": policy_summaries,
            "paired_comparisons": paired_vs_baselines,
        }

    return {
        "reference_policy": reference_policy,
        "bootstrap_resamples": n_resamples,
        "alpha": alpha,
        "metrics": list(metrics),
        "scenarios": scenarios_out,
    }


__all__ = [
    "DEFAULT_SUMMARY_METRICS",
    "METRIC_LOWER_IS_BETTER",
    "bootstrap_mean_ci",
    "compute_paired_statistics",
    "exact_wilcoxon_two_sided",
    "holm_bonferroni_adjust",
    "significance_stars",
]

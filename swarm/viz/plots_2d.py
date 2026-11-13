"""Publication-grade 300-DPI 2D scientific figures for City-Scale Neural DBN evaluation.

Generates five camera-ready figures using Matplotlib's thread/process-safe object-oriented
``Figure`` + ``FigureCanvasAgg`` backend with ``MPLCONFIGDIR`` isolated inside
``tempfile.gettempdir()`` (never polluting ``Path.cwd()``):

1. ``plot_scenario_comparison``: Multi-scenario cohort total delay & corridor spillback comparison
   with 10,000-resample 95% bootstrap error bars across In-Distribution, Zero-Shot 4x4 Incident OOD,
   and Zero-Shot Real-World Downtown Palo Alto OOD regimes.
2. ``plot_paired_forest``: Paired seed-level difference (``Neural DBN - Baseline``) forest plot with
   95% bootstrap CIs, Holm-Bonferroni Wilcoxon significance stars, and ``W/T/L`` seed counts.
3. ``plot_ablation_breakdown``: Controlled architectural ablation breakdown (``Full Neural DBN`` vs
   ``No Temporal Prior``, ``Static CPD``, ``Fixed Route``, ``Independent Belief``) across delay,
   spillback, and temporal graph edge-churn stability.
4. ``plot_calibration_and_roc_pr``: Held-out In-Distribution vs Zero-Shot OOD ROC curves,
   Precision-Recall curves, and 10-bin ECE reliability diagrams (pre- vs post-temperature scaling).
5. ``plot_latency_scaling``: Log-log CPU structure-update + inference latency scaling across
   ``N in {16, 32, 64, 128, 256}`` with median curves, IQR bands, p95 envelopes, and empirical
   complexity exponents.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Final

# Isolate Matplotlib cache/fontlist strictly inside system temp directory BEFORE importing matplotlib.
_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "swarm_mplconfig"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from ..contracts import ABLATION_POLICY_NAMES, EpisodeMetrics, MAIN_POLICY_NAMES  # noqa: E402
from ..logging_utils import (  # noqa: E402
    ConfigValidationError,
    NumericalInvariantError,
    get_logger,
    require_finite,
)

# Colorblind-safe publication palette (Okabe-Ito / Paul Tol high-contrast scientific scheme)
POLICY_COLORS: Final[dict[str, str]] = {
    "fixed_signal": "#7f8c8d",
    "actuated_signal": "#5c6bc0",
    "fcfs_reservation": "#e69f00",
    "independent_belief": "#cc79a7",
    "heuristic_radius_dbn": "#56b4e9",
    "online_bic_dbn": "#009e73",
    "neural_dbn": "#0b3c5d",
    "no_temporal_prior": "#d55e00",
    "static_cpd": "#f0ad4e",
    "fixed_route": "#4daf4a",
    "unconstrained_bic": "#b22222",
    "dense_bic_reference": "#b22222",
}

POLICY_LABELS: Final[dict[str, str]] = {
    "fixed_signal": "Fixed-Time TLS",
    "actuated_signal": "Actuated TLS",
    "fcfs_reservation": "AIM FCFS Reservation",
    "independent_belief": "Independent Belief",
    "heuristic_radius_dbn": "Heuristic-Radius DBN",
    "online_bic_dbn": "Online Pairwise BIC DBN",
    "neural_dbn": "Neural DBN Swarm (Ours)",
    "no_temporal_prior": "w/o Temporal Edge Prior",
    "static_cpd": "w/o Amortized Causal CPDs",
    "fixed_route": "w/o Dynamic Edge Routing",
    "unconstrained_bic": "Unconstrained O(N^2) BIC",
    "dense_bic_reference": "Unconstrained O(N^2) BIC",
}

SCENARIO_ORDER: Final[tuple[str, ...]] = (
    "grid3x3_nominal",
    "grid3x3_rush_asymmetric",
    "grid3x3_noisy_dropout",
    "grid4x4_incident",
    "osm_palo_alto_rush",
)

SCENARIO_LABELS: Final[dict[str, str]] = {
    "grid3x3_nominal": "3x3 Nominal\n[ID]",
    "grid3x3_rush_asymmetric": "3x3 Rush Wave\n[ID Spillback]",
    "grid3x3_noisy_dropout": "3x3 Sensor/V2X Drop\n[ID Stress]",
    "grid4x4_incident": "4x4 Grid + Closure\n[Zero-Shot OOD]",
    "osm_palo_alto_rush": "Downtown Palo Alto\n[Real-World OOD]",
}

ABLATION_ORDER: Final[tuple[str, ...]] = (
    "neural_dbn",
    "no_temporal_prior",
    "static_cpd",
    "fixed_route",
    "independent_belief",
)


def _style_axes(ax: Any, *, grid_axis: str = "y") -> None:
    """Apply crisp academic journal spine, tick, and subtle grid styling."""
    ax.set_facecolor("#fbfcfe")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(colors="#1e293b", labelsize=8.5)
    if grid_axis in ("x", "y", "both"):
        ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.55, color="#cbd5e1", alpha=0.75, zorder=0)


def _save_figure_atomic(fig: Figure, output_path: Path | str, *, dpi: int = 300) -> Path:
    """Render ``fig`` via ``FigureCanvasAgg`` and atomically write to ``output_path``."""
    if isinstance(dpi, bool) or dpi < 72:
        raise ConfigValidationError(f"dpi must be an integer >= 72, got {dpi!r}.")
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_p.suffix.lower().lstrip(".") or "png"
    if suffix not in ("png", "pdf", "svg"):
        raise ConfigValidationError(
            f"Unsupported figure extension {out_p.suffix!r} for {out_p}; expected .png, .pdf, or .svg."
        )
    canvas = FigureCanvasAgg(fig)
    tmp_p = out_p.parent / f".{out_p.stem}.pid{os.getpid()}.tmp.{suffix}"
    canvas.print_figure(str(tmp_p), dpi=dpi, format=suffix, bbox_inches="tight", facecolor="white")
    os.replace(tmp_p, out_p)
    return out_p


def _ordered_scenarios(keys: Sequence[str]) -> list[str]:
    present = set(keys)
    ordered = [s for s in SCENARIO_ORDER if s in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def _scenario_label(scen_id: str) -> str:
    if scen_id in SCENARIO_LABELS:
        return SCENARIO_LABELS[scen_id]
    clean = scen_id.replace("_", " ").title()
    if "ood" in scen_id.lower() or "4x4" in scen_id.lower() or "palo_alto" in scen_id.lower():
        return f"{clean}\n[Zero-Shot OOD]"
    return clean


def _policy_label(pol_name: str) -> str:
    return POLICY_LABELS.get(pol_name, pol_name.replace("_", " ").title())


def _policy_color(pol_name: str, fallback_idx: int = 0) -> str:
    if pol_name in POLICY_COLORS:
        return POLICY_COLORS[pol_name]
    cycle = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf")
    return cycle[fallback_idx % len(cycle)]


def _bootstrap_ci_1d(vals: np.ndarray, *, n_resamples: int = 10000, seed: int = 202611) -> tuple[float, float, float]:
    require_finite(vals, "bootstrap_ci_1d.vals")
    mean_v = float(np.mean(vals))
    if vals.size <= 1:
        return mean_v, mean_v, mean_v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_resamples, vals.size))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return mean_v, float(lo), float(hi)


def _holm_stars(p_values: Sequence[float]) -> list[tuple[float, str]]:
    m = len(p_values)
    if m == 0:
        return []
    arr = np.asarray(p_values, dtype=np.float64)
    require_finite(arr, "holm_stars.p_values")
    order = np.argsort(arr, kind="mergesort")
    adj_sorted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, orig in enumerate(order):
        running = max(running, min(1.0, float(arr[orig]) * (m - rank)))
        adj_sorted[rank] = running
    adj = np.empty(m, dtype=np.float64)
    adj[order] = adj_sorted
    out: list[tuple[float, str]] = []
    for p in adj:
        pf = float(p)
        stars = "***" if pf < 0.001 else ("**" if pf < 0.01 else ("*" if pf < 0.05 else "ns"))
        out.append((pf, stars))
    return out


def _summarize_from_episode_rows(episodes: Sequence[EpisodeMetrics | Mapping[str, Any]]) -> dict[str, Any]:
    """Build canonical ``scenarios`` summary mapping directly from raw episode rows if needed."""
    if not episodes:
        raise ConfigValidationError("Episode list in eval_report is empty.")
    grouped: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for item in episodes:
        row = item.to_dict() if isinstance(item, EpisodeMetrics) else dict(item)
        for k in ("scenario_id", "policy_name", "seed"):
            if k not in row:
                raise ConfigValidationError(f"Episode row missing required key {k!r}.")
        grouped.setdefault(str(row["scenario_id"]), {}).setdefault(str(row["policy_name"]), {})[int(row["seed"])] = row

    metrics_to_extract = (
        "cohort_mean_total_delay_s",
        "spillback_corridor_seconds",
        "edge_churn_hamming_rate",
        "throughput_veh_per_hr",
    )
    scenarios_out: dict[str, Any] = {}
    for scen_id, pol_map in sorted(grouped.items()):
        policies_summary: dict[str, Any] = {}
        for pol_name, seed_map in sorted(pol_map.items()):
            seeds = sorted(seed_map)
            m_dict: dict[str, Any] = {}
            for m_key in metrics_to_extract:
                if all(m_key in seed_map[s] for s in seeds):
                    arr = np.asarray([float(seed_map[s][m_key]) for s in seeds], dtype=np.float64)
                    mean_v, lo, hi = _bootstrap_ci_1d(arr)
                    m_dict[m_key] = {"mean": mean_v, "ci95_low": lo, "ci95_high": hi, "n_seeds": len(seeds)}
            policies_summary[pol_name] = m_dict

        paired_out: dict[str, dict[str, Any]] = {}
        if "neural_dbn" in pol_map:
            ref_map = pol_map["neural_dbn"]
            ref_seeds = sorted(ref_map)
            baselines = [p for p in sorted(pol_map) if p != "neural_dbn" and sorted(pol_map[p]) == ref_seeds]
            for b_pol in baselines:
                paired_out[b_pol] = {}
            for m_key in metrics_to_extract:
                if not all(m_key in ref_map[s] for s in ref_seeds):
                    continue
                ref_arr = np.asarray([float(ref_map[s][m_key]) for s in ref_seeds], dtype=np.float64)
                raw_ps: list[float] = []
                entries: list[tuple[str, dict[str, Any]]] = []
                lower_better = m_key != "throughput_veh_per_hr"
                for b_pol in baselines:
                    if not all(m_key in pol_map[b_pol][s] for s in ref_seeds):
                        continue
                    b_arr = np.asarray([float(pol_map[b_pol][s][m_key]) for s in ref_seeds], dtype=np.float64)
                    diffs = ref_arr - b_arr
                    md, dlo, dhi = _bootstrap_ci_1d(diffs)
                    nz = diffs[np.abs(diffs) > 1e-12]
                    if nz.size >= 2:
                        pval = float(wilcoxon(nz, zero_method="wilcox", alternative="two-sided", mode="auto").pvalue)
                    else:
                        pval = 1.0
                    ties = int(np.sum(np.abs(diffs) <= 1e-9))
                    wins = int(np.sum(diffs < -1e-9)) if lower_better else int(np.sum(diffs > 1e-9))
                    losses = int(np.sum(diffs > 1e-9)) if lower_better else int(np.sum(diffs < -1e-9))
                    entry = {
                        "mean_diff": md,
                        "ci95_low": dlo,
                        "ci95_high": dhi,
                        "lower_is_better": lower_better,
                        "p_value_raw": pval,
                        "wtl_summary": f"{wins}/{ties}/{losses}",
                    }
                    raw_ps.append(pval)
                    entries.append((b_pol, entry))
                for (b_pol, entry), (p_holm, stars) in zip(entries, _holm_stars(raw_ps), strict=True):
                    entry["p_value_holm"] = p_holm
                    entry["significance_stars"] = stars
                    paired_out[b_pol][m_key] = entry

        scenarios_out[scen_id] = {"policies": policies_summary, "paired_comparisons": paired_out}
    return scenarios_out


def _resolve_scenarios(eval_report: Mapping[str, Any]) -> dict[str, Any]:
    """Extract and validate canonical ``scenarios`` mapping from ``eval_report``."""
    if not isinstance(eval_report, Mapping):
        raise ConfigValidationError(f"eval_report must be a Mapping, got {type(eval_report).__name__}.")
    if "scenarios" in eval_report and isinstance(eval_report["scenarios"], Mapping):
        scenarios = dict(eval_report["scenarios"])
    elif (
        "statistics" in eval_report
        and isinstance(eval_report["statistics"], Mapping)
        and isinstance(eval_report["statistics"].get("scenarios"), Mapping)
    ):
        scenarios = dict(eval_report["statistics"]["scenarios"])
    elif "episodes" in eval_report and isinstance(eval_report["episodes"], Sequence):
        scenarios = _summarize_from_episode_rows(eval_report["episodes"])
    else:
        raise ConfigValidationError(
            "eval_report must contain 'scenarios', 'statistics.scenarios', or 'episodes'."
        )
    if not scenarios:
        raise ConfigValidationError("eval_report contains zero scenarios.")
    return scenarios


def _extract_mean_ci(stat_obj: Mapping[str, Any], label: str) -> tuple[float, float, float]:
    """Extract validated finite ``(mean, ci_low, ci_high)`` from a policy metric dictionary."""
    if not isinstance(stat_obj, Mapping) or "mean" not in stat_obj:
        raise ConfigValidationError(f"Missing 'mean' in metric stat entry for {label}.")
    mean_v = float(stat_obj["mean"])
    lo_v = float(stat_obj.get("ci95_low", stat_obj.get("ci_low", mean_v)))
    hi_v = float(stat_obj.get("ci95_high", stat_obj.get("ci_high", mean_v)))
    require_finite(mean_v, f"{label}.mean")
    require_finite(lo_v, f"{label}.ci_low")
    require_finite(hi_v, f"{label}.ci_high")
    if lo_v > hi_v + 1e-7:
        raise NumericalInvariantError(
            f"Inverted confidence interval [{lo_v}, {hi_v}] for {label}.",
            context={"label": label, "ci_low": lo_v, "ci_high": hi_v},
        )
    return mean_v, min(lo_v, mean_v), max(hi_v, mean_v)


# =============================================================================
# 1. Multi-Scenario Delay & Spillback Comparison (ID + Zero-Shot OOD)
# =============================================================================


def plot_scenario_comparison(
    eval_report: Mapping[str, Any],
    output_path: Path | str,
    *,
    dpi: int = 300,
) -> Path:
    """Plot multi-scenario cohort delay and corridor spillback with 95% bootstrap error bars."""
    scenarios = _resolve_scenarios(eval_report)
    scen_ids = _ordered_scenarios(list(scenarios.keys()))

    # Collect union of policies across scenarios; prioritize canonical 7 main policies when present
    all_policies_set: set[str] = set()
    for sid in scen_ids:
        pols = scenarios[sid].get("policies", scenarios[sid])
        if isinstance(pols, Mapping):
            all_policies_set.update(str(k) for k in pols.keys())

    main_present = [p for p in MAIN_POLICY_NAMES if p in all_policies_set]
    if main_present:
        ordered_policies = main_present
    else:
        ordered_policies = sorted(all_policies_set)
    if not ordered_policies:
        raise ConfigValidationError("No policies found across scenarios in plot_scenario_comparison.")

    fig = Figure(figsize=(13.2, 6.4), dpi=dpi)
    axes = fig.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.18, top=0.86, bottom=0.14, left=0.08, right=0.98)

    panel_specs = (
        ("cohort_mean_total_delay_s", "Cohort Mean Total Delay [s]\n(incl. unfinished & undeparted)", "(a) Multi-Scenario Travel Delay"),
        ("spillback_corridor_seconds", "Queue Spillback Exposure\n[corridor · s]", "(b) Upstream Corridor Spillback Cascades"),
    )

    n_scen = len(scen_ids)
    n_pol = len(ordered_policies)
    group_width = min(0.84, 0.13 * n_pol)
    bar_w = group_width / max(1, n_pol)
    x_centers = np.arange(n_scen, dtype=np.float64)

    for ax_idx, (metric_key, ylabel, panel_title) in enumerate(panel_specs):
        ax = axes[ax_idx]
        _style_axes(ax, grid_axis="y")

        # Highlight Zero-Shot OOD regime columns with a soft tint band
        for s_idx, sid in enumerate(scen_ids):
            if "4x4" in sid or "palo_alto" in sid or "ood" in sid.lower():
                ax.axvspan(s_idx - 0.46, s_idx + 0.46, color="#eef2ff", alpha=0.65, zorder=0)

        for p_idx, pol in enumerate(ordered_policies):
            offset = (p_idx - 0.5 * (n_pol - 1)) * bar_w
            xs: list[float] = []
            means: list[float] = []
            err_lo: list[float] = []
            err_hi: list[float] = []

            for s_idx, sid in enumerate(scen_ids):
                pol_map = scenarios[sid].get("policies", scenarios[sid])
                if not isinstance(pol_map, Mapping) or pol not in pol_map:
                    continue
                p_entry = pol_map[pol]
                if not isinstance(p_entry, Mapping) or metric_key not in p_entry:
                    raise ConfigValidationError(
                        f"Policy {pol!r} in scenario {sid!r} missing required metric {metric_key!r}."
                    )
                m_val, lo_val, hi_val = _extract_mean_ci(p_entry[metric_key], f"{sid}.{pol}.{metric_key}")
                xs.append(float(x_centers[s_idx] + offset))
                means.append(m_val)
                err_lo.append(max(0.0, m_val - lo_val))
                err_hi.append(max(0.0, hi_val - m_val))

            if not xs:
                continue
            is_ours = pol == "neural_dbn"
            color = _policy_color(pol, p_idx)
            ax.bar(
                xs,
                means,
                width=bar_w * 0.90,
                color=color,
                edgecolor="#091e3a" if is_ours else "#ffffff",
                linewidth=1.15 if is_ours else 0.45,
                alpha=0.96 if is_ours else 0.86,
                label=_policy_label(pol) if ax_idx == 0 else None,
                zorder=3 if is_ours else 2,
            )
            ax.errorbar(
                xs,
                means,
                yerr=[err_lo, err_hi],
                fmt="none",
                ecolor="#0f172a",
                elinewidth=0.95,
                capsize=2.4,
                capthick=0.85,
                zorder=4,
            )

        ax.set_ylabel(ylabel, fontsize=9.0, fontweight="semibold", color="#0f172a")
        ax.set_title(panel_title, loc="left", fontsize=10.0, fontweight="bold", color="#0f172a", pad=5)
        ax.set_ylim(bottom=0.0)

    axes[-1].set_xticks(x_centers)
    axes[-1].set_xticklabels([_scenario_label(sid) for sid in scen_ids], fontsize=8.8, fontweight="medium")
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.33),
        ncol=min(4, n_pol),
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        fontsize=8.3,
    )
    fig.suptitle(
        "City-Scale Autonomous Swarm Coordination: In-Distribution & Zero-Shot OOD Performance (95% Bootstrap CIs)",
        fontsize=11.5,
        fontweight="bold",
        color="#0f172a",
        y=0.97,
    )
    return _save_figure_atomic(fig, output_path, dpi=dpi)


# =============================================================================
# 2. Paired Difference Forest Plot (10,000-Resample 95% CIs + Holm Wilcoxon)
# =============================================================================


def plot_paired_forest(
    eval_report: Mapping[str, Any],
    output_path: Path | str,
    *,
    metric: str = "cohort_mean_total_delay_s",
    dpi: int = 300,
) -> Path:
    """Plot paired difference (Neural DBN - Baseline) forest plot with 95% CIs & Wilcoxon stars."""
    scenarios = _resolve_scenarios(eval_report)
    scen_ids = _ordered_scenarios(list(scenarios.keys()))

    rows: list[dict[str, Any]] = []
    for sid in scen_ids:
        paired = scenarios[sid].get("paired_comparisons", scenarios[sid].get("paired_vs_baselines", {}))
        if not isinstance(paired, Mapping):
            continue
        # Order main baselines first, then ablations
        ordered_b = [p for p in MAIN_POLICY_NAMES if p in paired and p != "neural_dbn"]
        ordered_b += [p for p in ABLATION_POLICY_NAMES if p in paired and p not in ordered_b]
        ordered_b += [p for p in sorted(paired.keys()) if p not in ordered_b and p != "neural_dbn"]

        for b_pol in ordered_b:
            b_metrics = paired[b_pol]
            if not isinstance(b_metrics, Mapping) or metric not in b_metrics:
                continue
            entry = b_metrics[metric]
            mean_d = float(entry["mean_diff"])
            lo_d = float(entry.get("ci95_low", entry.get("ci_low", mean_d)))
            hi_d = float(entry.get("ci95_high", entry.get("ci_high", mean_d)))
            require_finite(mean_d, f"forest.{sid}.{b_pol}.mean_diff")
            require_finite(lo_d, f"forest.{sid}.{b_pol}.ci_low")
            require_finite(hi_d, f"forest.{sid}.{b_pol}.ci_high")
            if lo_d > hi_d + 1e-7:
                raise NumericalInvariantError(
                    f"Inverted paired CI [{lo_d}, {hi_d}] on {sid} vs {b_pol}.",
                    context={"scenario_id": sid, "baseline": b_pol},
                )
            lower_better = bool(entry.get("lower_is_better", True))
            stars = str(entry.get("significance_stars", "ns"))
            wtl = str(entry.get("wtl_summary", f"{entry.get('wins', 0)}/{entry.get('ties', 0)}/{entry.get('losses', 0)}"))
            rows.append(
                {
                    "scenario_id": sid,
                    "baseline": b_pol,
                    "mean_diff": mean_d,
                    "ci_low": min(lo_d, mean_d),
                    "ci_high": max(hi_d, mean_d),
                    "lower_better": lower_better,
                    "stars": stars,
                    "wtl": wtl,
                }
            )

    if not rows:
        raise ConfigValidationError(
            f"No paired_comparisons found for metric {metric!r} in plot_paired_forest."
        )

    n_rows = len(rows)
    fig_h = max(4.4, 1.35 + 0.36 * n_rows)
    fig = Figure(figsize=(11.8, fig_h), dpi=dpi)
    ax = fig.subplots(1, 1)
    fig.subplots_adjust(left=0.28, right=0.69, top=0.88, bottom=0.13)
    _style_axes(ax, grid_axis="x")

    y_positions = np.arange(n_rows, dtype=np.float64)[::-1]
    y_labels: list[str] = []

    min_x = min(r["ci_low"] for r in rows)
    max_x = max(r["ci_high"] for r in rows)
    span_x = max(1.0, max_x - min_x, abs(min_x), abs(max_x))

    prev_scen = None
    band_toggle = False
    for idx, r in enumerate(rows):
        y = float(y_positions[idx])
        if r["scenario_id"] != prev_scen:
            if prev_scen is not None:
                ax.axhline(y + 0.5, color="#cbd5e1", linestyle="-", linewidth=0.75, zorder=1)
            prev_scen = r["scenario_id"]
            band_toggle = not band_toggle
        if band_toggle:
            ax.axhspan(y - 0.5, y + 0.5, color="#f1f5f9", alpha=0.55, zorder=0)

        scen_short = r["scenario_id"].replace("grid", "").replace("osm_", "").replace("_", " ")
        y_labels.append(f"{scen_short} | vs {_policy_label(r['baseline'])}")

        favors_ours = (r["ci_high"] < 0.0) if r["lower_better"] else (r["ci_low"] > 0.0)
        favors_base = (r["ci_low"] > 0.0) if r["lower_better"] else (r["ci_high"] < 0.0)
        pt_color = "#00796b" if favors_ours else ("#c62828" if favors_base else "#475569")

        ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=pt_color, linewidth=2.0, zorder=3)
        ax.plot([r["ci_low"], r["ci_low"]], [y - 0.14, y + 0.14], color=pt_color, linewidth=1.3, zorder=3)
        ax.plot([r["ci_high"], r["ci_high"]], [y - 0.14, y + 0.14], color=pt_color, linewidth=1.3, zorder=3)
        ax.scatter([r["mean_diff"]], [y], color=pt_color, edgecolor="#ffffff", linewidth=0.7, s=46, zorder=4)

        annot = f"{r['mean_diff']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] {r['stars']}  (W/T/L {r['wtl']})"
        ax.text(
            1.02,
            y,
            annot,
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=8.0,
            fontfamily="monospace",
            color="#0f172a",
        )

    ax.axvline(0.0, color="#0f172a", linestyle="--", linewidth=1.1, zorder=2)
    pad = 0.14 * span_x
    ax.set_xlim(min(0.0, min_x) - pad, max(0.0, max_x) + pad)
    ax.set_ylim(-0.65, n_rows - 0.35)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8.2)

    unit_lbl = "s" if metric.endswith("_s") else ""
    dir_note = "Negative Favors Neural DBN" if rows[0]["lower_better"] else "Positive Favors Neural DBN"
    ax.set_xlabel(
        f"Paired Difference Δ (Neural DBN − Baseline) in {metric} [{unit_lbl}]  ({dir_note})",
        fontsize=9.0,
        fontweight="semibold",
        color="#0f172a",
    )
    ax.set_title(
        "Paired Seed-Level Forest Plot: 10,000-Resample 95% Bootstrap CIs & Holm-Bonferroni Wilcoxon Tests",
        loc="left",
        fontsize=10.3,
        fontweight="bold",
        color="#0f172a",
        pad=10,
    )
    return _save_figure_atomic(fig, output_path, dpi=dpi)


# =============================================================================
# 3. Surgical Architectural Ablation Breakdown
# =============================================================================


def plot_ablation_breakdown(
    eval_report: Mapping[str, Any],
    output_path: Path | str,
    *,
    dpi: int = 300,
) -> Path:
    """Plot surgical ablation breakdown across delay, spillback, and temporal edge churn."""
    scenarios = _resolve_scenarios(eval_report)
    scen_ids = _ordered_scenarios(list(scenarios.keys()))

    present_ablations: list[str] = []
    for cand in ABLATION_ORDER:
        if any(cand in scenarios[sid].get("policies", scenarios[sid]) for sid in scen_ids):
            present_ablations.append(cand)

    if "neural_dbn" not in present_ablations or len(present_ablations) < 2:
        raise ConfigValidationError(
            "plot_ablation_breakdown requires 'neural_dbn' and at least one ablation comparator "
            f"({ABLATION_ORDER[1:]}), got {present_ablations}."
        )

    # Filter scenarios to those where neural_dbn and >=1 ablation comparator were evaluated
    ablation_scens = [
        sid
        for sid in scen_ids
        if "neural_dbn" in scenarios[sid].get("policies", scenarios[sid])
        and any(p in scenarios[sid].get("policies", scenarios[sid]) for p in present_ablations if p != "neural_dbn")
    ]
    if not ablation_scens:
        raise ConfigValidationError("No scenario in eval_report contains paired ablation policies.")

    # Select third panel metric dynamically if edge_churn_hamming_rate is absent
    sample_pol = scenarios[ablation_scens[0]].get("policies", scenarios[ablation_scens[0]])["neural_dbn"]
    third_metric = (
        "edge_churn_hamming_rate"
        if "edge_churn_hamming_rate" in sample_pol
        else ("throughput_veh_per_hr" if "throughput_veh_per_hr" in sample_pol else "cohort_mean_total_delay_s")
    )
    third_label = (
        "Edge Churn Hamming Rate"
        if third_metric == "edge_churn_hamming_rate"
        else ("Throughput [veh/hr]" if third_metric == "throughput_veh_per_hr" else "Cohort Mean Delay [s]")
    )

    panels = (
        ("cohort_mean_total_delay_s", "Cohort Mean Total Delay [s]", "(a) Travel Delay Impact"),
        ("spillback_corridor_seconds", "Spillback Exposure [corridor · s]", "(b) Spillback Gating Impact"),
        (third_metric, third_label, "(c) Structural Temporal Stability"),
    )

    fig = Figure(figsize=(14.0, 4.9), dpi=dpi)
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(wspace=0.25, top=0.80, bottom=0.20, left=0.06, right=0.98)

    n_scen = len(ablation_scens)
    n_pol = len(present_ablations)
    group_w = min(0.82, 0.16 * n_pol)
    bar_w = group_w / max(1, n_pol)
    x_pos = np.arange(n_scen, dtype=np.float64)

    for ax_idx, (m_key, y_lbl, p_title) in enumerate(panels):
        ax = axes[ax_idx]
        _style_axes(ax, grid_axis="y")
        for p_idx, pol in enumerate(present_ablations):
            offset = (p_idx - 0.5 * (n_pol - 1)) * bar_w
            xs: list[float] = []
            means: list[float] = []
            elo: list[float] = []
            ehi: list[float] = []
            for s_idx, sid in enumerate(ablation_scens):
                pol_map = scenarios[sid].get("policies", scenarios[sid])
                if pol not in pol_map or m_key not in pol_map[pol]:
                    continue
                mv, lv, hv = _extract_mean_ci(pol_map[pol][m_key], f"ablation.{sid}.{pol}.{m_key}")
                xs.append(float(x_pos[s_idx] + offset))
                means.append(mv)
                elo.append(max(0.0, mv - lv))
                ehi.append(max(0.0, hv - mv))
            if not xs:
                continue
            is_full = pol == "neural_dbn"
            ax.bar(
                xs,
                means,
                width=bar_w * 0.88,
                color=_policy_color(pol, p_idx),
                edgecolor="#091e3a" if is_full else "#ffffff",
                linewidth=1.1 if is_full else 0.45,
                label=_policy_label(pol) if ax_idx == 0 else None,
                zorder=3 if is_full else 2,
            )
            ax.errorbar(
                xs,
                means,
                yerr=[elo, ehi],
                fmt="none",
                ecolor="#0f172a",
                elinewidth=0.9,
                capsize=2.2,
                zorder=4,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [sid.replace("grid", "").replace("osm_", "").replace("_", "\n") for sid in ablation_scens],
            fontsize=8.0,
        )
        ax.set_ylabel(y_lbl, fontsize=8.8, fontweight="semibold")
        ax.set_title(p_title, loc="left", fontsize=9.6, fontweight="bold")
        ax.set_ylim(bottom=0.0)

    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(1.72, 1.26),
        ncol=min(5, n_pol),
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        fontsize=8.3,
    )
    fig.suptitle(
        "Controlled Neural DBN Component Ablation Study (Paired 95% Bootstrap CIs)",
        fontsize=11.2,
        fontweight="bold",
        color="#0f172a",
        y=0.97,
    )
    return _save_figure_atomic(fig, output_path, dpi=dpi)


# =============================================================================
# 4. Held-Out ID vs Zero-Shot OOD ROC, PR, & 10-Bin ECE Calibration
# =============================================================================


def _compute_roc_from_raw(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-y_prob, kind="mergesort")
    yt = y_true[order]
    pos = float(np.sum(yt > 0.5))
    neg = float(yt.size - pos)
    if pos <= 0.0 or neg <= 0.0:
        raise NumericalInvariantError("ROC computation requires at least one positive and one negative sample.")
    tps = np.concatenate(([0.0], np.cumsum(yt > 0.5)))
    fps = np.concatenate(([0.0], np.cumsum(yt <= 0.5)))
    tpr = tps / pos
    fpr = fps / neg
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def _compute_pr_from_raw(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-y_prob, kind="mergesort")
    yt = y_true[order]
    pos = float(np.sum(yt > 0.5))
    if pos <= 0.0:
        raise NumericalInvariantError("PR computation requires at least one positive sample.")
    tps = np.cumsum(yt > 0.5).astype(np.float64)
    ranks = np.arange(1, yt.size + 1, dtype=np.float64)
    precision = np.concatenate(([1.0], tps / ranks))
    recall = np.concatenate(([0.0], tps / pos))
    pr_auc = float(np.trapezoid(precision, recall))
    return recall, precision, pr_auc


def _compute_ece_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, float]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    confs: list[float] = []
    accs: list[float] = []
    ece = 0.0
    n_tot = float(max(1, y_true.size))
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (y_prob >= lo) & (y_prob <= hi if b == n_bins - 1 else y_prob < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            c_mean = float(np.mean(y_prob[mask]))
            a_mean = float(np.mean(y_true[mask]))
            confs.append(c_mean)
            accs.append(a_mean)
            ece += (cnt / n_tot) * abs(a_mean - c_mean)
    return np.asarray(confs, dtype=np.float64), np.asarray(accs, dtype=np.float64), float(ece)


def _extract_split_entries(predictive_report: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    if not isinstance(predictive_report, Mapping) or not predictive_report:
        raise ConfigValidationError("predictive_report must be a non-empty Mapping.")

    container = predictive_report.get("splits", predictive_report)
    if not isinstance(container, Mapping):
        raise ConfigValidationError("predictive_report.splits must be a Mapping.")

    # Direct single-split dictionary check
    metric_keys = {"roc_auc", "fpr", "y_true", "ece_post", "ece", "edge"}
    if any(k in container for k in metric_keys):
        if "edge" in container and isinstance(container["edge"], Mapping):
            return [("Held-Out Evaluation (Edge)", container["edge"])]
        return [("Held-Out Evaluation", container)]

    entries: list[tuple[str, Mapping[str, Any]]] = []
    for split_name, val in container.items():
        if not isinstance(val, Mapping):
            continue
        if "edge" in val and isinstance(val["edge"], Mapping):
            entries.append((str(split_name), val["edge"]))
        elif any(k in val for k in ("roc_auc", "fpr", "y_true", "ece_post", "ece")):
            entries.append((str(split_name), val))
    if not entries:
        raise ConfigValidationError(
            "predictive_report contains no recognizable split entries with ROC/PR/ECE metrics."
        )
    return entries


def plot_calibration_and_roc_pr(
    predictive_report: Mapping[str, Any],
    output_path: Path | str,
    *,
    dpi: int = 300,
) -> Path:
    """Plot Held-Out ID vs Zero-Shot OOD ROC curves, PR curves, and 10-bin ECE calibration."""
    splits = _extract_split_entries(predictive_report)
    split_colors = ("#0b3c5d", "#009e73", "#d55e00", "#5c6bc0", "#cc79a7")

    fig = Figure(figsize=(14.2, 4.5), dpi=dpi)
    ax_roc, ax_pr, ax_cal = fig.subplots(1, 3)
    fig.subplots_adjust(wspace=0.26, top=0.83, bottom=0.16, left=0.06, right=0.98)
    for ax in (ax_roc, ax_pr, ax_cal):
        _style_axes(ax, grid_axis="both")

    has_curve_data = False
    scalar_summary: list[tuple[str, float, float, float, float]] = []
    cal_plotted = False

    for idx, (s_name, s_data) in enumerate(splits):
        color = split_colors[idx % len(split_colors)]
        nice_name = s_name.replace("_", " ").title()

        fpr_arr: np.ndarray | None = None
        tpr_arr: np.ndarray | None = None
        rec_arr: np.ndarray | None = None
        prec_arr: np.ndarray | None = None
        roc_auc: float | None = None
        pr_auc: float | None = None

        if "y_true" in s_data and ("y_prob" in s_data or "probs" in s_data):
            yt = np.asarray(s_data["y_true"], dtype=np.float64)
            yp = np.asarray(s_data.get("y_prob", s_data.get("probs")), dtype=np.float64)
            require_finite(yt, f"{s_name}.y_true")
            require_finite(yp, f"{s_name}.y_prob")
            fpr_arr, tpr_arr, roc_auc = _compute_roc_from_raw(yt, yp)
            rec_arr, prec_arr, pr_auc = _compute_pr_from_raw(yt, yp)
            if not cal_plotted:
                if "y_prob_uncalibrated" in s_data or "pre_cal_probs" in s_data:
                    yp_pre = np.asarray(
                        s_data.get("y_prob_uncalibrated", s_data.get("pre_cal_probs")), dtype=np.float64
                    )
                    require_finite(yp_pre, f"{s_name}.y_prob_uncalibrated")
                    c_pre, a_pre, ece_pre = _compute_ece_bins(yt, yp_pre)
                    ax_cal.plot(
                        c_pre,
                        a_pre,
                        "--s",
                        color="#d55e00",
                        linewidth=1.6,
                        markersize=4.5,
                        label=f"Pre-T_cal (ECE={ece_pre:.3f})",
                    )
                c_post, a_post, ece_post = _compute_ece_bins(yt, yp)
                ax_cal.plot(
                    c_post,
                    a_post,
                    "-o",
                    color="#0b3c5d",
                    linewidth=2.0,
                    markersize=5.0,
                    label=f"Post-T_cal (ECE={ece_post:.3f})",
                )
                cal_plotted = True
        else:
            roc_sub = s_data.get("roc_curve", s_data)
            pr_sub = s_data.get("pr_curve", s_data)
            if isinstance(roc_sub, Mapping) and "fpr" in roc_sub and "tpr" in roc_sub:
                fpr_arr = np.asarray(roc_sub["fpr"], dtype=np.float64)
                tpr_arr = np.asarray(roc_sub["tpr"], dtype=np.float64)
                require_finite(fpr_arr, f"{s_name}.fpr")
                require_finite(tpr_arr, f"{s_name}.tpr")
                roc_auc = float(s_data.get("roc_auc", np.trapezoid(tpr_arr, fpr_arr)))
            if isinstance(pr_sub, Mapping) and "recall" in pr_sub and "precision" in pr_sub:
                rec_arr = np.asarray(pr_sub["recall"], dtype=np.float64)
                prec_arr = np.asarray(pr_sub["precision"], dtype=np.float64)
                require_finite(rec_arr, f"{s_name}.recall")
                require_finite(prec_arr, f"{s_name}.precision")
                pr_auc = float(s_data.get("pr_auc", np.trapezoid(prec_arr, rec_arr)))

            if not cal_plotted:
                rel = s_data.get("reliability_bins", s_data)
                if isinstance(rel, Mapping) and "bin_confidences" in rel and "bin_accuracies" in rel:
                    bc = np.asarray(rel["bin_confidences"], dtype=np.float64)
                    ba = np.asarray(rel["bin_accuracies"], dtype=np.float64)
                    require_finite(bc, f"{s_name}.bin_confidences")
                    require_finite(ba, f"{s_name}.bin_accuracies")
                    if "pre_bin_confidences" in rel and "pre_bin_accuracies" in rel:
                        pbc = np.asarray(rel["pre_bin_confidences"], dtype=np.float64)
                        pba = np.asarray(rel["pre_bin_accuracies"], dtype=np.float64)
                        require_finite(pbc, f"{s_name}.pre_bin_confidences")
                        require_finite(pba, f"{s_name}.pre_bin_accuracies")
                        ece_pre_v = float(s_data.get("ece_pre", s_data.get("ece_uncalibrated", 0.0)))
                        require_finite(ece_pre_v, f"{s_name}.ece_pre")
                        ax_cal.plot(
                            pbc,
                            pba,
                            "--s",
                            color="#d55e00",
                            linewidth=1.6,
                            markersize=4.5,
                            label=f"Pre-T_cal (ECE={ece_pre_v:.3f})",
                        )
                    ece_post_v = float(s_data.get("ece_post", s_data.get("ece", 0.0)))
                    require_finite(ece_post_v, f"{s_name}.ece_post")
                    ax_cal.plot(
                        bc,
                        ba,
                        "-o",
                        color="#0b3c5d",
                        linewidth=2.0,
                        markersize=5.0,
                        label=f"Post-T_cal (ECE={ece_post_v:.3f})",
                    )
                    cal_plotted = True

        if fpr_arr is not None and tpr_arr is not None and roc_auc is not None:
            require_finite(roc_auc, f"{s_name}.roc_auc")
            has_curve_data = True
            ax_roc.plot(fpr_arr, tpr_arr, color=color, linewidth=2.0, label=f"{nice_name} (AUC={roc_auc:.3f})")

        if rec_arr is not None and prec_arr is not None and pr_auc is not None:
            require_finite(pr_auc, f"{s_name}.pr_auc")
            has_curve_data = True
            ax_pr.plot(rec_arr, prec_arr, color=color, linewidth=2.0, label=f"{nice_name} (PR-AUC={pr_auc:.3f})")

        if "roc_auc" in s_data and "pr_auc" in s_data:
            r_s = float(s_data["roc_auc"])
            p_s = float(s_data["pr_auc"])
            e_pre = float(s_data.get("ece_pre", s_data.get("ece_uncalibrated", s_data.get("ece", 0.0))))
            e_post = float(s_data.get("ece_post", s_data.get("ece_calibrated", s_data.get("ece", 0.0))))
            for nm, vv in (("roc_auc", r_s), ("pr_auc", p_s), ("ece_pre", e_pre), ("ece_post", e_post)):
                require_finite(vv, f"{s_name}.{nm}")
                if vv < -1e-6 or vv > 1.0 + 1e-6:
                    raise NumericalInvariantError(f"{s_name}.{nm}={vv} lies outside [0, 1].")
            scalar_summary.append((nice_name, r_s, p_s, e_pre, e_post))

    if has_curve_data:
        ax_roc.plot([0.0, 1.0], [0.0, 1.0], "--", color="#94a3b8", linewidth=1.1, label="Random Chance")
        ax_roc.set_xlim(-0.02, 1.02)
        ax_roc.set_ylim(-0.02, 1.03)
        ax_roc.set_xlabel("False Positive Rate", fontsize=8.8, fontweight="semibold")
        ax_roc.set_ylabel("True Positive Rate", fontsize=8.8, fontweight="semibold")
        ax_roc.set_title("(a) ROC Curves (ID & Zero-Shot OOD)", loc="left", fontsize=9.6, fontweight="bold")
        ax_roc.legend(loc="lower right", fontsize=7.8, frameon=True)

        ax_pr.set_xlim(-0.02, 1.02)
        ax_pr.set_ylim(0.0, 1.04)
        ax_pr.set_xlabel("Recall", fontsize=8.8, fontweight="semibold")
        ax_pr.set_ylabel("Precision", fontsize=8.8, fontweight="semibold")
        ax_pr.set_title("(b) Precision-Recall Curves", loc="left", fontsize=9.6, fontweight="bold")
        ax_pr.legend(loc="lower left", fontsize=7.8, frameon=True)
    elif scalar_summary:
        xs = np.arange(len(scalar_summary), dtype=np.float64)
        names = [t[0] for t in scalar_summary]
        ax_roc.bar(xs, [t[1] for t in scalar_summary], width=0.52, color="#0b3c5d")
        ax_roc.set_xticks(xs)
        ax_roc.set_xticklabels(names, fontsize=8.0)
        ax_roc.set_ylim(0.0, 1.05)
        ax_roc.set_ylabel("ROC-AUC", fontsize=8.8, fontweight="semibold")
        ax_roc.set_title("(a) Held-Out ROC-AUC", loc="left", fontsize=9.6, fontweight="bold")

        ax_pr.bar(xs, [t[2] for t in scalar_summary], width=0.52, color="#009e73")
        ax_pr.set_xticks(xs)
        ax_pr.set_xticklabels(names, fontsize=8.0)
        ax_pr.set_ylim(0.0, 1.05)
        ax_pr.set_ylabel("PR-AUC", fontsize=8.8, fontweight="semibold")
        ax_pr.set_title("(b) Held-Out PR-AUC", loc="left", fontsize=9.6, fontweight="bold")
    else:
        raise ConfigValidationError(
            "predictive_report lacked valid ROC/PR curve arrays, raw y_true/y_prob, and scalar roc_auc/pr_auc."
        )

    if cal_plotted:
        ax_cal.plot([0.0, 1.0], [0.0, 1.0], "--", color="#64748b", linewidth=1.1, label="Perfect Calibration")
        ax_cal.set_xlim(-0.02, 1.02)
        ax_cal.set_ylim(-0.02, 1.03)
        ax_cal.set_xlabel("Mean Predicted Probability", fontsize=8.8, fontweight="semibold")
        ax_cal.set_ylabel("Empirical Positive Frequency", fontsize=8.8, fontweight="semibold")
        ax_cal.set_title("(c) 10-Bin Reliability (Pre vs Post T_cal)", loc="left", fontsize=9.6, fontweight="bold")
        ax_cal.legend(loc="upper left", fontsize=7.8, frameon=True)
    elif scalar_summary:
        xs = np.arange(len(scalar_summary), dtype=np.float64)
        names = [t[0] for t in scalar_summary]
        ax_cal.bar(xs - 0.17, [t[3] for t in scalar_summary], width=0.32, color="#d55e00", label="Pre-T_cal ECE")
        ax_cal.bar(xs + 0.17, [t[4] for t in scalar_summary], width=0.32, color="#0b3c5d", label="Post-T_cal ECE")
        ax_cal.set_xticks(xs)
        ax_cal.set_xticklabels(names, fontsize=8.0)
        ax_cal.set_ylabel("10-Bin ECE (Lower is Better)", fontsize=8.8, fontweight="semibold")
        ax_cal.set_title("(c) Temperature Scaling Calibration", loc="left", fontsize=9.6, fontweight="bold")
        ax_cal.legend(loc="upper right", fontsize=7.8, frameon=True)
    else:
        raise ConfigValidationError("predictive_report lacked reliability bin arrays and scalar ECE metrics.")

    fig.suptitle(
        "Causal Edge & Delay Predictive Discrimination and Post-Hoc Temperature Calibration",
        fontsize=11.2,
        fontweight="bold",
        color="#0f172a",
        y=0.97,
    )
    return _save_figure_atomic(fig, output_path, dpi=dpi)


# =============================================================================
# 5. Log-Log CPU Structure-Update Latency Scaling (N = 16 ... 256)
# =============================================================================


def _normalize_scaling_series(scaling_report: Mapping[str, Any]) -> dict[str, list[dict[str, float]]]:
    """Normalize scaling benchmark payloads into ``method -> sorted list of {n, median, q25, q75, p95}``."""
    if not isinstance(scaling_report, Mapping) or not scaling_report:
        raise ConfigValidationError("scaling_report must be a non-empty Mapping.")

    series_out: dict[str, list[dict[str, float]]] = {}

    if "records" in scaling_report or "rows" in scaling_report:
        raw_rows = scaling_report.get("records", scaling_report.get("rows"))
        if not isinstance(raw_rows, Sequence) or not raw_rows:
            raise ConfigValidationError("scaling_report records/rows must be a non-empty list.")
        for item in raw_rows:
            if not isinstance(item, Mapping):
                raise ConfigValidationError("Each scaling record must be a Mapping.")
            m_name = str(item.get("policy_name", item.get("method")))
            n_val = int(item.get("n_vehicles", item.get("swarm_size", item.get("n", 0))))
            med = float(item.get("median_ms", item.get("median", 0.0)))
            iqr = float(item.get("iqr_ms", item.get("iqr", 0.0)))
            q25 = float(item.get("q25_ms", item.get("p25_ms", max(1e-4, med - 0.5 * iqr))))
            q75 = float(item.get("q75_ms", item.get("p75_ms", med + 0.5 * iqr)))
            p95 = float(item.get("p95_ms", item.get("p95", q75)))
            series_out.setdefault(m_name, []).append(
                {"n": float(n_val), "median": med, "q25": q25, "q75": q75, "p95": p95}
            )
    else:
        sizes_raw = scaling_report.get("swarm_sizes", scaling_report.get("n_vehicles", scaling_report.get("sizes")))
        methods_raw = (
            scaling_report.get("methods")
            or scaling_report.get("policies")
            or scaling_report.get("series")
            or scaling_report.get("results")
        )
        if not isinstance(methods_raw, Mapping) or not methods_raw:
            raise ConfigValidationError(
                "scaling_report must contain 'methods'/'policies'/'results' Mapping or 'records' list."
            )
        for m_name, m_payload in methods_raw.items():
            pts: list[dict[str, float]] = []
            if isinstance(m_payload, Sequence) and not isinstance(m_payload, (str, bytes)):
                if not isinstance(sizes_raw, Sequence) and not all(
                    isinstance(x, Mapping) and ("n" in x or "n_vehicles" in x or "swarm_size" in x) for x in m_payload
                ):
                    raise ConfigValidationError(
                        f"scaling_report method {m_name!r} is a list without matching 'swarm_sizes'."
                    )
                for idx, entry in enumerate(m_payload):
                    if isinstance(entry, Mapping):
                        n_v = float(
                            entry.get(
                                "n_vehicles",
                                entry.get("swarm_size", entry.get("n", sizes_raw[idx] if sizes_raw else 0)),
                            )
                        )
                        med = float(entry.get("median_ms", entry.get("median")))
                        iqr = float(entry.get("iqr_ms", entry.get("iqr", 0.0)))
                        q25 = float(entry.get("q25_ms", entry.get("p25_ms", max(1e-4, med - 0.5 * iqr))))
                        q75 = float(entry.get("q75_ms", entry.get("p75_ms", med + 0.5 * iqr)))
                        p95 = float(entry.get("p95_ms", entry.get("p95", q75)))
                    else:
                        n_v = float(sizes_raw[idx])  # type: ignore[index]
                        med = float(entry)
                        q25, q75, p95 = med, med, med
                    pts.append({"n": n_v, "median": med, "q25": q25, "q75": q75, "p95": p95})
            elif isinstance(m_payload, Mapping):
                for n_key, entry in m_payload.items():
                    n_v = float(int(n_key))
                    if isinstance(entry, Mapping):
                        med = float(entry.get("median_ms", entry.get("median")))
                        iqr = float(entry.get("iqr_ms", entry.get("iqr", 0.0)))
                        q25 = float(entry.get("q25_ms", entry.get("p25_ms", max(1e-4, med - 0.5 * iqr))))
                        q75 = float(entry.get("q75_ms", entry.get("p75_ms", med + 0.5 * iqr)))
                        p95 = float(entry.get("p95_ms", entry.get("p95", q75)))
                    else:
                        med = float(entry)
                        q25, q75, p95 = med, med, med
                    pts.append({"n": n_v, "median": med, "q25": q25, "q75": q75, "p95": p95})
            if pts:
                series_out[str(m_name)] = pts

    if not series_out:
        raise ConfigValidationError("No valid latency series extracted from scaling_report.")

    for m_name, pts in series_out.items():
        pts.sort(key=lambda d: d["n"])
        for pt in pts:
            for field_name, val in pt.items():
                require_finite(val, f"scaling.{m_name}.{field_name}")
            if pt["n"] <= 0.0 or pt["median"] <= 0.0:
                raise NumericalInvariantError(
                    f"Scaling values must be strictly positive for log-log plot in {m_name}: {pt}."
                )
    return series_out


def plot_latency_scaling(
    scaling_report: Mapping[str, Any],
    output_path: Path | str,
    *,
    dpi: int = 300,
) -> Path:
    """Plot log-log CPU structure-update + inference latency scaling across N = 16 ... 256."""
    series_map = _normalize_scaling_series(scaling_report)

    fig = Figure(figsize=(7.6, 5.2), dpi=dpi)
    ax = fig.subplots(1, 1)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.14)
    _style_axes(ax, grid_axis="both")

    all_ns: set[int] = set()
    markers = ("o", "s", "^", "D", "v")
    preferred = ("neural_dbn", "online_bic_dbn", "heuristic_radius_dbn", "unconstrained_bic", "dense_bic_reference")
    ordered_methods = [m for m in preferred if m in series_map] + [
        m for m in sorted(series_map) if m not in preferred
    ]

    for idx, m_name in enumerate(ordered_methods):
        pts = series_map[m_name]
        ns = np.asarray([p["n"] for p in pts], dtype=np.float64)
        meds = np.asarray([p["median"] for p in pts], dtype=np.float64)
        q25s = np.asarray([max(1e-4, p["q25"]) for p in pts], dtype=np.float64)
        q75s = np.asarray([max(p["median"], p["q75"]) for p in pts], dtype=np.float64)
        p95s = np.asarray([max(p["median"], p["p95"]) for p in pts], dtype=np.float64)
        all_ns.update(int(round(x)) for x in ns)

        slope_str = ""
        if ns.size >= 2:
            slope, _ = np.polyfit(np.log(ns), np.log(meds), 1)
            slope_str = f" (~O(N^{slope:.2f}))"

        color = _policy_color(m_name, idx)
        mk = markers[idx % len(markers)]
        is_ours = m_name == "neural_dbn"

        ax.fill_between(ns, q25s, q75s, color=color, alpha=0.18, zorder=2)
        ax.plot(
            ns,
            meds,
            marker=mk,
            color=color,
            linewidth=2.3 if is_ours else 1.8,
            markersize=6.0 if is_ours else 5.0,
            label=f"{_policy_label(m_name)}{slope_str}",
            zorder=4 if is_ours else 3,
        )
        if np.any(p95s > meds * 1.02):
            ax.plot(ns, p95s, linestyle=":", color=color, linewidth=1.1, alpha=0.85, zorder=3)

    # Real-time 10% step budget reference line (50 ms for dt = 500 ms)
    ax.axhline(50.0, color="#dc2626", linestyle="--", linewidth=1.1, zorder=2, label="Real-Time Budget (50 ms @ 2 Hz)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    sorted_ticks = sorted(all_ns)
    ax.set_xticks(sorted_ticks)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Active Swarm Size N [vehicles]", fontsize=9.2, fontweight="semibold")
    ax.set_ylabel("CPU DBN Structure + Inference Latency [ms]", fontsize=9.2, fontweight="semibold")
    ax.set_title(
        "Log-Log Online DBN Update & Inference Latency Scaling (Median, IQR Band, Dotted p95)",
        loc="left",
        fontsize=9.9,
        fontweight="bold",
        pad=8,
    )
    ax.legend(loc="upper left", fontsize=8.0, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
    return _save_figure_atomic(fig, output_path, dpi=dpi)


# =============================================================================
# 6. Unified Publication Suite Orchestrator
# =============================================================================


def _load_mapping_if_json_path(obj: Mapping[str, Any] | Path | str | None, name: str) -> Mapping[str, Any] | None:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj
    if isinstance(obj, (Path, str)):
        p = Path(obj)
        if not p.is_file():
            raise ConfigValidationError(f"{name} file does not exist: {p}")
        try:
            loaded = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            get_logger("viz.plots_2d").error("Malformed JSON in %s (%s): %s", name, p, exc)
            raise ConfigValidationError(f"Malformed JSON in {name} ({p}): {exc}") from exc
        if not isinstance(loaded, Mapping):
            raise ConfigValidationError(f"{name} JSON must decode to an object, got {type(loaded).__name__}.")
        return loaded
    raise ConfigValidationError(f"Unsupported type {type(obj).__name__} for {name}.")


def render_publication_plots(
    eval_report: Mapping[str, Any] | Path | str | None = None,
    output_dir: Path | str = Path("outputs/plots"),
    *,
    ablation_report: Mapping[str, Any] | Path | str | None = None,
    predictive_report: Mapping[str, Any] | Path | str | None = None,
    training_report: Mapping[str, Any] | Path | str | None = None,
    scaling_report: Mapping[str, Any] | Path | str | None = None,
    dpi: int = 300,
) -> list[Path]:
    """Render all applicable 300-DPI publication figures into ``output_dir`` and return written paths."""
    eval_map = _load_mapping_if_json_path(eval_report, "eval_report")
    ablation_map = _load_mapping_if_json_path(ablation_report, "ablation_report")
    pred_map = _load_mapping_if_json_path(predictive_report, "predictive_report")
    train_map = _load_mapping_if_json_path(training_report, "training_report")
    scale_map = _load_mapping_if_json_path(scaling_report, "scaling_report")

    # Unpack bundled top-level experiment dictionary if caller passed a composite report
    if eval_map is not None:
        if pred_map is None:
            sub_pred = eval_map.get("predictive_report", eval_map.get("predictive_calibration"))
            if isinstance(sub_pred, Mapping):
                pred_map = sub_pred
        if scale_map is None:
            sub_scale = eval_map.get("scaling_report", eval_map.get("scaling_benchmark"))
            if isinstance(sub_scale, Mapping):
                scale_map = sub_scale
        if ablation_map is None:
            sub_abl = eval_map.get("ablation_report")
            if isinstance(sub_abl, Mapping):
                ablation_map = sub_abl
        if "eval_report" in eval_map and isinstance(eval_map["eval_report"], Mapping):
            eval_map = eval_map["eval_report"]

    if pred_map is None and train_map is not None:
        sub_p = train_map.get("predictive_metrics", train_map.get("calibration", train_map))
        if isinstance(sub_p, Mapping):
            pred_map = sub_p

    if all(x is None for x in (eval_map, ablation_map, pred_map, scale_map)):
        raise ConfigValidationError(
            "render_publication_plots requires at least one of eval_report, ablation_report, "
            "predictive_report, training_report, or scaling_report."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    if eval_map is not None:
        generated.append(plot_scenario_comparison(eval_map, out_dir / "fig1_scenario_delay_spillback.png", dpi=dpi))
        scenarios_check = _resolve_scenarios(eval_map)
        has_paired = any(
            bool(s.get("paired_comparisons") or s.get("paired_vs_baselines")) for s in scenarios_check.values()
        )
        if has_paired:
            generated.append(plot_paired_forest(eval_map, out_dir / "fig2_paired_forest_ci.png", dpi=dpi))

        if ablation_map is None:
            # Auto-detect whether eval_map also includes surgical ablation policies alongside neural_dbn
            all_pols: set[str] = set()
            for s in scenarios_check.values():
                pmap = s.get("policies", s)
                if isinstance(pmap, Mapping):
                    all_pols.update(str(k) for k in pmap.keys())
            ablation_comparators = {"no_temporal_prior", "static_cpd", "fixed_route", "independent_belief"}
            if "neural_dbn" in all_pols and bool(all_pols & ablation_comparators):
                ablation_map = eval_map

    if ablation_map is not None:
        generated.append(plot_ablation_breakdown(ablation_map, out_dir / "fig3_ablation_breakdown.png", dpi=dpi))

    if pred_map is not None:
        generated.append(plot_calibration_and_roc_pr(pred_map, out_dir / "fig4_calibration_roc_pr.png", dpi=dpi))

    if scale_map is not None:
        generated.append(plot_latency_scaling(scale_map, out_dir / "fig5_latency_scaling_loglog.png", dpi=dpi))

    return generated


__all__ = [
    "POLICY_COLORS",
    "POLICY_LABELS",
    "SCENARIO_LABELS",
    "plot_ablation_breakdown",
    "plot_calibration_and_roc_pr",
    "plot_latency_scaling",
    "plot_paired_forest",
    "plot_scenario_comparison",
    "render_publication_plots",
]

"""Headless scientific plots and a standalone interactive trajectory replay."""

import json
import os
from pathlib import Path


def pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd()/".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


COLORS = {"signal": "#8b99ac", "fcfs": "#e2a14c", "independent": "#aa85c6",
          "greedy": "#488bc9", "neural": "#21a390"}


def comparison_plot(report, path):
    plt = pyplot()
    scenarios = list(report["configs"])
    fig, axes = plt.subplots(len(scenarios), 3, figsize=(13, 3.5*len(scenarios)), squeeze=False)
    fields = [("mean_accrued_delay_s", "Accrued delay / arrival (s)"),
              ("throughput_veh_per_hour", "Throughput (vehicles / hour)"),
              ("stops_per_arrival", "Stops / arrival")]
    for row, scenario in enumerate(scenarios):
        groups = [r for r in report["summary"] if r["scenario"] == scenario]
        names = [r["policy"] for r in groups]
        for col, (metric, label) in enumerate(fields):
            ax = axes[row, col]
            ax.bar(names, [r[metric]["mean"] for r in groups],
                   yerr=[r[metric]["ci95_half_width"] or 0 for r in groups],
                   capsize=3, color=[COLORS[n] for n in names])
            ax.set_title(f"{scenario} / {label}", loc="left", fontsize=10)
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
    fig.suptitle("Intersection coordination: paired demand and noise seeds", fontsize=16, x=.04, ha="left")
    fig.text(.04, .015, "Error bars: normal-approximation 95% intervals across seeds. All policies use the same privileged safety shield.", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def training_plot(report, predictions_path, path):
    import numpy as np
    plt = pyplot()
    with np.load(predictions_path) as d:
        y, p = d["target"], d["probability"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    history = report["history"]
    axes[0].plot([h["epoch"] for h in history], [h["log_loss"] for h in history], color=COLORS["neural"])
    axes[0].set(xlabel="Epoch", ylabel="Validation log loss", title="Learning from simulated interactions")
    bins = np.linspace(0, 1, 11)
    means, fractions = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.any():
            means.append(p[mask].mean())
            fractions.append(y[mask].mean())
    axes[1].plot([0, 1], [0, 1], "--", color="#aaaaaa")
    axes[1].plot(means, fractions, "o-", color=COLORS["neural"])
    axes[1].set(xlabel="Predicted interaction probability", ylabel="Observed interaction frequency",
                title="Held-out synthetic scenes", xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def scaling_plot(report, path):
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in ("neural", "greedy"):
        rows = [r for r in report["rows"] if r["method"] == method]
        ax.plot([r["vehicles"] for r in rows], [r["median_ms"] for r in rows], "o-", label=method, color=COLORS[method])
    ax.set(xlabel="Vehicles", ylabel="Median structure update (ms)", yscale="log",
           title="Structure-update latency on synthetic snapshots")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.text(.1, .01, "Training and inference excluded; both methods score pairwise candidates.", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_replay(trace, path):
    template = Path(__file__).with_name("replay.html").read_text()
    payload = json.dumps(trace, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    Path(path).write_text(template.replace("__TRACE_DATA__", payload))


def trajectory_preview(trace, path):
    """Render an inspectable static frame with standard plotting tools."""
    plt = pyplot()
    from matplotlib.patches import Rectangle
    from matplotlib.collections import LineCollection
    populated = [frame for frame in trace["frames"] if frame["vehicles"]]
    frame = populated[len(populated)//2] if populated else trace["frames"][0]
    cfg = trace["config"]
    extent = cfg["approach_length"]+12
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0c151b")
    ax.set_facecolor("#101f27")
    ax.add_patch(Rectangle((-9, -extent), 18, 2*extent, color="#293e47"))
    ax.add_patch(Rectangle((-extent, -9), 2*extent, 18, color="#293e47"))
    for sign in (-1, 1):
        ax.plot([0, 0], [sign*12, sign*extent], "--", color="#77929c", lw=1)
        ax.plot([sign*12, sign*extent], [0, 0], "--", color="#77929c", lw=1)
    vehicles = {v["id"]: v for v in frame["vehicles"]}
    lines, colors = [], []
    for edge in frame["edges"]:
        a, b = vehicles.get(edge["source"]), vehicles.get(edge["target"])
        if a and b:
            lines.append([(a["x"], a["y"]), (b["x"], b["y"])])
            colors.append((.49, .87, .83, .15+.5*edge["probability"]))
    ax.add_collection(LineCollection(lines, colors=colors, linewidths=1))
    palette = ["#67d7bb", "#e6b86d", "#8ca8f0", "#dc94bf"]
    for v in vehicles.values():
        ax.scatter(v["x"], v["y"], s=60, c=palette[v["lane"]], zorder=3,
                   edgecolors="white" if v["reserved"] else "none", linewidths=1.5)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"SWARM / {trace['policy'].upper()}\n{frame['time']:.1f} s   |   {len(vehicles)} vehicles   |   {frame['completed']} trips completed",
                 color="#e9f3f5", loc="left", pad=20, fontsize=13)
    fig.text(.12, .035, "Lines: inferred dependencies   /   White rings: crossing reservations\nSynthetic intersection with a shared true-state safety shield", color="#9db1bb", fontsize=9)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

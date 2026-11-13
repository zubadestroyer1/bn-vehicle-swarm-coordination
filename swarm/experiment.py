"""Reproducible episodes, paired-policy experiments, and structure-update timing."""

from dataclasses import replace
import csv
import json
import math
from pathlib import Path
from time import perf_counter
import numpy as np
from .belief import BeliefTracker
from .bayes import GreedyStructure, NeuralStructure
from .policies import Controller
from .simulator import Intersection
from .training import sample_scene


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def run_episode(config, policy, seed=0, model=None, record=False):
    env = Intersection(config, seed)
    tracker = BeliefTracker(config)
    controller = Controller(policy, config, model, seed)
    commanded_accelerations, frames = {}, []
    start = perf_counter()
    for tick in range(math.ceil(config.duration/config.dt)):
        if tick:
            tracker.predict(commanded_accelerations)
        for registration in env.prepare():
            tracker.register(*registration)
        tracker.update(env.observe(), env.time)
        action = controller.act(tracker, env.time)
        # The filter knows commands, not the simulator's realized acceleration noise
        # or privileged safety-projected velocity.
        commanded_accelerations = {
            i: float(np.clip((action.target_speeds.get(i, 0)-t.speed)/config.dt,
                             -config.braking, config.acceleration)) for i, t in tracker.tracks.items()
        }
        if record and tick % 2 == 0:
            frames.append(env.frame(action, controller.network))
        departed, _ = env.step(action)
        for vehicle_id in departed:
            tracker.remove(vehicle_id)
    if record:
        frames.append(env.frame(action, controller.network))
    metrics = {"policy": policy, "seed": seed, **env.metrics(),
               "wall_time_s": perf_counter()-start,
               "mean_belief_update_ms": float(np.mean(controller.update_times)),
               "mean_structure_update_ms": float(np.mean(controller.structure_times)) if controller.structure_times else 0.0,
               "p95_structure_update_ms": float(np.percentile(controller.structure_times, 95)) if controller.structure_times else 0.0,
               "mean_planning_ms": float(np.mean(controller.control_times)),
               "mean_inference_ess": float(np.mean(controller.ess_values)) if controller.ess_values else None}
    return metrics, {"config": config.to_dict(), "policy": policy, "seed": seed, "metrics": metrics, "frames": frames}


SUMMARY_METRICS = ("throughput_veh_per_hour", "mean_accrued_delay_s", "mean_completed_delay_s",
                   "completion_fraction", "stops_per_arrival", "collision_pairs", "shield_interventions",
                   "messages_per_arrival", "mean_structure_update_ms", "mean_belief_update_ms",
                   "mean_planning_ms", "reward")


def summarize(rows):
    summary = []
    for scenario, policy in sorted({(r["scenario"], r["policy"]) for r in rows}):
        members = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
        group = {"scenario": scenario, "policy": policy, "runs": len(members)}
        for metric in SUMMARY_METRICS:
            values = [r[metric] for r in members if r[metric] is not None]
            group[metric] = {"mean": float(np.mean(values)) if values else None,
                             "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                             "ci95_half_width": float(1.96*np.std(values, ddof=1)/np.sqrt(len(values))) if len(values) > 1 else None,
                             "n": len(values)}
        summary.append(group)
    return summary


def paired_differences(rows):
    result = []
    index = {(r["scenario"], r["seed"], r["policy"]): r for r in rows}
    for scenario, policy in sorted({(r["scenario"], r["policy"]) for r in rows if r["policy"] != "neural"}):
        for metric in ("mean_accrued_delay_s", "throughput_veh_per_hour", "stops_per_arrival"):
            differences = []
            for row in rows:
                if row["scenario"] == scenario and row["policy"] == policy:
                    neural = index.get((scenario, row["seed"], "neural"))
                    if neural:
                        differences.append(neural[metric]-row[metric])
            if differences:
                result.append({"scenario": scenario, "baseline": policy, "metric": metric,
                               "definition": "neural minus baseline on identical demand seed",
                               "n": len(differences), "mean_difference": float(np.mean(differences)),
                               "ci95_half_width": float(1.96*np.std(differences, ddof=1)/np.sqrt(len(differences))) if len(differences) > 1 else None})
    return result


def evaluate(config, policies, seeds, model, output_dir, scenarios=("nominal",), progress=print):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = {
        "nominal": config,
        "busy": replace(config, arrival_rate=config.arrival_rate*1.8),
        "noisy": replace(config, position_noise=config.position_noise*2.5,
                         velocity_noise=config.velocity_noise*2, sensor_dropout=.35,
                         message_dropout=.4, max_message_delay=6),
        "no-communication": replace(config, communication_budget=0),
    }
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Use distinct, nonnegative seeds")
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("Use distinct policies")
    rows = []
    for scenario in scenarios:
        if scenario not in settings:
            raise ValueError(f"Unknown scenario {scenario}")
        for seed in seeds:
            for policy in policies:
                metrics, _ = run_episode(settings[scenario], policy, seed, model)
                rows.append({"scenario": scenario, **metrics})
                progress(f"{scenario:16s} seed={seed:<3d} {policy:11s} completed={metrics['completed']:3d}/{metrics['arrived']:3d} "
                         f"delay={metrics['mean_accrued_delay_s']:6.2f}s collisions={metrics['collision_pairs']}")
                # Preserve completed episodes if a long experiment is interrupted.
                write_json(output/"episodes.json", rows)
    with (output/"episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"configs": {name: settings[name].to_dict() for name in scenarios},
              "seeds": seeds, "policies": policies, "summary": summarize(rows),
              "paired_differences": paired_differences(rows),
              "interval_note": "Normal-approximation 95% intervals over independent episode seeds; exploratory at small n",
              "safety_note": "All controllers share a privileged true-state collision-prevention shield; collision counts are not a learned-safety guarantee"}
    write_json(output/"summary.json", report)
    from .render import comparison_plot
    comparison_plot(report, output/"comparison.png")
    return report


def benchmark_structure(config, model, sizes=(8, 16, 32, 64), repeats=8, seed=0):
    if repeats < 1 or any(n < 2 for n in sizes):
        raise ValueError("Benchmark needs repeats >= 1 and vehicle counts >= 2")
    rng, rows = np.random.default_rng(seed), []
    for n in sizes:
        tracks, _ = sample_scene(rng, config, n)
        # Synthetic correlated delay observations, same window and nodes for both
        # methods. This isolates structure-update latency, not total simulation.
        latent = rng.integers(2, size=(config.history_window, max(1, n//4)))
        history = []
        for time in range(config.history_window):
            values = np.array([latent[time, i % latent.shape[1]] for i in range(n)])
            values = np.logical_xor(values, rng.random(n) < .08)
            history.append({t.vehicle_id: float(value) for t, value in zip(tracks, values)})
        for t in tracks:
            t.mean[1] = 0.0 if history[-1][t.vehicle_id] else 8.0
            t.age = 0.0
        for name, structure in (("neural", NeuralStructure(model, config)), ("greedy", GreedyStructure(config))):
            elapsed = []
            for _ in range(repeats+1):
                if name == "greedy":
                    structure.history.clear()
                    structure.history.extend(history[:-1])
                start = perf_counter()
                adjacency = structure.update(tracks)
                elapsed.append((perf_counter()-start)*1000)
            rows.append({"vehicles": n, "method": name, "median_ms": float(np.median(elapsed[1:])),
                         "p95_ms": float(np.percentile(elapsed[1:], 95)), "repeats": repeats,
                         "edges": int(np.count_nonzero(adjacency))})
    return {"seed": seed, "history_window": config.history_window, "max_parents": config.max_parents,
            "note": "Warmed structure-update timings on synthetic snapshots; methods use different learning objectives; neural training cost excluded; pair scoring remains O(n^2)", "rows": rows}

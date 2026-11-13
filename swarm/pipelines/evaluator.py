"""Multi-process SUMO experiment matrix evaluator, SHA-256 provenance, and paired statistical reports.

Guarantees:
1. Strict pre-registered ``TEST_SEEDS`` (``3000..3014``) anti-leakage enforcement when
   ``require_test_seeds=True`` (:class:`~swarm.logging_utils.ConfigValidationError` on any train/val seed).
2. Strict neural checkpoint & preset validation (:class:`~swarm.logging_utils.CheckpointIntegrityError`
   if missing, uncalibrated when required, or trained with ``"ci_micro"`` unless ``allow_ci_micro=True``).
3. Multi-process ``spawn`` execution via :class:`concurrent.futures.ProcessPoolExecutor` (never threads
   for ``libsumo``!), incremental ``episodes.csv`` / ``episodes.json`` flushing, cryptographic SHA-256
   provenance manifest, and loud crash-report abort (:class:`~swarm.logging_utils.EpisodeExecutionError`)
   if any worker fails.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import fields, replace
import hashlib
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np

from ..config import (
    ControlConfig,
    DBNConfig,
    ExperimentConfig,
    ScenarioConfig,
    TEST_SEEDS,
    TRAIN_SEEDS,
    VAL_SEEDS,
    config_sha256,
    file_sha256,
    validate_sumo_backend,
)
from ..contracts import (
    ABLATION_POLICY_NAMES,
    DBNStateSnapshot,
    EpisodeMetrics,
    GroundTruthStepRecord,
    MAIN_POLICY_NAMES,
    NEURAL_POLICY_NAMES,
    ObservationBatch,
    POLICY_NAMES,
    RoadTopology,
    SCHEMA_VERSION,
    SwarmBeliefState,
    SwarmControlAction,
)
from ..logging_utils import (
    CheckpointIntegrityError,
    ConfigValidationError,
    EpisodeExecutionError,
    fail_loud_entrypoint,
    get_logger,
    require_finite,
)
from .statistics import compute_paired_statistics

EVAL_SUMMARY_METRICS: tuple[tuple[str, bool], ...] = (
    ("cohort_mean_total_delay_s", True),
    ("cohort_p95_total_delay_s", True),
    ("cohort_mean_completed_delay_s", True),
    ("throughput_veh_per_hr", False),
    ("mean_stops_per_veh", True),
    ("spillback_corridor_seconds", True),
    ("mean_co2_g_per_veh", True),
    ("emergency_brake_rate_per_veh_km", True),
    ("v2x_delivery_efficiency", False),
    ("edge_churn_hamming_rate", True),
)


def _bind_and_call(fn: Callable[..., Any], candidate_kwargs: Mapping[str, Any]) -> Any:
    sig = inspect.signature(fn)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return fn(**candidate_kwargs)
    bound = {k: v for k, v in candidate_kwargs.items() if k in params}
    return fn(**bound)


def _resolve_git_commit_sha(repo_root: Path | None = None) -> str:
    git_bin = shutil.which("git")
    if git_bin is None:
        return "unknown"
    cwd = str(repo_root) if repo_root is not None else str(Path.cwd())
    proc = subprocess.run(
        [git_bin, "rev-parse", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=3.0,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    return "unknown"


def validate_eval_seeds(
    seeds: Sequence[int],
    *,
    require_test_seeds: bool = True,
) -> tuple[int, ...]:
    """Validate evaluation seed list, enforcing ``TEST_SEEDS`` lock when ``require_test_seeds=True``."""
    if not seeds:
        raise ConfigValidationError("Evaluation seeds cannot be empty.")

    cleaned: list[int] = []
    for idx, s in enumerate(seeds):
        if isinstance(s, bool) or not isinstance(s, int) or s < 0:
            raise ConfigValidationError(
                f"Invalid evaluation seed at index {idx}: {s!r}.",
                context={"index": idx, "seed": repr(s)},
            )
        cleaned.append(s)

    if len(set(cleaned)) != len(cleaned):
        raise ConfigValidationError(
            "Duplicate seeds passed to evaluator.",
            context={"seeds": cleaned},
        )

    if require_test_seeds:
        leaked_train = sorted(set(cleaned) & set(TRAIN_SEEDS))
        leaked_val = sorted(set(cleaned) & set(VAL_SEEDS))
        non_test = sorted(set(cleaned) - set(TEST_SEEDS))
        if leaked_train or leaked_val or non_test:
            raise ConfigValidationError(
                "Anti-leakage lock triggered: evaluation requires pre-registered TEST_SEEDS (3000..3014) ONLY.",
                context={
                    "leaked_train_seeds": leaked_train,
                    "leaked_val_seeds": leaked_val,
                    "non_test_seeds": non_test,
                },
            )
    return tuple(cleaned)


def _resolve_checkpoint_target(checkpoint_path: Path | str) -> Path:
    ckpt_p = Path(checkpoint_path)
    if ckpt_p.is_dir() and (ckpt_p / "neural_dbn.pt").is_file():
        return ckpt_p / "neural_dbn.pt"
    return ckpt_p


def validate_and_hash_checkpoint(
    checkpoint_path: Path | str,
    *,
    allow_ci_micro: bool = False,
    require_calibrated: bool = True,
) -> str:
    """Validate checkpoint metadata/files and return a deterministic SHA-256 hex digest."""
    ckpt_p = _resolve_checkpoint_target(checkpoint_path)
    if not ckpt_p.exists():
        raise CheckpointIntegrityError(
            f"Neural checkpoint path does not exist: {ckpt_p}",
            context={"checkpoint_path": str(ckpt_p)},
        )

    if ckpt_p.is_file():
        from ..dbn.updater import load_dbn_checkpoint

        bundle = load_dbn_checkpoint(
            ckpt_p,
            require_calibrated=require_calibrated,
            device="cpu",
        )
        if bundle.model.preset_name == "ci_micro" and not allow_ci_micro:
            raise CheckpointIntegrityError(
                "Checkpoint with model_preset='ci_micro' is strictly blocked from evaluation claims "
                "(pass allow_ci_micro=True only in unit tests).",
                context={"checkpoint_path": str(ckpt_p), "model_preset": bundle.model.preset_name},
            )
        return bundle.checkpoint_sha256

    if not ckpt_p.is_dir():
        raise CheckpointIntegrityError(
            f"Unsupported checkpoint path type: {ckpt_p}",
            context={"checkpoint_path": str(ckpt_p)},
        )

    meta_path = ckpt_p / "checkpoint_meta.json"
    if not meta_path.is_file():
        raise CheckpointIntegrityError(
            f"Checkpoint directory missing checkpoint_meta.json: {ckpt_p}",
            context={"checkpoint_path": str(ckpt_p)},
        )

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        get_logger("pipelines.evaluator").error("Malformed checkpoint_meta.json in %s: %s", ckpt_p, exc)
        raise CheckpointIntegrityError(
            f"Malformed checkpoint_meta.json in {ckpt_p}: {exc}",
            context={"checkpoint_path": str(ckpt_p)},
        ) from exc

    if meta.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointIntegrityError(
            f"Checkpoint schema_version={meta.get('schema_version')} != {SCHEMA_VERSION}.",
            context={"checkpoint_path": str(ckpt_p), "schema_version": meta.get("schema_version")},
        )

    preset_name = meta.get("model_preset") or (
        meta.get("model_metadata", {}).get("preset_name")
        if isinstance(meta.get("model_metadata"), Mapping)
        else None
    )
    if preset_name == "ci_micro" and not allow_ci_micro:
        raise CheckpointIntegrityError(
            "Checkpoint with model_preset='ci_micro' is strictly blocked from evaluation claims "
            "(pass allow_ci_micro=True only in unit tests).",
            context={"checkpoint_path": str(ckpt_p), "model_preset": preset_name},
        )

    temps_obj = meta.get("temperatures")
    is_cal = bool(meta.get("is_calibrated", False)) or (
        isinstance(temps_obj, Mapping) and bool(temps_obj.get("calibrated", False))
    )
    if require_calibrated and not is_cal:
        raise CheckpointIntegrityError(
            f"Checkpoint at {ckpt_p} is uncalibrated, but DBNConfig.require_calibrated=True.",
            context={"checkpoint_path": str(ckpt_p)},
        )

    pt_files = sorted(p for p in ckpt_p.glob("*.pt") if p.is_file())
    if not pt_files:
        raise CheckpointIntegrityError(
            f"Checkpoint directory contains zero .pt weight files: {ckpt_p}",
            context={"checkpoint_path": str(ckpt_p)},
        )

    hasher = hashlib.sha256()
    hasher.update(b"meta:")
    hasher.update(file_sha256(meta_path).encode("ascii"))
    for pt_p in pt_files:
        hasher.update(f"|{pt_p.name}:".encode("utf-8"))
        hasher.update(file_sha256(pt_p).encode("ascii"))
    return hasher.hexdigest()


def _sync_dbn_preset_from_checkpoint(dbn_cfg: DBNConfig, checkpoint_path: Path | None) -> DBNConfig:
    """Synchronize ``dbn_cfg.model_preset`` with the checkpoint's architecture preset (e.g. ``sota_cloud_xl``)."""
    if checkpoint_path is None:
        return dbn_cfg
    ckpt_p = _resolve_checkpoint_target(checkpoint_path)
    if not ckpt_p.is_file():
        return dbn_cfg
    from ..dbn.updater import load_dbn_checkpoint

    bundle = load_dbn_checkpoint(
        ckpt_p,
        require_calibrated=dbn_cfg.require_calibrated,
        device="cpu",
    )
    if bundle.model.preset_name != dbn_cfg.model_preset:
        return replace(dbn_cfg, model_preset=bundle.model.preset_name)
    return dbn_cfg


def _default_topology_loader(scenario: ScenarioConfig) -> RoadTopology:
    from ..env import load_or_build_topology

    return load_or_build_topology(scenario)


def _default_adapter_factory(
    scenario: ScenarioConfig,
    topology: RoadTopology,
    seed: int,
    *,
    policy_name: str = "actuated_signal",
    control_config: ControlConfig | None = None,
) -> Any:
    from ..env import SumoSimulationAdapter

    return SumoSimulationAdapter(
        scenario,
        control_config=control_config,
        policy_name=policy_name,
        seed=seed,
        topology=topology,
    )


def _default_tracker_factory(scenario: ScenarioConfig, topology: RoadTopology) -> Any:
    from ..env import CityBeliefTracker

    return CityBeliefTracker(
        topology=topology,
        observation_config=scenario.observation,
    )


def _default_controller_factory(
    policy_name: str,
    topology: RoadTopology,
    exp_cfg: ExperimentConfig,
    checkpoint_path: Path | None,
    *,
    seed: int = 0,
) -> Any:
    from ..control import build_policy_controller

    return build_policy_controller(
        policy_name=policy_name,
        topology=topology,
        config=exp_cfg,
        checkpoint_path=checkpoint_path,
        seed=seed,
    )


def _default_trace_recorder_factory(
    scenario_id: str,
    policy_name: str,
    seed: int,
    topology: RoadTopology,
) -> Any:
    from ..viz.trace_recorder import EpisodeTraceRecorder

    return _bind_and_call(
        EpisodeTraceRecorder,
        {
            "scenario_id": scenario_id,
            "policy_name": policy_name,
            "seed": seed,
            "topology": topology,
            "network_id": topology.network_id,
        },
    )


def _compute_metrics_from_step_stream(
    *,
    scenario: ScenarioConfig,
    policy_name: str,
    seed: int,
    obs_history: Sequence[ObservationBatch],
    gt_history: Sequence[GroundTruthStepRecord],
    dbn_update_ms: Sequence[float],
    dbn_infer_ms: Sequence[float],
    ctrl_ms: Sequence[float],
    churn_rates: Sequence[float],
    clamp_total: int,
    active_veh_step_denom: int,
    deadzone_total: int,
    v2x_polls_total: int,
    v2x_delivered_total: int,
    wall_time_s: float,
    sim_steps: int,
    ckpt_sha: str | None,
    cfg_sha: str,
    net_sha: str | None,
) -> EpisodeMetrics:
    """Compute exact cohort & physical safety metrics directly from step records (used when adapter
    implements ``SimulationAdapterProtocol`` without a custom ``tripinfo.xml`` metrics method)."""
    warmup_s = scenario.warmup_duration_s
    cohort_end_s = scenario.cohort_end_s
    total_end_s = scenario.total_duration_s
    dt = scenario.step_length_s

    cohort_ids: set[str] = set()
    sched_depart: dict[str, float] = {}
    phys_depart: dict[str, float] = {}
    arrived_time: dict[str, float] = {}
    last_time_loss: dict[str, float] = {}
    last_rem_eta: dict[str, float] = {}
    stop_state: dict[str, bool] = {}
    stop_counts: dict[str, int] = {}
    co2_mg_by_veh: dict[str, float] = {}
    reroutes_cohort = 0
    total_veh_m = 0.0

    collisions = 0
    teleports = 0
    emerg_stops = 0
    emerg_brakes = 0
    ttc_near_misses = 0
    min_ttc = 999.0
    spillback_corridor_s = 0.0

    for obs, gt in zip(obs_history, gt_history, strict=True):
        t_now = float(obs.sim_time)
        for dep in obs.departed:
            phys_depart.setdefault(dep.veh_id, float(dep.entry_sim_time))
            if warmup_s <= dep.entry_sim_time < cohort_end_s:
                cohort_ids.add(dep.veh_id)

        for aid in obs.arrived_veh_ids:
            arrived_time.setdefault(aid, t_now)

        collisions += int(gt.collisions)
        teleports += int(gt.teleports)
        emerg_stops += int(gt.emergency_stops)
        emerg_brakes += int(gt.emergency_brakes)
        ttc_near_misses += int(gt.near_misses_ttc)
        if gt.min_ttc_s < min_ttc:
            min_ttc = float(gt.min_ttc_s)
        if t_now >= warmup_s:
            spillback_corridor_s += float(gt.spillback_corridor_count) * dt
            reroutes_cohort += int(gt.reroutes_applied)

        for rv in gt.raw_vehicles:
            vid = rv.veh_id
            s_dep = float(rv.scheduled_depart)
            sched_depart.setdefault(vid, s_dep)
            phys_depart.setdefault(vid, t_now)
            if warmup_s <= s_dep < cohort_end_s:
                cohort_ids.add(vid)
            last_time_loss[vid] = float(rv.time_loss)
            rem_dist = max(0.0, float(rv.lane_length - rv.lane_pos))
            last_rem_eta[vid] = rem_dist / 13.9
            total_veh_m += max(0.0, float(rv.speed)) * dt
            co2_mg_by_veh[vid] = co2_mg_by_veh.get(vid, 0.0) + float(rv.co2_mg_s) * dt

            was_stopped = stop_state.get(vid, False)
            if rv.speed < 0.2 and not was_stopped:
                stop_state[vid] = True
                stop_counts[vid] = stop_counts.get(vid, 0) + 1
            elif rv.speed > 1.0 and was_stopped:
                stop_state[vid] = False

    cohort_sorted = sorted(cohort_ids)
    n_cohort = len(cohort_sorted)
    total_delays: list[float] = []
    comp_delays: list[float] = []
    completed_n = 0
    unfinished_n = 0
    undeparted_n = 0

    for vid in cohort_sorted:
        t_sched = sched_depart.get(vid, phys_depart.get(vid, warmup_s))
        if vid not in phys_depart:
            undeparted_n += 1
            total_delays.append(max(0.0, total_end_s - t_sched))
            continue
        dep_delay = max(0.0, phys_depart[vid] - t_sched)
        t_loss = max(0.0, last_time_loss.get(vid, 0.0))
        if vid in arrived_time:
            completed_n += 1
            d_val = dep_delay + t_loss
            total_delays.append(d_val)
            comp_delays.append(d_val)
        else:
            unfinished_n += 1
            total_delays.append(dep_delay + t_loss + last_rem_eta.get(vid, 0.0))

    mean_tot_delay = float(np.mean(total_delays)) if total_delays else 0.0
    p95_tot_delay = float(np.percentile(total_delays, 95.0)) if total_delays else 0.0
    mean_comp_delay = float(np.mean(comp_delays)) if comp_delays else mean_tot_delay
    comp_rate = float(completed_n / max(1, n_cohort))
    meas_hrs = max(1e-6, (total_end_s - warmup_s) / 3600.0)
    throughput_vph = float(completed_n / meas_hrs)
    mean_stops = (
        float(np.mean([stop_counts.get(v, 0) for v in cohort_sorted])) if cohort_sorted else 0.0
    )
    mean_co2_g = (
        float(np.mean([co2_mg_by_veh.get(v, 0.0) / 1000.0 for v in cohort_sorted]))
        if cohort_sorted
        else 0.0
    )
    reroutes_pv = float(reroutes_cohort / max(1, n_cohort))
    veh_km = max(1e-6, total_veh_m / 1000.0)
    ebrake_rate = float(emerg_brakes / veh_km)
    clamp_rate = float(clamp_total / max(1, active_veh_step_denom))
    v2x_eff = float(v2x_delivered_total / max(1, v2x_polls_total)) if v2x_polls_total > 0 else 1.0

    return EpisodeMetrics(
        scenario_id=scenario.scenario_id,
        policy_name=policy_name,
        seed=int(seed),
        cohort_size=n_cohort,
        cohort_completed_count=completed_n,
        cohort_unfinished_count=unfinished_n,
        cohort_undeparted_count=undeparted_n,
        cohort_completion_rate=comp_rate,
        cohort_mean_total_delay_s=mean_tot_delay,
        cohort_p95_total_delay_s=p95_tot_delay,
        cohort_mean_completed_delay_s=mean_comp_delay,
        throughput_veh_per_hr=throughput_vph,
        mean_stops_per_veh=mean_stops,
        spillback_corridor_seconds=spillback_corridor_s,
        mean_co2_g_per_veh=mean_co2_g,
        reroutes_per_veh=reroutes_pv,
        collision_count=collisions,
        teleport_count=teleports,
        emergency_stop_count=emerg_stops,
        emergency_brake_count=emerg_brakes,
        emergency_brake_rate_per_veh_km=ebrake_rate,
        ttc_near_miss_count=ttc_near_misses,
        min_ttc_s=min_ttc,
        safety_filter_clamp_count=clamp_total,
        safety_filter_clamp_rate=clamp_rate,
        zero_speed_deadzone_count=deadzone_total,
        v2x_polls_requested=v2x_polls_total,
        v2x_packets_delivered=v2x_delivered_total,
        v2x_delivery_efficiency=v2x_eff,
        mean_dbn_update_latency_ms=float(np.mean(dbn_update_ms)) if dbn_update_ms else 0.0,
        mean_dbn_inference_latency_ms=float(np.mean(dbn_infer_ms)) if dbn_infer_ms else 0.0,
        mean_control_latency_ms=float(np.mean(ctrl_ms)) if ctrl_ms else 0.0,
        edge_churn_hamming_rate=float(np.mean(churn_rates)) if churn_rates else 0.0,
        wall_time_s=wall_time_s,
        sim_steps=sim_steps,
        steps_per_sec=float(sim_steps / max(wall_time_s, 1e-6)),
        checkpoint_sha256=ckpt_sha,
        config_sha256=cfg_sha,
        net_xml_sha256=net_sha,
    )


def run_single_episode(
    exp_cfg: ExperimentConfig,
    policy_name: str,
    seed: int,
    checkpoint_path: Path | str | None = None,
    *,
    record_trace: bool = False,
    trace_output_path: Path | str | None = None,
    allow_ci_micro: bool = False,
    adapter_factory: Callable[..., Any] | None = None,
    tracker_factory: Callable[..., Any] | None = None,
    controller_factory: Callable[..., Any] | None = None,
    topology_loader: Callable[..., RoadTopology] | None = None,
    trace_recorder_factory: Callable[..., Any] | None = None,
) -> tuple[EpisodeMetrics, dict[str, Any] | None]:
    """Execute one closed-loop episode and return ``(EpisodeMetrics, optional_3d_trace_dict)``."""
    if policy_name not in POLICY_NAMES:
        raise ConfigValidationError(
            f"Unknown policy_name={policy_name!r}; valid policies: {list(POLICY_NAMES)}.",
            context={"policy_name": policy_name, "valid_policies": list(POLICY_NAMES)},
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigValidationError(f"seed must be a non-negative int, got {seed!r}.")

    ckpt_resolved: Path | None = (
        _resolve_checkpoint_target(checkpoint_path) if checkpoint_path is not None else None
    )
    ckpt_sha: str | None = None
    if policy_name in NEURAL_POLICY_NAMES:
        if ckpt_resolved is None:
            raise CheckpointIntegrityError(
                f"Neural policy {policy_name!r} requires a non-null checkpoint_path.",
                context={"policy_name": policy_name},
            )
        ckpt_sha = validate_and_hash_checkpoint(
            ckpt_resolved,
            allow_ci_micro=allow_ci_micro,
            require_calibrated=exp_cfg.dbn.require_calibrated,
        )
        exp_cfg = replace(exp_cfg, dbn=_sync_dbn_preset_from_checkpoint(exp_cfg.dbn, ckpt_resolved))
    elif ckpt_resolved is not None and ckpt_resolved.exists():
        ckpt_sha = validate_and_hash_checkpoint(
            ckpt_resolved,
            allow_ci_micro=True,
            require_calibrated=False,
        )

    scenario = exp_cfg.scenario
    net_sha: str | None = None
    if adapter_factory is None:
        validate_sumo_backend(scenario.sumo_backend)
        net_p, _ = scenario.resolve_paths(must_exist=True)
        net_sha = file_sha256(net_p)
    else:
        cand_net = Path(scenario.net_xml_path)
        if cand_net.is_file():
            net_sha = file_sha256(cand_net)

    cfg_sha = config_sha256(exp_cfg)

    topo_fn = topology_loader or _default_topology_loader
    topology = topo_fn(scenario)

    if adapter_factory is None:
        adapter = _default_adapter_factory(
            scenario,
            topology,
            seed,
            policy_name=policy_name,
            control_config=exp_cfg.control,
        )
    else:
        adapter = adapter_factory(scenario, topology, seed)

    if controller_factory is None:
        controller = _default_controller_factory(
            policy_name,
            topology,
            exp_cfg,
            ckpt_resolved,
            seed=seed,
        )
    else:
        controller = controller_factory(policy_name, topology, exp_cfg, ckpt_resolved)

    trace_tracker: Any = None
    trace_recorder: Any = None
    if record_trace:
        trk_fn = tracker_factory or _default_tracker_factory
        trace_tracker = trk_fn(scenario, topology)
        trace_tracker.reset()
        rec_fn = trace_recorder_factory or _default_trace_recorder_factory
        trace_recorder = rec_fn(scenario.scenario_id, policy_name, seed, topology)

    obs_history: list[ObservationBatch] = []
    gt_history: list[GroundTruthStepRecord] = []
    dbn_update_ms: list[float] = []
    dbn_infer_ms: list[float] = []
    ctrl_ms: list[float] = []
    churn_rates: list[float] = []
    prev_edge_set: set[tuple[str, str]] | None = None

    clamp_total = 0
    active_veh_step_denom = 0
    deadzone_total = 0
    v2x_polls_total = 0
    v2x_delivered_total = 0

    wall_start = time.perf_counter()
    adapter_metrics: EpisodeMetrics | None = None
    sim_steps = 0

    try:
        controller.reset()
        obs, gt_rec = adapter.reset(seed=seed)

        for _ in range(scenario.total_steps):
            sim_steps += 1
            belief_for_trace: SwarmBeliefState | None = None
            if trace_tracker is not None:
                belief_for_trace = trace_tracker.observe(obs)

            t_act0 = time.perf_counter()
            action: SwarmControlAction = controller.act(obs)
            dt_act_ms = (time.perf_counter() - t_act0) * 1000.0
            eff_ctrl_ms = action.control_latency_ms if action.control_latency_ms > 0.0 else dt_act_ms
            ctrl_ms.append(float(eff_ctrl_ms))

            snap: DBNStateSnapshot | None = controller.last_dbn_snapshot
            if snap is not None:
                dbn_update_ms.append(float(snap.update_latency_ms))
                dbn_infer_ms.append(float(snap.inference_latency_ms))
                cur_edges = {(e.src_id, e.dst_id) for e in snap.edges}
                if prev_edge_set is not None:
                    union_n = len(cur_edges | prev_edge_set)
                    churn = float(len(cur_edges ^ prev_edge_set) / union_n) if union_n > 0 else 0.0
                    churn_rates.append(churn)
                prev_edge_set = cur_edges

            clamp_total += int(action.safety_clamp_count)
            active_veh_step_denom += len(obs.active_veh_ids)
            deadzone_total += int(action.zero_speed_deadzone_count)
            v2x_polls_total += len(action.v2x_polls)
            v2x_delivered_total += int(gt_rec.v2x_delivered)

            if trace_recorder is not None:
                ctrl_belief = (
                    getattr(controller, "last_belief_state", None)
                    or getattr(controller, "last_belief", None)
                    or belief_for_trace
                )
                trace_recorder.record_step(
                    gt_record=gt_rec,
                    belief=ctrl_belief,
                    dbn_snapshot=snap,
                    action=action,
                    tls_states=obs.tls_states,
                )
            if trace_tracker is not None:
                trace_tracker.predict(
                    dt=obs.dt,
                    post_safety_accels=action.post_safety_accels,
                    v2x_polled_ids=action.v2x_polls,
                    sim_time=obs.sim_time,
                )

            obs_history.append(obs)
            gt_history.append(gt_rec)

            obs, gt_rec = adapter.step(action)
    finally:
        adapter.close()

    wall_time_s = max(1e-6, time.perf_counter() - wall_start)

    # Query adapter's native tripinfo-backed metrics method after close() (when tripinfo.xml is flushed)
    for m_name in ("compute_episode_metrics", "finalize_episode_metrics", "get_episode_metrics"):
        m_fn = getattr(adapter, m_name, None)
        if callable(m_fn):
            adapter_metrics = _bind_and_call(
                m_fn,
                {
                    "policy_name": policy_name,
                    "seed": seed,
                    "mean_dbn_update_latency_ms": float(np.mean(dbn_update_ms)) if dbn_update_ms else 0.0,
                    "mean_dbn_inference_latency_ms": float(np.mean(dbn_infer_ms)) if dbn_infer_ms else 0.0,
                    "mean_control_latency_ms": float(np.mean(ctrl_ms)) if ctrl_ms else 0.0,
                    "edge_churn_hamming_rate": float(np.mean(churn_rates)) if churn_rates else 0.0,
                    "safety_filter_clamp_count": clamp_total,
                    "safety_filter_clamp_rate": float(clamp_total / max(1, active_veh_step_denom)),
                    "zero_speed_deadzone_count": deadzone_total,
                    "v2x_polls_requested": v2x_polls_total,
                    "wall_time_s": wall_time_s,
                    "sim_steps": sim_steps,
                    "checkpoint_sha256": ckpt_sha,
                    "config_sha256": cfg_sha,
                    "net_xml_sha256": net_sha,
                },
            )
            break

    if adapter_metrics is not None:
        metrics = replace(
            adapter_metrics,
            policy_name=policy_name,
            seed=int(seed),
            mean_dbn_update_latency_ms=(
                float(np.mean(dbn_update_ms))
                if dbn_update_ms
                else adapter_metrics.mean_dbn_update_latency_ms
            ),
            mean_dbn_inference_latency_ms=(
                float(np.mean(dbn_infer_ms))
                if dbn_infer_ms
                else adapter_metrics.mean_dbn_inference_latency_ms
            ),
            mean_control_latency_ms=(
                float(np.mean(ctrl_ms)) if ctrl_ms else adapter_metrics.mean_control_latency_ms
            ),
            edge_churn_hamming_rate=(
                float(np.mean(churn_rates))
                if churn_rates
                else adapter_metrics.edge_churn_hamming_rate
            ),
            wall_time_s=wall_time_s,
            sim_steps=sim_steps,
            steps_per_sec=float(sim_steps / wall_time_s),
            checkpoint_sha256=ckpt_sha,
            config_sha256=cfg_sha,
            net_xml_sha256=net_sha,
        )
    else:
        metrics = _compute_metrics_from_step_stream(
            scenario=scenario,
            policy_name=policy_name,
            seed=seed,
            obs_history=obs_history,
            gt_history=gt_history,
            dbn_update_ms=dbn_update_ms,
            dbn_infer_ms=dbn_infer_ms,
            ctrl_ms=ctrl_ms,
            churn_rates=churn_rates,
            clamp_total=clamp_total,
            active_veh_step_denom=active_veh_step_denom,
            deadzone_total=deadzone_total,
            v2x_polls_total=v2x_polls_total,
            v2x_delivered_total=v2x_delivered_total,
            wall_time_s=wall_time_s,
            sim_steps=sim_steps,
            ckpt_sha=ckpt_sha,
            cfg_sha=cfg_sha,
            net_sha=net_sha,
        )

    # Validate finite metrics before returning
    for fobj in fields(EpisodeMetrics):
        val = getattr(metrics, fobj.name)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            require_finite(float(val), f"EpisodeMetrics.{fobj.name}")

    trace_payload: dict[str, Any] | None = None
    if trace_recorder is not None:
        trace_payload = trace_recorder.finalize(metrics)
        if trace_output_path is not None:
            out_tp = Path(trace_output_path)
            out_tp.parent.mkdir(parents=True, exist_ok=True)
            tmp_tp = out_tp.parent / f".{out_tp.name}.pid{os.getpid()}.tmp"
            tmp_tp.write_text(json.dumps(trace_payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")
            os.replace(tmp_tp, out_tp)

    return metrics, trace_payload


def _eval_worker_task(
    exp_cfg: ExperimentConfig,
    policy_name: str,
    seed: int,
    ckpt_str: str | None,
    allow_ci_micro: bool,
    output_dir_str: str,
) -> EpisodeMetrics:
    out_dir = Path(output_dir_str)

    @fail_loud_entrypoint(out_dir)
    def _exec() -> EpisodeMetrics:
        m, _ = run_single_episode(
            exp_cfg=exp_cfg,
            policy_name=policy_name,
            seed=seed,
            checkpoint_path=Path(ckpt_str) if ckpt_str is not None else None,
            record_trace=False,
            allow_ci_micro=allow_ci_micro,
        )
        return m

    return _exec()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.pid{os.getpid()}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _flush_episodes_csv_and_json(out_dir: Path, completed: Sequence[EpisodeMetrics]) -> None:
    sorted_eps = sorted(completed, key=lambda m: (m.scenario_id, m.policy_name, m.seed))
    rows = [m.to_dict() for m in sorted_eps]
    _atomic_write_json(out_dir / "episodes.json", rows)

    field_names = [f.name for f in fields(EpisodeMetrics)]
    csv_path = out_dir / "episodes.csv"
    tmp_csv = out_dir / f".episodes.csv.pid{os.getpid()}.tmp"
    with tmp_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp_csv, csv_path)


def _summarize_policy_group(
    episodes: Sequence[EpisodeMetrics],
    scenario_ids: Sequence[str],
    primary_policy: str,
    group_policies: Sequence[str],
    n_resamples: int,
) -> dict[str, Any]:
    scen_set = set(scenario_ids)
    pol_set = set(group_policies)
    filtered = [m for m in episodes if m.scenario_id in scen_set and m.policy_name in pol_set]
    if not filtered:
        return {}
    stat_res = compute_paired_statistics(
        filtered,
        reference_policy=primary_policy,
        n_resamples=n_resamples,
    )
    scenarios_map = stat_res.get("scenarios", {})
    return {sid: scenarios_map[sid] for sid in scenario_ids if sid in scenarios_map}


def evaluate_experiment_matrix(
    scenarios: Sequence[ScenarioConfig],
    policies: Sequence[str],
    seeds: Sequence[int],
    output_dir: Path | str,
    *,
    checkpoint_path: Path | str | None = None,
    dbn_cfg: DBNConfig = DBNConfig(),
    control_cfg: ControlConfig = ControlConfig(),
    max_workers: int = 4,
    bootstrap_resamples: int = 10_000,
    require_test_seeds: bool = True,
    allow_ci_micro: bool = False,
    adapter_factory: Callable[..., Any] | None = None,
    tracker_factory: Callable[..., Any] | None = None,
    controller_factory: Callable[..., Any] | None = None,
    topology_loader: Callable[..., RoadTopology] | None = None,
) -> dict[str, Any]:
    """Run paired multi-scenario, multi-policy evaluation matrix and emit statistical reports."""
    if not scenarios:
        raise ConfigValidationError("evaluate_experiment_matrix requires at least one ScenarioConfig.")
    if not policies:
        raise ConfigValidationError("evaluate_experiment_matrix requires at least one policy name.")
    if max_workers < 1:
        raise ConfigValidationError(f"max_workers must be >= 1, got {max_workers}.")
    if bootstrap_resamples < 100:
        raise ConfigValidationError(f"bootstrap_resamples must be >= 100, got {bootstrap_resamples}.")

    for p in policies:
        if p not in POLICY_NAMES:
            raise ConfigValidationError(
                f"Unknown policy {p!r} in evaluate_experiment_matrix; allowed={list(POLICY_NAMES)}.",
                context={"policy": p},
            )

    clean_seeds = validate_eval_seeds(seeds, require_test_seeds=require_test_seeds)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("pipelines.evaluator")

    ckpt_resolved = Path(checkpoint_path) if checkpoint_path is not None else None
    ckpt_sha: str | None = None
    if any(p in NEURAL_POLICY_NAMES for p in policies):
        if ckpt_resolved is None:
            raise CheckpointIntegrityError(
                "Neural policy requested in evaluate_experiment_matrix, but checkpoint_path is None."
            )
        ckpt_sha = validate_and_hash_checkpoint(
            ckpt_resolved,
            allow_ci_micro=allow_ci_micro,
            require_calibrated=dbn_cfg.require_calibrated,
        )
        dbn_cfg = _sync_dbn_preset_from_checkpoint(dbn_cfg, ckpt_resolved)

    network_shas: dict[str, str] = {}
    scenario_shas: dict[str, str] = {}
    tasks: list[tuple[ExperimentConfig, str, int]] = []

    for scen in scenarios:
        scenario_shas[scen.scenario_id] = config_sha256(scen)
        if adapter_factory is None:
            validate_sumo_backend(scen.sumo_backend)
            net_p, _ = scen.resolve_paths(must_exist=True)
            network_shas[scen.scenario_id] = file_sha256(net_p)
        else:
            cand_p = Path(scen.net_xml_path)
            if cand_p.is_file():
                network_shas[scen.scenario_id] = file_sha256(cand_p)

        exp_cfg = ExperimentConfig(
            scenario=scen,
            dbn=dbn_cfg,
            control=control_cfg,
            max_parallel_workers=max_workers,
            bootstrap_resamples=bootstrap_resamples,
        )
        for pol in policies:
            for seed in clean_seeds:
                tasks.append((exp_cfg, pol, seed))

    completed: list[EpisodeMetrics] = []
    custom_hooks = any(
        h is not None for h in (adapter_factory, tracker_factory, controller_factory, topology_loader)
    )

    if max_workers == 1 or len(tasks) == 1 or custom_hooks:
        for exp_cfg, pol, seed in tasks:
            m, _ = run_single_episode(
                exp_cfg=exp_cfg,
                policy_name=pol,
                seed=seed,
                checkpoint_path=ckpt_resolved,
                record_trace=False,
                allow_ci_micro=allow_ci_micro,
                adapter_factory=adapter_factory,
                tracker_factory=tracker_factory,
                controller_factory=controller_factory,
                topology_loader=topology_loader,
            )
            completed.append(m)
            _flush_episodes_csv_and_json(out_dir, completed)
            logger.info(
                "[%d/%d] %s | %s | seed=%d | delay=%.2fs | stops=%.2f",
                len(completed),
                len(tasks),
                m.scenario_id,
                m.policy_name,
                m.seed,
                m.cohort_mean_total_delay_s,
                m.mean_stops_per_veh,
            )
    else:
        ctx = mp.get_context("spawn")
        ckpt_str = str(ckpt_resolved) if ckpt_resolved is not None else None
        with ProcessPoolExecutor(max_workers=min(max_workers, len(tasks)), mp_context=ctx) as pool:
            fut_map = {
                pool.submit(
                    _eval_worker_task,
                    exp_cfg,
                    pol,
                    seed,
                    ckpt_str,
                    allow_ci_micro,
                    str(out_dir),
                ): (exp_cfg.scenario.scenario_id, pol, seed)
                for exp_cfg, pol, seed in tasks
            }
            for fut in as_completed(fut_map):
                scen_id, pol, seed = fut_map[fut]
                try:
                    m = fut.result()
                except Exception as exc:
                    for pending in fut_map:
                        pending.cancel()
                    logger.error(
                        "Evaluation worker failed on scenario=%s policy=%s seed=%d: %s",
                        scen_id,
                        pol,
                        seed,
                        exc,
                    )
                    raise EpisodeExecutionError(
                        f"Evaluation worker failed on scenario={scen_id!r}, policy={pol!r}, seed={seed}: {exc}",
                        context={"scenario_id": scen_id, "policy_name": pol, "seed": seed},
                    ) from exc
                completed.append(m)
                _flush_episodes_csv_and_json(out_dir, completed)
                logger.info(
                    "[%d/%d] %s | %s | seed=%d | delay=%.2fs",
                    len(completed),
                    len(tasks),
                    m.scenario_id,
                    m.policy_name,
                    m.seed,
                    m.cohort_mean_total_delay_s,
                )

    scen_ids = [s.scenario_id for s in scenarios]
    primary_pol = "neural_dbn" if "neural_dbn" in policies else policies[0]
    main_pols = [p for p in policies if p in MAIN_POLICY_NAMES]
    if primary_pol not in main_pols:
        main_pols = [primary_pol, *main_pols]

    ablation_pols = [
        p for p in policies if p in ABLATION_POLICY_NAMES or p in ("neural_dbn", "independent_belief")
    ]

    main_comparison = _summarize_policy_group(
        episodes=completed,
        scenario_ids=scen_ids,
        primary_policy=primary_pol,
        group_policies=main_pols,
        n_resamples=bootstrap_resamples,
    )
    ablation_comparison = (
        _summarize_policy_group(
            episodes=completed,
            scenario_ids=scen_ids,
            primary_policy="neural_dbn",
            group_policies=ablation_pols,
            n_resamples=bootstrap_resamples,
        )
        if "neural_dbn" in ablation_pols and len(ablation_pols) > 1
        else {}
    )

    provenance: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": _resolve_git_commit_sha(),
        "checkpoint_path": str(ckpt_resolved) if ckpt_resolved is not None else None,
        "checkpoint_sha256": ckpt_sha,
        "dbn_config_sha256": config_sha256(dbn_cfg),
        "control_config_sha256": config_sha256(control_cfg),
        "scenario_config_sha256s": scenario_shas,
        "network_xml_sha256s": network_shas,
        "require_test_seeds": require_test_seeds,
        "seeds": list(clean_seeds),
        "policies": list(policies),
        "scenarios": scen_ids,
        "bootstrap_resamples": bootstrap_resamples,
        "num_episodes": len(completed),
    }
    _atomic_write_json(out_dir / "provenance.json", provenance)

    scenarios_plot_view = _summarize_policy_group(
        episodes=completed,
        scenario_ids=scen_ids,
        primary_policy=primary_pol,
        group_policies=list(policies),
        n_resamples=bootstrap_resamples,
    )

    summary_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "primary_policy": primary_pol,
        "scenario_ids": scen_ids,
        "scenarios": scenarios_plot_view,
        "policies": list(policies),
        "seeds": list(clean_seeds),
        "num_episodes": len(completed),
        "provenance": provenance,
        "main_comparison": main_comparison,
        "ablation_comparison": ablation_comparison,
    }
    _atomic_write_json(out_dir / "summary.json", summary_payload)
    return summary_payload


__all__ = [
    "EVAL_SUMMARY_METRICS",
    "evaluate_experiment_matrix",
    "run_single_episode",
    "validate_and_hash_checkpoint",
    "validate_eval_seeds",
]

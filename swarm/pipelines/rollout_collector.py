"""Zero-leakage SUMO rollout dataset collector for unrolled neural DBN training & evaluation.

Anti-Leakage Guarantees (``#9``, ``#13``, ``#14``, ``#16``, ``#26``):
1. ``split='train'`` ONLY permits ``TRAIN_SEEDS`` (``1000..1039``) on in-distribution ``grid3x3`` scenarios.
2. ``split='val'`` ONLY permits ``VAL_SEEDS`` (``2000..2009``) on in-distribution ``grid3x3`` scenarios.
3. ``split='eval_id'`` and ``split='eval_ood'`` ONLY permit ``TEST_SEEDS`` (``3000..3014``) and are
   blocked by ``trainer.py`` from ever fitting normalizers, model weights, or calibration temperatures.
4. All episode shards are stored as pure ``torch.Tensor`` + primitive dictionaries loaded strictly
   with ``torch.load(..., weights_only=True)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import inspect
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from typing import Any, Literal

import numpy as np
import torch

from ..config import (
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
    CORRIDOR_EDGE_FEATURE_DIM,
    CORRIDOR_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    ObservationBatch,
    RelationType,
    RoadTopology,
    SCHEMA_VERSION,
    SwarmBeliefState,
    SwarmControlAction,
)
from ..logging_utils import (
    ConfigValidationError,
    EpisodeExecutionError,
    fail_loud_entrypoint,
    get_logger,
    require_finite,
    require_shape,
    update_crash_context,
)
from .labels import compute_step_causal_labels

RolloutSplit = Literal["train", "val", "eval_id", "eval_ood", "test"]


def validate_rollout_split(
    scenario: ScenarioConfig,
    seeds: Sequence[int],
    split: RolloutSplit,
) -> tuple[int, ...]:
    """Enforce strict seed-split and ID/OOD topology separation; raise ``ConfigValidationError`` on leak."""
    if split not in ("train", "val", "eval_id", "eval_ood", "test"):
        raise ConfigValidationError(
            f"Invalid rollout split {split!r}; expected 'train', 'val', 'eval_id', 'eval_ood', or 'test'.",
            context={"split": split},
        )
    if not seeds:
        raise ConfigValidationError(f"Empty seed sequence passed for split={split!r}.")

    clean_seeds: list[int] = []
    for s in seeds:
        if isinstance(s, bool) or not isinstance(s, int):
            raise ConfigValidationError(
                f"Seed must be strict int, got {type(s).__name__}={s!r}.",
                context={"split": split, "seed": repr(s)},
            )
        clean_seeds.append(s)
    if len(set(clean_seeds)) != len(clean_seeds):
        raise ConfigValidationError(
            f"Duplicate seeds in rollout request for split={split!r}: {clean_seeds}.",
            context={"split": split, "seeds": clean_seeds},
        )

    seed_set = set(clean_seeds)
    net_tag = f"{scenario.network_id} {scenario.scenario_id}".lower()

    if split == "train":
        illegal = sorted(seed_set - set(TRAIN_SEEDS))
        if illegal:
            raise ConfigValidationError(
                f"Train split seed-leak guard: seeds {illegal} are outside TRAIN_SEEDS (1000..1039).",
                context={"split": split, "illegal_seeds": illegal, "scenario_id": scenario.scenario_id},
            )
        if scenario.is_ood_regime or "grid3x3" not in net_tag:
            raise ConfigValidationError(
                f"Train split topology-leak guard: scenario {scenario.scenario_id!r} "
                f"(network={scenario.network_id!r}, is_ood={scenario.is_ood_regime}) is forbidden in training; "
                "only in-distribution grid3x3 scenarios are allowed.",
                context={
                    "split": split,
                    "scenario_id": scenario.scenario_id,
                    "network_id": scenario.network_id,
                    "is_ood_regime": scenario.is_ood_regime,
                },
            )

    elif split == "val":
        illegal = sorted(seed_set - set(VAL_SEEDS))
        if illegal:
            raise ConfigValidationError(
                f"Validation split seed-leak guard: seeds {illegal} are outside VAL_SEEDS (2000..2009).",
                context={"split": split, "illegal_seeds": illegal, "scenario_id": scenario.scenario_id},
            )
        if scenario.is_ood_regime or "grid3x3" not in net_tag:
            raise ConfigValidationError(
                f"Validation split topology-leak guard: scenario {scenario.scenario_id!r} "
                f"(network={scenario.network_id!r}, is_ood={scenario.is_ood_regime}) is forbidden in validation; "
                "only in-distribution grid3x3 scenarios are allowed.",
                context={
                    "split": split,
                    "scenario_id": scenario.scenario_id,
                    "network_id": scenario.network_id,
                    "is_ood_regime": scenario.is_ood_regime,
                },
            )

    elif split == "eval_id":
        illegal = sorted(seed_set - set(TEST_SEEDS))
        if illegal:
            raise ConfigValidationError(
                f"Held-out ID eval seed-leak guard: seeds {illegal} are outside TEST_SEEDS (3000..3014).",
                context={"split": split, "illegal_seeds": illegal},
            )
        if scenario.is_ood_regime:
            raise ConfigValidationError(
                f"Scenario {scenario.scenario_id!r} has is_ood_regime=True; pass split='eval_ood' or 'test' instead.",
                context={"split": split, "scenario_id": scenario.scenario_id},
            )

    elif split == "eval_ood":
        illegal = sorted(seed_set - set(TEST_SEEDS))
        if illegal:
            raise ConfigValidationError(
                f"Held-out OOD eval seed-leak guard: seeds {illegal} are outside TEST_SEEDS (3000..3014).",
                context={"split": split, "illegal_seeds": illegal},
            )
        if not scenario.is_ood_regime:
            raise ConfigValidationError(
                f"Scenario {scenario.scenario_id!r} has is_ood_regime=False; pass split='eval_id' or 'test' instead.",
                context={"split": split, "scenario_id": scenario.scenario_id},
            )

    else:  # split == "test" (held-out predictive evaluation on TEST_SEEDS across ID or OOD; blocked from train/val)
        illegal = sorted(seed_set - set(TEST_SEEDS))
        if illegal:
            raise ConfigValidationError(
                f"Held-out test seed-leak guard: seeds {illegal} are outside TEST_SEEDS (3000..3014).",
                context={"split": split, "illegal_seeds": illegal},
            )

    return tuple(clean_seeds)


def select_behavior_policy(seed: int) -> tuple[str, bool]:
    """Deterministic mixed behavior schedule: 30% actuated, 30% FCFS, 20% heuristic, 20% exploratory."""
    bucket = int(seed) % 10
    if bucket in (0, 1, 2):
        return "actuated_signal", False
    if bucket in (3, 4, 5):
        return "fcfs_reservation", False
    if bucket in (6, 7):
        return "heuristic_radius_dbn", False
    return "heuristic_radius_dbn", True


def _apply_exploratory_perturbation(
    action: SwarmControlAction,
    obs: ObservationBatch,
    belief: SwarmBeliefState,
    topology: RoadTopology,
    rng: np.random.Generator,
) -> SwarmControlAction:
    """Inject safe bounded-slowdown & edge-cost perturbations on exploratory rollouts.

    Strictly respects ``ObservableSafetyFilter`` speed ceilings (``perturbed <= base_v``) and
    stopped-vehicle deadzone (``base_v > 0.15``), while keeping ``post_safety_accels`` synchronized
    with ``tracker.predict(...)``.
    """
    new_speeds = dict(action.target_speeds)
    new_accels = dict(action.post_safety_accels)
    dt_safe = max(float(obs.dt), 1e-3)
    for vid in obs.active_veh_ids:
        if rng.random() < 0.15 and vid in new_speeds and vid in belief.vehicles:
            base_v = float(new_speeds[vid])
            if base_v > 0.15:
                perturbed = float(base_v * rng.uniform(0.60, 1.0))
                if perturbed <= 0.15:
                    perturbed = 0.0
                new_speeds[vid] = perturbed
                v_est = float(belief.vehicles[vid].mean_v_mps)
                new_accels[vid] = float(np.clip((perturbed - v_est) / dt_safe, -4.5, 2.6))

    new_costs = dict(action.broadcast_edge_costs) if action.broadcast_edge_costs is not None else None
    if new_costs is not None and rng.random() < 0.25:
        for edge_id in list(new_costs):
            if edge_id in topology.edge_to_corridor:
                scale = float(np.clip(1.0 + rng.normal(0.0, 0.20), 0.65, 1.50))
                new_costs[edge_id] = float(new_costs[edge_id] * scale)

    return replace(
        action,
        target_speeds=new_speeds,
        post_safety_accels=new_accels,
        broadcast_edge_costs=new_costs,
    )


def _bind_and_call(func: Callable[..., Any], available: Mapping[str, Any]) -> Any:
    """Call ``func`` binding matching parameter names from ``available`` without swallowing errors."""
    sig = inspect.signature(func)
    kwargs: dict[str, Any] = {}
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    for pname, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if pname in available:
            kwargs[pname] = available[pname]
        elif param.default is inspect.Parameter.empty and not accepts_var_kw:
            raise ConfigValidationError(
                f"Cannot bind required parameter {pname!r} on {func.__qualname__}; "
                f"available keys={sorted(available)}."
            )
    return func(**kwargs)


def _build_corridor_turn_tables(
    belief: SwarmBeliefState,
    veh_ids: Sequence[str],
    corridor_ids: Sequence[str],
    topology: RoadTopology,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-vehicle corridor index, padded outgoing corridor indices/turn probs, and exit flags."""
    corr_to_idx = {cid: idx for idx, cid in enumerate(corridor_ids)}
    max_turns = max((len(topology.corridors[cid].turn_options) for cid in corridor_ids), default=1)
    max_turns = max(1, max_turns)
    n_v = len(veh_ids)

    veh_corr_idx = torch.zeros((n_v,), dtype=torch.long)
    out_corr_idx = torch.full((n_v, max_turns), -1, dtype=torch.long)
    out_turn_probs = torch.zeros((n_v, max_turns), dtype=torch.float32)
    is_exit_veh = torch.zeros((n_v,), dtype=torch.bool)

    for i, vid in enumerate(veh_ids):
        vb = belief.vehicles[vid]
        if vb.corridor_id not in corr_to_idx:
            raise ConfigValidationError(
                f"Vehicle {vid!r} references unknown corridor_id {vb.corridor_id!r}."
            )
        veh_corr_idx[i] = corr_to_idx[vb.corridor_id]
        spec = topology.corridors[vb.corridor_id]
        if spec.is_exit_fringe:
            # Boundary Lock A (#19 §1): exit-fringe corridor has zero downstream context by contract.
            is_exit_veh[i] = True
            continue
        if len(vb.turn_probs) != len(spec.turn_options):
            raise ConfigValidationError(
                f"Vehicle {vid!r} turn_probs length {len(vb.turn_probs)} != "
                f"turn_options length {len(spec.turn_options)} on corridor {vb.corridor_id!r}."
            )
        for m_idx, (opt, prob) in enumerate(zip(spec.turn_options, vb.turn_probs, strict=True)):
            out_corr_idx[i, m_idx] = corr_to_idx[opt.to_corridor]
            out_turn_probs[i, m_idx] = float(prob)

    return veh_corr_idx, out_corr_idx, out_turn_probs, is_exit_veh


def _unpack_extracted_features(
    raw_feats: Any,
    belief: SwarmBeliefState,
    topology: RoadTopology,
) -> dict[str, Any]:
    """Normalize output of ``swarm.dbn.extract_graph_features`` (``GraphFeatureBatch``) into tensors."""
    def _get(obj: Any, *names: str) -> Any:
        for n in names:
            if isinstance(obj, Mapping) and n in obj:
                return obj[n]
            if hasattr(obj, n):
                return getattr(obj, n)
        return None

    cand_graph = _get(raw_feats, "candidate_graph")
    x_v_raw = _get(raw_feats, "x_veh", "x_v", "X_V", "node_features", "veh_features")
    x_c_raw = _get(raw_feats, "x_corridor", "x_c", "X_C", "corridor_features")
    x_e_raw = _get(raw_feats, "x_edge", "x_e", "X_E", "edge_features")
    x_cc_raw = _get(raw_feats, "x_corridor_edge", "x_cc", "X_CC", "corridor_edge_features")

    if x_v_raw is None and isinstance(raw_feats, tuple) and len(raw_feats) >= 4:
        x_v_raw, x_c_raw, x_e_raw, x_cc_raw = raw_feats[:4]

    if x_v_raw is None or x_c_raw is None or x_e_raw is None or x_cc_raw is None:
        raise ConfigValidationError(
            f"Unsupported return value from extract_graph_features: {type(raw_feats).__name__}."
        )

    veh_ids_raw = (
        _get(cand_graph, "veh_ids")
        if cand_graph is not None
        else _get(raw_feats, "veh_ids", "vehicle_ids")
    )
    veh_ids = tuple(veh_ids_raw) if veh_ids_raw is not None else tuple(sorted(belief.vehicles))

    corr_ids_raw = _get(raw_feats, "corridor_ids")
    corridor_ids = tuple(corr_ids_raw) if corr_ids_raw is not None else tuple(sorted(topology.corridors))

    edge_index_raw = (
        _get(cand_graph, "edge_index")
        if cand_graph is not None
        else _get(raw_feats, "edge_index", "veh_edge_index")
    )
    edge_rel_raw = (
        _get(cand_graph, "relation_types")
        if cand_graph is not None
        else _get(raw_feats, "edge_relations", "relation_types", "relations")
    )
    p_conflict_raw = _get(cand_graph, "p_conflict") if cand_graph is not None else _get(raw_feats, "p_conflict")
    p_merge_raw = _get(cand_graph, "p_merge") if cand_graph is not None else _get(raw_feats, "p_merge")
    signed_gap_raw = (
        _get(cand_graph, "signed_path_gap_m")
        if cand_graph is not None
        else _get(raw_feats, "signed_path_gap_m")
    )

    cand_tuples_raw = _get(raw_feats, "candidate_edges", "candidates")
    corr_edge_index_raw = _get(raw_feats, "corridor_edge_index")

    x_v = torch.as_tensor(x_v_raw, dtype=torch.float32).cpu()
    x_c = torch.as_tensor(x_c_raw, dtype=torch.float32).cpu()
    x_e = torch.as_tensor(x_e_raw, dtype=torch.float32).cpu()
    x_cc = torch.as_tensor(x_cc_raw, dtype=torch.float32).cpu()

    require_shape(x_v, (len(veh_ids), NODE_FEATURE_DIM), "rollout.X_V")
    require_shape(x_c, (len(corridor_ids), CORRIDOR_FEATURE_DIM), "rollout.X_C")
    require_shape(x_e, (None, EDGE_FEATURE_DIM), "rollout.X_E")
    require_shape(x_cc, (None, CORRIDOR_EDGE_FEATURE_DIM), "rollout.X_CC")
    require_finite(x_v, "rollout.X_V")
    require_finite(x_c, "rollout.X_C")
    require_finite(x_e, "rollout.X_E")
    require_finite(x_cc, "rollout.X_CC")

    e_cand = int(x_e.shape[0])
    if edge_index_raw is not None:
        edge_index = torch.as_tensor(edge_index_raw, dtype=torch.long).cpu()
        require_shape(edge_index, (2, e_cand), "rollout.edge_index")
    elif cand_tuples_raw is not None:
        v_map = {vid: idx for idx, vid in enumerate(veh_ids)}
        if len(cand_tuples_raw) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            src_idx = [v_map[c[0] if isinstance(c, tuple) else c.src_id] for c in cand_tuples_raw]
            dst_idx = [v_map[c[1] if isinstance(c, tuple) else c.dst_id] for c in cand_tuples_raw]
            edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
    else:
        raise ConfigValidationError("extract_graph_features output missing edge_index / candidate_edges.")

    if edge_rel_raw is not None:
        edge_relations = torch.as_tensor(edge_rel_raw, dtype=torch.long).cpu()
        require_shape(edge_relations, (e_cand,), "rollout.edge_relations")
    else:
        edge_relations = (
            torch.argmax(x_e[:, :4], dim=-1).to(torch.long)
            if e_cand > 0
            else torch.zeros((0,), dtype=torch.long)
        )

    p_conflict = (
        torch.as_tensor(p_conflict_raw, dtype=torch.float32).cpu()
        if p_conflict_raw is not None
        else x_e[:, 8].clone()
    )
    p_merge = (
        torch.as_tensor(p_merge_raw, dtype=torch.float32).cpu()
        if p_merge_raw is not None
        else x_e[:, 9].clone()
    )
    signed_gap_m = (
        torch.as_tensor(signed_gap_raw, dtype=torch.float32).cpu()
        if signed_gap_raw is not None
        else (x_e[:, 6] * 50.0).clone()
    )

    cand_triples: list[tuple[str, str, int]] = []
    for k in range(e_cand):
        s_i = int(edge_index[0, k].item())
        d_i = int(edge_index[1, k].item())
        r_i = int(edge_relations[k].item())
        cand_triples.append((veh_ids[s_i], veh_ids[d_i], r_i))

    if corr_edge_index_raw is not None:
        corridor_edge_index = torch.as_tensor(corr_edge_index_raw, dtype=torch.long).cpu()
        require_shape(corridor_edge_index, (2, int(x_cc.shape[0])), "rollout.corridor_edge_index")
    else:
        raise ConfigValidationError("extract_graph_features output missing corridor_edge_index.")

    topo_prec_raw = _get(raw_feats, "topo_precedence")
    topo_precedence = (
        torch.as_tensor(topo_prec_raw, dtype=torch.float32).cpu()
        if topo_prec_raw is not None
        else torch.tensor(
            [-belief.vehicles[v].free_eta_s + 15.0 * belief.vehicles[v].tail_blockage_score for v in veh_ids],
            dtype=torch.float32,
        )
    )

    speed_ev_raw = _get(raw_feats, "speed_evidence_pos")
    if speed_ev_raw is not None:
        speed_evidence_pos = torch.as_tensor(speed_ev_raw, dtype=torch.float32).cpu()
    else:
        ev_vals: list[float] = []
        for vid in veh_ids:
            vb = belief.vehicles[vid]
            vmax = topology.corridors[vb.corridor_id].speed_limit_mps
            sig = 1.0 / (1.0 + math.exp(-((0.45 * vmax - vb.mean_v_mps) / 1.2)))
            ev_vals.append(0.5 + 0.42 * math.exp(-vb.obs_age_s / 3.0) * (2.0 * sig - 1.0))
        speed_evidence_pos = torch.tensor(ev_vals, dtype=torch.float32)

    turn_src_raw = _get(raw_feats, "veh_turn_src_idx")
    turn_out_raw = _get(raw_feats, "veh_turn_out_corridor_idx")
    turn_prob_raw = _get(raw_feats, "veh_turn_prob")
    if turn_src_raw is not None and turn_out_raw is not None and turn_prob_raw is not None:
        veh_turn_src_idx = torch.as_tensor(turn_src_raw, dtype=torch.long).cpu()
        veh_turn_out_corridor_idx = torch.as_tensor(turn_out_raw, dtype=torch.long).cpu()
        veh_turn_prob = torch.as_tensor(turn_prob_raw, dtype=torch.float32).cpu()
    else:
        corr_to_idx = {cid: idx for idx, cid in enumerate(corridor_ids)}
        ts_l: list[int] = []
        to_l: list[int] = []
        tp_l: list[float] = []
        for vi, vid in enumerate(veh_ids):
            vb = belief.vehicles[vid]
            spec = topology.corridors[vb.corridor_id]
            if spec.is_exit_fringe:
                continue
            for opt, pr in zip(spec.turn_options, vb.turn_probs, strict=True):
                ts_l.append(vi)
                to_l.append(corr_to_idx[opt.to_corridor])
                tp_l.append(float(pr))
        veh_turn_src_idx = torch.tensor(ts_l, dtype=torch.long)
        veh_turn_out_corridor_idx = torch.tensor(to_l, dtype=torch.long)
        veh_turn_prob = torch.tensor(tp_l, dtype=torch.float32)

    return {
        "veh_ids": veh_ids,
        "corridor_ids": corridor_ids,
        "cand_triples": tuple(cand_triples),
        "x_v": x_v,
        "x_c": x_c,
        "x_e": x_e,
        "x_cc": x_cc,
        "edge_index": edge_index,
        "edge_relations": edge_relations,
        "p_conflict": p_conflict,
        "p_merge": p_merge,
        "signed_path_gap_m": signed_gap_m,
        "corridor_edge_index": corridor_edge_index,
        "topo_precedence": topo_precedence,
        "speed_evidence_pos": speed_evidence_pos,
        "veh_turn_src_idx": veh_turn_src_idx,
        "veh_turn_out_corridor_idx": veh_turn_out_corridor_idx,
        "veh_turn_prob": veh_turn_prob,
    }


def _select_contiguous_window_indices(
    n_active: int,
    *,
    max_windows_per_episode: int | None,
    window_len_steps: int,
    horizon_steps: int,
) -> list[int]:
    """Select evenly spaced non-overlapping contiguous ``window_len_steps`` segments across an episode."""
    if max_windows_per_episode is None:
        return list(range(n_active))
    if max_windows_per_episode < 1 or window_len_steps < 1:
        raise ConfigValidationError(
            f"Require max_windows_per_episode >= 1 and window_len_steps >= 1, "
            f"got ({max_windows_per_episode}, {window_len_steps})."
        )

    # Prefer placing windows inside [0, n_active - horizon_steps) when long enough so every supervised
    # frame has a full un-truncated future horizon [t, t + H_p].
    usable_end = (
        n_active - horizon_steps
        if n_active >= max_windows_per_episode * window_len_steps + horizon_steps
        else n_active
    )
    if usable_end <= max_windows_per_episode * window_len_steps:
        return list(range(min(n_active, max_windows_per_episode * window_len_steps)))

    max_start = usable_end - window_len_steps
    if max_windows_per_episode == 1:
        starts = [max_start // 2]
    else:
        raw_starts = np.linspace(0, max_start, num=max_windows_per_episode)
        starts = []
        next_free = 0
        for rs in raw_starts:
            st = max(next_free, int(round(float(rs))))
            if st + window_len_steps > usable_end:
                break
            starts.append(st)
            next_free = st + window_len_steps

    indices: list[int] = []
    for st in starts:
        indices.extend(range(st, st + window_len_steps))
    return indices


def collect_single_rollout_episode(
    exp_cfg: ExperimentConfig,
    seed: int,
    split: RolloutSplit,
    output_dir: Path,
    *,
    behavior_policy_override: str | None = None,
    max_windows_per_episode: int | None = None,
    window_len_steps: int = 16,
    adapter_factory: Callable[..., Any] | None = None,
    tracker_factory: Callable[..., Any] | None = None,
    controller_factory: Callable[..., Any] | None = None,
    topology_loader: Callable[..., RoadTopology] | None = None,
    feature_extractor_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Execute one SUMO rollout episode, compute causal future labels, and write a ``.pt`` shard."""
    scenario = exp_cfg.scenario
    validate_rollout_split(scenario, [seed], split)
    update_crash_context(
        command="collect",
        scenario_id=scenario.scenario_id,
        seed=seed,
        split=split,
    )

    if adapter_factory is None:
        validate_sumo_backend(scenario.sumo_backend)

    if feature_extractor_fn is None:
        from .. import dbn as swarm_dbn

        feature_extractor_fn = swarm_dbn.extract_graph_features

    if behavior_policy_override is not None:
        policy_name, exploratory = behavior_policy_override, False
    else:
        policy_name, exploratory = select_behavior_policy(seed)

    update_crash_context(policy_name=policy_name)

    if topology_loader is None:
        from .. import env as swarm_env

        topology = swarm_env.load_or_build_topology(scenario)
    else:
        topology = _bind_and_call(
            topology_loader,
            {"source": scenario, "scenario": scenario, "scenario_cfg": scenario, "config": scenario},
        )
    work_dir = output_dir / f".sumo_work_{scenario.scenario_id}_seed{seed}_pid{os.getpid()}"
    work_dir.mkdir(parents=True, exist_ok=True)

    if adapter_factory is None:
        from .. import env as swarm_env

        adapter = swarm_env.SumoSimulationAdapter(
            scenario,
            control_config=exp_cfg.control,
            policy_name=policy_name,
            seed=seed,
            work_dir=work_dir,
            topology=topology,
        )
    else:
        adapter = _bind_and_call(
            adapter_factory,
            {
                "scenario": scenario,
                "scenario_cfg": scenario,
                "config": scenario,
                "control_config": exp_cfg.control,
                "policy_name": policy_name,
                "topology": topology,
                "seed": seed,
                "backend": scenario.sumo_backend,
                "sumo_backend": scenario.sumo_backend,
                "work_dir": work_dir,
            },
        )
    if tracker_factory is None:
        from .. import env as swarm_env

        tracker = swarm_env.CityBeliefTracker(
            topology=topology,
            observation_config=scenario.observation,
            prediction_horizon_s=exp_cfg.dbn.prediction_horizon_s,
        )
    else:
        tracker = _bind_and_call(
            tracker_factory,
            {
                "topology": topology,
                "observation_config": scenario.observation,
                "obs_cfg": scenario.observation,
                "config": scenario.observation,
                "seed": seed,
            },
        )
    if controller_factory is None:
        from .. import control as swarm_control

        controller = swarm_control.build_policy_controller(
            policy_name=policy_name,
            topology=topology,
            config=exp_cfg,
            checkpoint_path=None,
            seed=seed,
        )
    else:
        controller = _bind_and_call(
            controller_factory,
            {
                "policy_name": policy_name,
                "cfg": exp_cfg,
                "exp_cfg": exp_cfg,
                "config": exp_cfg,
                "topology": topology,
                "checkpoint_path": None,
                "seed": seed,
            },
        )

    exp_rng = np.random.default_rng(seed * 1009 + 17)
    belief_steps: list[SwarmBeliefState] = []
    gt_steps = []
    prev_snaps: list[Any] = []

    tracker.reset()
    controller.reset()

    try:
        obs, gt_rec = _bind_and_call(adapter.reset, {"seed": seed})
        total_steps = scenario.total_steps

        for step_idx in range(total_steps):
            update_crash_context(
                sim_time=obs.sim_time,
                step_idx=step_idx,
                active_veh_count=len(obs.active_veh_ids),
            )
            belief = tracker.observe(obs)
            prev_snap = controller.last_dbn_snapshot
            action = controller.act(obs)
            if exploratory:
                action = _apply_exploratory_perturbation(action, obs, belief, topology, exp_rng)

            # Record full 0.5s trajectory belief & ground-truth steps where vehicles are present
            if obs.sim_time >= scenario.warmup_duration_s and len(belief.vehicles) > 0:
                belief_steps.append(belief)
                gt_steps.append(gt_rec)
                prev_snaps.append(prev_snap)

            tracker.predict(
                dt=obs.dt,
                post_safety_accels=action.post_safety_accels,
                v2x_polled_ids=action.v2x_polls,
                sim_time=obs.sim_time,
            )
            obs, gt_rec = adapter.step(action)
    finally:
        adapter.close()
        if work_dir.exists():
            shutil.rmtree(work_dir)

    if not belief_steps:
        raise EpisodeExecutionError(
            f"Rollout produced zero active post-warmup steps for {scenario.scenario_id!r} seed={seed}.",
            context={"scenario_id": scenario.scenario_id, "seed": seed, "split": split},
        )

    horizon_steps = max(1, int(round(exp_cfg.dbn.prediction_horizon_s / max(scenario.step_length_s, 0.1))))
    selected_indices = _select_contiguous_window_indices(
        len(belief_steps),
        max_windows_per_episode=max_windows_per_episode,
        window_len_steps=window_len_steps,
        horizon_steps=horizon_steps,
    )

    unpacked_steps: list[dict[str, Any]] = []
    for t_idx in selected_indices:
        b_t = belief_steps[t_idx]
        ps_t = prev_snaps[t_idx]
        prev_edge_probs = ps_t.raw_edge_probs if ps_t is not None else {}
        prev_delay_post = ps_t.delay_post if ps_t is not None else {}
        prev_spillback = ps_t.corridor_spillback_pred if ps_t is not None else {}
        raw_feats = _bind_and_call(
            feature_extractor_fn,
            {
                "belief": b_t,
                "topology": topology,
                "dbn_cfg": exp_cfg.dbn,
                "config": exp_cfg.dbn,
                "prev_edge_probs": prev_edge_probs,
                "prev_delay_post": prev_delay_post,
                "prev_spillback_prob": prev_spillback,
                "prev_snapshot": ps_t,
            },
        )
        unpacked_steps.append(_unpack_extracted_features(raw_feats, b_t, topology))

    corridor_ids = tuple(unpacked_steps[0]["corridor_ids"])
    corridor_edge_index = unpacked_steps[0]["corridor_edge_index"].to(dtype=torch.int32)

    serialized_steps: list[dict[str, Any]] = []
    for t_idx, unpacked in zip(selected_indices, unpacked_steps, strict=True):
        b_step = belief_steps[t_idx]
        veh_ids = tuple(unpacked["veh_ids"])
        lbl = compute_step_causal_labels(
            step_idx=t_idx,
            belief_steps=belief_steps,
            gt_steps=gt_steps,
            topology=topology,
            candidate_edges=tuple(unpacked["cand_triples"]),
            corridor_order=corridor_ids,
            veh_order=veh_ids,
            horizon_s=exp_cfg.dbn.prediction_horizon_s,
            delay_threshold_s=exp_cfg.dbn.delay_threshold_s,
        )

        x_v: torch.Tensor = unpacked["x_v"]
        x_c: torch.Tensor = unpacked["x_c"]
        x_e: torch.Tensor = unpacked["x_e"]
        x_cc: torch.Tensor = unpacked["x_cc"]
        edge_index: torch.Tensor = unpacked["edge_index"].to(dtype=torch.int32)

        veh_corr_idx, _, _, _ = _build_corridor_turn_tables(
            b_step, veh_ids, corridor_ids, topology
        )

        serialized_steps.append(
            {
                "sim_time": float(b_step.sim_time),
                "veh_ids": veh_ids,
                "veh_corridor_idx": veh_corr_idx.to(dtype=torch.int32),
                "veh_turn_src_idx": unpacked["veh_turn_src_idx"].to(dtype=torch.int32),
                "veh_turn_out_corridor_idx": unpacked["veh_turn_out_corridor_idx"].to(dtype=torch.int32),
                "veh_turn_prob": unpacked["veh_turn_prob"],
                "topo_precedence": unpacked["topo_precedence"],
                "speed_evidence_pos": unpacked["speed_evidence_pos"],
                "x_veh": x_v,
                "x_corridor": x_c,
                "x_edge": x_e,
                "x_corridor_edge": x_cc,
                "edge_index": edge_index,
                "signed_path_gap_m": unpacked["signed_path_gap_m"],
                "y_delay": torch.from_numpy(lbl.y_delay),
                "tau_delay_s": torch.from_numpy(lbl.tau_delay_s),
                "y_spillback": torch.from_numpy(lbl.y_spillback),
                "y_edge": torch.from_numpy(lbl.y_edge),
                "rank_pairs": torch.from_numpy(lbl.rank_pairs).to(dtype=torch.int32),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_name = f"episode_{scenario.scenario_id}_seed{seed}.pt"
    shard_path = output_dir / shard_name
    tmp_path = output_dir / f".{shard_name}.pid{os.getpid()}.tmp"

    shard_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "scenario_id": scenario.scenario_id,
        "network_id": scenario.network_id,
        "seed": int(seed),
        "behavior_policy": f"{policy_name}{'_exploratory' if exploratory else ''}",
        "dt": float(scenario.step_length_s),
        "corridor_ids": corridor_ids,
        "corridor_edge_index": corridor_edge_index,
        "steps": serialized_steps,
    }
    torch.save(shard_payload, tmp_path)
    os.replace(tmp_path, shard_path)

    pos_delay = sum(int((s["y_delay"] > 0.5).sum().item()) for s in serialized_steps)
    tot_nodes = sum(int(s["y_delay"].numel()) for s in serialized_steps)
    pos_edges = sum(int((s["y_edge"] > 0.5).sum().item()) for s in serialized_steps)
    tot_edges = sum(int(s["y_edge"].numel()) for s in serialized_steps)

    return {
        "shard_file": shard_name,
        "shard_sha256": file_sha256(shard_path),
        "scenario_id": scenario.scenario_id,
        "network_id": scenario.network_id,
        "seed": int(seed),
        "split": split,
        "behavior_policy": shard_payload["behavior_policy"],
        "num_steps": len(serialized_steps),
        "total_vehicle_nodes": tot_nodes,
        "positive_delay_nodes": pos_delay,
        "total_candidate_edges": tot_edges,
        "positive_causal_edges": pos_edges,
    }


def _rollout_worker_task(
    exp_cfg: ExperimentConfig,
    seed: int,
    split: RolloutSplit,
    output_dir_str: str,
    behavior_policy_override: str | None,
    max_windows_per_episode: int | None,
    window_len_steps: int,
) -> dict[str, Any]:
    out_dir = Path(output_dir_str)

    @fail_loud_entrypoint(out_dir)
    def _exec() -> dict[str, Any]:
        return collect_single_rollout_episode(
            exp_cfg=exp_cfg,
            seed=seed,
            split=split,
            output_dir=out_dir,
            behavior_policy_override=behavior_policy_override,
            max_windows_per_episode=max_windows_per_episode,
            window_len_steps=window_len_steps,
        )

    return _exec()


def collect_rollout_dataset(
    scenarios: Sequence[ScenarioConfig],
    seeds: Sequence[int],
    split: RolloutSplit,
    output_dir: Path | str,
    *,
    dbn_cfg: DBNConfig = DBNConfig(),
    max_workers: int = 4,
    behavior_policy_override: str | None = None,
    max_windows_per_episode: int | None = None,
    window_len_steps: int = 16,
    adapter_factory: Callable[..., Any] | None = None,
    tracker_factory: Callable[..., Any] | None = None,
    controller_factory: Callable[..., Any] | None = None,
    topology_loader: Callable[..., RoadTopology] | None = None,
    feature_extractor_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Collect a zero-leakage multi-episode rollout dataset and write ``manifest.json``."""
    if not scenarios:
        raise ConfigValidationError("collect_rollout_dataset requires at least one ScenarioConfig.")
    if max_workers < 1:
        raise ConfigValidationError(f"max_workers must be >= 1, got {max_workers}.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("pipelines.rollout")

    # Validate every (scenario, seeds, split) combination up front before launching workers
    tasks: list[tuple[ExperimentConfig, int]] = []
    clean_seeds_union: list[int] = []
    for idx, scen in enumerate(scenarios):
        validated_seeds = validate_rollout_split(scen, seeds, split)
        if idx == 0:
            clean_seeds_union = list(validated_seeds)

    # Distribute seeds across scenarios (if multiple ID scenarios e.g. nominal/rush/noisy are given,
    # round-robin seeds across scenarios so each seed runs cleanly, or if 1 scenario run all seeds).
    ordered_seeds = list(validate_rollout_split(scenarios[0], seeds, split))
    for s_idx, seed in enumerate(ordered_seeds):
        scen = scenarios[s_idx % len(scenarios)]
        exp_cfg = ExperimentConfig(scenario=scen, dbn=dbn_cfg)
        tasks.append((exp_cfg, seed))

    shard_records: list[dict[str, Any]] = []
    custom_hooks = any(
        h is not None
        for h in (adapter_factory, tracker_factory, controller_factory, topology_loader, feature_extractor_fn)
    )

    if max_workers == 1 or len(tasks) == 1 or custom_hooks:
        for exp_cfg, seed in tasks:
            rec = collect_single_rollout_episode(
                exp_cfg=exp_cfg,
                seed=seed,
                split=split,
                output_dir=out_dir,
                behavior_policy_override=behavior_policy_override,
                max_windows_per_episode=max_windows_per_episode,
                window_len_steps=window_len_steps,
                adapter_factory=adapter_factory,
                tracker_factory=tracker_factory,
                controller_factory=controller_factory,
                topology_loader=topology_loader,
                feature_extractor_fn=feature_extractor_fn,
            )
            shard_records.append(rec)
            logger.info(
                "Collected %s shard %s (steps=%d, pos_edges=%d/%d)",
                split,
                rec["shard_file"],
                rec["num_steps"],
                rec["positive_causal_edges"],
                rec["total_candidate_edges"],
            )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(max_workers, len(tasks)), mp_context=ctx) as pool:
            future_map = {
                pool.submit(
                    _rollout_worker_task,
                    exp_cfg,
                    seed,
                    split,
                    str(out_dir),
                    behavior_policy_override,
                    max_windows_per_episode,
                    window_len_steps,
                ): (exp_cfg.scenario.scenario_id, seed)
                for exp_cfg, seed in tasks
            }
            for fut in as_completed(future_map):
                scen_id, seed = future_map[fut]
                try:
                    rec = fut.result()
                except Exception as exc:
                    for pending in future_map:
                        pending.cancel()
                    logger.error("Rollout worker failed on %s seed=%d: %s", scen_id, seed, exc)
                    raise EpisodeExecutionError(
                        f"Rollout worker failed on scenario={scen_id!r}, seed={seed}: {exc}",
                        context={"scenario_id": scen_id, "seed": seed, "split": split},
                    ) from exc
                shard_records.append(rec)
                logger.info("Collected %s shard %s", split, rec["shard_file"])

    shard_records.sort(key=lambda r: (r["seed"], r["scenario_id"]))
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "scenarios": [s.scenario_id for s in scenarios],
        "scenario_config_sha256s": {s.scenario_id: config_sha256(s) for s in scenarios},
        "dbn_config_sha256": config_sha256(dbn_cfg),
        "seeds": clean_seeds_union,
        "num_episodes": len(shard_records),
        "num_steps_total": sum(int(r["num_steps"]) for r in shard_records),
        "shards": shard_records,
    }
    manifest_path = out_dir / "manifest.json"
    tmp_manifest = out_dir / f".manifest.json.pid{os.getpid()}.tmp"
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)
    return manifest


def load_rollout_shards(
    dataset_dir: Path | str,
    *,
    expected_split: RolloutSplit | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and verify ``manifest.json`` and all ``.pt`` shards via ``torch.load(..., weights_only=True)``."""
    base = Path(dataset_dir)
    manifest_path = base / "manifest.json"
    if not manifest_path.is_file():
        raise ConfigValidationError(
            f"Rollout dataset manifest missing: {manifest_path}",
            context={"dataset_dir": str(base)},
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ConfigValidationError(
            f"Rollout manifest schema_version={manifest.get('schema_version')} != {SCHEMA_VERSION}.",
            context={"manifest_path": str(manifest_path)},
        )
    actual_split = manifest.get("split")
    if expected_split is not None and actual_split != expected_split:
        raise ConfigValidationError(
            f"Dataset split mismatch in {base}: expected {expected_split!r}, found {actual_split!r}.",
            context={"dataset_dir": str(base), "expected_split": expected_split, "actual_split": actual_split},
        )

    # Secondary seed-leak check on manifest load
    seeds = [int(s) for s in manifest.get("seeds", [])]
    if actual_split == "train" and (set(seeds) - set(TRAIN_SEEDS)):
        raise ConfigValidationError(
            "Corrupted/leaked train manifest contains non-TRAIN_SEEDS!",
            context={"seeds": seeds},
        )
    if actual_split == "val" and (set(seeds) - set(VAL_SEEDS)):
        raise ConfigValidationError(
            "Corrupted/leaked val manifest contains non-VAL_SEEDS!",
            context={"seeds": seeds},
        )
    if actual_split in ("eval_id", "eval_ood", "test") and (set(seeds) - set(TEST_SEEDS)):
        raise ConfigValidationError(
            "Corrupted/leaked held-out eval manifest contains non-TEST_SEEDS!",
            context={"seeds": seeds},
        )

    episodes: list[dict[str, Any]] = []
    for rec in manifest.get("shards", []):
        shard_path = base / rec["shard_file"]
        if not shard_path.is_file():
            raise ConfigValidationError(
                f"Rollout shard file missing: {shard_path}",
                context={"shard_file": rec["shard_file"]},
            )
        expected_hash = rec.get("shard_sha256")
        if expected_hash is not None and file_sha256(shard_path) != expected_hash:
            raise ConfigValidationError(
                f"SHA-256 mismatch on rollout shard {shard_path}.",
                context={"shard_path": str(shard_path)},
            )
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        if not isinstance(shard, dict) or "steps" not in shard:
            raise ConfigValidationError(f"Malformed shard structure in {shard_path}.")
        episodes.append(shard)

    if not episodes:
        raise ConfigValidationError(f"Zero rollout episodes loaded from {base}.")
    return manifest, episodes


__all__ = [
    "RolloutSplit",
    "collect_rollout_dataset",
    "collect_single_rollout_episode",
    "load_rollout_shards",
    "select_behavior_policy",
    "validate_rollout_split",
]

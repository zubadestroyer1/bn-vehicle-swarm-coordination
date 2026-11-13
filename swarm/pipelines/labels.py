"""Offline future-horizon causal supervision extractor for SUMO swarm rollout trajectories.

FIREWALL NOTICE:
This module reads privileged :class:`~swarm.contracts.GroundTruthStepRecord` trajectories
**strictly offline** after an episode finishes during dataset creation (or held-out predictive
calibration evaluation). It is NEVER imported or called by any runtime perception, DBN, or
control module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ..contracts import (
    GroundTruthStepRecord,
    RelationType,
    RoadTopology,
    SwarmBeliefState,
    TurnOption,
    VehicleRawTelemetry,
)
from ..logging_utils import (
    ConfigValidationError,
    TopologyIntegrityError,
    require_finite,
    require_shape,
)


@dataclass(frozen=True)
class StepSupervisionLabels:
    """Future-window ``[t, t + H_p]`` supervision targets aligned to active nodes, corridors, and edges."""

    sim_time: float
    veh_ids: tuple[str, ...]
    corridor_ids: tuple[str, ...]
    candidate_edges: tuple[tuple[str, str, int], ...]
    y_delay: np.ndarray          # float32 shape (N_t,) in {0.0, 1.0}
    tau_delay_s: np.ndarray      # float32 shape (N_t,) in [0.0, H_p]
    y_spillback: np.ndarray      # float32 shape (N_C,) in {0.0, 1.0}
    y_edge: np.ndarray           # float32 shape (E_cand,) in {0.0, 1.0}
    rank_pairs: np.ndarray       # int64 shape (M, 2) indices (winner_idx, loser_idx) into veh_ids

    def validate(self) -> None:
        """Enforce strict shape and finiteness invariants across all label arrays."""
        n_v = len(self.veh_ids)
        n_c = len(self.corridor_ids)
        n_e = len(self.candidate_edges)
        require_shape(self.y_delay, (n_v,), "StepSupervisionLabels.y_delay")
        require_shape(self.tau_delay_s, (n_v,), "StepSupervisionLabels.tau_delay_s")
        require_shape(self.y_spillback, (n_c,), "StepSupervisionLabels.y_spillback")
        require_shape(self.y_edge, (n_e,), "StepSupervisionLabels.y_edge")
        require_shape(self.rank_pairs, (None, 2), "StepSupervisionLabels.rank_pairs")
        require_finite(self.y_delay, "StepSupervisionLabels.y_delay")
        require_finite(self.tau_delay_s, "StepSupervisionLabels.tau_delay_s")
        require_finite(self.y_spillback, "StepSupervisionLabels.y_spillback")
        require_finite(self.y_edge, "StepSupervisionLabels.y_edge")


def _raw_corridor_and_pos(
    raw_veh: VehicleRawTelemetry,
    topology: RoadTopology,
) -> tuple[str | None, float]:
    """Map raw SUMO ``road_id`` + ``lane_pos`` onto unified ``(corridor_id, corridor_pos_m)``."""
    if raw_veh.road_id.startswith(":"):
        return None, float(raw_veh.lane_pos)
    cid = topology.edge_to_corridor.get(raw_veh.road_id)
    if cid is None:
        return None, float(raw_veh.lane_pos)
    offset = topology.edge_offset_in_corridor[raw_veh.road_id]
    return cid, float(offset + raw_veh.lane_pos)


def _realized_next_corridor(
    raw_veh: VehicleRawTelemetry,
    current_cid: str | None,
    topology: RoadTopology,
) -> tuple[str | None, TurnOption | None]:
    """Identify vehicle's true next outgoing corridor & matching ``TurnOption`` from ``route_edges``."""
    if current_cid is None:
        return None, None
    spec = topology.corridors.get(current_cid)
    if spec is None or spec.is_exit_fringe:
        return None, None

    for idx in range(max(0, raw_veh.route_index), len(raw_veh.route_edges)):
        edge_id = raw_veh.route_edges[idx]
        next_cid = topology.edge_to_corridor.get(edge_id)
        if next_cid is not None and next_cid != current_cid:
            for opt in spec.turn_options:
                if opt.to_corridor == next_cid:
                    return next_cid, opt
            return next_cid, None
    return None, None


def compute_step_causal_labels(
    step_idx: int,
    belief_steps: Sequence[SwarmBeliefState],
    gt_steps: Sequence[GroundTruthStepRecord],
    topology: RoadTopology,
    candidate_edges: Sequence[tuple[str, str, RelationType | int]],
    *,
    corridor_order: Sequence[str] | None = None,
    veh_order: Sequence[str] | None = None,
    horizon_s: float = 6.0,
    delay_threshold_s: float = 1.5,
    spillback_tail_m: float = 22.0,
    spillback_crawl_mps: float = 1.5,
    jam_spacing_m: float = 7.0,
) -> StepSupervisionLabels:
    """Compute future-window ``[t, t + H_p]`` causal delay, spillback, edge, and precedence labels."""
    if len(belief_steps) != len(gt_steps):
        raise ConfigValidationError(
            f"belief_steps ({len(belief_steps)}) and gt_steps ({len(gt_steps)}) length mismatch."
        )
    if step_idx < 0 or step_idx >= len(belief_steps):
        raise ConfigValidationError(
            f"step_idx={step_idx} out of range [0, {len(belief_steps)})."
        )
    if horizon_s <= 0.0 or delay_threshold_s <= 0.0:
        raise ConfigValidationError("horizon_s and delay_threshold_s must be positive.")

    b_now = belief_steps[step_idx]
    gt_now = gt_steps[step_idx]
    t0 = float(b_now.sim_time)
    t_limit = t0 + float(horizon_s) + 1e-6

    # Collect window indices w in [t, t + H_p]
    win_indices: list[int] = []
    for w in range(step_idx, len(gt_steps)):
        if float(gt_steps[w].sim_time) <= t_limit:
            win_indices.append(w)
        else:
            break
    if not win_indices:
        win_indices = [step_idx]

    ordered_corridors = tuple(corridor_order) if corridor_order is not None else tuple(sorted(topology.corridors))
    for cid in ordered_corridors:
        if cid not in topology.corridors:
            raise TopologyIntegrityError(f"Unknown corridor_id {cid!r} in compute_step_causal_labels.")

    ordered_vehs = tuple(veh_order) if veh_order is not None else tuple(sorted(b_now.vehicles))
    veh_to_idx = {vid: idx for idx, vid in enumerate(ordered_vehs)}

    # Index raw ground-truth telemetry by (w, veh_id)
    gt_by_step_veh: list[dict[str, VehicleRawTelemetry]] = [
        {rv.veh_id: rv for rv in gt_steps[w].raw_vehicles} for w in win_indices
    ]
    now_raw_map = gt_by_step_veh[0]

    # -------------------------------------------------------------------------
    # 1. Per-Vehicle Continuous Delay tau_delay_s & Binary Delay Onset y_delay
    # -------------------------------------------------------------------------
    n_v = len(ordered_vehs)
    tau_delay_s = np.zeros((n_v,), dtype=np.float32)
    y_delay = np.zeros((n_v,), dtype=np.float32)

    # Track junction entry timestamps per vehicle inside [t, t + H_p] for conflict precedence
    junction_entry_time: dict[str, tuple[str, float]] = {}
    decel_triggered: dict[str, bool] = {}
    min_speed_in_win: dict[str, float] = {}

    for idx, vid in enumerate(ordered_vehs):
        vb = b_now.vehicles.get(vid)
        if vb is None:
            raise ConfigValidationError(f"Vehicle {vid!r} not present in belief_steps[{step_idx}].")
        corr_spec = topology.corridors.get(vb.corridor_id)
        v_free = corr_spec.speed_limit_mps if corr_spec is not None else max(vb.mean_v_mps, 13.89)

        rv_start = now_raw_map.get(vid)
        if rv_start is None:
            # Vehicle exited on this boundary tick -> zero future delay
            tau_delay_s[idx] = 0.0
            y_delay[idx] = 0.0
            decel_triggered[vid] = False
            min_speed_in_win[vid] = float(vb.mean_v_mps)
            continue

        last_rv = rv_start
        last_sim_t = float(gt_steps[win_indices[0]].sim_time)
        prev_xy = rv_start.center_xy
        dist_traveled = 0.0
        had_decel = rv_start.accel < -0.45
        v_min = float(rv_start.speed)

        for local_k, w in enumerate(win_indices):
            rv_w = gt_by_step_veh[local_k].get(vid)
            if rv_w is None:
                break
            sim_w = float(gt_steps[w].sim_time)
            if local_k > 0:
                step_dist = math.hypot(rv_w.center_xy[0] - prev_xy[0], rv_w.center_xy[1] - prev_xy[1])
                dist_traveled += step_dist
            prev_xy = rv_w.center_xy
            last_rv = rv_w
            last_sim_t = sim_w
            if rv_w.accel < -0.45:
                had_decel = True
            if rv_w.speed < v_min:
                v_min = float(rv_w.speed)
            if rv_w.road_id.startswith(":") and vid not in junction_entry_time:
                # Extract junction ID from SUMO internal edge `:J_id_...`
                j_guess = vb.in_junction_id or (corr_spec.to_junction if corr_spec is not None else "")
                junction_entry_time[vid] = (j_guess, sim_w)

        decel_triggered[vid] = had_decel
        min_speed_in_win[vid] = v_min

        dt_eff = max(0.0, last_sim_t - float(gt_steps[win_indices[0]].sim_time))
        delta_timeloss = max(0.0, float(last_rv.time_loss) - float(rv_start.time_loss))
        kinematic_loss = max(0.0, dt_eff - dist_traveled / max(v_free, 1.0))
        realized_delay = min(float(horizon_s), max(delta_timeloss, kinematic_loss))
        tau_delay_s[idx] = float(realized_delay)
        y_delay[idx] = 1.0 if realized_delay >= delay_threshold_s else 0.0

    # -------------------------------------------------------------------------
    # 2. Unified Corridor Spillback Labels y_spillback (`#13` §2)
    # -------------------------------------------------------------------------
    n_c = len(ordered_corridors)
    y_spillback = np.zeros((n_c,), dtype=np.float32)
    corr_to_cidx = {cid: i for i, cid in enumerate(ordered_corridors)}

    # Also record which vehicles occupy upstream tail (<= 25m, speed < 2.0 m/s) of each corridor in window
    corridor_tail_halted_vehs: dict[str, set[str]] = {cid: set() for cid in ordered_corridors}

    for local_k, w in enumerate(win_indices):
        b_w = belief_steps[w]
        counts_by_corr: dict[str, int] = {cid: 0 for cid in ordered_corridors}
        speeds_by_corr: dict[str, list[float]] = {cid: [] for cid in ordered_corridors}

        for rv in gt_steps[w].raw_vehicles:
            cid, cpos = _raw_corridor_and_pos(rv, topology)
            if cid is None or cid not in corr_to_cidx:
                continue
            counts_by_corr[cid] += 1
            speeds_by_corr[cid].append(float(rv.speed))
            if cpos <= spillback_tail_m and rv.speed < spillback_crawl_mps:
                y_spillback[corr_to_cidx[cid]] = 1.0
            if cpos <= 25.0 and rv.speed < 2.0:
                corridor_tail_halted_vehs[cid].add(rv.veh_id)

        for cid, cidx in corr_to_cidx.items():
            if y_spillback[cidx] >= 0.5:
                continue
            cnt = counts_by_corr[cid]
            if cnt == 0:
                continue
            cap = topology.effective_capacity(cid, b_w.open_lanes_by_edge, jam_spacing_m=jam_spacing_m)
            mean_v = sum(speeds_by_corr[cid]) / len(speeds_by_corr[cid])
            if cnt >= max(2, int(math.ceil(0.85 * cap))) and mean_v < 2.5:
                y_spillback[cidx] = 1.0

    # -------------------------------------------------------------------------
    # 3. Directed Causal Edge Labels y_edge & Precedence Pairs (j > i)
    # -------------------------------------------------------------------------
    n_e = len(candidate_edges)
    y_edge = np.zeros((n_e,), dtype=np.float32)
    normalized_cands: list[tuple[str, str, int]] = []
    positive_directed: set[tuple[int, int]] = set()

    for e_idx, (src_id, dst_id, rel_raw) in enumerate(candidate_edges):
        rel = int(rel_raw)
        normalized_cands.append((src_id, dst_id, rel))
        if src_id not in veh_to_idx or dst_id not in veh_to_idx or src_id == dst_id:
            continue

        src_i = veh_to_idx[src_id]
        dst_i = veh_to_idx[dst_id]
        rv_src = now_raw_map.get(src_id)
        rv_dst = now_raw_map.get(dst_id)
        vb_src = b_now.vehicles[src_id]
        vb_dst = b_now.vehicles[dst_id]

        if rv_src is None or rv_dst is None:
            continue

        cid_src, pos_src = _raw_corridor_and_pos(rv_src, topology)
        cid_dst, pos_dst = _raw_corridor_and_pos(rv_dst, topology)
        is_causal = False

        if rel == int(RelationType.SAME_CORRIDOR):
            if cid_src is not None and cid_src == cid_dst and pos_src > pos_dst:
                gap_m = pos_src - pos_dst - 0.5 * (rv_src.length + rv_dst.length)
                same_or_adj_lane = abs(rv_src.lane_index - rv_dst.lane_index) <= 1
                headway_thresh = max(14.0, 2.5 * max(float(rv_dst.speed), 1.0))
                spec = topology.corridors[cid_src]
                dst_impeded = (
                    decel_triggered.get(dst_id, False)
                    or min_speed_in_win.get(dst_id, rv_dst.speed) < 0.75 * spec.speed_limit_mps
                    or tau_delay_s[dst_i] >= 0.75
                )
                if same_or_adj_lane and gap_m <= headway_thresh and dst_impeded:
                    is_causal = True

        elif rel == int(RelationType.SAME_JUNCTION_CONFLICT):
            spec_src = topology.corridors.get(cid_src) if cid_src is not None else None
            spec_dst = topology.corridors.get(cid_dst) if cid_dst is not None else None
            j_src = vb_src.in_junction_id or (spec_src.to_junction if spec_src is not None else None)
            j_dst = vb_dst.in_junction_id or (spec_dst.to_junction if spec_dst is not None else None)
            if j_src is not None and j_src == j_dst and j_src in topology.junctions:
                out_src, opt_src = (
                    _realized_next_corridor(rv_src, cid_src, topology)
                    if spec_src is not None and spec_src.to_junction == j_src
                    else (cid_src, None)
                )
                out_dst, opt_dst = (
                    _realized_next_corridor(rv_dst, cid_dst, topology)
                    if spec_dst is not None and spec_dst.to_junction == j_src
                    else (cid_dst, None)
                )
                paths_conflict = False
                if opt_src is not None and opt_dst is not None:
                    paths_conflict = topology.turn_conflict(j_src, opt_src, opt_dst) or (out_src == out_dst)
                elif out_src is not None and out_src == out_dst:
                    paths_conflict = True
                elif rv_src.road_id.startswith(":") or vb_src.in_junction_id == j_src:
                    paths_conflict = True

                if paths_conflict:
                    ent_src = junction_entry_time.get(src_id, (j_src, t0 + vb_src.free_eta_s))[1]
                    ent_dst = junction_entry_time.get(dst_id, (j_dst, t0 + vb_dst.free_eta_s))[1]
                    if ent_src <= ent_dst + 0.25 and abs(ent_dst - ent_src) <= 4.0:
                        if decel_triggered.get(dst_id, False) or tau_delay_s[dst_i] >= 0.6 or vb_dst.free_eta_s <= 8.0:
                            is_causal = True

        elif rel == int(RelationType.DOWNSTREAM_SPILLBACK):
            out_dst, _ = _realized_next_corridor(rv_dst, cid_dst, topology)
            effective_targets = {out_dst} if out_dst is not None else set(
                opt.to_corridor for opt in topology.corridors[vb_dst.corridor_id].turn_options
            )
            if cid_src in effective_targets and src_id in corridor_tail_halted_vehs.get(cid_src, set()):
                if vb_dst.free_eta_s <= 10.0 and (decel_triggered.get(dst_id, False) or tau_delay_s[dst_i] >= 0.75):
                    is_causal = True

        elif rel == int(RelationType.UPSTREAM_FEEDER):
            out_src, _ = _realized_next_corridor(rv_src, cid_src, topology)
            if out_src is not None and out_src == cid_dst:
                if vb_src.free_eta_s <= 6.0 and pos_dst <= 45.0:
                    is_causal = True

        if is_causal:
            y_edge[e_idx] = 1.0
            positive_directed.add((src_i, dst_i))

    # Strict asymmetric ordering pairs (j > i) where y_{j->i}=1 and y_{i->j}=0
    rank_list: list[tuple[int, int]] = [
        (j_i, i_i) for (j_i, i_i) in sorted(positive_directed) if (i_i, j_i) not in positive_directed
    ]
    rank_pairs = (
        np.asarray(rank_list, dtype=np.int64)
        if rank_list
        else np.zeros((0, 2), dtype=np.int64)
    )

    out = StepSupervisionLabels(
        sim_time=t0,
        veh_ids=ordered_vehs,
        corridor_ids=ordered_corridors,
        candidate_edges=tuple(normalized_cands),
        y_delay=y_delay,
        tau_delay_s=tau_delay_s,
        y_spillback=y_spillback,
        y_edge=y_edge,
        rank_pairs=rank_pairs,
    )
    out.validate()
    return out


def compute_episode_causal_labels(
    belief_steps: Sequence[SwarmBeliefState],
    gt_steps: Sequence[GroundTruthStepRecord],
    topology: RoadTopology,
    candidate_edges_by_step: Sequence[Sequence[tuple[str, str, RelationType | int]]],
    *,
    corridor_order: Sequence[str] | None = None,
    veh_orders_by_step: Sequence[Sequence[str]] | None = None,
    horizon_s: float = 6.0,
    delay_threshold_s: float = 1.5,
    spillback_tail_m: float = 22.0,
    spillback_crawl_mps: float = 1.5,
) -> list[StepSupervisionLabels]:
    """Compute :class:`StepSupervisionLabels` across an entire recorded rollout episode."""
    n_steps = len(belief_steps)
    if len(gt_steps) != n_steps or len(candidate_edges_by_step) != n_steps:
        raise ConfigValidationError(
            f"Episode step count mismatch: belief={n_steps}, gt={len(gt_steps)}, "
            f"cands={len(candidate_edges_by_step)}."
        )
    if veh_orders_by_step is not None and len(veh_orders_by_step) != n_steps:
        raise ConfigValidationError("veh_orders_by_step length does not match belief_steps.")

    ordered_corridors = tuple(corridor_order) if corridor_order is not None else tuple(sorted(topology.corridors))
    return [
        compute_step_causal_labels(
            step_idx=t,
            belief_steps=belief_steps,
            gt_steps=gt_steps,
            topology=topology,
            candidate_edges=candidate_edges_by_step[t],
            corridor_order=ordered_corridors,
            veh_order=veh_orders_by_step[t] if veh_orders_by_step is not None else None,
            horizon_s=horizon_s,
            delay_threshold_s=delay_threshold_s,
            spillback_tail_m=spillback_tail_m,
            spillback_crawl_mps=spillback_crawl_mps,
        )
        for t in range(n_steps)
    ]


__all__ = [
    "StepSupervisionLabels",
    "compute_episode_causal_labels",
    "compute_step_causal_labels",
]

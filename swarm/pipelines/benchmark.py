"""Warmed CPU DBN structure-update & inference latency scaling benchmark across N = 16 ... 256.

Benchmarks online per-step latency (structure scoring + DAG pruning + Sum-Product LBP inference)
on CPU across swarm sizes ``N in {16, 32, 64, 128, 256}`` for:
1. ``neural_dbn`` (``STGraphTransformerDBN`` + division-free prefix-suffix LBP in ``inference_mode``)
2. ``online_bic_dbn`` (candidate-pruned sliding-window empirical BIC + LBP)
3. ``heuristic_radius_dbn`` (kinematic radius/conflict heuristic DBN + LBP)
4. ``unconstrained_bic`` (dense ``O(N^2)`` all-pairs sliding-window BIC reference + LBP)

Outputs ``scaling.json`` directly compatible with :func:`swarm.viz.plots_2d.plot_latency_scaling`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import inspect
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from ..config import DBNConfig, ModelPresetName, get_model_preset
from ..contracts import (
    CorridorBeliefStats,
    CorridorSpec,
    DBNStateSnapshot,
    JunctionSpec,
    RoadTopology,
    SCHEMA_VERSION,
    SwarmBeliefState,
    TurnOption,
    VehicleBelief,
)
from ..logging_utils import (
    CheckpointIntegrityError,
    ConfigValidationError,
    get_logger,
    require_finite,
)

DEFAULT_SWARM_SIZES: tuple[int, ...] = (16, 32, 64, 128, 256)
DEFAULT_BENCHMARK_METHODS: tuple[str, ...] = (
    "neural_dbn",
    "online_bic_dbn",
    "heuristic_radius_dbn",
    "unconstrained_bic",
)


def _bind_and_call(fn: Callable[..., Any], candidate_kwargs: Mapping[str, Any]) -> Any:
    sig = inspect.signature(fn)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return fn(**candidate_kwargs)
    bound = {k: v for k, v in candidate_kwargs.items() if k in params}
    return fn(**bound)


def build_benchmark_grid_topology(grid_dim: int = 4, block_length_m: float = 150.0) -> RoadTopology:
    """Construct a valid Boundary-Lock-A-compliant synthetic urban grid :class:`RoadTopology`
    for standalone CPU scaling benchmarks when no SUMO ``.net.xml`` is supplied."""
    if grid_dim < 2:
        raise ConfigValidationError(f"grid_dim must be >= 2, got {grid_dim}.")

    corridors: dict[str, CorridorSpec] = {}
    edge_to_corridor: dict[str, str] = {}
    edge_offset: dict[str, float] = {}
    junctions_coords: dict[str, tuple[float, float]] = {}

    # Interior junctions J_r_c for r, c in [0 .. grid_dim-1]
    for r in range(grid_dim):
        for c in range(grid_dim):
            junctions_coords[f"J_{r}_{c}"] = (float(c * block_length_m), float(r * block_length_m))

    # Perimeter fringe junctions
    for r in range(grid_dim):
        junctions_coords[f"F_W_{r}"] = (-block_length_m, float(r * block_length_m))
        junctions_coords[f"F_E_{r}"] = (float(grid_dim * block_length_m), float(r * block_length_m))
    for c in range(grid_dim):
        junctions_coords[f"F_S_{c}"] = (float(c * block_length_m), -block_length_m)
        junctions_coords[f"F_N_{c}"] = (float(c * block_length_m), float(grid_dim * block_length_m))

    raw_corridors: list[tuple[str, str, str, tuple[float, float], bool, bool]] = []

    def _add_pair(u: str, v: str) -> None:
        ux, uy = junctions_coords[u]
        vx, vy = junctions_coords[v]
        ang = math.atan2(vy - uy, vx - ux)
        cid = f"C_{u}_to_{v}"
        raw_corridors.append((cid, u, v, (math.cos(ang), math.sin(ang)), u.startswith("F_"), v.startswith("F_")))

    for r in range(grid_dim):
        for c in range(grid_dim - 1):
            _add_pair(f"J_{r}_{c}", f"J_{r}_{c + 1}")
            _add_pair(f"J_{r}_{c + 1}", f"J_{r}_{c}")
        _add_pair(f"F_W_{r}", f"J_{r}_0")
        _add_pair(f"J_{r}_0", f"F_W_{r}")
        _add_pair(f"J_{r}_{grid_dim - 1}", f"F_E_{r}")
        _add_pair(f"F_E_{r}", f"J_{r}_{grid_dim - 1}")

    for c in range(grid_dim):
        for r in range(grid_dim - 1):
            _add_pair(f"J_{r}_{c}", f"J_{r + 1}_{c}")
            _add_pair(f"J_{r + 1}_{c}", f"J_{r}_{c}")
        _add_pair(f"F_S_{c}", f"J_0_{c}")
        _add_pair(f"J_0_{c}", f"F_S_{c}")
        _add_pair(f"J_{grid_dim - 1}_{c}", f"F_N_{c}")
        _add_pair(f"F_N_{c}", f"J_{grid_dim - 1}_{c}")

    inc_by_junc: dict[str, list[tuple[str, tuple[float, float], bool]]] = {j: [] for j in junctions_coords}
    out_by_junc: dict[str, list[tuple[str, tuple[float, float], bool, str]]] = {j: [] for j in junctions_coords}
    for cid, u, v, dvec, is_in, is_out in raw_corridors:
        inc_by_junc[v].append((cid, dvec, is_in))
        out_by_junc[u].append((cid, dvec, is_out, v))

    # Assign per-junction turn options & realistic crossing/merging foe matrices
    junc_specs: dict[str, JunctionSpec] = {}
    turn_map_by_cid: dict[str, tuple[TurnOption, ...]] = {}

    for jid, xy in junctions_coords.items():
        if jid.startswith("F_"):
            inc_cids = tuple(x[0] for x in inc_by_junc[jid])
            out_cids = tuple(x[0] for x in out_by_junc[jid])
            for cid, _, _ in inc_by_junc[jid]:
                turn_map_by_cid[cid] = ()
            junc_specs[jid] = JunctionSpec(
                junction_id=jid,
                tls_id=None,
                coord_xy=xy,
                incoming_corridors=inc_cids,
                outgoing_corridors=out_cids,
                num_tls_links=0,
                tls_link_foe_matrix=np.zeros((0, 0), dtype=bool),
                req_foe_matrix=np.zeros((0, 0), dtype=bool),
            )
            continue

        # Interior 4-way signalized junction
        tls_id = f"TLS_{jid}"
        all_turns: list[tuple[str, str, tuple[float, float], tuple[float, float], float]] = []
        inc_list = sorted(inc_by_junc[jid], key=lambda t: t[0])
        out_list = sorted(out_by_junc[jid], key=lambda t: t[0])

        per_inc_turns: dict[str, list[tuple[str, float, int]]] = {}
        for icid, ivec, _ in inc_list:
            sub: list[tuple[str, float, tuple[float, float]]] = []
            for ocid, ovec, _, ov in out_list:
                # Skip U-turns back to reverse of same pair
                dot = ivec[0] * ovec[0] + ivec[1] * ovec[1]
                cross = ivec[0] * ovec[1] - ivec[1] * ovec[0]
                if dot < -0.95:
                    continue
                ang = math.atan2(cross, dot)
                sub.append((ocid, ang, ovec))
            # Sort by signed CCW turn angle descending (left > thru > right)
            sub.sort(key=lambda x: x[1], reverse=True)
            per_inc_turns[icid] = []
            for ocid, ang, ovec in sub:
                g_idx = len(all_turns)
                all_turns.append((icid, ocid, ivec, ovec, ang))
                per_inc_turns[icid].append((ocid, ang, g_idx))

        n_links = len(all_turns)
        foe = np.zeros((n_links, n_links), dtype=bool)
        for a_i in range(n_links):
            ic_a, oc_a, iv_a, _, ang_a = all_turns[a_i]
            for b_i in range(a_i + 1, n_links):
                ic_b, oc_b, iv_b, _, ang_b = all_turns[b_i]
                if ic_a == ic_b:
                    continue
                if oc_a == oc_b:
                    # Merging into same outgoing corridor conflicts
                    foe[a_i, b_i] = foe[b_i, a_i] = True
                    continue
                dot_in = iv_a[0] * iv_b[0] + iv_a[1] * iv_b[1]
                if abs(dot_in) < 0.5:
                    # Perpendicular approaches conflict unless both are right turns into distinct corridors
                    if not (ang_a < -0.5 and ang_b < -0.5):
                        foe[a_i, b_i] = foe[b_i, a_i] = True
                elif dot_in < -0.5:
                    # Opposing approaches conflict if either turns left across oncoming thru/right
                    if ang_a > 0.35 or ang_b > 0.35:
                        foe[a_i, b_i] = foe[b_i, a_i] = True

        for icid, entries in per_inc_turns.items():
            opts: list[TurnOption] = []
            for t_idx, (ocid, ang, g_idx) in enumerate(entries):
                opts.append(
                    TurnOption(
                        turn_idx=t_idx,
                        from_edge=f"{icid}_e0",
                        to_corridor=ocid,
                        to_edge=f"{ocid}_e0",
                        angle_rad=ang,
                        tls_id=tls_id,
                        tls_link_indices=(g_idx,),
                        junction_req_indices=(g_idx,),
                    )
                )
            turn_map_by_cid[icid] = tuple(opts)

        junc_specs[jid] = JunctionSpec(
            junction_id=jid,
            tls_id=tls_id,
            coord_xy=xy,
            incoming_corridors=tuple(x[0] for x in inc_list),
            outgoing_corridors=tuple(x[0] for x in out_list),
            num_tls_links=n_links,
            tls_link_foe_matrix=foe,
            req_foe_matrix=foe.copy(),
        )

    succ: dict[str, tuple[str, ...]] = {}
    pred_lists: dict[str, list[str]] = {cid: [] for cid, *_ in raw_corridors}

    for cid, u, v, _, is_in, is_out in raw_corridors:
        eid = f"{cid}_e0"
        t_opts = turn_map_by_cid[cid]
        corridors[cid] = CorridorSpec(
            corridor_id=cid,
            edges=(eid,),
            edge_offsets_m=(0.0,),
            from_junction=u,
            to_junction=v,
            length_m=block_length_m,
            base_lanes=2,
            speed_limit_mps=13.9,
            turn_options=t_opts,
            is_entry_fringe=is_in,
            is_exit_fringe=is_out,
        )
        edge_to_corridor[eid] = cid
        edge_offset[eid] = 0.0
        s_targets = tuple(dict.fromkeys(opt.to_corridor for opt in t_opts))
        succ[cid] = s_targets
        for tc in s_targets:
            pred_lists[tc].append(cid)

    pred = {cid: tuple(sorted(set(plist))) for cid, plist in pred_lists.items()}

    return RoadTopology(
        network_id=f"bench_grid{grid_dim}x{grid_dim}",
        net_xml_path=f"synthetic_grid{grid_dim}x{grid_dim}.net.xml",
        poly_xml_path=None,
        corridors=corridors,
        edge_to_corridor=edge_to_corridor,
        edge_offset_in_corridor=edge_offset,
        junctions=junc_specs,
        successor_corridors=succ,
        predecessor_corridors=pred,
    )


def _generate_belief_sequence(
    topology: RoadTopology,
    n_vehicles: int,
    num_steps: int,
    *,
    seed: int,
    dt: float = 0.5,
) -> list[SwarmBeliefState]:
    """Generate a realistic kinematics-consistent sequence of ``SwarmBeliefState`` ticks for size ``N``."""
    rng = np.random.default_rng(seed)
    non_exit_cids = sorted(cid for cid, spec in topology.corridors.items() if not spec.is_exit_fringe)
    if not non_exit_cids:
        non_exit_cids = sorted(topology.corridors)

    veh_ids = [f"veh_{idx:04d}" for idx in range(n_vehicles)]
    assigned_cid = [non_exit_cids[idx % len(non_exit_cids)] for idx in range(n_vehicles)]
    s_pos = np.array(
        [
            float(rng.uniform(8.0, max(12.0, topology.corridors[c].length_m - 5.0)))
            for c in assigned_cid
        ],
        dtype=np.float64,
    )
    speeds = rng.uniform(1.5, 13.0, size=n_vehicles).astype(np.float64)
    turn_prefs: list[tuple[float, ...]] = []
    for cid in assigned_cid:
        k_t = len(topology.corridors[cid].turn_options)
        if k_t == 0:
            turn_prefs.append(())
        else:
            raw_p = rng.dirichlet(np.ones(k_t, dtype=np.float64) * 0.8)
            turn_prefs.append(tuple(float(x) for x in raw_p))

    tls_states: dict[str, str] = {}
    for jspec in topology.junctions.values():
        if jspec.tls_id is not None and jspec.num_tls_links > 0:
            half = max(1, jspec.num_tls_links // 2)
            tls_states[jspec.tls_id] = "G" * half + "r" * (jspec.num_tls_links - half)

    open_lanes = {eid: spec.base_lanes for cid, spec in topology.corridors.items() for eid in spec.edges}

    sequence: list[SwarmBeliefState] = []
    for step_idx in range(num_steps):
        sim_t = float(step_idx * dt)
        vehs: dict[str, VehicleBelief] = {}
        by_corr: dict[str, list[VehicleBelief]] = {cid: [] for cid in topology.corridors}

        # Advance vehicles smoothly with realistic stochastic fluctuations & periodic platoon waves
        accel_wave = rng.normal(0.0, 0.9, size=n_vehicles)
        speeds = np.clip(speeds + accel_wave * dt, 0.0, 14.5)
        s_pos = s_pos + speeds * dt

        for i, vid in enumerate(veh_ids):
            cid = assigned_cid[i]
            spec = topology.corridors[cid]
            if s_pos[i] >= spec.length_m - 2.0 and spec.turn_options:
                # Transition to highest-probability non-exit successor if possible
                next_cid = spec.turn_options[int(np.argmax(turn_prefs[i]))].to_corridor
                if topology.corridors[next_cid].is_exit_fringe:
                    next_cid = non_exit_cids[(i + step_idx) % len(non_exit_cids)]
                assigned_cid[i] = next_cid
                cid = next_cid
                spec = topology.corridors[cid]
                s_pos[i] = float(rng.uniform(2.0, 15.0))
                k_new = len(spec.turn_options)
                turn_prefs[i] = (
                    tuple(float(x) for x in rng.dirichlet(np.ones(k_new) * 0.8)) if k_new > 0 else ()
                )

            rem_d = max(0.0, spec.length_m - float(s_pos[i]))
            free_eta = rem_d / max(float(speeds[i]), 3.5)
            pos_var = float(rng.uniform(0.6, 2.5))
            vel_var = float(rng.uniform(0.15, 0.8))
            cov = np.array([[pos_var, 0.05], [0.05, vel_var]], dtype=np.float64)
            tail_b = float(
                1.0 / (1.0 + math.exp(-((22.0 - float(s_pos[i])) / 4.0)))
                * (1.0 / (1.0 + math.exp(-((2.5 - float(speeds[i])) / 0.8))))
            )
            obs_age = float((step_idx + i) % 3) * dt if rng.random() < 0.15 else 0.0

            vb = VehicleBelief(
                veh_id=vid,
                entered_at=0.0,
                corridor_id=cid,
                edge_id=spec.edges[0],
                in_junction_id=None,
                lane_index=i % spec.base_lanes,
                length_m=4.5,
                width_m=1.85,
                mean_s_m=float(np.clip(s_pos[i], 0.0, spec.length_m)),
                mean_v_mps=float(speeds[i]),
                cov_2x2=cov,
                accel_est_mps2=float(accel_wave[i]),
                obs_age_s=obs_age,
                wait_time_s=float(step_idx * dt * 0.2) if speeds[i] < 1.0 else 0.0,
                turn_probs=turn_prefs[i],
                free_eta_s=float(free_eta),
                eta_std_s=0.45,
                tail_blockage_score=tail_b if not spec.is_entry_fringe else 0.0,
                leader_id=None,
                noisy_leader_gap_m=None,
                noisy_leader_speed_mps=None,
                last_v2x_poll_time=-10.0,
            )
            vehs[vid] = vb
            by_corr[cid].append(vb)

        inflow_acc: dict[str, float] = {cid: 0.0 for cid in topology.corridors}
        for vb in vehs.values():
            spec = topology.corridors[vb.corridor_id]
            if not spec.is_exit_fringe and vb.free_eta_s <= 6.0:
                for opt, pr in zip(spec.turn_options, vb.turn_probs, strict=True):
                    inflow_acc[opt.to_corridor] += float(pr)

        corr_stats: dict[str, CorridorBeliefStats] = {}
        for cid, spec in topology.corridors.items():
            cvs = by_corr[cid]
            cap = topology.effective_capacity(cid, open_lanes)
            cnt = len(cvs)
            occ = min(1.0, cnt / float(max(1, cap)))
            surv = 1.0
            for v_item in cvs:
                surv *= 1.0 - v_item.tail_blockage_score
            t_prob = 0.0 if spec.is_entry_fringe else float(1.0 - surv)
            m_spd = float(sum(v_item.mean_v_mps for v_item in cvs) / cnt) if cnt > 0 else spec.speed_limit_mps
            corr_stats[cid] = CorridorBeliefStats(
                corridor_id=cid,
                active_count=cnt,
                effective_lanes=spec.base_lanes,
                storage_capacity=cap,
                occupancy_ratio=occ,
                tail_blockage_prob=t_prob,
                expected_inflow=inflow_acc[cid],
                mean_speed_mps=m_spd,
            )

        sequence.append(
            SwarmBeliefState(
                sim_time=sim_t,
                dt=dt,
                vehicles=vehs,
                corridors=corr_stats,
                open_lanes_by_edge=dict(open_lanes),
                tls_states=dict(tls_states),
            )
        )

    return sequence


class _UnconstrainedAllPairsBICBenchmarkUpdater:
    """Reference ``O(N^2)`` unconstrained all-pairs sliding-window BIC + LBP benchmark updater.

    Evaluates empirical pairwise log-likelihood gains across ALL ``N * (N - 1) / 2`` ordered
    precedence-ordered pairs without ``O(N)`` spatial-corridor candidate bucketing, quantifying
    the exact quadratic scaling wall avoided by ``STGraphTransformerDBN``.
    """

    def __init__(self, dbn_cfg: DBNConfig) -> None:
        self._cfg = dbn_cfg
        self._history: deque[dict[str, float]] = deque(maxlen=dbn_cfg.bic_window_steps)

    def reset(self) -> None:
        self._history.clear()

    def step_latency_and_edges(self, belief: SwarmBeliefState, topology: RoadTopology) -> tuple[float, int]:
        from ..dbn.cpd import gate_and_prune_edges, generalized_noisy_or_base_survival
        from ..dbn.inference import multiplicative_sum_product_lbp

        t0 = time.perf_counter()
        veh_ids = tuple(sorted(belief.vehicles))
        n_v = len(veh_ids)

        # Record current kinematic delay proxy observation into sliding window
        curr_slow: dict[str, float] = {}
        prec_list: list[float] = []
        ev_list: list[float] = []
        for vid in veh_ids:
            vb = belief.vehicles[vid]
            vmax = topology.corridors[vb.corridor_id].speed_limit_mps
            is_delayed = 1.0 if (vb.mean_v_mps < 0.45 * vmax and vb.obs_age_s <= 1.0) else (
                0.0 if vb.obs_age_s <= 1.0 else -1.0
            )
            curr_slow[vid] = is_delayed
            prec_list.append(-vb.free_eta_s + 15.0 * vb.tail_blockage_score)
            sig = 1.0 / (1.0 + math.exp(-((0.45 * vmax - vb.mean_v_mps) / 1.2)))
            ev_list.append(0.5 + 0.42 * math.exp(-vb.obs_age_s / 3.0) * (2.0 * sig - 1.0))

        prev_hist = list(self._history)
        self._history.append(curr_slow)

        # Dense all-pairs precedence ordering & sliding-window sufficiency scan O(W * N^2)
        prec_np = np.asarray(prec_list, dtype=np.float64)
        w_len = len(prev_hist)
        src_edges: list[int] = []
        dst_edges: list[int] = []
        omegas: list[float] = []

        if w_len >= 2 and n_v >= 2:
            # Build observation matrix [W, N] (-1 denotes missing due to dropout/unborn)
            obs_mat = np.full((w_len, n_v), -1.0, dtype=np.float32)
            for t_i, h_dict in enumerate(prev_hist):
                for v_i, vid in enumerate(veh_ids):
                    val = h_dict.get(vid, -1.0)
                    if val >= 0.0:
                        obs_mat[t_i, v_i] = val

            prev_valid = obs_mat[:-1] >= 0.0  # [W-1, N]
            prev_one = obs_mat[:-1] > 0.5
            next_valid = obs_mat[1:] >= 0.0   # [W-1, N]
            next_one = obs_mat[1:] > 0.5

            # Evaluate all N*(N-1) ordered pairs respecting precedence order
            for j in range(n_v):
                vj_val = prev_valid[:, j]
                vj_one = prev_one[:, j]
                for i in range(n_v):
                    if i == j:
                        continue
                    if not (prec_np[j] > prec_np[i] or (prec_np[j] == prec_np[i] and j < i)):
                        continue
                    joint_mask = vj_val & next_valid[:, i]
                    n_ji = int(np.count_nonzero(joint_mask))
                    if n_ji < self._cfg.bic_min_pair_samples:
                        continue
                    yi = next_one[joint_mask, i]
                    xj = vj_one[joint_mask]
                    n1 = int(np.count_nonzero(xj))
                    n0 = n_ji - n1
                    c_tot = int(np.count_nonzero(yi))
                    p_marg = np.clip(c_tot / float(n_ji), 1e-4, 1.0 - 1e-4)
                    ll_null = c_tot * math.log(p_marg) + (n_ji - c_tot) * math.log(1.0 - p_marg)

                    c11 = int(np.count_nonzero(xj & yi))
                    c01 = c_tot - c11
                    p1 = np.clip(c11 / float(max(1, n1)), 1e-4, 1.0 - 1e-4) if n1 > 0 else p_marg
                    p0 = np.clip(c01 / float(max(1, n0)), 1e-4, 1.0 - 1e-4) if n0 > 0 else p_marg
                    ll_alt = (
                        (c11 * math.log(p1) + (n1 - c11) * math.log(1.0 - p1) if n1 > 0 else 0.0)
                        + (c01 * math.log(p0) + (n0 - c01) * math.log(1.0 - p0) if n0 > 0 else 0.0)
                    )
                    bic_gain = (ll_alt - ll_null) - self._cfg.bic_penalty_weight * 0.5 * math.log(n_ji)
                    if bic_gain > 0.0 and p1 >= p0:
                        omega = 1.0 / (1.0 + math.exp(-min(20.0, 2.0 * bic_gain)))
                        if omega >= self._cfg.min_edge_prob:
                            src_edges.append(j)
                            dst_edges.append(i)
                            omegas.append(omega)

        with torch.inference_mode():
            prec_t = torch.tensor(prec_list, dtype=torch.float32)
            if src_edges:
                edge_idx_t = torch.tensor([src_edges, dst_edges], dtype=torch.long)
                omega_t = torch.tensor(omegas, dtype=torch.float32)
                kappa_t = torch.full_like(omega_t, 0.25)
                gamma_t = torch.full_like(omega_t, 0.55)
            else:
                edge_idx_t = torch.zeros((2, 0), dtype=torch.long)
                omega_t = torch.zeros((0,), dtype=torch.float32)
                kappa_t = torch.zeros((0,), dtype=torch.float32)
                gamma_t = torch.zeros((0,), dtype=torch.float32)

            pruned = gate_and_prune_edges(
                edge_index=edge_idx_t,
                precedence=prec_t,
                omega_raw=omega_t,
                kappa=kappa_t,
                gamma=gamma_t,
                num_vehicles=n_v,
                max_parents_k=self._cfg.max_parents_k,
                min_edge_prob=self._cfg.min_edge_prob,
                training=False,
            )
            z0 = generalized_noisy_or_base_survival(
                leak_prob=torch.full((n_v,), 0.06, dtype=torch.float32),
                self_persist_prob=torch.full((n_v,), 0.40, dtype=torch.float32),
                prev_delay_post=torch.full((n_v,), 0.10, dtype=torch.float32),
                spill_weight=torch.full((n_v,), 0.45, dtype=torch.float32),
                expected_down_spill=torch.zeros((n_v,), dtype=torch.float32),
            )
            _ = multiplicative_sum_product_lbp(
                base_survival_z0=z0,
                pruned_dag=pruned,
                speed_evidence_pos=torch.tensor(ev_list, dtype=torch.float32),
                lbp_sweeps=self._cfg.lbp_sweeps,
                lbp_damping=self._cfg.lbp_damping,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return float(elapsed_ms), int(pruned.active_edge_indices.numel())


def _summarize_timings(
    method_name: str,
    n_vehicles: int,
    latencies_ms: Sequence[float],
    active_edges: Sequence[int],
) -> dict[str, Any]:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    require_finite(arr, f"benchmark.{method_name}.N{n_vehicles}")
    q25 = float(np.percentile(arr, 25.0))
    med = float(np.percentile(arr, 50.0))
    q75 = float(np.percentile(arr, 75.0))
    p95 = float(np.percentile(arr, 95.0))
    return {
        "method": method_name,
        "policy_name": method_name,
        "n_vehicles": int(n_vehicles),
        "repeats": int(arr.size),
        "median_ms": max(1e-4, med),
        "iqr_ms": max(0.0, q75 - q25),
        "q25_ms": max(1e-4, q25),
        "q75_ms": max(1e-4, q75),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "p95_ms": max(1e-4, p95),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "mean_active_edges": float(np.mean(active_edges)) if active_edges else 0.0,
    }


def run_scaling_benchmark(
    output_path: Path | str | None = None,
    *,
    swarm_sizes: Sequence[int] = DEFAULT_SWARM_SIZES,
    methods: Sequence[str] = DEFAULT_BENCHMARK_METHODS,
    checkpoint_path: Path | str | None = None,
    model_preset: ModelPresetName = "sota_local",
    dbn_cfg: DBNConfig | None = None,
    topology: RoadTopology | None = None,
    warmup_repeats: int = 5,
    repeats: int = 20,
    allow_ci_micro: bool = False,
    seed: int = 2026,
) -> dict[str, Any]:
    """Measure warmed CPU structure-update + LBP inference latency across ``swarm_sizes``."""
    if not swarm_sizes:
        raise ConfigValidationError("swarm_sizes cannot be empty.")
    for n_val in swarm_sizes:
        if isinstance(n_val, bool) or not isinstance(n_val, int) or n_val < 2:
            raise ConfigValidationError(f"Each swarm size must be an int >= 2, got {n_val!r}.")
    if warmup_repeats < 1 or repeats < 1:
        raise ConfigValidationError(
            f"Require warmup_repeats >= 1 and repeats >= 1, got ({warmup_repeats}, {repeats})."
        )

    ckpt_resolved = (
        (Path(checkpoint_path) / "neural_dbn.pt")
        if checkpoint_path is not None
        and Path(checkpoint_path).is_dir()
        and (Path(checkpoint_path) / "neural_dbn.pt").is_file()
        else (Path(checkpoint_path) if checkpoint_path is not None else None)
    )
    from ..dbn import (
        FeatureNormalizer,
        NeuralDBNUpdater,
        STGraphTransformerDBN,
        TemperatureScales,
        build_dbn_updater,
        load_dbn_checkpoint,
    )

    if ckpt_resolved is not None:
        ckpt_bundle = load_dbn_checkpoint(
            ckpt_resolved,
            require_calibrated=(dbn_cfg.require_calibrated if dbn_cfg is not None else True),
            device="cpu",
        )
        model_preset = ckpt_bundle.model.preset_name  # type: ignore[assignment]

    preset_spec = get_model_preset(model_preset)
    if preset_spec.name == "ci_micro" and not allow_ci_micro:
        raise CheckpointIntegrityError(
            "Benchmarking with model_preset='ci_micro' is blocked (pass allow_ci_micro=True only in unit tests)."
        )

    cfg = (
        replace(dbn_cfg, model_preset=model_preset)
        if dbn_cfg is not None
        else DBNConfig(model_preset=model_preset, inference_device="cpu")
    )
    topo = topology if topology is not None else build_benchmark_grid_topology(grid_dim=4)
    logger = get_logger("pipelines.benchmark")

    synthetic_neural_updater: NeuralDBNUpdater | None = None
    if "neural_dbn" in methods and ckpt_resolved is None:
        torch.manual_seed(seed)
        net = STGraphTransformerDBN(preset=preset_spec, prediction_horizon_s=cfg.prediction_horizon_s)
        synthetic_neural_updater = NeuralDBNUpdater(
            model=net,
            normalizer=FeatureNormalizer.identity(),
            temperatures=TemperatureScales(t_edge=1.0, t_delay=1.0, t_link=1.0, calibrated=True),
            dbn_cfg=cfg,
            policy_name="neural_dbn",
        )

    methods_out: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}
    flat_records: list[dict[str, Any]] = []

    for n_veh in swarm_sizes:
        # Pre-generate history window (for BIC sufficiency) + warmup + timed steps
        hist_steps = cfg.bic_window_steps if ("online_bic_dbn" in methods or "unconstrained_bic" in methods) else 2
        total_seq_len = hist_steps + warmup_repeats + repeats
        belief_seq = _generate_belief_sequence(topo, n_veh, total_seq_len, seed=seed + n_veh)

        for m_name in methods:
            latencies: list[float] = []
            edge_counts: list[int] = []

            if m_name == "unconstrained_bic":
                dense_bic = _UnconstrainedAllPairsBICBenchmarkUpdater(cfg)
                dense_bic.reset()
                for s_idx, b_state in enumerate(belief_seq):
                    ms, n_e = dense_bic.step_latency_and_edges(b_state, topo)
                    if s_idx >= hist_steps + warmup_repeats:
                        latencies.append(ms)
                        edge_counts.append(n_e)
            else:
                if m_name == "neural_dbn" and ckpt_resolved is None and synthetic_neural_updater is not None:
                    upd: Any = synthetic_neural_updater
                else:
                    upd = _bind_and_call(
                        build_dbn_updater,
                        {
                            "policy_name": m_name,
                            "dbn_cfg": cfg,
                            "checkpoint_path": ckpt_resolved,
                        },
                    )
                upd.reset()
                prev_snap: DBNStateSnapshot | None = None
                start_idx = 0 if m_name == "online_bic_dbn" else hist_steps
                measure_start = (
                    (hist_steps + warmup_repeats)
                    if m_name == "online_bic_dbn"
                    else (start_idx + warmup_repeats)
                )
                for s_idx in range(start_idx, total_seq_len):
                    t0 = time.perf_counter()
                    prev_snap = upd.update_and_infer(belief_seq[s_idx], prev_snap, topo)
                    wall_ms = (time.perf_counter() - t0) * 1000.0
                    rep_ms = (
                        prev_snap.update_latency_ms + prev_snap.inference_latency_ms
                        if (prev_snap.update_latency_ms + prev_snap.inference_latency_ms) > 0.0
                        else wall_ms
                    )
                    if s_idx >= measure_start:
                        latencies.append(float(rep_ms))
                        edge_counts.append(len(prev_snap.edges))

            stat_row = _summarize_timings(m_name, n_veh, latencies, edge_counts)
            methods_out[m_name].append(stat_row)
            flat_records.append(stat_row)
            logger.info(
                "Benchmark %s @ N=%d: median=%.3f ms (IQR=%.3f ms, p95=%.3f ms)",
                m_name,
                n_veh,
                stat_row["median_ms"],
                stat_row["iqr_ms"],
                stat_row["p95_ms"],
            )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "device": "cpu",
        "model_preset": preset_spec.name,
        "warmup_repeats": warmup_repeats,
        "repeats": repeats,
        "swarm_sizes": [int(n) for n in swarm_sizes],
        "methods": methods_out,
        "records": flat_records,
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = out_p.parent / f".{out_p.name}.pid{os.getpid()}.tmp"
        tmp_p.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_p, out_p)

    return report


__all__ = [
    "DEFAULT_BENCHMARK_METHODS",
    "DEFAULT_SWARM_SIZES",
    "build_benchmark_grid_topology",
    "run_scaling_benchmark",
]

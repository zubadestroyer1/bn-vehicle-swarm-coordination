"""Factored 5-iteration Damped Max-Sum multi-intersection coordination over sparse DBN graphs.

Replaces unscalable ``O(H!)`` permutation enumeration and passive single-scalar ``+1.5 * delays``
slot bloat (closing defects D2 and D5) with vectorized ``O(L_ms * |U|^2 * |E_t|)`` Damped Max-Sum
message passing over :class:`~swarm.contracts.LongitudinalMode`, driven by 2-hop DBN
``cascade_score`` platoon relief and ``corridor_spillback_pred`` backpressure gating.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..config import ControlConfig
from ..contracts import (
    CorridorSpec,
    DBNStateSnapshot,
    LongitudinalMode,
    RelationType,
    RoadTopology,
    SwarmBeliefState,
    VehicleBelief,
)
from ..logging_utils import (
    NumericalInvariantError,
    TopologyIntegrityError,
    require_dag,
    require_finite,
    require_probability_range,
    require_probability_simplex,
)

_NUM_MODES: int = 4
_JUNCTION_TRAVERSAL_WIDTH_M: float = 14.0
_Z95_ONE_SIDED: float = 1.645


@dataclass(frozen=True)
class MaxSumResult:
    """Decoded longitudinal coordination modes, node Q-values, and crossing preference margins."""

    modes: dict[str, LongitudinalMode]
    q_values: dict[str, tuple[float, float, float, float]]
    net_crossing_preference: dict[str, float]
    expected_out_spillback: dict[str, float]


def compute_expected_outgoing_spillback(
    veh: VehicleBelief,
    corr: CorridorSpec,
    belief: SwarmBeliefState,
    dbn_snapshot: DBNStateSnapshot | None,
) -> float:
    """Compute belief-weighted expected outgoing corridor spillback ``q_bar_out(i) in [0, 1]``.

    Enforces Boundary Lock A: exit-fringe corridors with ``len(turn_options) == 0`` evaluate
    to ``0.0`` identically; non-exit corridors validate ``turn_probs`` against ``turn_options``.
    """
    if corr.is_exit_fringe and len(corr.turn_options) == 0:
        return 0.0
    if len(corr.turn_options) == 0:
        raise TopologyIntegrityError(
            f"Boundary Lock A violation: non-exit corridor {corr.corridor_id!r} has empty turn_options.",
            context={"corridor_id": corr.corridor_id, "veh_id": veh.veh_id},
        )
    if len(veh.turn_probs) != len(corr.turn_options):
        raise NumericalInvariantError(
            f"Vehicle {veh.veh_id!r} turn_probs len {len(veh.turn_probs)} != "
            f"Corridor {corr.corridor_id!r} turn_options len {len(corr.turn_options)}.",
            context={
                "veh_id": veh.veh_id,
                "corridor_id": corr.corridor_id,
                "turn_probs_len": len(veh.turn_probs),
                "turn_options_len": len(corr.turn_options),
            },
        )
    require_probability_simplex(veh.turn_probs, f"VehicleBelief[{veh.veh_id}].turn_probs")

    expected_q = 0.0
    for opt in corr.turn_options:
        prob_m = float(veh.turn_probs[opt.turn_idx])
        out_cid = opt.to_corridor
        if dbn_snapshot is not None and out_cid in dbn_snapshot.corridor_spillback_pred:
            q_out = float(dbn_snapshot.corridor_spillback_pred[out_cid])
            require_probability_range(q_out, f"corridor_spillback_pred[{out_cid}]")
        elif out_cid in belief.corridors:
            q_out = float(belief.corridors[out_cid].tail_blockage_prob)
            require_probability_range(q_out, f"CorridorBeliefStats[{out_cid}].tail_blockage_prob")
        else:
            q_out = 0.0
        expected_q += prob_m * max(0.0, min(1.0, q_out))

    return max(0.0, min(1.0, expected_q))


def _mode_kinematics(
    veh: VehicleBelief,
    corr: CorridorSpec,
    cfg: ControlConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(v_modes[4], t_in[4], t_out[4], clear_dur[4])`` using ONLY physical clearance + 1.645*eta_std."""
    require_finite(veh.mean_s_m, f"VehicleBelief[{veh.veh_id}].mean_s_m")
    require_finite(veh.mean_v_mps, f"VehicleBelief[{veh.veh_id}].mean_v_mps")
    require_finite(veh.free_eta_s, f"VehicleBelief[{veh.veh_id}].free_eta_s")
    require_finite(veh.eta_std_s, f"VehicleBelief[{veh.veh_id}].eta_std_s")
    require_finite(veh.length_m, f"VehicleBelief[{veh.veh_id}].length_m")

    v_max = corr.speed_limit_mps
    v_cur = min(v_max, max(0.0, float(veh.mean_v_mps)))
    v_min_glide = min(v_max, cfg.min_glide_speed_mps)

    v_go = v_max
    v_cruise = min(v_max, max(v_min_glide, max(v_cur, 0.85 * v_max)))
    v_glide = min(v_max, max(v_min_glide, 0.52 * v_max))
    v_modes = np.array([v_go, v_cruise, v_glide, 0.0], dtype=np.float64)

    d_rem = max(0.0, corr.length_m - float(veh.mean_s_m))
    eta_free = max(0.0, float(veh.free_eta_s))
    eta_std = max(0.0, float(veh.eta_std_s))

    if veh.in_junction_id is not None:
        t_in = np.zeros(_NUM_MODES, dtype=np.float64)
    else:
        t_in = np.array(
            [
                eta_free,
                max(eta_free, d_rem / max(v_cruise, 1.0)),
                max(eta_free + 0.8, d_rem / max(v_glide, 1.0)),
                1e6,
            ],
            dtype=np.float64,
        )

    # Physical crossing clearance duration + 95% one-sided ETA uncertainty ONLY (Zero +1.5*delays bloat).
    phys_dist = _JUNCTION_TRAVERSAL_WIDTH_M + float(veh.length_m) + cfg.idm_min_gap_m
    clear_dur = np.array(
        [
            phys_dist / max(v_go, 3.0) + _Z95_ONE_SIDED * eta_std,
            phys_dist / max(v_cruise, 3.0) + _Z95_ONE_SIDED * eta_std,
            phys_dist / max(v_glide, 3.0) + _Z95_ONE_SIDED * eta_std,
            0.0,
        ],
        dtype=np.float64,
    )
    t_out = t_in + clear_dur
    return v_modes, t_in, t_out, clear_dur


def run_damped_maxsum(
    belief: SwarmBeliefState,
    dbn_snapshot: DBNStateSnapshot,
    topology: RoadTopology,
    control_config: ControlConfig,
) -> MaxSumResult:
    """Execute vectorized Damped Max-Sum coordination over ``dbn_snapshot`` and ``belief``."""
    veh_ids = sorted(belief.vehicles.keys())
    n_veh = len(veh_ids)
    if n_veh == 0:
        return MaxSumResult(
            modes={},
            q_values={},
            net_crossing_preference={},
            expected_out_spillback={},
        )

    require_dag(
        dbn_snapshot.veh_ids,
        dbn_snapshot.edges,
        dbn_snapshot.precedence_scores,
        context={"sim_time": belief.sim_time},
    )

    vid_to_idx = {vid: idx for idx, vid in enumerate(veh_ids)}
    unary = np.zeros((n_veh, _NUM_MODES), dtype=np.float64)
    v_modes_arr = np.zeros((n_veh, _NUM_MODES), dtype=np.float64)
    t_in_arr = np.zeros((n_veh, _NUM_MODES), dtype=np.float64)
    t_out_arr = np.zeros((n_veh, _NUM_MODES), dtype=np.float64)
    clear_dur_arr = np.zeros((n_veh, _NUM_MODES), dtype=np.float64)
    expected_out_spill: dict[str, float] = {}

    cfg = control_config
    theta_spill = cfg.spillback_threshold
    c_spill = cfg.spillback_penalty

    for idx, vid in enumerate(veh_ids):
        veh = belief.vehicles[vid]
        corr = topology.corridors.get(veh.corridor_id)
        if corr is None:
            raise TopologyIntegrityError(
                f"Vehicle {vid!r} references unknown corridor {veh.corridor_id!r} in Max-Sum.",
                context={"veh_id": vid, "corridor_id": veh.corridor_id},
            )
        require_finite(veh.wait_time_s, f"VehicleBelief[{vid}].wait_time_s")
        psi_casc = float(dbn_snapshot.cascade_score.get(vid, 0.0))
        require_finite(psi_casc, f"DBNStateSnapshot.cascade_score[{vid}]")
        if psi_casc < -1e-6:
            raise NumericalInvariantError(
                f"Negative cascade_score={psi_casc} for vehicle {vid!r}.",
                context={"veh_id": vid, "cascade_score": psi_casc},
            )
        psi_casc = max(0.0, psi_casc)

        q_out = compute_expected_outgoing_spillback(veh, corr, belief, dbn_snapshot)
        expected_out_spill[vid] = q_out

        v_m, t_in, t_out, c_dur = _mode_kinematics(veh, corr, cfg)
        v_modes_arr[idx] = v_m
        t_in_arr[idx] = t_in
        t_out_arr[idx] = t_out
        clear_dur_arr[idx] = c_dur

        prio_mult = 1.0 + cfg.wait_weight * min(60.0, max(0.0, float(veh.wait_time_s))) + cfg.cascade_weight * psi_casc

        if veh.in_junction_id is not None:
            # Vehicles inside a junction box must clear expeditiously to prevent gridlock.
            unary[idx, int(LongitudinalMode.GO_WAVE)] = 4.0 * prio_mult
            unary[idx, int(LongitudinalMode.CRUISE)] = 3.0 * prio_mult
            unary[idx, int(LongitudinalMode.ECO_GLIDE)] = 0.5 * prio_mult
            unary[idx, int(LongitudinalMode.YIELD)] = -6.0 * prio_mult
            continue

        spill_pen_cross = 0.0
        spill_pen_glide = 0.0
        if q_out > theta_spill and not corr.is_exit_fringe:
            excess = (q_out - theta_spill) / max(1e-6, 1.0 - theta_spill)
            # Scale spillback backpressure above prio_mult so downstream blockage reliably holds entry.
            spill_pen_cross = -c_spill * q_out * (1.0 + 1.5 * excess) * max(1.0, 0.55 * prio_mult)
            spill_pen_glide = -0.35 * c_spill * excess

        moving_stop_cost = 0.45 if float(veh.mean_v_mps) > 2.0 else 0.05
        far_cruise_bonus = 0.25 if float(veh.free_eta_s) > 8.0 and q_out <= theta_spill else 0.0

        unary[idx, int(LongitudinalMode.GO_WAVE)] = 1.35 * prio_mult + spill_pen_cross
        unary[idx, int(LongitudinalMode.CRUISE)] = 1.18 * prio_mult + far_cruise_bonus + spill_pen_cross
        unary[idx, int(LongitudinalMode.ECO_GLIDE)] = 0.62 * prio_mult + spill_pen_glide
        unary[idx, int(LongitudinalMode.YIELD)] = -0.35 * prio_mult - moving_stop_cost

    # Filter active edges whose endpoints are both in belief.vehicles and omega > 0.
    active_edges = [
        e
        for e in dbn_snapshot.edges
        if e.src_id in vid_to_idx and e.dst_id in vid_to_idx and float(e.omega) > 0.0
    ]
    m_edges = len(active_edges)

    if m_edges == 0:
        q_final = unary
    else:
        src_indices = np.empty(m_edges, dtype=np.int64)
        dst_indices = np.empty(m_edges, dtype=np.int64)
        payoffs = np.zeros((m_edges, _NUM_MODES, _NUM_MODES), dtype=np.float64)

        for e_idx, edge in enumerate(active_edges):
            require_probability_range(edge.omega, f"DBNEdge[{edge.src_id}->{edge.dst_id}].omega")
            require_probability_range(edge.kappa, f"DBNEdge[{edge.src_id}->{edge.dst_id}].kappa")
            require_probability_range(edge.gamma, f"DBNEdge[{edge.src_id}->{edge.dst_id}].gamma")
            require_probability_range(edge.p_conflict, f"DBNEdge[{edge.src_id}->{edge.dst_id}].p_conflict")
            require_probability_range(edge.p_merge, f"DBNEdge[{edge.src_id}->{edge.dst_id}].p_merge")

            s_i = vid_to_idx[edge.src_id]
            d_i = vid_to_idx[edge.dst_id]
            src_indices[e_idx] = s_i
            dst_indices[e_idx] = d_i

            omega = max(0.0, min(1.0, float(edge.omega)))
            kappa = max(0.0, min(1.0, float(edge.kappa)))
            gamma = max(0.0, min(1.0, float(edge.gamma)))
            p_conf_eff = max(float(edge.p_conflict), 0.75 * float(edge.p_merge))
            d_src_post = float(dbn_snapshot.delay_post.get(edge.src_id, dbn_snapshot.delay_pred.get(edge.src_id, 0.0)))
            require_probability_range(d_src_post, f"delay_post[{edge.src_id}]")

            mat = np.zeros((_NUM_MODES, _NUM_MODES), dtype=np.float64)

            # A. Intra-junction conflict / cross-approach outgoing-corridor merge mutual exclusion.
            if edge.relation == RelationType.SAME_JUNCTION_CONFLICT:
                eff_risk = max(p_conf_eff, 0.25)
                if eff_risk > cfg.conflict_risk_threshold:
                    for u_src in range(3):
                        for u_dst in range(3):
                            ov = min(t_out_arr[s_i, u_src], t_out_arr[d_i, u_dst]) - max(
                                t_in_arr[s_i, u_src], t_in_arr[d_i, u_dst]
                            )
                            if ov > 0.0:
                                min_dur = max(0.5, min(clear_dur_arr[s_i, u_src], clear_dur_arr[d_i, u_dst]))
                                norm_ov = ov / min_dur
                                mat[u_src, u_dst] -= cfg.conflict_penalty * omega * eff_risk * (0.6 + norm_ov)
                    # Precedence tie-break bonus when src precedes dst cleanly without overlap.
                    mat[0, 2] += 0.08 * omega * eff_risk
                    mat[0, 3] += 0.06 * omega * eff_risk

            # B. Same-corridor platoon & upstream-feeder green-wave speed harmonization.
            if edge.relation in (RelationType.SAME_CORRIDOR, RelationType.UPSTREAM_FEEDER):
                v_diff = np.abs(v_modes_arr[s_i, :, None] - v_modes_arr[d_i, None, :])
                mat += cfg.platoon_harmony_weight * omega * np.exp(-v_diff / 3.5)
                # Discourage follower from charging at GO_WAVE/CRUISE into a YIELDing leader.
                mat[int(LongitudinalMode.YIELD), int(LongitudinalMode.GO_WAVE)] -= 2.8 * omega * (0.4 + 0.6 * kappa)
                mat[int(LongitudinalMode.YIELD), int(LongitudinalMode.CRUISE)] -= 2.2 * omega * (0.4 + 0.6 * kappa)

            # C. Inter-junction downstream spillback tail blockage coupling.
            if edge.relation == RelationType.DOWNSTREAM_SPILLBACK:
                spill_strength = omega * (kappa + (1.0 - kappa) * gamma * max(0.0, min(1.0, d_src_post)))
                mat[:, int(LongitudinalMode.GO_WAVE)] -= cfg.spillback_penalty * spill_strength
                mat[:, int(LongitudinalMode.CRUISE)] -= 0.9 * cfg.spillback_penalty * spill_strength
                mat[:, int(LongitudinalMode.ECO_GLIDE)] -= 0.25 * cfg.spillback_penalty * spill_strength

            payoffs[e_idx] = mat

        msg_fwd = np.zeros((m_edges, _NUM_MODES), dtype=np.float64)
        msg_rev = np.zeros((m_edges, _NUM_MODES), dtype=np.float64)
        lam = cfg.maxsum_damping

        for _ in range(cfg.maxsum_iterations):
            node_bel = unary.copy()
            np.add.at(node_bel, dst_indices, msg_fwd)
            np.add.at(node_bel, src_indices, msg_rev)

            cav_src = node_bel[src_indices] - msg_rev
            cav_dst = node_bel[dst_indices] - msg_fwd

            raw_fwd = np.max(cav_src[:, :, None] + payoffs, axis=1)
            raw_rev = np.max(cav_dst[:, None, :] + payoffs, axis=2)

            raw_fwd -= np.mean(raw_fwd, axis=1, keepdims=True)
            raw_rev -= np.mean(raw_rev, axis=1, keepdims=True)

            msg_fwd = lam * msg_fwd + (1.0 - lam) * raw_fwd
            msg_rev = lam * msg_rev + (1.0 - lam) * raw_rev

        q_final = unary.copy()
        np.add.at(q_final, dst_indices, msg_fwd)
        np.add.at(q_final, src_indices, msg_rev)

    require_finite(q_final, "MaxSumResult.q_final")

    modes: dict[str, LongitudinalMode] = {}
    q_values: dict[str, tuple[float, float, float, float]] = {}
    net_pref: dict[str, float] = {}

    for idx, vid in enumerate(veh_ids):
        row = q_final[idx]
        best_mode = LongitudinalMode(int(np.argmax(row)))
        modes[vid] = best_mode
        q_values[vid] = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        cross_q = max(float(row[int(LongitudinalMode.GO_WAVE)]), float(row[int(LongitudinalMode.CRUISE)]))
        hold_q = max(float(row[int(LongitudinalMode.ECO_GLIDE)]), float(row[int(LongitudinalMode.YIELD)]))
        net_pref[vid] = cross_q - hold_q

    return MaxSumResult(
        modes=modes,
        q_values=q_values,
        net_crossing_preference=net_pref,
        expected_out_spillback=expected_out_spill,
    )


__all__ = [
    "MaxSumResult",
    "compute_expected_outgoing_spillback",
    "run_damped_maxsum",
]

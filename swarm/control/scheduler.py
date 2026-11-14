"""Sequential Priority-DAG conflict-free TLS link and time-slot reservation decoder.

Decodes either factored Damped Max-Sum preferences (with 2-hop DBN cascade priority and
spillback backpressure gating) or First-Come-First-Served (Dresner-Stone AIM) arrivals into:
1. Guaranteed conflict-free TLS green link sets ``tls_desired_greens`` verified against
   ``JunctionSpec.tls_link_foe_matrix``,
2. Per-vehicle conflict-zone reservation windows ``[t_start, t_end]`` using physical traversal
   clearance + ``1.645 * eta_std_s`` (closing D2), and
3. ``granted_vehicles`` vs ``must_hold_stopline`` sets enforcing downstream effective storage caps
   and spillback gating so cross-streets flow freely when an orthogonal corridor spills back.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import heapq
import math

import numpy as np

from ..config import ControlConfig
from ..contracts import (
    CorridorSpec,
    DBNStateSnapshot,
    JunctionSpec,
    LongitudinalMode,
    RoadTopology,
    SwarmBeliefState,
    TurnOption,
    VehicleBelief,
)
from ..logging_utils import (
    NumericalInvariantError,
    TopologyIntegrityError,
    require_finite,
    require_probability_range,
    require_probability_simplex,
)
from .maxsum import MaxSumResult, compute_expected_outgoing_spillback

_PLANNING_HORIZON_S: float = 12.0
_IMMEDIATE_GRANT_SLACK_S: float = 0.85
_JUNCTION_TRAVERSAL_WIDTH_M: float = 14.0
_Z95_ONE_SIDED: float = 1.645


@dataclass(frozen=True)
class ScheduleResult:
    """Conflict-free TLS green sets, granted crossing IDs, hold set, and slot start/end times."""

    tls_desired_greens: dict[str, frozenset[int]]
    granted_vehicles: frozenset[str]
    must_hold_stopline: frozenset[str]
    scheduled_start_s: dict[str, float]
    scheduled_end_s: dict[str, float]


@dataclass
class _TlsHoldRecord:
    green_links: frozenset[int]
    requested_at: float
    green_started_at: float | None = None


def _are_links_live_green(
    links: Iterable[int],
    tls_id: str,
    tls_states: Mapping[str, str],
) -> bool:
    """Return ``True`` when all ``links`` are currently ``'G'``/``'g'`` in ``tls_states`` (or untracked)."""
    ryg = tls_states.get(tls_id)
    if ryg is None:
        return True
    link_list = list(links)
    if not link_list:
        return True
    return all(0 <= idx < len(ryg) and ryg[idx] in ("G", "g") for idx in link_list)


def are_tls_links_compatible(
    candidate_links: Iterable[int],
    existing_greens: frozenset[int] | set[int],
    foe_matrix: np.ndarray,
    junction_id: str,
) -> bool:
    """Return ``True`` iff ``candidate_links`` have zero internal or external conflicts in ``foe_matrix``."""
    cand_list = list(dict.fromkeys(int(idx) for idx in candidate_links))
    n_links = foe_matrix.shape[0]
    for idx, l_a in enumerate(cand_list):
        if l_a < 0 or l_a >= n_links:
            raise TopologyIntegrityError(
                f"TLS link index {l_a} out of bounds [0, {n_links}) at junction {junction_id!r}.",
                context={"junction_id": junction_id, "link_index": l_a, "num_tls_links": n_links},
            )
        for l_b in cand_list[idx + 1 :]:
            if l_b < 0 or l_b >= n_links:
                raise TopologyIntegrityError(
                    f"TLS link index {l_b} out of bounds [0, {n_links}) at junction {junction_id!r}.",
                    context={"junction_id": junction_id, "link_index": l_b, "num_tls_links": n_links},
                )
            if bool(foe_matrix[l_a, l_b]) or bool(foe_matrix[l_b, l_a]):
                return False
        for l_prev in existing_greens:
            if l_a == l_prev:
                continue
            if bool(foe_matrix[l_a, l_prev]) or bool(foe_matrix[l_prev, l_a]):
                return False
    return True


def verify_conflict_free_greens(
    junction: JunctionSpec,
    green_links: frozenset[int] | set[int],
) -> None:
    """Fail loudly if any pair of granted links in ``green_links`` conflicts in ``tls_link_foe_matrix``."""
    ordered = sorted(green_links)
    n_links = junction.num_tls_links
    for idx, l_a in enumerate(ordered):
        if l_a < 0 or l_a >= n_links:
            raise TopologyIntegrityError(
                f"Granted TLS link {l_a} out of bounds [0, {n_links}) on {junction.junction_id!r}.",
                context={"junction_id": junction.junction_id, "link_index": l_a, "num_tls_links": n_links},
            )
        for l_b in ordered[idx + 1 :]:
            if bool(junction.tls_link_foe_matrix[l_a, l_b]) or bool(junction.tls_link_foe_matrix[l_b, l_a]):
                raise NumericalInvariantError(
                    f"Conflicting TLS links ({l_a}, {l_b}) simultaneously granted at {junction.junction_id!r}.",
                    context={"junction_id": junction.junction_id, "link_a": l_a, "link_b": l_b},
                )


def _pair_turn_conflict_prob(
    topology: RoadTopology,
    junction_id: str,
    veh_a: VehicleBelief,
    corr_a: CorridorSpec,
    veh_b: VehicleBelief,
    corr_b: CorridorSpec,
    open_lanes_by_edge: Mapping[str, int] | None = None,
) -> float:
    """Compute belief-expected turn conflict probability ``P(conflict(a, b))`` at ``junction_id``."""
    junc = topology.junctions.get(junction_id)
    if junc is None:
        raise TopologyIntegrityError(
            f"Unknown junction_id {junction_id!r} in _pair_turn_conflict_prob.",
            context={"junction_id": junction_id},
        )

    a_clearing = veh_a.in_junction_id == junction_id and corr_a.to_junction != junction_id
    b_clearing = veh_b.in_junction_id == junction_id and corr_b.to_junction != junction_id

    if not a_clearing and corr_a.to_junction != junction_id:
        raise TopologyIntegrityError(
            f"Vehicle {veh_a.veh_id!r} corridor {corr_a.corridor_id!r} targets junction "
            f"{corr_a.to_junction!r}, not {junction_id!r} (in_junction_id={veh_a.in_junction_id!r}).",
            context={
                "junction_id": junction_id,
                "veh_id": veh_a.veh_id,
                "corridor_id": corr_a.corridor_id,
                "corridor_to_junction": corr_a.to_junction,
                "in_junction_id": veh_a.in_junction_id,
            },
        )
    if not b_clearing and corr_b.to_junction != junction_id:
        raise TopologyIntegrityError(
            f"Vehicle {veh_b.veh_id!r} corridor {corr_b.corridor_id!r} targets junction "
            f"{corr_b.to_junction!r}, not {junction_id!r} (in_junction_id={veh_b.in_junction_id!r}).",
            context={
                "junction_id": junction_id,
                "veh_id": veh_b.veh_id,
                "corridor_id": corr_b.corridor_id,
                "corridor_to_junction": corr_b.to_junction,
                "in_junction_id": veh_b.in_junction_id,
            },
        )

    if a_clearing and b_clearing:
        return 0.0

    is_signalized_tls = junc.tls_id is not None and junc.num_tls_links > 0
    open_lanes = open_lanes_by_edge or {}

    if a_clearing or b_clearing:
        if is_signalized_tls:
            # At signalized TLS junctions, verify_conflict_free_greens + 3.0s yellow clearance
            # enforce cross-phase separation; longitudinal spacing is enforced by leader headway & IDM.
            return 0.0

        veh_app, corr_app, corr_clear = (
            (veh_b, corr_b, corr_a) if a_clearing else (veh_a, corr_a, corr_b)
        )
        if not corr_app.turn_options:
            return 0.0
        clearing_feeders = [
            (in_cid, opt_c)
            for in_cid in junc.incoming_corridors
            if not topology.corridors[in_cid].is_exit_fringe
            for opt_c in topology.corridors[in_cid].turn_options
            if opt_c.to_corridor == corr_clear.corridor_id
        ]
        if not clearing_feeders:
            return 0.0
        clear_open_lanes = topology.effective_lanes(corr_clear.corridor_id, open_lanes)
        total_clear = 0.0
        for opt_app in corr_app.turn_options:
            p_app = float(veh_app.turn_probs[opt_app.turn_idx])
            if p_app <= 1e-6:
                continue
            conflicts = any(
                (
                    in_cid != corr_app.corridor_id
                    and (
                        opt_c.to_corridor == opt_app.to_corridor
                        or topology.turn_conflict(junction_id, opt_c, opt_app)
                    )
                )
                or (
                    in_cid == corr_app.corridor_id
                    and (
                        topology.turn_conflict(junction_id, opt_c, opt_app)
                        or (opt_c.to_corridor == opt_app.to_corridor and clear_open_lanes <= 1)
                    )
                )
                for in_cid, opt_c in clearing_feeders
            )
            if conflicts:
                total_clear += p_app
        return max(0.0, min(1.0, total_clear))

    if not corr_a.turn_options or not corr_b.turn_options:
        return 0.0

    same_corridor = corr_a.corridor_id == corr_b.corridor_id
    if same_corridor and veh_a.lane_index == veh_b.lane_index:
        # Same-lane longitudinal headway (idm_time_headway_s) is enforced directly on lane leaders.
        return 0.0

    total = 0.0
    for opt_a in corr_a.turn_options:
        p_a = float(veh_a.turn_probs[opt_a.turn_idx])
        if p_a <= 1e-6:
            continue
        for opt_b in corr_b.turn_options:
            p_b = float(veh_b.turn_probs[opt_b.turn_idx])
            if p_b <= 1e-6:
                continue
            if same_corridor:
                # Parallel lanes on the same approach discharge side-by-side into multi-lane outgoing
                # corridors; conflict ONLY on crossing turn paths or single-lane merge bottlenecks.
                has_conflict = topology.turn_conflict(junction_id, opt_a, opt_b) or (
                    opt_a.to_corridor == opt_b.to_corridor
                    and topology.effective_lanes(opt_a.to_corridor, open_lanes) <= 1
                )
            else:
                has_conflict = opt_a.to_corridor == opt_b.to_corridor or topology.turn_conflict(
                    junction_id, opt_a, opt_b
                )
            if has_conflict:
                total += p_a * p_b
    return max(0.0, min(1.0, total))


def _physical_slot_duration(
    veh: VehicleBelief,
    corr: CorridorSpec,
    cfg: ControlConfig,
) -> float:
    """Compute physical intersection occupancy duration + 1.645 * eta_std_s (zero +1.5*delays bloat)."""
    v_cross = max(3.5, min(corr.speed_limit_mps, max(float(veh.mean_v_mps), 0.75 * corr.speed_limit_mps)))
    phys_len = _JUNCTION_TRAVERSAL_WIDTH_M + float(veh.length_m) + cfg.idm_min_gap_m
    return phys_len / v_cross + _Z95_ONE_SIDED * max(0.0, float(veh.eta_std_s))


class PriorityDAGScheduler:
    """Multi-intersection conflict-free TLS link & reservation decoder."""

    def __init__(self, control_config: ControlConfig) -> None:
        self._cfg = control_config
        self._tls_holds: dict[str, _TlsHoldRecord] = {}
        self._tls_red_elapsed_s: dict[str, np.ndarray] = {}
        self._last_decode_time: float | None = None

    def reset(self) -> None:
        """Clear per-TLS minimum-green hold and red-starvation state across episodes."""
        self._tls_holds.clear()
        self._tls_red_elapsed_s.clear()
        self._last_decode_time = None

    def decode(
        self,
        belief: SwarmBeliefState,
        topology: RoadTopology,
        *,
        maxsum_result: MaxSumResult | None = None,
        dbn_snapshot: DBNStateSnapshot | None = None,
        enable_spillback_gate: bool = True,
    ) -> ScheduleResult:
        """Decode conflict-free per-junction TLS green sets and vehicle slot reservations.

        When ``maxsum_result`` is provided, vehicles are scheduled by Damped Max-Sum preference
        and learned DBN ``precedence_scores``; when ``None``, vehicles are scheduled in
        First-Come-First-Served (Dresner-Stone AIM) order ``(entered_at, free_eta_s, veh_id)``.
        """
        require_finite(belief.sim_time, "SwarmBeliefState.sim_time")
        now_sim = float(belief.sim_time)
        dt_red = (
            max(0.0, float(belief.dt))
            if self._last_decode_time is None
            else max(0.0, min(5.0, now_sim - self._last_decode_time))
        )
        self._last_decode_time = now_sim
        for tls_id, ryg in belief.tls_states.items():
            red_arr = self._tls_red_elapsed_s.get(tls_id)
            if red_arr is None or red_arr.shape != (len(ryg),):
                red_arr = np.zeros(len(ryg), dtype=np.float64)
                self._tls_red_elapsed_s[tls_id] = red_arr
            for idx, ch in enumerate(ryg):
                if ch in ("G", "g"):
                    red_arr[idx] = 0.0
                else:
                    red_arr[idx] += dt_red

        cfg = self._cfg
        theta_spill = cfg.spillback_threshold
        theta_risk = cfg.conflict_risk_threshold

        # Precompute corridor spillback risks and remaining storage capacities.
        corridor_q_spill: dict[str, float] = {}
        corridor_rem_slots: dict[str, int] = {}
        active_by_corridor: dict[str, list[VehicleBelief]] = {cid: [] for cid in topology.corridors}
        in_junction_by_jid: dict[str, list[VehicleBelief]] = {jid: [] for jid in topology.junctions}

        for vid in sorted(belief.vehicles):
            veh = belief.vehicles[vid]
            if veh.corridor_id not in topology.corridors:
                raise TopologyIntegrityError(
                    f"Vehicle {vid!r} references unknown corridor {veh.corridor_id!r} in scheduler.",
                    context={"veh_id": vid, "corridor_id": veh.corridor_id},
                )
            if veh.in_junction_id is not None:
                if veh.in_junction_id not in in_junction_by_jid:
                    raise TopologyIntegrityError(
                        f"Vehicle {vid!r} references unknown in_junction_id {veh.in_junction_id!r}.",
                        context={"veh_id": vid, "in_junction_id": veh.in_junction_id},
                    )
                in_junction_by_jid[veh.in_junction_id].append(veh)
            else:
                active_by_corridor[veh.corridor_id].append(veh)

        for cid, corr in topology.corridors.items():
            cap = topology.effective_capacity(cid, belief.open_lanes_by_edge)
            if cid in belief.corridors:
                occ_count = int(belief.corridors[cid].active_count)
                base_tail_q = float(belief.corridors[cid].tail_blockage_prob)
                require_probability_range(base_tail_q, f"CorridorBeliefStats[{cid}].tail_blockage_prob")
            else:
                occ_count = len(active_by_corridor[cid])
                base_tail_q = 0.0

            if dbn_snapshot is not None and cid in dbn_snapshot.corridor_spillback_pred:
                q_pred = float(dbn_snapshot.corridor_spillback_pred[cid])
                require_probability_range(q_pred, f"corridor_spillback_pred[{cid}]")
                corridor_q_spill[cid] = max(0.0, min(1.0, q_pred))
            else:
                corridor_q_spill[cid] = max(0.0, min(1.0, base_tail_q))

            if enable_spillback_gate and corridor_q_spill[cid] > theta_spill and not corr.is_exit_fringe:
                corridor_rem_slots[cid] = 0
            else:
                corridor_rem_slots[cid] = max(0, cap - occ_count)

        tls_desired_greens: dict[str, frozenset[int]] = {}
        granted_vehicles: set[str] = set()
        must_hold_stopline: set[str] = set()
        scheduled_start_s: dict[str, float] = {}
        scheduled_end_s: dict[str, float] = {}

        # Vehicles currently inside junctions or cruising on exit-fringe corridors are free to proceed.
        for veh in belief.vehicles.values():
            corr = topology.corridors[veh.corridor_id]
            if veh.in_junction_id is not None or corr.is_exit_fringe:
                granted_vehicles.add(veh.veh_id)
                scheduled_start_s[veh.veh_id] = 0.0
                scheduled_end_s[veh.veh_id] = _physical_slot_duration(veh, corr, cfg)

        # Decode each junction independently across the city road network.
        for jid in sorted(topology.junctions):
            junc = topology.junctions[jid]
            approaching: list[VehicleBelief] = []
            far_cruising: list[VehicleBelief] = []
            for in_cid in junc.incoming_corridors:
                in_corr = topology.corridors[in_cid]
                if in_corr.is_exit_fringe:
                    continue
                for veh in active_by_corridor[in_cid]:
                    if not in_corr.is_exit_fringe and len(veh.turn_probs) != len(in_corr.turn_options):
                        raise NumericalInvariantError(
                            f"Vehicle {veh.veh_id!r} turn_probs len {len(veh.turn_probs)} != "
                            f"{len(in_corr.turn_options)} on corridor {in_cid!r}.",
                            context={"veh_id": veh.veh_id, "corridor_id": in_cid},
                        )
                    require_probability_simplex(veh.turn_probs, f"VehicleBelief[{veh.veh_id}].turn_probs")
                    if float(veh.free_eta_s) <= _PLANNING_HORIZON_S:
                        approaching.append(veh)
                    else:
                        far_cruising.append(veh)

            # Far vehicles (> planning horizon) cruise freely unless outgoing corridor is spilled back.
            for veh in far_cruising:
                in_corr = topology.corridors[veh.corridor_id]
                q_out = (
                    maxsum_result.expected_out_spillback[veh.veh_id]
                    if maxsum_result is not None and veh.veh_id in maxsum_result.expected_out_spillback
                    else compute_expected_outgoing_spillback(veh, in_corr, belief, dbn_snapshot)
                )
                if not (enable_spillback_gate and q_out > theta_spill):
                    granted_vehicles.add(veh.veh_id)
                scheduled_start_s[veh.veh_id] = float(veh.free_eta_s)
                scheduled_end_s[veh.veh_id] = float(veh.free_eta_s) + _physical_slot_duration(veh, in_corr, cfg)

            # Establish same-lane physical leader map so followers never jump ahead of a held same-lane leader.
            lane_leader: dict[str, str] = {}
            by_lane: dict[tuple[str, int], list[VehicleBelief]] = {}
            for veh in approaching:
                by_lane.setdefault((veh.corridor_id, veh.lane_index), []).append(veh)
            for lane_vehs in by_lane.values():
                if len(lane_vehs) > 1:
                    # Closest to stop-line has largest mean_s_m.
                    desc_s = sorted(lane_vehs, key=lambda v: (-float(v.mean_s_m), v.veh_id))
                    for idx in range(1, len(desc_s)):
                        lane_leader[desc_s[idx].veh_id] = desc_s[idx - 1].veh_id

            ordered_approaching = self._sort_approaching_vehicles(
                approaching=approaching,
                junc=junc,
                topology=topology,
                belief=belief,
                lane_leader=lane_leader,
                corridor_q_spill=corridor_q_spill,
                corridor_rem_slots=corridor_rem_slots,
                enable_spillback_gate=enable_spillback_gate,
                maxsum_result=maxsum_result,
                dbn_snapshot=dbn_snapshot,
            )

            is_signalized_tls = junc.tls_id is not None and junc.num_tls_links > 0
            active_green_links: set[int] = set()
            if is_signalized_tls:
                active_green_links.update(
                    self._retained_min_hold_greens(
                        junc=junc,
                        topology=topology,
                        belief=belief,
                        approaching=approaching,
                        corridor_q_spill=corridor_q_spill,
                        corridor_rem_slots=corridor_rem_slots,
                        sim_time=float(belief.sim_time),
                        enable_spillback_gate=enable_spillback_gate,
                    )
                )

            scheduled_at_junc: list[VehicleBelief] = list(in_junction_by_jid[jid])
            for in_v in in_junction_by_jid[jid]:
                in_c = topology.corridors[in_v.corridor_id]
                scheduled_start_s[in_v.veh_id] = 0.0
                scheduled_end_s[in_v.veh_id] = max(0.6, 0.5 * _physical_slot_duration(in_v, in_c, cfg))

            for veh in ordered_approaching:
                vid = veh.veh_id
                corr = topology.corridors[veh.corridor_id]
                opt_map = max(corr.turn_options, key=lambda o: (float(veh.turn_probs[o.turn_idx]), -o.turn_idx))
                out_cid = opt_map.to_corridor
                out_corr = topology.corridors[out_cid]

                q_out_expected = (
                    maxsum_result.expected_out_spillback[vid]
                    if maxsum_result is not None and vid in maxsum_result.expected_out_spillback
                    else compute_expected_outgoing_spillback(veh, corr, belief, dbn_snapshot)
                )
                q_out_map = corridor_q_spill.get(out_cid, 0.0)
                spilled_out = (
                    enable_spillback_gate
                    and not out_corr.is_exit_fringe
                    and (q_out_expected > theta_spill or q_out_map > theta_spill)
                )
                no_storage = not out_corr.is_exit_fringe and corridor_rem_slots.get(out_cid, 1) <= 0
                same_lane_lead_id = lane_leader.get(vid)
                same_lane_blocked = same_lane_lead_id is not None and same_lane_lead_id in must_hold_stopline

                if spilled_out or no_storage or same_lane_blocked:
                    must_hold_stopline.add(vid)
                    scheduled_start_s[vid] = math.inf
                    scheduled_end_s[vid] = math.inf
                    continue

                # Compute earliest conflict-free crossing start time relative to now.
                t_start = max(0.0, float(veh.free_eta_s))
                if same_lane_lead_id is not None and same_lane_lead_id in scheduled_start_s:
                    lead_start = scheduled_start_s[same_lane_lead_id]
                    if math.isfinite(lead_start):
                        t_start = max(t_start, lead_start + cfg.idm_time_headway_s)
                    else:
                        must_hold_stopline.add(vid)
                        scheduled_start_s[vid] = math.inf
                        scheduled_end_s[vid] = math.inf
                        continue

                for earlier in scheduled_at_junc:
                    earlier_corr = topology.corridors[earlier.corridor_id]
                    p_conf = _pair_turn_conflict_prob(
                        topology,
                        jid,
                        veh,
                        corr,
                        earlier,
                        earlier_corr,
                        open_lanes_by_edge=belief.open_lanes_by_edge,
                    )
                    if p_conf > theta_risk:
                        t_start = max(t_start, scheduled_end_s[earlier.veh_id])

                dur = _physical_slot_duration(veh, corr, cfg)
                t_end = t_start + dur

                mode_allows_entry = (
                    True
                    if maxsum_result is None
                    else maxsum_result.modes.get(vid, LongitudinalMode.CRUISE) != LongitudinalMode.YIELD
                )

                if is_signalized_tls:
                    assert junc.tls_id is not None
                    chosen_links = self._select_vehicle_tls_links(
                        veh=veh,
                        corr=corr,
                        opt_map=opt_map,
                        junc=junc,
                        active_green_links=active_green_links,
                        corridor_q_spill=corridor_q_spill,
                        corridor_rem_slots=corridor_rem_slots,
                        topology=topology,
                        enable_spillback_gate=enable_spillback_gate,
                    )
                    tls_ok = not (opt_map.tls_link_indices and not chosen_links)
                    if tls_ok:
                        active_green_links.update(chosen_links)
                        if _are_links_live_green(chosen_links, junc.tls_id, belief.tls_states):
                            granted_vehicles.add(vid)
                            if not out_corr.is_exit_fringe and float(veh.free_eta_s) <= 2.0:
                                corridor_rem_slots[out_cid] = max(0, corridor_rem_slots.get(out_cid, 1) - 1)
                            scheduled_start_s[vid] = t_start
                            scheduled_end_s[vid] = t_end
                            scheduled_at_junc.append(veh)
                        else:
                            # Cross-street is clearing yellow in SUMO; TLS state 'r' physically holds stopline
                            # while keeping LC2013 cooperative turn-pocket lane-changing active upstream.
                            wait_green_start = max(t_start, cfg.tls_yellow_clearance_s)
                            scheduled_start_s[vid] = wait_green_start
                            scheduled_end_s[vid] = wait_green_start + dur
                    else:
                        # Red TLS link physically governs stopline without sticky setSpeed overriding LC2013.
                        scheduled_start_s[vid] = math.inf
                        scheduled_end_s[vid] = math.inf
                else:
                    if mode_allows_entry and t_start <= float(veh.free_eta_s) + _IMMEDIATE_GRANT_SLACK_S:
                        granted_vehicles.add(vid)
                        if not out_corr.is_exit_fringe and float(veh.free_eta_s) <= 2.0:
                            corridor_rem_slots[out_cid] = max(0, corridor_rem_slots.get(out_cid, 1) - 1)
                        scheduled_start_s[vid] = t_start
                        scheduled_end_s[vid] = t_end
                        scheduled_at_junc.append(veh)
                    else:
                        scheduled_start_s[vid] = t_start
                        scheduled_end_s[vid] = t_end
                        scheduled_at_junc.append(veh)
                        if t_start > float(veh.free_eta_s) + _IMMEDIATE_GRANT_SLACK_S or not mode_allows_entry:
                            must_hold_stopline.add(vid)

            # Complete non-conflicting compatible green phase for dual-through progression & idle junctions.
            if is_signalized_tls:
                assert junc.tls_id is not None
                self._fill_compatible_greens(
                    junc=junc,
                    topology=topology,
                    belief=belief,
                    approaching=approaching,
                    maxsum_result=maxsum_result,
                    dbn_snapshot=dbn_snapshot,
                    active_green_links=active_green_links,
                    corridor_q_spill=corridor_q_spill,
                    corridor_rem_slots=corridor_rem_slots,
                    enable_spillback_gate=enable_spillback_gate,
                )
                final_greens = frozenset(active_green_links)
                verify_conflict_free_greens(junc, final_greens)
                now_t = float(belief.sim_time)
                prev_hold = self._tls_holds.get(junc.tls_id)
                live_green_now = _are_links_live_green(final_greens, junc.tls_id, belief.tls_states)
                if prev_hold is None or not are_tls_links_compatible(
                    final_greens,
                    prev_hold.green_links,
                    junc.tls_link_foe_matrix,
                    junc.junction_id,
                ):
                    self._tls_holds[junc.tls_id] = _TlsHoldRecord(
                        green_links=final_greens,
                        requested_at=now_t,
                        green_started_at=now_t if live_green_now else None,
                    )
                else:
                    started_at = prev_hold.green_started_at
                    if started_at is None and live_green_now:
                        started_at = now_t
                    if (
                        started_at is not None
                        and (now_t - started_at) >= cfg.tls_min_green_hold_s - 1e-9
                        and final_greens != prev_hold.green_links
                    ):
                        self._tls_holds[junc.tls_id] = _TlsHoldRecord(
                            green_links=final_greens,
                            requested_at=now_t,
                            green_started_at=now_t if live_green_now else None,
                        )
                    else:
                        prev_hold.green_links = final_greens
                        prev_hold.green_started_at = started_at
                tls_desired_greens[junc.tls_id] = final_greens

        return ScheduleResult(
            tls_desired_greens=tls_desired_greens,
            granted_vehicles=frozenset(granted_vehicles),
            must_hold_stopline=frozenset(must_hold_stopline),
            scheduled_start_s=scheduled_start_s,
            scheduled_end_s=scheduled_end_s,
        )

    def _sort_approaching_vehicles(
        self,
        approaching: list[VehicleBelief],
        junc: JunctionSpec,
        topology: RoadTopology,
        belief: SwarmBeliefState,
        lane_leader: Mapping[str, str],
        corridor_q_spill: Mapping[str, float],
        corridor_rem_slots: Mapping[str, int],
        enable_spillback_gate: bool,
        maxsum_result: MaxSumResult | None,
        dbn_snapshot: DBNStateSnapshot | None,
    ) -> list[VehicleBelief]:
        if not approaching:
            return []

        if maxsum_result is None:
            def _rank_key(v: VehicleBelief) -> tuple[int, float, float, float, str]:
                # Dresner-Stone AIM First-Come-First-Served order (strictly observable entered_at).
                return (0, float(v.entered_at), float(v.free_eta_s), 0.0, v.veh_id)
        else:
            is_sig = junc.tls_id is not None and junc.num_tls_links > 0
            theta_spill = self._cfg.spillback_threshold
            movement_own_pressure: dict[tuple[str, int], float] = {}
            movement_queued_count: dict[tuple[str, int], int] = {}
            movement_close_count: dict[tuple[str, int], int] = {}
            movement_links: dict[tuple[str, int], tuple[int, ...]] = {}
            movement_spilled: dict[tuple[str, int], bool] = {}
            veh_map_turn: dict[str, tuple[str, int]] = {}
            for v_item in approaching:
                in_c = topology.corridors[v_item.corridor_id]
                if not in_c.turn_options:
                    continue
                opt_m = max(in_c.turn_options, key=lambda o: (float(v_item.turn_probs[o.turn_idx]), -o.turn_idx))
                m_key = (v_item.corridor_id, opt_m.turn_idx)
                veh_map_turn[v_item.veh_id] = m_key
                movement_links[m_key] = opt_m.tls_link_indices
                out_c = topology.corridors[opt_m.to_corridor]
                q_exp = maxsum_result.expected_out_spillback.get(v_item.veh_id, 0.0)
                is_spilled = (
                    enable_spillback_gate
                    and not out_c.is_exit_fringe
                    and (
                        q_exp > theta_spill
                        or corridor_q_spill.get(opt_m.to_corridor, 0.0) > theta_spill
                        or corridor_rem_slots.get(opt_m.to_corridor, 1) <= 0
                    )
                )
                movement_spilled[m_key] = is_spilled
                if is_spilled:
                    continue
                movement_own_pressure.setdefault(m_key, 0.0)

                w_s = min(60.0, max(0.0, float(v_item.wait_time_s)))
                c_s = (
                    max(0.0, float(dbn_snapshot.cascade_score.get(v_item.veh_id, 0.0)))
                    if dbn_snapshot is not None
                    else 0.0
                )
                is_slow_queue = float(v_item.mean_v_mps) < 2.5
                if is_slow_queue or w_s > 1.0:
                    movement_queued_count[m_key] = movement_queued_count.get(m_key, 0) + 1
                if float(v_item.free_eta_s) <= 4.5:
                    movement_close_count[m_key] = movement_close_count.get(m_key, 0) + 1

                if float(v_item.free_eta_s) <= 10.0 or is_slow_queue:
                    q_bonus = 0.85 if is_slow_queue else 0.0
                    movement_own_pressure[m_key] = (
                        movement_own_pressure.get(m_key, 0.0) + q_bonus + 0.12 * w_s + 0.45 * c_s
                    )

            if is_sig and junc.tls_id is not None:
                red_arr = self._tls_red_elapsed_s.get(junc.tls_id)
                prev_hold = self._tls_holds.get(junc.tls_id)
                now_t = float(belief.sim_time)
                for m_key, links_a in movement_links.items():
                    if movement_spilled.get(m_key, False) or not links_a:
                        continue
                    # Scale movement pressure by multi-lane saturation service capacity (Max-Pressure mu_m * Q_m).
                    mu_scale = min(2.0, 0.65 + 0.35 * float(len(links_a)))
                    base_mp = movement_own_pressure.get(m_key, 0.0) * mu_scale
                    # Long-red anti-starvation floor (> 24s) so minor single-lane turns never starve.
                    if red_arr is not None and movement_queued_count.get(m_key, 0) > 0:
                        max_red_s = max(
                            (float(red_arr[idx]) for idx in links_a if 0 <= idx < len(red_arr)),
                            default=0.0,
                        )
                        if max_red_s > 24.0:
                            base_mp += 0.30 * (min(70.0, max_red_s) - 24.0)
                    # Smooth green-wave continuation up to 26s while active platoon crosses (< 4.5s).
                    if (
                        prev_hold is not None
                        and prev_hold.green_started_at is not None
                        and (now_t - prev_hold.green_started_at) < 26.0
                        and movement_close_count.get(m_key, 0) > 0
                        and set(links_a).issubset(prev_hold.green_links)
                    ):
                        base_mp += 1.60 * mu_scale
                    movement_own_pressure[m_key] = base_mp

            coalition_pressure: dict[tuple[str, int], float] = {}
            for m_key, own_p in movement_own_pressure.items():
                total_p = own_p
                links_a = movement_links.get(m_key, ())
                if is_sig and links_a:
                    for other_key, other_p in movement_own_pressure.items():
                        if other_key == m_key or other_p <= 0.0:
                            continue
                        links_b = movement_links.get(other_key, ())
                        if links_b and are_tls_links_compatible(
                            links_b,
                            frozenset(links_a),
                            junc.tls_link_foe_matrix,
                            junc.junction_id,
                        ):
                            weight_b = 0.25 if other_key[0] == m_key[0] else 0.90
                            total_p += weight_b * other_p
                coalition_pressure[m_key] = total_p

            def _rank_key(v: VehicleBelief) -> tuple[int, float, float, float, str]:
                vid = v.veh_id
                m_key = veh_map_turn.get(vid)
                is_spilled = movement_spilled.get(m_key, False) if m_key is not None else False
                if is_sig:
                    # At signalized TLS junctions, unspilled red-phase queues still compete for green
                    # phase transition; cross-phase separation is enforced by verify_conflict_free_greens.
                    wants_cross = 0 if is_spilled else 1
                    raw_pref = float(maxsum_result.net_crossing_preference.get(vid, 0.0))
                    base_pref = raw_pref if is_spilled else max(-1.5, raw_pref)
                else:
                    mode = maxsum_result.modes.get(vid, LongitudinalMode.CRUISE)
                    wants_cross = 1 if (not is_spilled and mode in (LongitudinalMode.GO_WAVE, LongitudinalMode.CRUISE)) else 0
                    base_pref = float(maxsum_result.net_crossing_preference.get(vid, 0.0))

                coal_p = coalition_pressure.get(m_key, 0.0) if m_key is not None else 0.0
                net_q = base_pref + coal_p
                prec = (
                    float(dbn_snapshot.precedence_scores[vid])
                    if dbn_snapshot is not None and vid in dbn_snapshot.precedence_scores
                    else -float(v.free_eta_s)
                )
                return (-wants_cross, -net_q, -prec, float(v.free_eta_s), vid)

        # Enforce physical same-lane FIFO partial order (lane head MUST precede its followers)
        # while sorting across competing lanes/corridors by priority key.
        veh_by_id = {v.veh_id: v for v in approaching}
        same_lane_follower: dict[str, str] = {}
        for follower_id, lead_id in lane_leader.items():
            if follower_id in veh_by_id and lead_id in veh_by_id:
                same_lane_follower[lead_id] = follower_id

        heap: list[tuple[tuple[int, float, float, float, str], VehicleBelief]] = []
        for v in approaching:
            if v.veh_id not in lane_leader or lane_leader[v.veh_id] not in veh_by_id:
                heapq.heappush(heap, (_rank_key(v), v))

        ordered: list[VehicleBelief] = []
        while heap:
            _, cur = heapq.heappop(heap)
            ordered.append(cur)
            nxt_id = same_lane_follower.get(cur.veh_id)
            if nxt_id is not None:
                nxt_veh = veh_by_id[nxt_id]
                heapq.heappush(heap, (_rank_key(nxt_veh), nxt_veh))
        return ordered

    def _retained_min_hold_greens(
        self,
        junc: JunctionSpec,
        topology: RoadTopology,
        belief: SwarmBeliefState,
        approaching: list[VehicleBelief],
        corridor_q_spill: Mapping[str, float],
        corridor_rem_slots: Mapping[str, int],
        sim_time: float,
        enable_spillback_gate: bool,
    ) -> set[int]:
        assert junc.tls_id is not None
        hold = self._tls_holds.get(junc.tls_id)
        if hold is None or not hold.green_links:
            return set()

        if hold.green_started_at is None and _are_links_live_green(
            hold.green_links, junc.tls_id, belief.tls_states
        ):
            hold.green_started_at = sim_time

        if hold.green_started_at is not None:
            if (sim_time - hold.green_started_at) >= self._cfg.tls_min_green_hold_s - 1e-9:
                return set()
        else:
            max_transition_hold = (
                2.0 * self._cfg.tls_min_green_hold_s + self._cfg.tls_yellow_clearance_s + 1.0
            )
            if (sim_time - hold.requested_at) >= max_transition_hold - 1e-9:
                return set()

        # Retain held unspilled links through tls_min_green_hold_s so scheduler and SumoSimulationAdapter
        # _apply_tls_guard stay synchronized (never braking vehicles in front of a locked live green).
        del approaching
        theta_spill = self._cfg.spillback_threshold

        valid_links: set[int] = set()
        for in_cid in junc.incoming_corridors:
            in_corr = topology.corridors[in_cid]
            for opt in in_corr.turn_options:
                out_corr = topology.corridors[opt.to_corridor]
                spilled = (
                    enable_spillback_gate
                    and not out_corr.is_exit_fringe
                    and (
                        corridor_q_spill.get(opt.to_corridor, 0.0) > theta_spill
                        or corridor_rem_slots.get(opt.to_corridor, 1) <= 0
                    )
                )
                if spilled:
                    continue
                if opt.tls_link_indices and set(opt.tls_link_indices).issubset(hold.green_links):
                    if are_tls_links_compatible(
                        opt.tls_link_indices,
                        valid_links,
                        junc.tls_link_foe_matrix,
                        junc.junction_id,
                    ):
                        valid_links.update(opt.tls_link_indices)
        return valid_links

    def _select_vehicle_tls_links(
        self,
        veh: VehicleBelief,
        corr: CorridorSpec,
        opt_map: TurnOption,
        junc: JunctionSpec,
        active_green_links: set[int],
        corridor_q_spill: Mapping[str, float],
        corridor_rem_slots: Mapping[str, int],
        topology: RoadTopology,
        enable_spillback_gate: bool,
    ) -> tuple[int, ...]:
        del veh, corr, corridor_q_spill, corridor_rem_slots, topology, enable_spillback_gate
        if not opt_map.tls_link_indices:
            return ()
        if not are_tls_links_compatible(
            opt_map.tls_link_indices,
            active_green_links,
            junc.tls_link_foe_matrix,
            junc.junction_id,
        ):
            return ()
        return tuple(opt_map.tls_link_indices)

    def _fill_compatible_greens(
        self,
        junc: JunctionSpec,
        topology: RoadTopology,
        belief: SwarmBeliefState,
        approaching: list[VehicleBelief],
        maxsum_result: MaxSumResult | None,
        dbn_snapshot: DBNStateSnapshot | None,
        active_green_links: set[int],
        corridor_q_spill: Mapping[str, float],
        corridor_rem_slots: Mapping[str, int],
        enable_spillback_gate: bool,
    ) -> None:
        theta_spill = self._cfg.spillback_threshold
        turn_pressure: dict[tuple[str, int], float] = {}
        for veh in approaching:
            in_corr = topology.corridors[veh.corridor_id]
            base_p = 1.0 + 0.08 * min(60.0, max(0.0, float(veh.wait_time_s)))
            if maxsum_result is not None:
                base_p += max(0.0, float(maxsum_result.net_crossing_preference.get(veh.veh_id, 0.0)))
            if dbn_snapshot is not None:
                base_p += 0.5 * max(0.0, float(dbn_snapshot.cascade_score.get(veh.veh_id, 0.0)))
            for opt in in_corr.turn_options:
                prob_t = float(veh.turn_probs[opt.turn_idx])
                if prob_t > 1e-6:
                    key = (veh.corridor_id, opt.turn_idx)
                    turn_pressure[key] = turn_pressure.get(key, 0.0) + base_p * prob_t

        candidates: list[tuple[int, float, float, str, int, tuple[int, ...]]] = []

        for in_cid in junc.incoming_corridors:
            in_corr = topology.corridors[in_cid]
            occ = float(belief.corridors[in_cid].occupancy_ratio) if in_cid in belief.corridors else 0.0
            n_opts = len(in_corr.turn_options)
            for opt in in_corr.turn_options:
                if not opt.tls_link_indices:
                    continue
                out_corr = topology.corridors[opt.to_corridor]
                if not out_corr.is_exit_fringe:
                    if enable_spillback_gate and corridor_q_spill.get(opt.to_corridor, 0.0) > theta_spill:
                        continue
                    if corridor_rem_slots.get(opt.to_corridor, 1) <= 0:
                        continue
                is_prot_left = 1 if (n_opts >= 3 and float(opt.angle_rad) > 0.45) else 0
                mu_links = min(2.0, 0.65 + 0.35 * float(len(opt.tls_link_indices)))
                pressure = (turn_pressure.get((in_cid, opt.turn_idx), 0.0) + 0.5 * occ) * mu_links
                # Activate compatible Dual-Through + Right movements before protected left turns.
                candidates.append(
                    (is_prot_left, -pressure, abs(float(opt.angle_rad)), in_cid, opt.turn_idx, opt.tls_link_indices)
                )

        candidates.sort()
        for _, _, _, _, _, link_tuple in candidates:
            if are_tls_links_compatible(
                link_tuple,
                active_green_links,
                junc.tls_link_foe_matrix,
                junc.junction_id,
            ):
                active_green_links.update(link_tuple)


__all__ = [
    "PriorityDAGScheduler",
    "ScheduleResult",
    "are_tls_links_compatible",
    "verify_conflict_free_greens",
]

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
    since_sim_time: float


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
) -> float:
    """Compute belief-expected turn conflict probability ``P(conflict(a, b))`` at ``junction_id``."""
    if not corr_a.turn_options or not corr_b.turn_options:
        return 0.0
    if corr_a.corridor_id == corr_b.corridor_id and veh_a.lane_index == veh_b.lane_index:
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
            if opt_a.to_corridor == opt_b.to_corridor or topology.turn_conflict(junction_id, opt_a, opt_b):
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

    def reset(self) -> None:
        """Clear per-TLS minimum-green hold state across episodes."""
        self._tls_holds.clear()

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
                lane_leader=lane_leader,
                maxsum_result=maxsum_result,
                dbn_snapshot=dbn_snapshot,
            )

            active_green_links: set[int] = set()
            if junc.tls_id is not None and junc.num_tls_links > 0:
                active_green_links.update(
                    self._retained_min_hold_greens(
                        junc=junc,
                        topology=topology,
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
                    p_conf = _pair_turn_conflict_prob(topology, jid, veh, corr, earlier, earlier_corr)
                    if p_conf > theta_risk:
                        t_start = max(t_start, scheduled_end_s[earlier.veh_id])

                dur = _physical_slot_duration(veh, corr, cfg)
                t_end = t_start + dur

                mode_allows_entry = (
                    True
                    if maxsum_result is None
                    else maxsum_result.modes.get(vid, LongitudinalMode.CRUISE) != LongitudinalMode.YIELD
                )

                # Check TLS green-link compatibility for immediate crossing grant.
                tls_ok = True
                chosen_links: tuple[int, ...] = ()
                if junc.tls_id is not None and junc.num_tls_links > 0:
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
                    if opt_map.tls_link_indices and not chosen_links:
                        tls_ok = False

                if mode_allows_entry and tls_ok and t_start <= float(veh.free_eta_s) + _IMMEDIATE_GRANT_SLACK_S:
                    granted_vehicles.add(vid)
                    active_green_links.update(chosen_links)
                    if not out_corr.is_exit_fringe and float(veh.free_eta_s) <= 4.5:
                        corridor_rem_slots[out_cid] = max(0, corridor_rem_slots.get(out_cid, 1) - 1)
                    scheduled_start_s[vid] = t_start
                    scheduled_end_s[vid] = t_end
                    scheduled_at_junc.append(veh)
                else:
                    scheduled_start_s[vid] = t_start
                    scheduled_end_s[vid] = t_end
                    scheduled_at_junc.append(veh)
                    # Hold at stop-line if wait is substantial or close to stop-line without TLS green.
                    if not tls_ok or t_start > float(veh.free_eta_s) + 2.5 or float(veh.free_eta_s) < 1.8:
                        must_hold_stopline.add(vid)

            # Complete non-conflicting compatible green phase for dual-through progression & idle junctions.
            if junc.tls_id is not None and junc.num_tls_links > 0:
                self._fill_compatible_greens(
                    junc=junc,
                    topology=topology,
                    belief=belief,
                    active_green_links=active_green_links,
                    corridor_q_spill=corridor_q_spill,
                    corridor_rem_slots=corridor_rem_slots,
                    enable_spillback_gate=enable_spillback_gate,
                )
                final_greens = frozenset(active_green_links)
                verify_conflict_free_greens(junc, final_greens)
                prev_hold = self._tls_holds.get(junc.tls_id)
                if prev_hold is None or prev_hold.green_links != final_greens:
                    self._tls_holds[junc.tls_id] = _TlsHoldRecord(
                        green_links=final_greens,
                        since_sim_time=float(belief.sim_time),
                    )
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
        lane_leader: Mapping[str, str],
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
            def _rank_key(v: VehicleBelief) -> tuple[int, float, float, float, str]:
                vid = v.veh_id
                mode = maxsum_result.modes.get(vid, LongitudinalMode.CRUISE)
                wants_cross = 1 if mode in (LongitudinalMode.GO_WAVE, LongitudinalMode.CRUISE) else 0
                net_q = float(maxsum_result.net_crossing_preference.get(vid, 0.0))
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
        approaching: list[VehicleBelief],
        corridor_q_spill: Mapping[str, float],
        corridor_rem_slots: Mapping[str, int],
        sim_time: float,
        enable_spillback_gate: bool,
    ) -> set[int]:
        assert junc.tls_id is not None
        hold = self._tls_holds.get(junc.tls_id)
        if hold is None or (sim_time - hold.since_sim_time) >= self._cfg.tls_min_green_hold_s - 1e-9:
            return set()

        # Keep held green links ONLY while approaching demand exists within 6s AND target corridor is unspilled.
        demanded_corridors = {
            v.corridor_id for v in approaching if float(v.free_eta_s) <= 6.0
        }
        if not demanded_corridors:
            return set()

        valid_links: set[int] = set()
        theta_spill = self._cfg.spillback_threshold
        for in_cid in junc.incoming_corridors:
            if in_cid not in demanded_corridors:
                continue
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
        theta_spill = self._cfg.spillback_threshold
        theta_risk = self._cfg.conflict_risk_threshold

        if not opt_map.tls_link_indices:
            return ()
        if not are_tls_links_compatible(
            opt_map.tls_link_indices,
            active_green_links,
            junc.tls_link_foe_matrix,
            junc.junction_id,
        ):
            return ()

        chosen: list[int] = list(opt_map.tls_link_indices)
        # Also activate secondary plausible turns (>= conflict_risk_threshold) if compatible and unspilled.
        for opt in corr.turn_options:
            if opt.turn_idx == opt_map.turn_idx or not opt.tls_link_indices:
                continue
            if float(veh.turn_probs[opt.turn_idx]) < theta_risk:
                continue
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
            combined = list(dict.fromkeys(chosen + list(opt.tls_link_indices)))
            if are_tls_links_compatible(
                combined,
                active_green_links,
                junc.tls_link_foe_matrix,
                junc.junction_id,
            ):
                chosen = combined

        return tuple(chosen)

    def _fill_compatible_greens(
        self,
        junc: JunctionSpec,
        topology: RoadTopology,
        belief: SwarmBeliefState,
        active_green_links: set[int],
        corridor_q_spill: Mapping[str, float],
        corridor_rem_slots: Mapping[str, int],
        enable_spillback_gate: bool,
    ) -> None:
        theta_spill = self._cfg.spillback_threshold
        candidates: list[tuple[float, float, str, int, tuple[int, ...]]] = []

        for in_cid in junc.incoming_corridors:
            in_corr = topology.corridors[in_cid]
            occ = float(belief.corridors[in_cid].occupancy_ratio) if in_cid in belief.corridors else 0.0
            for opt in in_corr.turn_options:
                if not opt.tls_link_indices:
                    continue
                out_corr = topology.corridors[opt.to_corridor]
                if not out_corr.is_exit_fringe:
                    if enable_spillback_gate and corridor_q_spill.get(opt.to_corridor, 0.0) > theta_spill:
                        continue
                    if corridor_rem_slots.get(opt.to_corridor, 1) <= 0:
                        continue
                # Prefer high-occupancy incoming corridors and straight-through movements.
                candidates.append(
                    (-occ, abs(float(opt.angle_rad)), in_cid, opt.turn_idx, opt.tls_link_indices)
                )

        candidates.sort()
        for _, _, _, _, link_tuple in candidates:
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

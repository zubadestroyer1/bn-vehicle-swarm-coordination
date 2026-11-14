"""Comprehensive unit and invariant test suite for ``swarm.control`` (Wave 2B Slice 3).

Verifies:
1. Zero stopped-queue creep under noisy stopped leaders & sensor dropouts (closing D6).
2. Strict conflict-free ``tls_desired_greens`` across multi-link junctions & dual-through phases.
3. Predictive DBN spillback backpressure holding upstream approaches while cross-streets flow.
4. 2-hop ``cascade_score`` platoon priority winning over conflicting isolated arrivals, plus
   zero ``+1.5 * delays`` slot bloat (closing D2 & D5).
5. Firewall-safe DBN dynamic route cost broadcasting & ``fixed_route`` ablation gate.
6. Shared ``B_comm`` active-pool VoI V2X polling symmetry & cooldown enforcement (closing D7).
7. ``PolicyControllerProtocol`` compliance across all 10 ``POLICY_NAMES``, ``no_temporal_prior``
   isolation, deterministic ``reset()``, fail-loud error guards, and Boundary Lock C AST audit.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

import swarm.control as control_pkg
from swarm.config import ControlConfig, DBNConfig, ExperimentConfig, ObservationConfig, ScenarioConfig
from swarm.contracts import (
    CorridorBeliefStats,
    CorridorSpec,
    DBNEdge,
    DBNStateSnapshot,
    DBNTemporalMemory,
    DepartedVehicleEvent,
    JunctionSpec,
    LongitudinalMode,
    NoisyVehicleObservation,
    ObservationBatch,
    POLICY_NAMES,
    PolicyControllerProtocol,
    RelationType,
    RoadTopology,
    SwarmBeliefState,
    TurnOption,
    VehicleBelief,
)
from swarm.control import (
    ActuatedSignalController,
    FCFSReservationController,
    FactoredSwarmCoordinator,
    FixedSignalController,
    ObservableSafetyFilter,
    PriorityDAGScheduler,
    VoIV2XSelector,
    build_policy_controller,
    compute_broadcast_edge_costs,
    compute_ecoglide_speeds,
    compute_expected_outgoing_spillback,
    run_damped_maxsum,
    verify_conflict_free_greens,
)
from swarm.control.safety import select_actuated_target_speeds
from swarm.control.scheduler import _pair_turn_conflict_prob
from swarm.env.belief import CityBeliefTracker
from swarm.logging_utils import (
    CheckpointIntegrityError,
    ConfigValidationError,
    DAGInvariantError,
    NumericalInvariantError,
    TopologyIntegrityError,
)


def _build_multi_link_test_topology() -> RoadTopology:
    """Construct a 4-way 12-link central signalized junction + split-pocket east arterial."""
    # TLS link layout at J_center (12 links):
    # North approach (C_N_in): 0=Right(->W_out), 1=Thru(->S_out), 2=Left(->E_conn)
    # South approach (C_S_in): 3=Right(->E_conn), 4=Thru(->N_out), 5=Left(->W_out)
    # West  approach (C_W_in): 6=Right(->S_out), 7=Thru(->E_conn), 8=Left(->N_out)
    # East  approach (C_E_in): 9=Right(->N_out), 10=Thru(->W_out), 11=Left(->S_out)
    foe = np.zeros((12, 12), dtype=bool)

    def _add_conflict(a: int, b: int) -> None:
        foe[a, b] = True
        foe[b, a] = True

    ns_links = {0, 1, 2, 3, 4, 5}
    ew_links = {6, 7, 8, 9, 10, 11}
    # All NS movements conflict with EW movements except non-merging right-turn pairs if desired;
    # mark full NS vs EW crossing/merging conflicts rigorously:
    for a in ns_links:
        for b in ew_links:
            _add_conflict(a, b)

    # Left turns conflict with opposing through and opposing right turns merging into same exit:
    _add_conflict(2, 4)  # N-left vs S-thru
    _add_conflict(2, 3)  # N-left vs S-right (both enter C_E_conn)
    _add_conflict(5, 1)  # S-left vs N-thru
    _add_conflict(5, 0)  # S-left vs N-right (both enter C_W_out)
    _add_conflict(2, 5)  # opposing lefts intersect inside compact box
    _add_conflict(8, 10)  # W-left vs E-thru
    _add_conflict(8, 9)  # W-left vs E-right (both enter C_N_out)
    _add_conflict(11, 7)  # E-left vs W-thru
    _add_conflict(11, 6)  # E-left vs W-right (both enter C_S_out)
    _add_conflict(8, 11)  # opposing EW lefts

    # Downstream J_east has 1 TLS link (0: C_E_conn -> C_E_far_out)
    foe_east = np.zeros((1, 1), dtype=bool)
    empty_foe = np.zeros((0, 0), dtype=bool)

    junctions = {
        "J_N_fringe": JunctionSpec("J_N_fringe", None, (0.0, 150.0), (), ("C_N_in",), 0, empty_foe, empty_foe),
        "J_S_fringe": JunctionSpec("J_S_fringe", None, (0.0, -150.0), ("C_S_out",), ("C_S_in",), 0, empty_foe, empty_foe),
        "J_W_fringe": JunctionSpec("J_W_fringe", None, (-150.0, 0.0), ("C_W_out",), ("C_W_in",), 0, empty_foe, empty_foe),
        "J_E_fringe": JunctionSpec("J_E_fringe", None, (300.0, 0.0), ("C_E_far_out",), ("C_E_in",), 0, empty_foe, empty_foe),
        "J_N_exit": JunctionSpec("J_N_exit", None, (10.0, 150.0), ("C_N_out",), (), 0, empty_foe, empty_foe),
        "J_center": JunctionSpec(
            junction_id="J_center",
            tls_id="tls_center",
            coord_xy=(0.0, 0.0),
            incoming_corridors=("C_N_in", "C_S_in", "C_W_in", "C_E_in"),
            outgoing_corridors=("C_N_out", "C_S_out", "C_W_out", "C_E_conn"),
            num_tls_links=12,
            tls_link_foe_matrix=foe,
            req_foe_matrix=foe.copy(),
        ),
        "J_east": JunctionSpec(
            junction_id="J_east",
            tls_id="tls_east",
            coord_xy=(140.0, 0.0),
            incoming_corridors=("C_E_conn",),
            outgoing_corridors=("C_E_far_out",),
            num_tls_links=1,
            tls_link_foe_matrix=foe_east,
            req_foe_matrix=foe_east.copy(),
        ),
    }

    corridors = {
        "C_N_in": CorridorSpec(
            corridor_id="C_N_in",
            edges=("e_N_in",),
            edge_offsets_m=(0.0,),
            from_junction="J_N_fringe",
            to_junction="J_center",
            length_m=140.0,
            base_lanes=2,
            speed_limit_mps=13.9,
            turn_options=(
                TurnOption(0, "e_N_in", "C_W_out", "e_W_out", -math.pi / 2, "tls_center", (0,), (0,)),
                TurnOption(1, "e_N_in", "C_S_out", "e_S_out", 0.0, "tls_center", (1,), (1,)),
                TurnOption(2, "e_N_in", "C_E_conn", "e_E_trunk", math.pi / 2, "tls_center", (2,), (2,)),
            ),
            is_entry_fringe=True,
            is_exit_fringe=False,
        ),
        "C_S_in": CorridorSpec(
            corridor_id="C_S_in",
            edges=("e_S_in",),
            edge_offsets_m=(0.0,),
            from_junction="J_S_fringe",
            to_junction="J_center",
            length_m=140.0,
            base_lanes=2,
            speed_limit_mps=13.9,
            turn_options=(
                TurnOption(0, "e_S_in", "C_E_conn", "e_E_trunk", -math.pi / 2, "tls_center", (3,), (3,)),
                TurnOption(1, "e_S_in", "C_N_out", "e_N_out", 0.0, "tls_center", (4,), (4,)),
                TurnOption(2, "e_S_in", "C_W_out", "e_W_out", math.pi / 2, "tls_center", (5,), (5,)),
            ),
            is_entry_fringe=True,
            is_exit_fringe=False,
        ),
        "C_W_in": CorridorSpec(
            corridor_id="C_W_in",
            edges=("e_W_in",),
            edge_offsets_m=(0.0,),
            from_junction="J_W_fringe",
            to_junction="J_center",
            length_m=140.0,
            base_lanes=2,
            speed_limit_mps=13.9,
            turn_options=(
                TurnOption(0, "e_W_in", "C_S_out", "e_S_out", -math.pi / 2, "tls_center", (6,), (6,)),
                TurnOption(1, "e_W_in", "C_E_conn", "e_E_trunk", 0.0, "tls_center", (7,), (7,)),
                TurnOption(2, "e_W_in", "C_N_out", "e_N_out", math.pi / 2, "tls_center", (8,), (8,)),
            ),
            is_entry_fringe=True,
            is_exit_fringe=False,
        ),
        "C_E_in": CorridorSpec(
            corridor_id="C_E_in",
            edges=("e_E_in",),
            edge_offsets_m=(0.0,),
            from_junction="J_E_fringe",
            to_junction="J_center",
            length_m=140.0,
            base_lanes=2,
            speed_limit_mps=13.9,
            turn_options=(
                TurnOption(0, "e_E_in", "C_N_out", "e_N_out", -math.pi / 2, "tls_center", (9,), (9,)),
                TurnOption(1, "e_E_in", "C_W_out", "e_W_out", 0.0, "tls_center", (10,), (10,)),
                TurnOption(2, "e_E_in", "C_S_out", "e_S_out", math.pi / 2, "tls_center", (11,), (11,)),
            ),
            is_entry_fringe=True,
            is_exit_fringe=False,
        ),
        # Split trunk + turn-pocket arterial connecting J_center -> J_east:
        "C_E_conn": CorridorSpec(
            corridor_id="C_E_conn",
            edges=("e_E_trunk", "e_E_pocket"),
            edge_offsets_m=(0.0, 90.0),
            from_junction="J_center",
            to_junction="J_east",
            length_m=140.0,
            base_lanes=2,
            speed_limit_mps=13.9,
            turn_options=(
                TurnOption(0, "e_E_pocket", "C_E_far_out", "e_E_far_out", 0.0, "tls_east", (0,), (0,)),
            ),
            is_entry_fringe=False,
            is_exit_fringe=False,
        ),
        # Exit-fringe corridors (Boundary Lock A: is_exit_fringe=True, turn_options=()):
        "C_N_out": CorridorSpec(
            corridor_id="C_N_out",
            edges=("e_N_out",),
            edge_offsets_m=(0.0,),
            from_junction="J_center",
            to_junction="J_N_exit",
            length_m=120.0,
            base_lanes=2,
            speed_limit_mps=13.9,
            turn_options=(),
            is_entry_fringe=False,
            is_exit_fringe=True,
        ),
        "C_S_out": CorridorSpec(
            corridor_id="C_S_out",
            edges=("e_S_out",),
            edge_offsets_m=(0.0,),
            from_junction="J_center",
            to_junction="J_S_fringe",
            length_m=120.0,
            base_lanes=2,
            speed_limit_mps=13.9,
            turn_options=(),
            is_entry_fringe=False,
            is_exit_fringe=True,
        ),
        "C_W_out": CorridorSpec(
            corridor_id="C_W_out",
            edges=("e_W_out",),
            edge_offsets_m=(0.0,),
            from_junction="J_center",
            to_junction="J_W_fringe",
            length_m=120.0,
            base_lanes=2,
            speed_limit_mps=13.9,
            turn_options=(),
            is_entry_fringe=False,
            is_exit_fringe=True,
        ),
        "C_E_far_out": CorridorSpec(
            corridor_id="C_E_far_out",
            edges=("e_E_far_out",),
            edge_offsets_m=(0.0,),
            from_junction="J_east",
            to_junction="J_E_fringe",
            length_m=120.0,
            base_lanes=2,
            speed_limit_mps=13.9,
            turn_options=(),
            is_entry_fringe=False,
            is_exit_fringe=True,
        ),
    }

    edge_to_corridor: dict[str, str] = {}
    edge_offset: dict[str, float] = {}
    successor_corridors: dict[str, tuple[str, ...]] = {}
    predecessor_lists: dict[str, list[str]] = {cid: [] for cid in corridors}

    for cid, spec in corridors.items():
        for eid, off in zip(spec.edges, spec.edge_offsets_m, strict=True):
            edge_to_corridor[eid] = cid
            edge_offset[eid] = off
        succ = tuple(dict.fromkeys(opt.to_corridor for opt in spec.turn_options))
        successor_corridors[cid] = succ
        for sc in succ:
            predecessor_lists[sc].append(cid)

    predecessor_corridors = {cid: tuple(preds) for cid, preds in predecessor_lists.items()}
    return RoadTopology(
        network_id="test_grid_center_east",
        net_xml_path="synthetic_test.net.xml",
        poly_xml_path=None,
        corridors=corridors,
        edge_to_corridor=edge_to_corridor,
        edge_offset_in_corridor=edge_offset,
        junctions=junctions,
        successor_corridors=successor_corridors,
        predecessor_corridors=predecessor_corridors,
    )


def _make_vehicle_belief(
    veh_id: str,
    corridor_id: str,
    edge_id: str,
    *,
    entered_at: float = 0.0,
    lane_index: int = 0,
    mean_s_m: float = 100.0,
    mean_v_mps: float = 10.0,
    pos_var: float = 0.8,
    vel_var: float = 0.25,
    obs_age_s: float = 0.0,
    wait_time_s: float = 0.0,
    turn_probs: tuple[float, ...] = (0.1, 0.8, 0.1),
    free_eta_s: float = 3.0,
    eta_std_s: float = 0.25,
    tail_blockage_score: float = 0.0,
    leader_id: str | None = None,
    noisy_leader_gap_m: float | None = None,
    noisy_leader_speed_mps: float | None = None,
    last_v2x_poll_time: float = -1e9,
    in_junction_id: str | None = None,
) -> VehicleBelief:
    cov = np.array([[pos_var, 0.0], [0.0, vel_var]], dtype=np.float64)
    return VehicleBelief(
        veh_id=veh_id,
        entered_at=entered_at,
        corridor_id=corridor_id,
        edge_id=edge_id,
        in_junction_id=in_junction_id,
        lane_index=lane_index,
        length_m=4.5,
        width_m=1.85,
        mean_s_m=mean_s_m,
        mean_v_mps=mean_v_mps,
        cov_2x2=cov,
        accel_est_mps2=0.0,
        obs_age_s=obs_age_s,
        wait_time_s=wait_time_s,
        turn_probs=turn_probs,
        free_eta_s=free_eta_s,
        eta_std_s=eta_std_s,
        tail_blockage_score=tail_blockage_score,
        leader_id=leader_id,
        noisy_leader_gap_m=noisy_leader_gap_m,
        noisy_leader_speed_mps=noisy_leader_speed_mps,
        last_v2x_poll_time=last_v2x_poll_time,
    )


def _make_belief_state(
    topology: RoadTopology,
    vehicles: list[VehicleBelief],
    *,
    sim_time: float = 10.0,
    dt: float = 0.5,
    open_lanes_by_edge: dict[str, int] | None = None,
    tail_overrides: dict[str, float] | None = None,
) -> SwarmBeliefState:
    open_lanes = open_lanes_by_edge or {}
    veh_map = {v.veh_id: v for v in vehicles}
    by_corr: dict[str, list[VehicleBelief]] = {cid: [] for cid in topology.corridors}
    for v in vehicles:
        by_corr[v.corridor_id].append(v)

    corr_stats: dict[str, CorridorBeliefStats] = {}
    for cid, spec in topology.corridors.items():
        c_vehs = by_corr[cid]
        eff_lanes = topology.effective_lanes(cid, open_lanes)
        cap = topology.effective_capacity(cid, open_lanes)
        occ = min(1.0, len(c_vehs) / float(max(1, cap)))
        tail_p = (tail_overrides or {}).get(cid, 0.0)
        mean_v = (
            sum(v.mean_v_mps for v in c_vehs) / len(c_vehs)
            if c_vehs
            else spec.speed_limit_mps
        )
        corr_stats[cid] = CorridorBeliefStats(
            corridor_id=cid,
            active_count=len(c_vehs),
            effective_lanes=eff_lanes,
            storage_capacity=cap,
            occupancy_ratio=occ,
            tail_blockage_prob=tail_p,
            expected_inflow=0.0,
            mean_speed_mps=mean_v,
        )

    return SwarmBeliefState(
        sim_time=sim_time,
        dt=dt,
        vehicles=veh_map,
        corridors=corr_stats,
        open_lanes_by_edge=dict(open_lanes),
        tls_states={"tls_center": "GGGGGGGGGGGG", "tls_east": "G"},
    )


def _make_dbn_snapshot(
    belief: SwarmBeliefState,
    topology: RoadTopology,
    *,
    edges: tuple[DBNEdge, ...] = (),
    precedence_scores: dict[str, float] | None = None,
    cascade_score: dict[str, float] | None = None,
    corridor_spillback_pred: dict[str, float] | None = None,
    delay_post: dict[str, float] | None = None,
    expected_delay_s: dict[str, float] | None = None,
    with_memory: bool = False,
) -> DBNStateSnapshot:
    vids = tuple(sorted(belief.vehicles))
    prec = precedence_scores or {vid: -float(belief.vehicles[vid].free_eta_s) for vid in vids}
    by_dst: dict[str, list[DBNEdge]] = {vid: [] for vid in vids}
    by_src: dict[str, list[DBNEdge]] = {vid: [] for vid in vids}
    for e in edges:
        by_dst.setdefault(e.dst_id, []).append(e)
        by_src.setdefault(e.src_id, []).append(e)

    d_post = delay_post or {vid: 0.15 for vid in vids}
    exp_d = expected_delay_s or {vid: 1.0 for vid in vids}
    casc = cascade_score or {vid: 0.0 for vid in vids}
    spill = {
        cid: (corridor_spillback_pred or {}).get(cid, belief.corridors[cid].tail_blockage_prob)
        for cid in topology.corridors
    }
    mem = (
        DBNTemporalMemory(
            sim_time=belief.sim_time,
            veh_hidden={vid: np.ones((4,), dtype=np.float32) for vid in vids},
            corridor_hidden={},
            edge_hidden={},
        )
        if with_memory
        else None
    )
    return DBNStateSnapshot(
        sim_time=belief.sim_time,
        veh_ids=vids,
        precedence_scores=prec,
        edges=edges,
        edges_by_dst={k: tuple(v) for k, v in by_dst.items()},
        edges_by_src={k: tuple(v) for k, v in by_src.items()},
        delay_pred=dict(d_post),
        delay_post=dict(d_post),
        expected_delay_s=dict(exp_d),
        cascade_score=dict(casc),
        corridor_spillback_pred=spill,
        raw_edge_probs={(e.src_id, e.dst_id): e.omega for e in edges},
        update_latency_ms=0.8,
        inference_latency_ms=0.4,
        temporal_memory=mem,
    )


class _StubDBNUpdater:
    """Deterministic protocol-compliant ``DBNUpdaterProtocol`` stub for unit testing."""

    def __init__(
        self,
        *,
        corridor_spillback_pred: dict[str, float] | None = None,
        cascade_score: dict[str, float] | None = None,
        edges: tuple[DBNEdge, ...] = (),
        precedence_scores: dict[str, float] | None = None,
    ) -> None:
        self.corridor_spillback_pred = corridor_spillback_pred or {}
        self.cascade_score = cascade_score or {}
        self.edges = edges
        self.precedence_scores = precedence_scores
        self.received_prev_snapshots: list[DBNStateSnapshot | None] = []
        self.reset_calls: int = 0

    def reset(self) -> None:
        self.received_prev_snapshots.clear()
        self.reset_calls += 1

    def update_and_infer(
        self,
        belief: SwarmBeliefState,
        prev_snapshot: DBNStateSnapshot | None,
        topology: RoadTopology,
    ) -> DBNStateSnapshot:
        self.received_prev_snapshots.append(prev_snapshot)
        active_edges = tuple(
            e for e in self.edges if e.src_id in belief.vehicles and e.dst_id in belief.vehicles
        )
        return _make_dbn_snapshot(
            belief=belief,
            topology=topology,
            edges=active_edges,
            precedence_scores=self.precedence_scores,
            cascade_score={vid: self.cascade_score.get(vid, 0.0) for vid in belief.vehicles},
            corridor_spillback_pred=self.corridor_spillback_pred,
            with_memory=True,
        )


def test_zero_queue_creep_under_noisy_stopped_leader() -> None:
    """Verify zero position creep and deadzone activation behind a noisy stopped leader (D6)."""
    topology = _build_multi_link_test_topology()
    ctrl_cfg = ControlConfig()
    obs_cfg = ObservationConfig()
    safety_filter = ObservableSafetyFilter(ctrl_cfg)
    tracker = CityBeliefTracker(topology=topology, observation_config=obs_cfg)
    rng = np.random.default_rng(42)

    dt = 0.5
    true_lead_s = 132.0
    true_follow_s = 125.5  # bumper gap = 132.0 - 125.5 - 4.5 = 2.0m (exact SUMO standstill min_gap s_0)
    true_follow_v = 0.08  # residual sub-deadzone crawl

    total_deadzone = 0
    total_clamps = 0
    filtered_positions: list[float] = []
    warmup_ticks = 6

    for step in range(warmup_ticks + 24):
        sim_t = step * dt
        departed = (
            (
                DepartedVehicleEvent("lead", 0.0, "e_W_in", 0, 4.5, 1.85),
                DepartedVehicleEvent("follow", 0.0, "e_W_in", 0, 4.5, 1.85),
            )
            if step == 0
            else ()
        )

        # Simulate Gaussian radar noise + intermittent radar dropout every 4th post-warmup step
        radar_list: list[NoisyVehicleObservation] = [
            NoisyVehicleObservation(
                veh_id="lead",
                obs_time=sim_t,
                edge_id="e_W_in",
                corridor_id="C_W_in",
                in_junction_id=None,
                lane_index=0,
                noisy_corridor_pos_m=float(true_lead_s + rng.normal(0.0, 0.45)),
                noisy_speed_mps=float(max(0.0, rng.normal(0.0, 0.18))),
                pos_noise_std=obs_cfg.pos_noise_std_m,
                speed_noise_std=obs_cfg.speed_noise_std_mps,
                turn_likelihood=(0.1, 0.8, 0.1),
                noisy_leader_gap_m=None,
                noisy_leader_speed_mps=None,
                leader_id=None,
            )
        ]
        is_dropout = step >= warmup_ticks and (step - warmup_ticks) % 4 == 3
        if not is_dropout:
            noisy_gap = max(1.6, min(2.3, (true_lead_s - true_follow_s - 4.5) + float(rng.normal(0.0, 0.18))))
            radar_list.append(
                NoisyVehicleObservation(
                    veh_id="follow",
                    obs_time=sim_t,
                    edge_id="e_W_in",
                    corridor_id="C_W_in",
                    in_junction_id=None,
                    lane_index=0,
                    noisy_corridor_pos_m=float(true_follow_s + rng.normal(0.0, 0.40)),
                    noisy_speed_mps=float(max(0.0, true_follow_v + rng.normal(0.0, 0.15))),
                    pos_noise_std=obs_cfg.pos_noise_std_m,
                    speed_noise_std=obs_cfg.speed_noise_std_mps,
                    turn_likelihood=(0.1, 0.8, 0.1),
                    noisy_leader_gap_m=noisy_gap,
                    noisy_leader_speed_mps=float(max(0.0, rng.normal(0.0, 0.12))),
                    leader_id="lead",
                )
            )

        batch = ObservationBatch(
            sim_time=sim_t,
            dt=dt,
            active_veh_ids=("follow", "lead"),
            departed=departed,
            arrived_veh_ids=(),
            radar_obs=tuple(radar_list),
            v2x_packets=(),
            open_lanes_by_edge={"e_W_in": 2},
            tls_states={"tls_center": "rrrrrrrrrrrr", "tls_east": "G"},
        )
        belief = tracker.observe(batch)
        # Even if upstream planner requests full 13.9 m/s speed, safety filter must lock standstill:
        res = safety_filter.apply(
            belief=belief,
            topology=topology,
            desired_speeds={"lead": 0.0, "follow": 13.9},
            must_hold_stopline=frozenset({"lead"}),
        )

        assert res.target_speeds["lead"] == 0.0
        if step >= warmup_ticks:
            assert res.target_speeds["follow"] == 0.0
            assert -ctrl_cfg.max_decel_mps2 <= res.post_safety_accels["follow"] <= 0.0
            true_follow_v = res.target_speeds["follow"]
            true_follow_s += true_follow_v * dt
            total_deadzone += res.zero_speed_deadzone_count
            total_clamps += res.safety_clamp_count
            filtered_positions.append(belief.vehicles["follow"].mean_s_m)

        tracker.predict(dt=dt, post_safety_accels=res.post_safety_accels, sim_time=sim_t)

    assert total_deadzone > 0
    assert total_clamps == 24
    assert true_follow_s == pytest.approx(125.5, abs=1e-9)
    assert max(filtered_positions) < true_lead_s - 4.5

    # Verify immediate standstill queue discharge once the leader accelerates on green startup:
    moving_lead = _make_vehicle_belief("m_lead", "C_W_in", "e_W_in", mean_s_m=132.25, mean_v_mps=1.2)
    queued_follow = _make_vehicle_belief(
        "q_follow",
        "C_W_in",
        "e_W_in",
        mean_s_m=125.5,
        mean_v_mps=0.0,
        leader_id="m_lead",
        noisy_leader_gap_m=2.25,
        noisy_leader_speed_mps=1.2,
    )
    rel_res = safety_filter.apply(
        _make_belief_state(topology, [queued_follow, moving_lead]),
        topology,
        desired_speeds={"m_lead": 13.9, "q_follow": 13.9},
    )
    assert rel_res.target_speeds["q_follow"] > ctrl_cfg.zero_speed_deadzone_mps

    # Also verify hard physical deceleration clamp [-max_decel_mps2, comfort_accel_mps2] at high speed:
    fast_follow = _make_vehicle_belief(
        "fast_f", "C_W_in", "e_W_in",
        mean_s_m=118.0, mean_v_mps=12.0,
        leader_id="stopped_l", noisy_leader_gap_m=3.0, noisy_leader_speed_mps=0.0,
    )
    stopped_lead = _make_vehicle_belief("stopped_l", "C_W_in", "e_W_in", mean_s_m=125.5, mean_v_mps=0.0)
    em_res = safety_filter.apply(
        _make_belief_state(topology, [fast_follow, stopped_lead]),
        topology,
        desired_speeds={"fast_f": 13.9, "stopped_l": 0.0},
    )
    assert em_res.post_safety_accels["fast_f"] == pytest.approx(-ctrl_cfg.max_decel_mps2)
    assert em_res.safety_clamp_count >= 1


def test_conflict_free_tls_greens_on_multilink_junction() -> None:
    """Verify pairwise conflict-free TLS green link sets across dense 4-approach demand."""
    topology = _build_multi_link_test_topology()
    junc_center = topology.junctions["J_center"]

    obs_list = [
        NoisyVehicleObservation("v_N_thru", 5.0, "e_N_in", "C_N_in", None, 0, 128.0, 11.0, 0.8, 0.3, (0.05, 0.90, 0.05), None, None, None),
        NoisyVehicleObservation("v_N_left", 5.0, "e_N_in", "C_N_in", None, 1, 126.0, 10.5, 0.8, 0.3, (0.05, 0.10, 0.85), None, None, None),
        NoisyVehicleObservation("v_S_thru", 5.0, "e_S_in", "C_S_in", None, 0, 129.0, 11.2, 0.8, 0.3, (0.05, 0.90, 0.05), None, None, None),
        NoisyVehicleObservation("v_W_thru", 5.0, "e_W_in", "C_W_in", None, 0, 125.0, 10.0, 0.8, 0.3, (0.05, 0.90, 0.05), None, None, None),
        NoisyVehicleObservation("v_E_left", 5.0, "e_E_in", "C_E_in", None, 1, 124.0, 9.5, 0.8, 0.3, (0.05, 0.10, 0.85), None, None, None),
    ]
    departed = tuple(
        DepartedVehicleEvent(o.veh_id, 5.0, o.edge_id, o.lane_index, 4.5, 1.85) for o in obs_list
    )
    batch = ObservationBatch(
        sim_time=5.0,
        dt=0.5,
        active_veh_ids=tuple(o.veh_id for o in obs_list),
        departed=departed,
        arrived_veh_ids=(),
        radar_obs=tuple(obs_list),
        v2x_packets=(),
        open_lanes_by_edge={},
        tls_states={"tls_center": "GGGGGGGGGGGG", "tls_east": "G"},
    )

    fcfs_ctrl = FCFSReservationController(topology=topology)
    dbn_ctrl = FactoredSwarmCoordinator(
        policy_name="neural_dbn",
        topology=topology,
        dbn_updater=_StubDBNUpdater(),
    )

    for ctrl in (fcfs_ctrl, dbn_ctrl):
        act = ctrl.act(batch)
        assert act.tls_desired_greens is not None
        greens = frozenset(act.tls_desired_greens["tls_center"])
        assert len(greens) > 0
        verify_conflict_free_greens(junc_center, greens)
        for l_a in greens:
            for l_b in greens:
                if l_a != l_b:
                    assert not bool(junc_center.tls_link_foe_matrix[l_a, l_b])

    # Compatible opposing N->S (link 1) and S->N (link 4) through links can be co-activated cleanly:
    dbn_greens = dbn_ctrl.act(
        ObservationBatch(
            sim_time=5.5,
            dt=0.5,
            active_veh_ids=tuple(o.veh_id for o in obs_list),
            departed=(),
            arrived_veh_ids=(),
            radar_obs=tuple(
                NoisyVehicleObservation(
                    o.veh_id, 5.5, o.edge_id, o.corridor_id, None, o.lane_index,
                    o.noisy_corridor_pos_m + 2.0, o.noisy_speed_mps, 0.8, 0.3, o.turn_likelihood,
                    None, None, None,
                )
                for o in obs_list
            ),
            v2x_packets=(),
            open_lanes_by_edge={},
            tls_states={"tls_center": "GGGGGGGGGGGG", "tls_east": "G"},
        )
    ).tls_desired_greens
    assert dbn_greens is not None
    assert {1, 4}.issubset(dbn_greens["tls_center"])
    # Conflicting N-left (link 2) and W-thru (link 7) are strictly excluded while {1, 4} are green:
    assert 2 not in dbn_greens["tls_center"]
    assert 7 not in dbn_greens["tls_center"]


def test_spillback_backpressure_holds_upstream_and_releases_cross_street() -> None:
    """Verify predictive spillback q_out > theta_spill holds W->E approach and grants N->S cross-street."""
    topology = _build_multi_link_test_topology()
    # West->East vehicle arrived earlier & is closer to stop-line, heading into C_E_conn (link 7).
    # North->South cross-street vehicle arrived later, heading into unspilled C_S_out (link 1).
    batch = ObservationBatch(
        sim_time=10.0,
        dt=0.5,
        active_veh_ids=("v_W_early", "v_N_cross"),
        departed=(
            DepartedVehicleEvent("v_W_early", 10.0, "e_W_in", 0, 4.5, 1.85),
            DepartedVehicleEvent("v_N_cross", 10.0, "e_N_in", 0, 4.5, 1.85),
        ),
        arrived_veh_ids=(),
        radar_obs=(
            NoisyVehicleObservation(
                "v_W_early", 10.0, "e_W_in", "C_W_in", None, 0,
                126.0, 11.5, 0.5, 0.2, (0.02, 0.96, 0.02), None, None, None,
            ),
            NoisyVehicleObservation(
                "v_N_cross", 10.0, "e_N_in", "C_N_in", None, 0,
                116.0, 11.0, 0.5, 0.2, (0.02, 0.96, 0.02), None, None, None,
            ),
        ),
        v2x_packets=(),
        open_lanes_by_edge={},
        tls_states={"tls_center": "GGGGGGGGGGGG", "tls_east": "G"},
    )

    # DBN predicts severe downstream spillback on C_E_conn (0.88 > theta_spill=0.55) BEFORE storage is full!
    spilled_updater = _StubDBNUpdater(
        corridor_spillback_pred={"C_E_conn": 0.88, "C_S_out": 0.0},
        precedence_scores={"v_W_early": 2.0, "v_N_cross": 1.0},
    )
    coord = FactoredSwarmCoordinator(
        policy_name="neural_dbn",
        topology=topology,
        dbn_updater=spilled_updater,
    )
    act_dbn = coord.act(batch)
    sched_dbn = coord.last_schedule
    assert sched_dbn is not None
    assert act_dbn.tls_desired_greens is not None

    # W->E link 7 is withheld; v_W_early is held behind stop-line; N->S cross-street (link 1) gets green!
    assert "v_W_early" in sched_dbn.must_hold_stopline
    assert "v_W_early" not in sched_dbn.granted_vehicles
    assert 7 not in act_dbn.tls_desired_greens["tls_center"]
    assert "v_N_cross" in sched_dbn.granted_vehicles
    assert 1 in act_dbn.tls_desired_greens["tls_center"]

    # Contrast: FCFS AIM without predictive DBN spillback grants W->E link 7 and blocks N->S link 1!
    fcfs = FCFSReservationController(topology=topology)
    act_fcfs = fcfs.act(batch)
    assert act_fcfs.tls_desired_greens is not None
    assert 7 in act_fcfs.tls_desired_greens["tls_center"]
    assert 1 not in act_fcfs.tls_desired_greens["tls_center"]


def test_platoon_cascade_score_priority_wins_over_isolated_vehicle_and_no_delay_slot_bloat() -> None:
    """Verify 2-hop cascade_score flips conflict priority to platoon leader and D2 slot duration is unbloated."""
    topology = _build_multi_link_test_topology()
    ctrl_cfg = ControlConfig()

    # Isolated N->S vehicle slightly closer (free_eta_s=2.2s) vs W->E platoon of 4 vehicles (lead free_eta_s=2.5s)
    v_iso = _make_vehicle_belief(
        "v_iso", "C_N_in", "e_N_in",
        mean_s_m=112.0, mean_v_mps=12.5, free_eta_s=2.2, turn_probs=(0.02, 0.96, 0.02),
    )
    v_plat_0 = _make_vehicle_belief(
        "v_plat_0", "C_W_in", "e_W_in",
        mean_s_m=109.0, mean_v_mps=12.4, free_eta_s=2.5, turn_probs=(0.02, 0.96, 0.02),
    )
    platoon_followers = [
        _make_vehicle_belief(
            f"v_plat_{k}", "C_W_in", "e_W_in",
            mean_s_m=109.0 - 11.0 * k,
            mean_v_mps=12.4,
            free_eta_s=2.5 + 0.9 * k,
            turn_probs=(0.02, 0.96, 0.02),
            leader_id=f"v_plat_{k - 1}",
            noisy_leader_gap_m=6.5,
            noisy_leader_speed_mps=12.4,
        )
        for k in (1, 2, 3)
    ]
    belief = _make_belief_state(topology, [v_iso, v_plat_0, *platoon_followers])

    # 1. Independent belief (0 edges, cascade_score=0): isolated earlier vehicle wins
    snap_indep = _make_dbn_snapshot(belief, topology, edges=(), cascade_score={})
    ms_indep = run_damped_maxsum(belief, snap_indep, topology, ctrl_cfg)
    sched_indep = PriorityDAGScheduler(ctrl_cfg).decode(
        belief, topology, maxsum_result=ms_indep, dbn_snapshot=snap_indep
    )
    assert "v_iso" in sched_indep.granted_vehicles
    assert 1 in sched_indep.tls_desired_greens["tls_center"]

    # 2. Neural DBN with active platoon edges & 2-hop cascade_score on v_plat_0: platoon wins!
    platoon_edges = (
        DBNEdge("v_plat_0", "v_iso", RelationType.SAME_JUNCTION_CONFLICT, 0.90, 0.75, 0.65, 0.92, 0.0),
        DBNEdge("v_plat_0", "v_plat_1", RelationType.SAME_CORRIDOR, 0.92, 0.80, 0.85, 0.0, 0.92),
        DBNEdge("v_plat_1", "v_plat_2", RelationType.SAME_CORRIDOR, 0.90, 0.80, 0.85, 0.0, 0.92),
        DBNEdge("v_plat_2", "v_plat_3", RelationType.SAME_CORRIDOR, 0.88, 0.80, 0.85, 0.0, 0.92),
    )
    snap_platoon = _make_dbn_snapshot(
        belief,
        topology,
        edges=platoon_edges,
        precedence_scores={
            "v_plat_0": 5.0,
            "v_plat_1": 4.0,
            "v_plat_2": 3.0,
            "v_plat_3": 2.0,
            "v_iso": 1.0,
        },
        cascade_score={"v_plat_0": 3.6, "v_plat_1": 1.9, "v_plat_2": 0.8, "v_plat_3": 0.0, "v_iso": 0.0},
    )
    ms_platoon = run_damped_maxsum(belief, snap_platoon, topology, ctrl_cfg)
    assert ms_platoon.net_crossing_preference["v_plat_0"] > ms_platoon.net_crossing_preference["v_iso"]

    sched_platoon = PriorityDAGScheduler(ctrl_cfg).decode(
        belief, topology, maxsum_result=ms_platoon, dbn_snapshot=snap_platoon
    )
    assert "v_plat_0" in sched_platoon.granted_vehicles
    assert 7 in sched_platoon.tls_desired_greens["tls_center"]
    assert 1 not in sched_platoon.tls_desired_greens["tls_center"]

    # Eco-Glide assigns smooth non-zero glide or yield speed to deferred v_iso
    glide = compute_ecoglide_speeds(belief, topology, sched_platoon, ctrl_cfg, ms_platoon.modes)
    assert 0.0 <= glide.desired_speeds["v_iso"] <= topology.corridors["C_N_in"].speed_limit_mps

    # 3. Verify D2 fix: inflating expected_delay_s by 100s NEVER bloats scheduled crossing slot duration!
    snap_huge_delay = _make_dbn_snapshot(
        belief,
        topology,
        edges=platoon_edges,
        precedence_scores=snap_platoon.precedence_scores,
        cascade_score=snap_platoon.cascade_score,
        expected_delay_s={vid: 100.0 for vid in belief.vehicles},
    )
    sched_huge_delay = PriorityDAGScheduler(ctrl_cfg).decode(
        belief, topology, maxsum_result=ms_platoon, dbn_snapshot=snap_huge_delay
    )
    dur_normal = sched_platoon.scheduled_end_s["v_plat_0"] - sched_platoon.scheduled_start_s["v_plat_0"]
    dur_huge = sched_huge_delay.scheduled_end_s["v_plat_0"] - sched_huge_delay.scheduled_start_s["v_plat_0"]
    assert dur_normal == pytest.approx(dur_huge, abs=1e-12)


def test_dynamic_routing_costs_split_corridors_and_fixed_route_ablation() -> None:
    """Verify fixed_route disables broadcast_edge_costs while neural_dbn emits valid split-edge costs."""
    topology = _build_multi_link_test_topology()
    ctrl_cfg = ControlConfig(reroute_interval_s=5.0)
    veh = _make_vehicle_belief("v1", "C_E_conn", "e_E_trunk", turn_probs=(1.0,))
    belief = _make_belief_state(topology, [veh], sim_time=10.0)
    snap_clear = _make_dbn_snapshot(belief, topology, corridor_spillback_pred={"C_E_conn": 0.05})
    snap_spilled = _make_dbn_snapshot(belief, topology, corridor_spillback_pred={"C_E_conn": 0.90})

    for static_pol in ("fixed_route", "fixed_signal", "actuated_signal", "fcfs_reservation"):
        assert compute_broadcast_edge_costs(
            belief, snap_spilled, topology, ctrl_cfg, policy_name=static_pol
        ) is None

    costs_clear = compute_broadcast_edge_costs(
        belief, snap_clear, topology, ctrl_cfg, policy_name="neural_dbn"
    )
    costs_spilled = compute_broadcast_edge_costs(
        belief, snap_spilled, topology, ctrl_cfg, policy_name="neural_dbn"
    )
    assert costs_clear is not None and costs_spilled is not None
    assert set(costs_clear.keys()) == set(topology.edge_to_corridor.keys())
    assert all(math.isfinite(val) and val > 0.0 for val in costs_clear.values())

    # Split corridor C_E_conn = ("e_E_trunk" [90m], "e_E_pocket" [50m]) responds monotonically to spillback:
    clear_conn_total = costs_clear["e_E_trunk"] + costs_clear["e_E_pocket"]
    spilled_conn_total = costs_spilled["e_E_trunk"] + costs_spilled["e_E_pocket"]
    assert spilled_conn_total > clear_conn_total + 10.0
    assert costs_spilled["e_E_trunk"] / costs_spilled["e_E_pocket"] == pytest.approx(90.0 / 50.0, rel=1e-6)

    # Closed-loop coordinator check: fixed_route returns None, neural_dbn respects reroute_interval_s
    updater = _StubDBNUpdater(corridor_spillback_pred={"C_E_conn": 0.8})
    fixed_coord = FactoredSwarmCoordinator("fixed_route", topology, updater, control_config=ctrl_cfg)
    neural_coord = FactoredSwarmCoordinator("neural_dbn", topology, _StubDBNUpdater(), control_config=ctrl_cfg)

    obs_t0 = ObservationBatch(
        sim_time=0.0,
        dt=0.5,
        active_veh_ids=("v1",),
        departed=(DepartedVehicleEvent("v1", 0.0, "e_W_in", 0, 4.5, 1.85),),
        arrived_veh_ids=(),
        radar_obs=(
            NoisyVehicleObservation("v1", 0.0, "e_W_in", "C_W_in", None, 0, 80.0, 10.0, 0.8, 0.3, (0.1, 0.8, 0.1), None, None, None),
        ),
        v2x_packets=(),
        open_lanes_by_edge={},
        tls_states={"tls_center": "GGGGGGGGGGGG", "tls_east": "G"},
    )
    obs_t1 = ObservationBatch(
        sim_time=1.0,
        dt=0.5,
        active_veh_ids=("v1",),
        departed=(),
        arrived_veh_ids=(),
        radar_obs=(
            NoisyVehicleObservation("v1", 1.0, "e_W_in", "C_W_in", None, 0, 90.0, 10.0, 0.8, 0.3, (0.1, 0.8, 0.1), None, None, None),
        ),
        v2x_packets=(),
        open_lanes_by_edge={},
        tls_states={"tls_center": "GGGGGGGGGGGG", "tls_east": "G"},
    )
    assert fixed_coord.act(obs_t0).broadcast_edge_costs is None
    assert neural_coord.act(obs_t0).broadcast_edge_costs is not None
    # Within reroute_interval_s (1.0s < 5.0s), broadcast is suppressed:
    assert neural_coord.act(obs_t1).broadcast_edge_costs is None


def test_identical_b_comm_v2x_budget_and_active_pool_fairness() -> None:
    """Verify identical B_comm cap, non-head follower eligibility, and cooldown across policies (D7)."""
    topology = _build_multi_link_test_topology()
    obs_cfg = ObservationConfig(v2x_budget_per_step=4, v2x_poll_cooldown_s=1.5)
    ctrl_cfg = ControlConfig()

    # 10 vehicles on C_W_in: followers behind corridor head have high obs_age & uncertain turn probs
    radar_obs: list[NoisyVehicleObservation] = []
    departed: list[DepartedVehicleEvent] = []
    for i in range(10):
        vid = f"veh_{i:02d}"
        departed.append(DepartedVehicleEvent(vid, 0.0, "e_W_in", i % 2, 4.5, 1.85))
        # Give mid-queue followers higher position noise and maximum 3-way turn entropy:
        is_follower = i >= 2
        t_like = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0) if is_follower else (0.01, 0.98, 0.01)
        pos_std = 3.5 if is_follower else 0.3
        radar_obs.append(
            NoisyVehicleObservation(
                veh_id=vid,
                obs_time=0.0,
                edge_id="e_W_in",
                corridor_id="C_W_in",
                in_junction_id=None,
                lane_index=i % 2,
                noisy_corridor_pos_m=130.0 - 10.0 * i,
                noisy_speed_mps=9.0,
                pos_noise_std=pos_std,
                speed_noise_std=0.4,
                turn_likelihood=t_like,
                noisy_leader_gap_m=None,
                noisy_leader_speed_mps=None,
                leader_id=None,
            )
        )

    batch_0 = ObservationBatch(
        sim_time=0.0,
        dt=0.5,
        active_veh_ids=tuple(d.veh_id for d in departed),
        departed=tuple(departed),
        arrived_veh_ids=(),
        radar_obs=tuple(radar_obs),
        v2x_packets=(),
        open_lanes_by_edge={},
        tls_states={"tls_center": "GGGGGGGGGGGG", "tls_east": "G"},
    )
    batch_1 = ObservationBatch(
        sim_time=0.5,
        dt=0.5,
        active_veh_ids=batch_0.active_veh_ids,
        departed=(),
        arrived_veh_ids=(),
        radar_obs=(),
        v2x_packets=(),
        open_lanes_by_edge={},
        tls_states={"tls_center": "GGGGGGGGGGGG", "tls_east": "G"},
    )

    v2x_controllers = [
        build_policy_controller(
            pol,
            topology,
            control_config=ctrl_cfg,
            observation_config=obs_cfg,
            dbn_updater=_StubDBNUpdater() if pol not in ("fixed_signal", "actuated_signal", "fcfs_reservation") else None,
        )
        for pol in POLICY_NAMES
    ]

    for ctrl in v2x_controllers:
        a0 = ctrl.act(batch_0)
        assert len(a0.v2x_polls) <= obs_cfg.v2x_budget_per_step
        if ctrl.policy_name in ("fixed_signal", "actuated_signal"):
            assert a0.v2x_polls == ()
            assert a0.tls_desired_greens is None
        else:
            # Active V2X-enabled controllers all saturate exact B_comm=4 cap from the 10-vehicle pool
            assert len(a0.v2x_polls) == obs_cfg.v2x_budget_per_step
            # Non-head followers with high uncertainty ARE polled (not restricted to 2 lane heads!)
            assert any(vid not in ("veh_00", "veh_01") for vid in a0.v2x_polls)
            a1 = ctrl.act(batch_1)
            assert len(a1.v2x_polls) == obs_cfg.v2x_budget_per_step
            # Cooldown (0.5s < 1.5s) prevents re-polling step-0 vehicles on step 1
            assert set(a0.v2x_polls).isdisjoint(set(a1.v2x_polls))


def test_protocols_ablations_reset_determinism_and_fail_loud_guards() -> None:
    """Verify PolicyControllerProtocol, no_temporal_prior isolation, reset(), and fail-loud errors."""
    topology = _build_multi_link_test_topology()
    exp_cfg = ExperimentConfig(
        scenario=ScenarioConfig(
            scenario_id="grid3x3_test",
            network_id=topology.network_id,
            net_xml_path="synthetic_test.net.xml",
        ),
        dbn=DBNConfig(),
        control=ControlConfig(),
    )

    for pol in POLICY_NAMES:
        stub = None if pol in ("fixed_signal", "actuated_signal", "fcfs_reservation") else _StubDBNUpdater()
        ctrl = build_policy_controller(pol, topology, config=exp_cfg, dbn_updater=stub)
        assert isinstance(ctrl, PolicyControllerProtocol)
        assert ctrl.policy_name == pol

    # Neural policy without dbn_updater and without checkpoint_path MUST fail loudly:
    with pytest.raises(CheckpointIntegrityError):
        build_policy_controller("neural_dbn", topology, config=exp_cfg, checkpoint_path=None, dbn_updater=None)

    with pytest.raises(ConfigValidationError):
        build_policy_controller("nonexistent_policy", topology, config=exp_cfg)

    # Verify no_temporal_prior strips prev_snapshot while neural_dbn forwards prev_snapshot:
    batch_a = ObservationBatch(
        sim_time=0.0,
        dt=0.5,
        active_veh_ids=("v1",),
        departed=(DepartedVehicleEvent("v1", 0.0, "e_N_in", 0, 4.5, 1.85),),
        arrived_veh_ids=(),
        radar_obs=(
            NoisyVehicleObservation("v1", 0.0, "e_N_in", "C_N_in", None, 0, 95.0, 11.0, 0.6, 0.2, (0.1, 0.8, 0.1), None, None, None),
        ),
        v2x_packets=(),
        open_lanes_by_edge={},
        tls_states={"tls_center": "GGGGGGGGGGGG", "tls_east": "G"},
    )
    batch_b = ObservationBatch(
        sim_time=0.5,
        dt=0.5,
        active_veh_ids=("v1",),
        departed=(),
        arrived_veh_ids=(),
        radar_obs=(
            NoisyVehicleObservation("v1", 0.5, "e_N_in", "C_N_in", None, 0, 100.5, 11.0, 0.6, 0.2, (0.1, 0.8, 0.1), None, None, None),
        ),
        v2x_packets=(),
        open_lanes_by_edge={},
        tls_states={"tls_center": "GGGGGGGGGGGG", "tls_east": "G"},
    )

    stub_no_prior = _StubDBNUpdater()
    coord_no_prior = FactoredSwarmCoordinator("no_temporal_prior", topology, stub_no_prior)
    coord_no_prior.act(batch_a)
    coord_no_prior.act(batch_b)
    assert stub_no_prior.received_prev_snapshots == [None, None]

    stub_neural = _StubDBNUpdater()
    coord_neural = FactoredSwarmCoordinator("neural_dbn", topology, stub_neural)
    act_1_first = coord_neural.act(batch_a)
    act_2_first = coord_neural.act(batch_b)
    assert stub_neural.received_prev_snapshots[0] is None
    assert stub_neural.received_prev_snapshots[1] is not None

    # Non-monotonic sim_time without reset() fails loudly:
    with pytest.raises(NumericalInvariantError):
        coord_neural.act(batch_a)

    # After reset(), re-running the episode produces identical actions:
    coord_neural.reset()
    act_1_second = coord_neural.act(batch_a)
    act_2_second = coord_neural.act(batch_b)
    assert act_1_first.target_speeds == act_1_second.target_speeds
    assert act_2_first.target_speeds == act_2_second.target_speeds
    assert act_1_first.tls_desired_greens == act_1_second.tls_desired_greens
    assert act_2_first.v2x_polls == act_2_second.v2x_polls

    # Cyclic DBNEdge graph passed to run_damped_maxsum fails loudly with DAGInvariantError:
    v_a = _make_vehicle_belief("va", "C_N_in", "e_N_in")
    v_b = _make_vehicle_belief("vb", "C_W_in", "e_W_in")
    bel_ab = _make_belief_state(topology, [v_a, v_b])
    cyc_snap = _make_dbn_snapshot(
        bel_ab,
        topology,
        edges=(
            DBNEdge("va", "vb", RelationType.SAME_JUNCTION_CONFLICT, 0.8, 0.7, 0.5, 0.8, 0.0),
            DBNEdge("vb", "va", RelationType.SAME_JUNCTION_CONFLICT, 0.8, 0.7, 0.5, 0.8, 0.0),
        ),
        precedence_scores={"va": 1.0, "vb": 2.0},
    )
    with pytest.raises(DAGInvariantError):
        run_damped_maxsum(bel_ab, cyc_snap, topology, ControlConfig())

    # Boundary Lock A: exit-fringe vehicle with empty turn_options=() evaluates outgoing spillback to 0.0:
    v_exit = _make_vehicle_belief("v_exit", "C_S_out", "e_S_out", turn_probs=())
    assert compute_expected_outgoing_spillback(
        v_exit, topology.corridors["C_S_out"], bel_ab, None
    ) == 0.0


def test_lazy_build_policy_controller_integration_with_real_dbn_updaters() -> None:
    """Verify build_policy_controller(dbn_updater=None) lazily wires swarm.dbn.build_dbn_updater."""
    topology = _build_multi_link_test_topology()
    batch = ObservationBatch(
        sim_time=2.0,
        dt=0.5,
        active_veh_ids=("v_w", "v_n"),
        departed=(
            DepartedVehicleEvent("v_w", 2.0, "e_W_in", 0, 4.5, 1.85),
            DepartedVehicleEvent("v_n", 2.0, "e_N_in", 0, 4.5, 1.85),
        ),
        arrived_veh_ids=(),
        radar_obs=(
            NoisyVehicleObservation("v_w", 2.0, "e_W_in", "C_W_in", None, 0, 122.0, 10.5, 0.7, 0.3, (0.05, 0.90, 0.05), None, None, None),
            NoisyVehicleObservation("v_n", 2.0, "e_N_in", "C_N_in", None, 0, 120.0, 10.0, 0.7, 0.3, (0.05, 0.90, 0.05), None, None, None),
        ),
        v2x_packets=(),
        open_lanes_by_edge={},
        tls_states={"tls_center": "GGGGGGGGGGGG", "tls_east": "G"},
    )
    for pol in ("independent_belief", "heuristic_radius_dbn", "online_bic_dbn"):
        ctrl = build_policy_controller(pol, topology, dbn_updater=None)
        act = ctrl.act(batch)
        assert ctrl.last_dbn_snapshot is not None
        assert act.tls_desired_greens is not None
        verify_conflict_free_greens(topology.junctions["J_center"], frozenset(act.tls_desired_greens["tls_center"]))
        assert act.broadcast_edge_costs is not None


def test_boundary_lock_c_static_ast_firewall_on_swarm_control() -> None:
    """Verify zero simulator/ground-truth imports, zero nan_to_num, and zero silent excepts in swarm/control."""
    control_dir = Path(control_pkg.__file__).resolve().parent
    py_files = sorted(control_dir.glob("*.py"))
    assert len(py_files) == 9

    forbidden_modules = {"libsumo", "traci", "sumolib"}
    forbidden_names = {
        "VehicleRawTelemetry",
        "GroundTruthStepRecord",
        "EpisodeMetrics",
        "route_goal",
        "departDelay",
        "nan_to_num",
    }

    for py_path in py_files:
        source = py_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    assert root_mod not in forbidden_modules, f"Forbidden import {alias.name} in {py_path.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    parts = set(node.module.split("."))
                    assert parts.isdisjoint(forbidden_modules), (
                        f"Forbidden from-import {node.module} in {py_path.name}"
                    )
                for alias in node.names:
                    assert alias.name not in forbidden_names, (
                        f"Forbidden symbol import {alias.name} in {py_path.name}"
                    )
            elif isinstance(node, ast.Name):
                assert node.id not in forbidden_names, f"Forbidden symbol {node.id} in {py_path.name}"
            elif isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_names, f"Forbidden attribute {node.attr} in {py_path.name}"
            elif isinstance(node, ast.ExceptHandler):
                assert node.type is not None, f"Bare except prohibited in {py_path.name}"
                has_raise = any(isinstance(sub, ast.Raise) for sub in ast.walk(node))
                assert has_raise, f"Silent non-reraising except prohibited in {py_path.name}"


def test_in_junction_clearing_guard_and_palo_alto_rush_rollout(tmp_path: Path) -> None:
    """Verify clearing-vehicle outgoing-corridor guard (t=0..65s+ on osm_palo_alto_rush) & loud mismatch check."""
    topology = _build_multi_link_test_topology()
    v_approach = _make_vehicle_belief("v_app", "C_N_in", "e_N_in", free_eta_s=2.0)
    # Actively clearing J_center into outgoing C_E_conn (whose to_junction is J_east != J_center)
    v_clearing = _make_vehicle_belief(
        "v_clr",
        "C_E_conn",
        "e_E_trunk",
        turn_probs=(1.0,),
        free_eta_s=0.0,
        in_junction_id="J_center",
    )
    # At signalized J_center, cross-phase separation is enforced by verify_conflict_free_greens +
    # 3.0s yellow clearance, so an in-junction clearing vehicle returns 0.0 without indexing
    # downstream C_E_conn.turn_options at J_center:
    assert _pair_turn_conflict_prob(
        topology,
        "J_center",
        v_approach,
        topology.corridors["C_N_in"],
        v_clearing,
        topology.corridors["C_E_conn"],
    ) == pytest.approx(0.0)

    # At an unsignalized junction (tls_id=None), cross-corridor feeders into C_E_conn DO conflict:
    unsig_junc = dict(topology.junctions)
    j_c = unsig_junc["J_center"]
    unsig_junc["J_center"] = JunctionSpec(
        junction_id="J_center",
        tls_id=None,
        coord_xy=j_c.coord_xy,
        incoming_corridors=j_c.incoming_corridors,
        outgoing_corridors=j_c.outgoing_corridors,
        num_tls_links=0,
        tls_link_foe_matrix=np.zeros((0, 0), dtype=bool),
        req_foe_matrix=j_c.req_foe_matrix,
    )
    unsig_topo = RoadTopology(
        network_id=topology.network_id,
        net_xml_path=topology.net_xml_path,
        poly_xml_path=topology.poly_xml_path,
        corridors=topology.corridors,
        edge_to_corridor=topology.edge_to_corridor,
        edge_offset_in_corridor=topology.edge_offset_in_corridor,
        junctions=unsig_junc,
        successor_corridors=topology.successor_corridors,
        predecessor_corridors=topology.predecessor_corridors,
    )
    assert _pair_turn_conflict_prob(
        unsig_topo,
        "J_center",
        v_approach,
        unsig_topo.corridors["C_N_in"],
        v_clearing,
        unsig_topo.corridors["C_E_conn"],
    ) >= 0.85

    # Non-clearing vehicle on C_E_conn passed against J_center MUST fail loudly with TopologyIntegrityError
    v_mismatch = _make_vehicle_belief(
        "v_mis",
        "C_E_conn",
        "e_E_trunk",
        turn_probs=(1.0,),
        free_eta_s=2.0,
        in_junction_id=None,
    )
    with pytest.raises(TopologyIntegrityError):
        _pair_turn_conflict_prob(
            topology,
            "J_center",
            v_approach,
            topology.corridors["C_N_in"],
            v_mismatch,
            topology.corridors["C_E_conn"],
        )

    # Full closed-loop SUMO rollout on osm_palo_alto_rush across t = 0..65s+ with FactoredSwarmCoordinator
    from swarm.config import load_scenario_config
    from swarm.env import SumoSimulationAdapter, load_or_build_topology

    repo_root = Path(__file__).resolve().parents[1]
    pa_cfg = load_scenario_config(repo_root / "configs" / "scenarios" / "osm_palo_alto_rush.json", repo_root=repo_root)
    pa_topo = load_or_build_topology(pa_cfg, repo_root=repo_root)
    exp_cfg = ExperimentConfig(scenario=pa_cfg)
    controller = build_policy_controller("heuristic_radius_dbn", pa_topo, config=exp_cfg, seed=3000)

    saw_clearing_in_junction = False
    with SumoSimulationAdapter(
        pa_cfg,
        control_config=exp_cfg.control,
        policy_name="heuristic_radius_dbn",
        seed=3000,
        work_dir=tmp_path / "pa_ctrl",
        repo_root=repo_root,
        topology=pa_topo,
    ) as adapter:
        obs, gt = adapter.reset(seed=3000)
        while obs.sim_time < 65.0:
            if any(r.in_junction_id is not None for r in obs.radar_obs):
                saw_clearing_in_junction = True
            act = controller.act(obs)
            assert act.tls_desired_greens is not None
            for tls_id, g_links in act.tls_desired_greens.items():
                junc = next(j for j in pa_topo.junctions.values() if j.tls_id == tls_id)
                verify_conflict_free_greens(junc, frozenset(g_links))
            obs, gt = adapter.step(act)
            assert gt.collisions == 0
            assert gt.teleports == 0

    assert saw_clearing_in_junction


def test_multilane_parallel_discharge_clearing_nonblocking_and_min_green_invariant() -> None:
    """Verify HCM min-green >= yellow guard, parallel multi-lane simultaneous discharge, & unspilled Max-Sum."""
    ctrl_cfg = ControlConfig()
    assert ctrl_cfg.tls_min_green_hold_s >= ctrl_cfg.tls_yellow_clearance_s
    assert ctrl_cfg.tls_min_green_hold_s == pytest.approx(6.0)

    with pytest.raises(ConfigValidationError):
        ControlConfig(tls_min_green_hold_s=2.5, tls_yellow_clearance_s=3.0)

    topology = _build_multi_link_test_topology()

    # Two parallel lanes on C_W_in heading straight into 2-lane C_E_conn + one clearing vehicle inside J_center:
    v_lane0 = _make_vehicle_belief(
        "v_w_lane0", "C_W_in", "e_W_in", lane_index=0, mean_s_m=132.0, mean_v_mps=11.0,
        free_eta_s=0.6, turn_probs=(0.0, 1.0, 0.0),
    )
    v_lane1 = _make_vehicle_belief(
        "v_w_lane1", "C_W_in", "e_W_in", lane_index=1, mean_s_m=131.5, mean_v_mps=11.0,
        free_eta_s=0.65, turn_probs=(0.0, 1.0, 0.0),
    )
    v_clearing = _make_vehicle_belief(
        "v_w_clearing", "C_E_conn", "e_E_trunk", lane_index=0, mean_s_m=4.0, mean_v_mps=11.5,
        free_eta_s=0.0, turn_probs=(1.0,), in_junction_id="J_center",
    )
    v_down_far = _make_vehicle_belief(
        "v_e_far", "C_E_conn", "e_E_pocket", lane_index=0, mean_s_m=134.0, mean_v_mps=0.0,
        free_eta_s=0.5, turn_probs=(1.0,),
    )

    # Multi-lane open_lanes=2 => zero conflict between parallel straight lanes:
    assert _pair_turn_conflict_prob(
        topology, "J_center", v_lane0, topology.corridors["C_W_in"], v_lane1, topology.corridors["C_W_in"],
        open_lanes_by_edge={"e_E_trunk": 2, "e_E_pocket": 2},
    ) == pytest.approx(0.0)

    # Single-lane incident closure (open_lanes=1) => merge conflict = 1.0:
    assert _pair_turn_conflict_prob(
        topology, "J_center", v_lane0, topology.corridors["C_W_in"], v_lane1, topology.corridors["C_W_in"],
        open_lanes_by_edge={"e_E_trunk": 1, "e_E_pocket": 2},
    ) == pytest.approx(1.0)

    belief = _make_belief_state(
        topology,
        [v_lane0, v_lane1, v_clearing, v_down_far],
        open_lanes_by_edge={"e_E_trunk": 2, "e_E_pocket": 2},
    )
    # Downstream C_E_conn has 0.02 tail spillback (far stopline car does NOT spill back to entry tail):
    edges = (
        DBNEdge("v_e_far", "v_w_lane0", RelationType.DOWNSTREAM_SPILLBACK, 0.90, 0.85, 0.80, 0.0, 0.95),
        DBNEdge("v_e_far", "v_w_lane1", RelationType.DOWNSTREAM_SPILLBACK, 0.90, 0.85, 0.80, 0.0, 0.95),
    )
    snap = _make_dbn_snapshot(
        belief,
        topology,
        edges=edges,
        precedence_scores={"v_w_clearing": 5.0, "v_e_far": 4.0, "v_w_lane0": 3.0, "v_w_lane1": 2.9},
        corridor_spillback_pred={"C_E_conn": 0.02},
        delay_post={"v_e_far": 0.95, "v_w_clearing": 0.05, "v_w_lane0": 0.1, "v_w_lane1": 0.1},
    )
    ms = run_damped_maxsum(belief, snap, topology, ctrl_cfg)
    assert ms.modes["v_w_lane0"] != LongitudinalMode.YIELD
    assert ms.modes["v_w_lane1"] != LongitudinalMode.YIELD

    sched = PriorityDAGScheduler(ctrl_cfg).decode(belief, topology, maxsum_result=ms, dbn_snapshot=snap)
    assert {"v_w_lane0", "v_w_lane1", "v_w_clearing"}.issubset(sched.granted_vehicles)
    assert "v_w_lane0" not in sched.must_hold_stopline
    assert "v_w_lane1" not in sched.must_hold_stopline

    # Unthrottled desired-speed passthrough when no safety hazard is active (zero bogus clamp count):
    sf = ObservableSafetyFilter(ctrl_cfg)
    accel_veh = _make_vehicle_belief(
        "v_open_accel", "C_W_in", "e_W_in", lane_index=0, mean_s_m=50.0, mean_v_mps=1.2,
        free_eta_s=6.5, turn_probs=(0.0, 1.0, 0.0),
    )
    sf_res = sf.apply(
        _make_belief_state(topology, [accel_veh]),
        topology,
        desired_speeds={"v_open_accel": 13.9},
    )
    assert sf_res.target_speeds["v_open_accel"] == pytest.approx(13.9)
    assert sf_res.safety_clamp_count == 0
    assert sf_res.post_safety_accels["v_open_accel"] == pytest.approx(ctrl_cfg.comfort_accel_mps2)

    # Signalized red-queue anti-starvation handover: queued NS platoon wins green even under EW->NS conflict edge:
    v_ew_1 = _make_vehicle_belief(
        "v_ew_1", "C_W_in", "e_W_in", mean_s_m=128.0, mean_v_mps=11.0, free_eta_s=1.0, turn_probs=(0.0, 1.0, 0.0),
    )
    ns_queue = [
        _make_vehicle_belief(
            f"v_ns_{idx}",
            "C_N_in" if idx % 2 == 0 else "C_S_in",
            "e_N_in" if idx % 2 == 0 else "e_S_in",
            lane_index=(idx // 2) % 2,
            mean_s_m=132.0 - 7.0 * (idx // 2),
            mean_v_mps=0.0,
            wait_time_s=18.0,
            free_eta_s=1.2 + 0.8 * (idx // 2),
            turn_probs=(0.0, 1.0, 0.0),
        )
        for idx in range(6)
    ]
    starv_belief = SwarmBeliefState(
        sim_time=25.0,
        dt=0.5,
        vehicles={v.veh_id: v for v in [v_ew_1, *ns_queue]},
        corridors=_make_belief_state(topology, [v_ew_1, *ns_queue], sim_time=25.0).corridors,
        open_lanes_by_edge={},
        tls_states={"tls_center": "rrrrrrGGGGGG", "tls_east": "G"},
    )
    conf_edges = tuple(
        DBNEdge("v_ew_1", qv.veh_id, RelationType.SAME_JUNCTION_CONFLICT, 0.90, 0.80, 0.70, 0.95, 0.0)
        for qv in ns_queue[:2]
    )
    starv_snap = _make_dbn_snapshot(
        starv_belief,
        topology,
        edges=conf_edges,
        precedence_scores={"v_ew_1": 10.0, **{qv.veh_id: float(5 - idx) for idx, qv in enumerate(ns_queue)}},
    )
    ms_starv = run_damped_maxsum(starv_belief, starv_snap, topology, ctrl_cfg)
    sched_starv = PriorityDAGScheduler(ctrl_cfg).decode(
        starv_belief, topology, maxsum_result=ms_starv, dbn_snapshot=starv_snap
    )
    assert {1, 4}.issubset(sched_starv.tls_desired_greens["tls_center"])
    assert 7 not in sched_starv.tls_desired_greens["tls_center"]

    # Sticky setSpeed release: cruising/granted vehicles omit setSpeed (preserving LC2013 lane-changing),
    # spillback-held pocket vehicles receive safe braking speed, and clearing spillback emits -1.0 release:
    v_pocket = _make_vehicle_belief(
        "v_pocket_hold", "C_E_conn", "e_E_pocket", lane_index=0, mean_s_m=125.0, mean_v_mps=8.0,
        free_eta_s=1.8, turn_probs=(1.0,),
    )
    v_trunk = _make_vehicle_belief(
        "v_trunk_merge", "C_E_conn", "e_E_trunk", lane_index=0, mean_s_m=30.0, mean_v_mps=11.5,
        free_eta_s=9.5, turn_probs=(1.0,),
    )
    rel_bel = _make_belief_state(topology, [v_pocket, v_trunk])
    sf_hold = sf.apply(
        rel_bel,
        topology,
        desired_speeds={"v_pocket_hold": 0.0, "v_trunk_merge": 13.9},
        must_hold_stopline=frozenset({"v_pocket_hold", "v_trunk_merge"}),
    )
    speeds_t0, ctrl_t0 = select_actuated_target_speeds(
        belief=rel_bel,
        topology=topology,
        safety=sf_hold,
        control_config=ctrl_cfg,
        must_hold_stopline=frozenset({"v_pocket_hold", "v_trunk_merge"}),
        granted_vehicles=frozenset(),
        previously_controlled=frozenset(),
    )
    assert "v_trunk_merge" not in speeds_t0
    assert "v_pocket_hold" in speeds_t0 and speeds_t0["v_pocket_hold"] >= 0.0
    assert ctrl_t0 == {"v_pocket_hold"}

    sf_clear = sf.apply(
        rel_bel,
        topology,
        desired_speeds={"v_pocket_hold": 13.9, "v_trunk_merge": 13.9},
        must_hold_stopline=frozenset(),
    )
    speeds_t1, ctrl_t1 = select_actuated_target_speeds(
        belief=rel_bel,
        topology=topology,
        safety=sf_clear,
        control_config=ctrl_cfg,
        must_hold_stopline=frozenset(),
        granted_vehicles=frozenset({"v_pocket_hold", "v_trunk_merge"}),
        previously_controlled=ctrl_t0,
    )
    assert speeds_t1 == {"v_pocket_hold": -1.0}
    assert ctrl_t1 == set()





"""Comprehensive unit & integration tests for ``swarm.env`` (Slice 1: SUMO, Corridors, Sensing, Belief)."""

from __future__ import annotations

import ast
from dataclasses import replace
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from swarm.config import (
    IncidentSpec,
    ObservationConfig,
    ScenarioConfig,
    load_scenario_config,
)
from swarm.contracts import (
    BeliefTrackerProtocol,
    CorridorSpec,
    DepartedVehicleEvent,
    NoisyVehicleObservation,
    ObservationBatch,
    SimulationAdapterProtocol,
    SwarmControlAction,
    V2XPacket,
)
from swarm.env import (
    CityBeliefTracker,
    SumoSimulationAdapter,
    generate_episode_routes,
    load_or_build_topology,
)
from swarm.logging_utils import (
    CausalityFirewallError,
    ConfigValidationError,
    RoutingIntegrityError,
    SimulationLifecycleError,
    TopologyIntegrityError,
    require_psd_covariance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _passive_action(
    sim_time: float,
    *,
    tls_desired_greens: dict[str, set[int]] | None = None,
    broadcast_edge_costs: dict[str, float] | None = None,
    v2x_polls: tuple[str, ...] = (),
) -> SwarmControlAction:
    return SwarmControlAction(
        sim_time=sim_time,
        target_speeds={},
        post_safety_accels={},
        modes={},
        tls_desired_greens=tls_desired_greens,
        broadcast_edge_costs=broadcast_edge_costs,
        reroute_hysteresis_s=1.0,
        v2x_polls=v2x_polls,
    )


def test_all_five_scenario_jsons_and_topologies() -> None:
    expected = {
        "grid3x3_nominal": ("grid3x3", 48, 9, False),
        "grid3x3_rush_asymmetric": ("grid3x3", 48, 9, False),
        "grid3x3_noisy_dropout": ("grid3x3", 48, 9, False),
        "grid4x4_incident": ("grid4x4", 80, 16, True),
        "osm_palo_alto_rush": ("palo_alto_clean", 275, 26, True),
    }

    for scen_name, (net_id, exp_corrs, exp_tls, exp_ood) in expected.items():
        scen_path = REPO_ROOT / "configs" / "scenarios" / f"{scen_name}.json"
        cfg = load_scenario_config(scen_path, repo_root=REPO_ROOT)
        assert cfg.scenario_id == scen_name
        assert cfg.network_id == net_id
        assert cfg.warmup_duration_s == 60.0
        assert cfg.cohort_end_s == 300.0
        assert 180.0 <= cfg.drain_duration_s <= 240.0
        assert cfg.is_ood_regime is exp_ood

        topo = load_or_build_topology(cfg, repo_root=REPO_ROOT)
        assert len(topo.corridors) == exp_corrs
        tls_juncs = [j for j in topo.junctions.values() if j.tls_id is not None]
        assert len(tls_juncs) == exp_tls

        if scen_name == "grid4x4_incident":
            assert len(cfg.incidents) >= 1
            for eid in cfg.incidents[0].edge_ids:
                assert eid in topo.edge_to_corridor
        if scen_name == "osm_palo_alto_rush":
            assert topo.poly_xml_path is not None
            assert Path(topo.poly_xml_path).is_file()


def test_corridor_trunk_pocket_merging_and_turn_sorting() -> None:
    cfg = load_scenario_config(REPO_ROOT / "configs" / "scenarios" / "grid3x3_nominal.json", repo_root=REPO_ROOT)
    topo = load_or_build_topology(cfg, repo_root=REPO_ROOT)

    # Interior block A0 -> A1 (150m center-to-center minus 2x16m intersection footprints = 118m)
    spec = topo.corridors["A0A1"]
    assert spec.edges == ("A0A1", "A0A1.105.00")
    assert len(spec.edge_offsets_m) == 2
    assert spec.edge_offsets_m[0] == pytest.approx(0.0)
    assert spec.edge_offsets_m[1] == pytest.approx(89.0, abs=5.0)
    assert spec.length_m == pytest.approx(118.0, abs=5.0)
    assert topo.edge_to_corridor["A0A1"] == "A0A1"
    assert topo.edge_to_corridor["A0A1.105.00"] == "A0A1"

    for c_spec in topo.corridors.values():
        angles = [t.angle_rad for t in c_spec.turn_options]
        assert angles == sorted(angles), f"Unsorted turn_options on {c_spec.corridor_id}: {angles}"
        if not c_spec.is_exit_fringe:
            assert len(c_spec.turn_options) >= 1


def test_junction_foe_matrices_grid3x3_and_palo_alto() -> None:
    for scen_file in ("grid3x3_nominal.json", "osm_palo_alto_rush.json"):
        cfg = load_scenario_config(REPO_ROOT / "configs" / "scenarios" / scen_file, repo_root=REPO_ROOT)
        topo = load_or_build_topology(cfg, repo_root=REPO_ROOT)

        signalized = [j for j in topo.junctions.values() if j.tls_id is not None]
        assert len(signalized) >= 9
        has_nontrivial_conflict = False
        for junc in signalized:
            fmat = junc.tls_link_foe_matrix
            rmat = junc.req_foe_matrix
            assert fmat.shape == (junc.num_tls_links, junc.num_tls_links)
            assert np.array_equal(fmat, fmat.T)
            assert not np.any(np.diag(fmat))
            assert np.array_equal(rmat, rmat.T)
            assert not np.any(np.diag(rmat))
            if bool(fmat.any()):
                has_nontrivial_conflict = True
        assert has_nontrivial_conflict

    # Verify crossing straight corridors conflict at central intersection B1 on grid3x3
    grid_cfg = load_scenario_config(REPO_ROOT / "configs" / "scenarios" / "grid3x3_nominal.json", repo_root=REPO_ROOT)
    grid_topo = load_or_build_topology(grid_cfg, repo_root=REPO_ROOT)
    # B0B1 goes South->North through B1 to B1B2; A1B1 goes West->East through B1 to B1C1
    sn_thru = next(t for t in grid_topo.corridors["B0B1"].turn_options if t.to_corridor == "B1B2")
    we_thru = next(t for t in grid_topo.corridors["A1B1"].turn_options if t.to_corridor == "B1C1")
    assert grid_topo.turn_conflict("B1", sn_thru, we_thru) is True
    assert grid_topo.turn_conflict("B1", sn_thru, sn_thru) is False


def test_boundary_lock_a_fail_loud_topology_integrity(tmp_path: Path) -> None:
    # 1. Direct CorridorSpec Boundary Lock A violation
    with pytest.raises(TopologyIntegrityError):
        CorridorSpec(
            corridor_id="broken_interior",
            edges=("e0",),
            edge_offsets_m=(0.0,),
            from_junction="J0",
            to_junction="J1",
            length_m=100.0,
            base_lanes=2,
            speed_limit_mps=13.89,
            turn_options=(),
            is_entry_fringe=False,
            is_exit_fringe=False,
        )

    # 2. Missing <request> foe entries on multi-approach junction in XML must fail loudly
    src_net = REPO_ROOT / "configs" / "networks" / "grid3x3.net.xml"
    broken_net = tmp_path / "broken_no_requests.net.xml"
    tree = ET.parse(src_net)
    root = tree.getroot()
    for j_elem in root.findall("junction"):
        if j_elem.attrib.get("id") == "B1":
            for req in list(j_elem.findall("request")):
                j_elem.remove(req)
    tree.write(broken_net, encoding="utf-8", xml_declaration=True)

    with pytest.raises(TopologyIntegrityError):
        load_or_build_topology(broken_net, network_id="broken_grid")


def test_deterministic_seeded_route_generation(tmp_path: Path) -> None:
    cfg = load_scenario_config(REPO_ROOT / "configs" / "scenarios" / "grid3x3_nominal.json", repo_root=REPO_ROOT)
    topo = load_or_build_topology(cfg, repo_root=REPO_ROOT)

    rou_a, sched_a = generate_episode_routes(cfg, 1000, tmp_path / "a", topology=topo, repo_root=REPO_ROOT)
    rou_b, sched_b = generate_episode_routes(cfg, 1000, tmp_path / "b", topology=topo, repo_root=REPO_ROOT)
    rou_c, sched_c = generate_episode_routes(cfg, 1001, tmp_path / "c", topology=topo, repo_root=REPO_ROOT)

    assert sched_a == sched_b
    assert sched_a != sched_c
    # Compare parsed vehicle route chains ignoring timestamp header comments written by duarouter
    vehs_a = [(el.attrib["id"], el.attrib["depart"], el.find("route").attrib["edges"]) for el in ET.parse(rou_a).getroot().findall("vehicle")]  # type: ignore[union-attr]
    vehs_b = [(el.attrib["id"], el.attrib["depart"], el.find("route").attrib["edges"]) for el in ET.parse(rou_b).getroot().findall("vehicle")]  # type: ignore[union-attr]
    vehs_c = [(el.attrib["id"], el.attrib["depart"], el.find("route").attrib["edges"]) for el in ET.parse(rou_c).getroot().findall("vehicle")]  # type: ignore[union-attr]
    assert vehs_a == vehs_b
    assert vehs_a != vehs_c


def test_belief_tracker_protocol_oosm_lag_and_stopped_deadzone() -> None:
    cfg = load_scenario_config(REPO_ROOT / "configs" / "scenarios" / "grid3x3_nominal.json", repo_root=REPO_ROOT)
    topo = load_or_build_topology(cfg, repo_root=REPO_ROOT)
    obs_cfg = ObservationConfig()

    tracker_fresh = CityBeliefTracker(topo, obs_cfg)
    tracker_lagged = CityBeliefTracker(topo, obs_cfg)
    assert isinstance(tracker_fresh, BeliefTrackerProtocol)

    cid = "A0A1"
    entry_edge = topo.corridors[cid].edges[0]
    k_turns = len(topo.corridors[cid].turn_options)
    uniform_like = tuple(1.0 / k_turns for _ in range(k_turns))
    open_lanes = {e: 2 for c in topo.corridors.values() for e in c.edges}

    # Initialize both trackers identically at t = 8.5s with a radar sighting
    init_batch = ObservationBatch(
        sim_time=8.5,
        dt=0.5,
        active_veh_ids=("veh_0",),
        departed=(
            DepartedVehicleEvent(
                veh_id="veh_0",
                entry_sim_time=8.5,
                entry_edge_id=entry_edge,
                entry_lane_index=0,
                length_m=4.5,
                width_m=1.85,
            ),
        ),
        arrived_veh_ids=(),
        radar_obs=(
            NoisyVehicleObservation(
                veh_id="veh_0",
                obs_time=8.5,
                edge_id=entry_edge,
                corridor_id=cid,
                in_junction_id=None,
                lane_index=0,
                noisy_corridor_pos_m=40.0,
                noisy_speed_mps=10.0,
                pos_noise_std=1.2,
                speed_noise_std=0.45,
                turn_likelihood=uniform_like,
                noisy_leader_gap_m=None,
                noisy_leader_speed_mps=None,
                leader_id=None,
            ),
        ),
        v2x_packets=(),
        open_lanes_by_edge=open_lanes,
        tls_states={},
    )
    tracker_fresh.observe(init_batch)
    tracker_lagged.observe(init_batch)

    for _ in range(3):
        tracker_fresh.predict(0.5, {"veh_0": 0.0})
        tracker_lagged.predict(0.5, {"veh_0": 0.0})

    # At t = 10.0s: tracker_fresh receives zero-lag V2X (sent=10.0), tracker_lagged receives OOSM V2X (sent=8.5, lag=1.5s)
    pkt_zero_lag = V2XPacket(
        veh_id="veh_0",
        sent_time=10.0,
        received_time=10.0,
        edge_id=entry_edge,
        corridor_id=cid,
        lane_index=0,
        noisy_corridor_pos_m=55.0,
        noisy_speed_mps=10.0,
        pos_noise_std=0.30,
        speed_noise_std=0.12,
        turn_likelihood=uniform_like,
    )
    pkt_oosm_lag = replace(pkt_zero_lag, sent_time=8.5, noisy_corridor_pos_m=40.0)

    state_zero = tracker_fresh.observe(
        ObservationBatch(
            sim_time=10.0,
            dt=0.5,
            active_veh_ids=("veh_0",),
            departed=(),
            arrived_veh_ids=(),
            radar_obs=(),
            v2x_packets=(pkt_zero_lag,),
            open_lanes_by_edge=open_lanes,
            tls_states={},
        )
    )
    state_lag = tracker_lagged.observe(
        ObservationBatch(
            sim_time=10.0,
            dt=0.5,
            active_veh_ids=("veh_0",),
            departed=(),
            arrived_veh_ids=(),
            radar_obs=(),
            v2x_packets=(pkt_oosm_lag,),
            open_lanes_by_edge=open_lanes,
            tls_states={},
        )
    )

    cov_zero = state_zero.vehicles["veh_0"].cov_2x2
    cov_lag = state_lag.vehicles["veh_0"].cov_2x2
    require_psd_covariance(cov_zero, "cov_zero")
    require_psd_covariance(cov_lag, "cov_lag")
    assert float(np.trace(cov_lag)) > float(np.trace(cov_zero))
    assert float(cov_lag[0, 0]) > float(cov_zero[0, 0])
    assert state_lag.vehicles["veh_0"].obs_age_s == pytest.approx(1.5)

    # Stopped-queue deadzone check: v = 0.0 < 0.15 and accel <= 0 => zero forward creep across 20 dropout steps
    stopped_tracker = CityBeliefTracker(topo, obs_cfg)
    stopped_obs = ObservationBatch(
        sim_time=1.0,
        dt=0.5,
        active_veh_ids=("veh_stop",),
        departed=(
            DepartedVehicleEvent(
                veh_id="veh_stop",
                entry_sim_time=1.0,
                entry_edge_id=entry_edge,
                entry_lane_index=0,
                length_m=4.5,
                width_m=1.85,
            ),
        ),
        arrived_veh_ids=(),
        radar_obs=(
            NoisyVehicleObservation(
                veh_id="veh_stop",
                obs_time=1.0,
                edge_id=entry_edge,
                corridor_id=cid,
                in_junction_id=None,
                lane_index=0,
                noisy_corridor_pos_m=80.0,
                noisy_speed_mps=0.0,
                pos_noise_std=0.05,
                speed_noise_std=0.05,
                turn_likelihood=uniform_like,
                noisy_leader_gap_m=None,
                noisy_leader_speed_mps=None,
                leader_id=None,
            ),
        ),
        v2x_packets=(),
        open_lanes_by_edge=open_lanes,
        tls_states={},
    )
    b0 = stopped_tracker.observe(stopped_obs)
    pos_init = b0.vehicles["veh_stop"].mean_s_m
    prev_trace = float(np.trace(b0.vehicles["veh_stop"].cov_2x2))

    t_cur = 1.0
    for _ in range(20):
        stopped_tracker.predict(0.5, {"veh_stop": 0.0})
        t_cur += 0.5
        b_next = stopped_tracker.observe(
            ObservationBatch(
                sim_time=t_cur,
                dt=0.5,
                active_veh_ids=("veh_stop",),
                departed=(),
                arrived_veh_ids=(),
                radar_obs=(),
                v2x_packets=(),
                open_lanes_by_edge=open_lanes,
                tls_states={},
            )
        )
        vb = b_next.vehicles["veh_stop"]
        assert vb.mean_v_mps == pytest.approx(0.0)
        assert vb.mean_s_m == pytest.approx(pos_init, abs=1e-9)
        cur_trace = float(np.trace(vb.cov_2x2))
        assert cur_trace > prev_trace
        prev_trace = cur_trace

    # CausalityFirewallError when DepartedVehicleEvent leaks non-current entry_sim_time
    with pytest.raises(CausalityFirewallError):
        stopped_tracker.observe(
            ObservationBatch(
                sim_time=25.0,
                dt=0.5,
                active_veh_ids=("veh_leak",),
                departed=(
                    DepartedVehicleEvent(
                        veh_id="veh_leak",
                        entry_sim_time=21.5,  # != 25.0
                        entry_edge_id=entry_edge,
                        entry_lane_index=0,
                        length_m=4.5,
                        width_m=1.85,
                    ),
                ),
                arrived_veh_ids=(),
                radar_obs=(),
                v2x_packets=(),
                open_lanes_by_edge=open_lanes,
                tls_states={},
            )
        )


def test_libsumo_bit_identical_determinism_and_protocol(tmp_path: Path) -> None:
    cfg = load_scenario_config(REPO_ROOT / "configs" / "scenarios" / "grid3x3_nominal.json", repo_root=REPO_ROOT)

    def _run_trace(seed: int, sub: str) -> tuple[list[tuple[str, ...]], list[tuple[float, ...]], object]:
        active_seq: list[tuple[str, ...]] = []
        radar_seq: list[tuple[float, ...]] = []
        with SumoSimulationAdapter(cfg, seed=seed, work_dir=tmp_path / sub, repo_root=REPO_ROOT) as adapter:
            assert isinstance(adapter, SimulationAdapterProtocol)
            obs, gt = adapter.reset(seed=seed)
            for _ in range(45):
                polls = obs.active_veh_ids[:2]
                obs, gt = adapter.step(_passive_action(obs.sim_time, v2x_polls=polls))
                active_seq.append(obs.active_veh_ids)
                radar_seq.append(tuple(round(r.noisy_corridor_pos_m, 8) for r in obs.radar_obs))
                assert gt.collisions == 0
                assert gt.teleports == 0
            metrics = adapter.compute_episode_metrics()
        return active_seq, radar_seq, metrics

    act_1, rad_1, m_1 = _run_trace(1005, "det_1")
    act_2, rad_2, m_2 = _run_trace(1005, "det_2")
    act_3, rad_3, _ = _run_trace(1006, "det_3")

    assert act_1 == act_2
    assert rad_1 == rad_2
    assert (act_1, rad_1) != (act_3, rad_3)
    assert m_1.cohort_size > 0  # type: ignore[attr-defined]
    assert (
        m_1.cohort_completed_count + m_1.cohort_unfinished_count + m_1.cohort_undeparted_count  # type: ignore[attr-defined]
        == m_1.cohort_size  # type: ignore[attr-defined]
    )
    assert m_1.config_sha256 == m_2.config_sha256  # type: ignore[attr-defined]
    assert m_1.net_xml_sha256 == m_2.net_xml_sha256  # type: ignore[attr-defined]


def test_yellow_clearance_guard_zero_emergency_stops_and_conflict_rejection(tmp_path: Path) -> None:
    cfg = load_scenario_config(REPO_ROOT / "configs" / "scenarios" / "grid3x3_nominal.json", repo_root=REPO_ROOT)

    with SumoSimulationAdapter(cfg, seed=1010, work_dir=tmp_path / "yguard", repo_root=REPO_ROOT) as adapter:
        obs, _ = adapter.reset(seed=1010)
        topo = adapter.topology

        # Precompute two mutually conflicting singleton link sets per TLS junction
        phase_a: dict[str, set[int]] = {}
        phase_b: dict[str, set[int]] = {}
        conflicting_pair_sample: tuple[str, set[int]] | None = None

        for junc in topo.junctions.values():
            if junc.tls_id is None:
                continue
            fmat = junc.tls_link_foe_matrix
            found = False
            for l1 in range(junc.num_tls_links):
                for l2 in range(l1 + 1, junc.num_tls_links):
                    if bool(fmat[l1, l2]):
                        phase_a[junc.tls_id] = {l1}
                        phase_b[junc.tls_id] = {l2}
                        if conflicting_pair_sample is None:
                            conflicting_pair_sample = (junc.tls_id, {l1, l2})
                        found = True
                        break
                if found:
                    break

        assert conflicting_pair_sample is not None

        # Verify simultaneous conflicting green request raises TopologyIntegrityError immediately
        bad_tls, bad_pair = conflicting_pair_sample
        with pytest.raises(TopologyIntegrityError):
            adapter.step(_passive_action(obs.sim_time, tls_desired_greens={bad_tls: bad_pair}))

        # Adversarially flip desired conflicting phases every 0.5s across 70 steps
        prev_states = dict(obs.tls_states)
        yellow_run_len: dict[tuple[str, int], int] = {}

        for step_i in range(70):
            desired = phase_a if (step_i % 2 == 0) else phase_b
            obs, gt = adapter.step(_passive_action(obs.sim_time, tls_desired_greens=desired))
            assert gt.collisions == 0
            assert gt.teleports == 0
            assert gt.emergency_stops == 0

            for tls_id, cur_ryg in obs.tls_states.items():
                prev_ryg = prev_states[tls_id]
                junc = next(j for j in topo.junctions.values() if j.tls_id == tls_id)
                fmat = junc.tls_link_foe_matrix

                for idx, (p_ch, c_ch) in enumerate(zip(prev_ryg, cur_ryg, strict=True)):
                    # Never jump directly from Green ('G'/'g') to Red ('r') without Yellow ('y')!
                    assert not (p_ch in ("G", "g") and c_ch == "r"), (
                        f"Illegal direct G->r jump at {tls_id}[{idx}] on step {step_i}"
                    )
                    if c_ch in ("y", "Y"):
                        yellow_run_len[(tls_id, idx)] = yellow_run_len.get((tls_id, idx), 0) + 1
                    elif p_ch in ("y", "Y") and c_ch == "r":
                        # Completed yellow clearance must have lasted >= 6 steps (3.0s at dt=0.5s)
                        assert yellow_run_len.get((tls_id, idx), 0) >= 6
                        yellow_run_len[(tls_id, idx)] = 0

                # Verify no two conflicting links are simultaneously green
                green_indices = [i for i, ch in enumerate(cur_ryg) if ch in ("G", "g")]
                for i_pos, g1 in enumerate(green_indices):
                    for g2 in green_indices[i_pos + 1 :]:
                        assert not bool(fmat[g1, g2]), (
                            f"Simultaneous conflicting greens ({g1}, {g2}) at {tls_id}"
                        )
            prev_states = dict(obs.tls_states)


def test_mid_episode_incident_and_onboard_rerouter_and_route_guard(tmp_path: Path) -> None:
    base_cfg = load_scenario_config(
        REPO_ROOT / "configs" / "scenarios" / "grid4x4_incident.json", repo_root=REPO_ROOT
    )
    short_inc_cfg = replace(
        base_cfg,
        demand_rate_veh_per_s=2.4,
        incidents=(
            IncidentSpec(
                edge_ids=("B1C1", "B2C2"),
                start_time_s=2.0,
                end_time_s=5.0,
                closed_lane_indices=(0,),
                reduced_speed_mps=4.5,
            ),
        ),
    )

    with SumoSimulationAdapter(short_inc_cfg, seed=1012, work_dir=tmp_path / "inc", repo_root=REPO_ROOT) as adapter:
        obs, _ = adapter.reset(seed=1012)
        # At t = 0.5s (< 2.0s), both lanes are open
        assert obs.open_lanes_by_edge["B1C1"] == 2

        saw_closed = False
        saw_reopened = False
        total_reroutes = 0

        for _ in range(35):
            # Broadcast heavy congestion cost on central edges to trigger onboard Dijkstra detours
            costs: dict[str, float] | None = None
            if obs.sim_time >= 6.0:
                costs = {"B1C1": 600.0, "B2C2": 600.0, "B1B2": 600.0, "C1C2": 600.0}
                # Also penalize each active vehicle's immediate next corridor edge to ensure a detour exists
                for vid in obs.active_veh_ids[:6]:
                    for robs in obs.radar_obs:
                        if robs.veh_id == vid:
                            for opt in adapter.topology.corridors[robs.corridor_id].turn_options:
                                costs[opt.to_edge] = 800.0
                                break

            obs, gt = adapter.step(_passive_action(obs.sim_time, broadcast_edge_costs=costs))
            total_reroutes += gt.reroutes_applied
            if 2.0 <= obs.sim_time < 5.0:
                assert obs.open_lanes_by_edge["B1C1"] == 1
                assert obs.open_lanes_by_edge["B2C2"] == 1
                saw_closed = True
            elif obs.sim_time >= 5.0:
                assert obs.open_lanes_by_edge["B1C1"] == 2
                assert obs.open_lanes_by_edge["B2C2"] == 2
                saw_reopened = True

        assert saw_closed and saw_reopened
        assert total_reroutes >= 1

        # Direct disconnected route validation guard must raise RoutingIntegrityError (never pass silently!)
        normal_veh = next(
            rv for rv in gt.raw_vehicles if not rv.road_id.startswith(":") and len(rv.route_edges) >= 2
        )
        with pytest.raises(RoutingIntegrityError):
            adapter.validate_and_apply_route(
                normal_veh.veh_id,
                [normal_veh.road_id, "NON_EXISTENT_EDGE", normal_veh.route_goal],
            )


def test_fail_loud_lifecycle_backend_and_ast_firewall(tmp_path: Path) -> None:
    cfg = load_scenario_config(REPO_ROOT / "configs" / "scenarios" / "grid3x3_nominal.json", repo_root=REPO_ROOT)

    # 1. Invalid backend rejected immediately
    with pytest.raises(ConfigValidationError):
        ScenarioConfig(
            scenario_id="bad_backend",
            network_id="grid3x3",
            net_xml_path=cfg.net_xml_path,
            sumo_backend="invalid_sim",  # type: ignore[arg-type]
        )

    adapter1 = SumoSimulationAdapter(cfg, seed=1000, work_dir=tmp_path / "lc1", repo_root=REPO_ROOT)
    # 2. step() before reset() fails loudly
    with pytest.raises(SimulationLifecycleError):
        adapter1.step(_passive_action(0.5))

    adapter1.start(seed=1000)
    try:
        # 3. Double start() on active instance fails loudly
        with pytest.raises(SimulationLifecycleError):
            adapter1.start(seed=1000)

        # 4. Concurrent second libsumo instance fails loudly
        adapter2 = SumoSimulationAdapter(cfg, seed=1001, work_dir=tmp_path / "lc2", repo_root=REPO_ROOT)
        with pytest.raises(SimulationLifecycleError):
            adapter2.reset(seed=1001)
    finally:
        adapter1.close()

    # 5. Static AST check across all swarm/env/*.py files enforcing Boundary Lock C
    env_dir = REPO_ROOT / "swarm" / "env"
    forbidden_outside_adapter = {
        "libsumo",
        "traci",
        "VehicleRawTelemetry",
        "GroundTruthStepRecord",
        "EpisodeMetrics",
    }

    for py_file in sorted(env_dir.glob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))

        for node in ast.walk(tree):
            # Zero nan_to_num or weights_only=False anywhere
            if isinstance(node, ast.Attribute) and node.attr == "nan_to_num":
                pytest.fail(f"Forbidden nan_to_num in {py_file}")
            if isinstance(node, ast.Name) and node.id == "nan_to_num":
                pytest.fail(f"Forbidden nan_to_num in {py_file}")
            if isinstance(node, ast.keyword) and node.arg == "weights_only":
                if isinstance(node.value, ast.Constant) and node.value.value is False:
                    pytest.fail(f"Forbidden weights_only=False in {py_file}")

            # Zero bare except or silent except: pass / continue
            if isinstance(node, ast.ExceptHandler):
                assert node.type is not None, f"Bare except prohibited in {py_file}:{node.lineno}"
                for stmt in node.body:
                    assert not isinstance(stmt, (ast.Pass, ast.Continue)), (
                        f"Silent except pass/continue prohibited in {py_file}:{node.lineno}"
                    )

            # Firewall check on non-adapter modules
            if py_file.name != "sumo_adapter.py":
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_mod = alias.name.split(".")[0]
                        assert root_mod not in forbidden_outside_adapter, (
                            f"Forbidden import {alias.name!r} in {py_file}"
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        root_mod = node.module.split(".")[0]
                        assert root_mod not in ("libsumo", "traci"), (
                            f"Forbidden from-import {node.module!r} in {py_file}"
                        )
                    for alias in node.names:
                        assert alias.name not in forbidden_outside_adapter, (
                            f"Forbidden symbol {alias.name!r} imported in {py_file}"
                        )


def test_palo_alto_osm_rollout_and_fixed_vs_actuated_tls(tmp_path: Path) -> None:
    # 1. Real-world Downtown Palo Alto OSM end-to-end adapter -> CityBeliefTracker loop
    pa_cfg = load_scenario_config(
        REPO_ROOT / "configs" / "scenarios" / "osm_palo_alto_rush.json", repo_root=REPO_ROOT
    )
    with SumoSimulationAdapter(pa_cfg, seed=1015, work_dir=tmp_path / "pa", repo_root=REPO_ROOT) as adapter:
        tracker = CityBeliefTracker(adapter.topology, pa_cfg.observation)
        obs, gt = adapter.reset(seed=1015)
        belief = tracker.observe(obs)
        for _ in range(90):
            tracker.predict(obs.dt, {}, v2x_polled_ids=(), sim_time=obs.sim_time)
            obs, gt = adapter.step(_passive_action(obs.sim_time, v2x_polls=obs.active_veh_ids[:3]))
            belief = tracker.observe(obs)
            assert gt.collisions == 0
            assert gt.teleports == 0
        assert len(belief.vehicles) == len(obs.active_veh_ids)
        assert len(belief.corridors) == 275
        pa_metrics = adapter.compute_episode_metrics()
        assert pa_metrics.cohort_size > 0
        assert math.isfinite(pa_metrics.cohort_mean_total_delay_s)

    # 2. Verify fixed_signal ('fixed' static program) and actuated_signal ('0' gap-actuated program)
    # diverge under asymmetric rush demand when tls_desired_greens=None
    rush_cfg = load_scenario_config(
        REPO_ROOT / "configs" / "scenarios" / "grid3x3_rush_asymmetric.json", repo_root=REPO_ROOT
    )
    tls_trace_fixed: list[dict[str, str]] = []
    tls_trace_actuated: list[dict[str, str]] = []

    with SumoSimulationAdapter(
        rush_cfg, policy_name="fixed_signal", seed=1020, work_dir=tmp_path / "fix", repo_root=REPO_ROOT
    ) as ad_fix:
        obs_f, _ = ad_fix.reset(seed=1020)
        for _ in range(90):
            obs_f, _ = ad_fix.step(_passive_action(obs_f.sim_time))
            tls_trace_fixed.append(obs_f.tls_states)

    with SumoSimulationAdapter(
        rush_cfg, policy_name="actuated_signal", seed=1020, work_dir=tmp_path / "act", repo_root=REPO_ROOT
    ) as ad_act:
        obs_a, _ = ad_act.reset(seed=1020)
        for _ in range(90):
            obs_a, _ = ad_act.step(_passive_action(obs_a.sim_time))
            tls_trace_actuated.append(obs_a.tls_states)

    assert tls_trace_fixed != tls_trace_actuated


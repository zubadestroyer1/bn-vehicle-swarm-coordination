"""Comprehensive unit tests for ``swarm.viz`` (3D WebGL2 City & DBN Viewer + 2D Publication Figures).

Verifies:
- Boundary Lock C AST rules & 100% offline zero-CDN purity across ``swarm/viz/*``.
- 2D concave polygon ear-clipping triangulation exactness.
- Real ``sumolib`` + ``netgenerate`` 3D road ribbon, junction polygon, TLS gantry, procedural city
  block, and OSM ``.poly.xml`` 3D building extraction.
- ``EpisodeTraceRecorder`` conformance to ``TraceRecorderProtocol``, ``DBNTemporalMemory`` exclusion,
  and fail-loud monotonicity / numerical guards.
- Single-policy and synchronized split-screen 3D HTML compilation + ``node --check`` JS syntax verification.
- All 5 300-DPI publication 2D Matplotlib figures with zero ``.mplconfig`` side effects in ``Path.cwd()``.
"""

from __future__ import annotations

import ast
import base64
import gzip
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

# Ensure Matplotlib cache lives strictly in system temp directory prior to any imports.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "swarm_mplconfig"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import sumo  # noqa: E402
import sumolib  # noqa: E402

from swarm.contracts import (  # noqa: E402
    CorridorBeliefStats,
    CorridorSpec,
    DBNEdge,
    DBNStateSnapshot,
    DBNTemporalMemory,
    EpisodeMetrics,
    GroundTruthStepRecord,
    JunctionSpec,
    LongitudinalMode,
    RelationType,
    RoadTopology,
    SwarmBeliefState,
    SwarmControlAction,
    TraceRecorderProtocol,
    TurnOption,
    VehicleBelief,
    VehicleRawTelemetry,
)
from swarm.logging_utils import (  # noqa: E402
    ConfigValidationError,
    NumericalInvariantError,
    SimulationLifecycleError,
)
from swarm.viz import (  # noqa: E402
    EpisodeTraceRecorder,
    PAYLOAD_FORMAT_VERSION,
    SCENE_SCHEMA_VERSION,
    decode_viewer_html_payload,
    export_scene_3d,
    load_trace_json,
    pack_viewer_payload,
    plot_ablation_breakdown,
    plot_calibration_and_roc_pr,
    plot_latency_scaling,
    plot_paired_forest,
    plot_scenario_comparison,
    render_publication_plots,
    triangulate_polygon_2d,
    unpack_viewer_payload,
    write_city_3d_viewer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ_DIR = REPO_ROOT / "swarm" / "viz"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _build_synthetic_topology(tmp_dir: Path, *, poly_xml_path: Path | None = None) -> RoadTopology:
    """Generate a real 2x2 signalized multi-lane SUMO grid via ``netgenerate`` in ``tmp_dir``."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    netgen_bin = Path(sumo.SUMO_HOME) / "bin" / "netgenerate"
    assert netgen_bin.is_file(), f"Missing SUMO netgenerate binary at {netgen_bin}"

    net_path = tmp_dir / "synthetic_grid2x2.net.xml"
    cmd = [
        str(netgen_bin),
        "--grid",
        "--grid.number=2",
        "--grid.length=140",
        "--grid.attach-length=100",
        "--default.lanenumber=2",
        "--default.speed=13.89",
        "--tls.guess=true",
        "--no-turnarounds=true",
        f"--output-file={net_path}",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    net = sumolib.net.readNet(str(net_path), withInternal=True)
    normal_edges = [e for e in net.getEdges() if not e.isSpecial()]

    junctions: dict[str, JunctionSpec] = {}
    for node in net.getNodes():
        nid = node.getID()
        if nid.startswith(":"):
            continue
        cx, cy = node.getCoord()
        inc_ids = tuple(e.getID() for e in node.getIncoming() if not e.isSpecial())
        out_ids = tuple(e.getID() for e in node.getOutgoing() if not e.isSpecial())
        junctions[nid] = JunctionSpec(
            junction_id=nid,
            tls_id=nid if "traffic_light" in (node.getType() or "") else None,
            coord_xy=(float(cx), float(cy)),
            incoming_corridors=inc_ids,
            outgoing_corridors=out_ids,
            num_tls_links=0,
            tls_link_foe_matrix=np.zeros((0, 0), dtype=bool),
            req_foe_matrix=np.zeros((0, 0), dtype=bool),
        )

    corridors: dict[str, CorridorSpec] = {}
    edge_to_corridor: dict[str, str] = {}
    edge_offset: dict[str, float] = {}
    succ_map: dict[str, tuple[str, ...]] = {}
    pred_lists: dict[str, list[str]] = {e.getID(): [] for e in normal_edges}

    for edge in normal_edges:
        eid = edge.getID()
        edge_to_corridor[eid] = eid
        edge_offset[eid] = 0.0
        out_edges_set: list[str] = []
        for conn in edge.getOutgoing().keys():
            if not conn.isSpecial() and conn.getID() not in out_edges_set:
                out_edges_set.append(conn.getID())
        turns = tuple(
            TurnOption(
                turn_idx=i,
                from_edge=eid,
                to_corridor=target_eid,
                to_edge=target_eid,
                angle_rad=0.0,
                tls_id=None,
                tls_link_indices=(),
                junction_req_indices=(),
            )
            for i, target_eid in enumerate(out_edges_set)
        )
        for target_eid in out_edges_set:
            pred_lists[target_eid].append(eid)

        is_exit = len(turns) == 0
        is_entry = len([inc for inc in edge.getFromNode().getIncoming() if not inc.isSpecial()]) == 0
        corridors[eid] = CorridorSpec(
            corridor_id=eid,
            edges=(eid,),
            edge_offsets_m=(0.0,),
            from_junction=edge.getFromNode().getID(),
            to_junction=edge.getToNode().getID(),
            length_m=float(edge.getLength()),
            base_lanes=len(edge.getLanes()),
            speed_limit_mps=float(edge.getSpeed()),
            turn_options=turns,
            is_entry_fringe=is_entry,
            is_exit_fringe=is_exit,
        )
        succ_map[eid] = tuple(out_edges_set)

    pred_map = {k: tuple(v) for k, v in pred_lists.items()}
    return RoadTopology(
        network_id="synthetic_grid2x2",
        net_xml_path=str(net_path),
        poly_xml_path=str(poly_xml_path) if poly_xml_path is not None else None,
        corridors=corridors,
        edge_to_corridor=edge_to_corridor,
        edge_offset_in_corridor=edge_offset,
        junctions=junctions,
        successor_corridors=succ_map,
        predecessor_corridors=pred_map,
    )


def _make_episode_metrics(scenario_id: str, policy_name: str, seed: int, delay_s: float) -> EpisodeMetrics:
    return EpisodeMetrics(
        scenario_id=scenario_id,
        policy_name=policy_name,
        seed=seed,
        cohort_size=80,
        cohort_completed_count=79,
        cohort_unfinished_count=1,
        cohort_undeparted_count=0,
        cohort_completion_rate=79.0 / 80.0,
        cohort_mean_total_delay_s=delay_s,
        cohort_p95_total_delay_s=delay_s * 1.65,
        cohort_mean_completed_delay_s=delay_s * 0.96,
        throughput_veh_per_hr=1180.0,
        mean_stops_per_veh=1.15,
        spillback_corridor_seconds=delay_s * 0.85,
        mean_co2_g_per_veh=142.0,
        reroutes_per_veh=0.22,
        collision_count=0,
        teleport_count=0,
        emergency_stop_count=0,
        emergency_brake_count=1,
        emergency_brake_rate_per_veh_km=0.02,
        ttc_near_miss_count=2,
        min_ttc_s=1.85,
        safety_filter_clamp_count=9,
        safety_filter_clamp_rate=0.03,
        zero_speed_deadzone_count=14,
        v2x_polls_requested=90,
        v2x_packets_delivered=82,
        v2x_delivery_efficiency=82.0 / 90.0,
        mean_dbn_update_latency_ms=1.12,
        mean_dbn_inference_latency_ms=0.44,
        mean_control_latency_ms=0.31,
        edge_churn_hamming_rate=0.07,
        wall_time_s=2.4,
        sim_steps=240,
        steps_per_sec=100.0,
    )


def test_viz_ast_and_offline_purity() -> None:
    """Verify Boundary Lock C AST rules and zero external URLs across ``swarm/viz``."""
    forbidden_attr = "_".join(("nan", "to", "num"))
    py_files = sorted(VIZ_DIR.glob("*.py"))
    assert len(py_files) >= 5

    for py_path in py_files:
        src = py_path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                assert node.type is not None, f"Bare except prohibited in {py_path}"
                has_raise = any(isinstance(sub, ast.Raise) for sub in ast.walk(node))
                assert has_raise, f"Swallowing except handler without raise in {py_path}"
            elif isinstance(node, ast.Attribute):
                assert node.attr != forbidden_attr, f"Forbidden attribute in {py_path}"
            elif isinstance(node, ast.Name):
                assert node.id != forbidden_attr, f"Forbidden identifier in {py_path}"
            elif isinstance(node, ast.keyword) and node.arg == "weights_only":
                assert not (isinstance(node.value, ast.Constant) and node.value.value is False)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in ("traci", "libsumo"), f"Unexpected SUMO runtime import in {py_path}"
            elif isinstance(node, ast.ImportFrom):
                assert node.module not in ("traci", "libsumo"), f"Unexpected SUMO runtime import in {py_path}"

    template_html = (VIZ_DIR / "templates" / "city_dbn_3d.html").read_text(encoding="utf-8")
    engine_js = (VIZ_DIR / "assets" / "city_engine_webgl2.js").read_text(encoding="utf-8")
    url_re = re.compile(r"https?://", flags=re.IGNORECASE)
    assert url_re.findall(template_html) == []
    assert url_re.findall(engine_js) == []


def test_triangulate_concave_polygon_2d() -> None:
    """Verify ear-clipping triangulation preserves exact area on non-convex L-shaped footprints."""
    # L-shaped 6-vertex concave polygon: 10x4 base (40) + 4x6 vertical stem (24) = 64.0 m^2
    l_shape = [(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (4.0, 4.0), (4.0, 10.0), (0.0, 10.0)]
    indices = triangulate_polygon_2d(l_shape)
    assert len(indices) == 4 * 3
    assert all(0 <= idx < len(l_shape) for idx in indices)

    total_area = 0.0
    for i in range(0, len(indices), 3):
        ax, ay = l_shape[indices[i]]
        bx, by = l_shape[indices[i + 1]]
        cx, cy = l_shape[indices[i + 2]]
        signed_twice = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        assert signed_twice > 0.0, "Every ear-clipped triangle must be CCW-oriented"
        total_area += 0.5 * signed_twice

    assert total_area == pytest.approx(64.0, abs=1e-6)


def test_export_scene_3d_procedural_and_osm_poly(tmp_path: Path) -> None:
    """Verify 3D lane ribbons, junction polygons, TLS poles, procedural blocks, and OSM .poly.xml parsing."""
    topo_proc = _build_synthetic_topology(tmp_path, poly_xml_path=None)
    scene_json_path = tmp_path / "scene_proc.json"
    scene_proc = export_scene_3d(topo_proc, output_path=scene_json_path)

    assert scene_json_path.is_file()
    assert scene_proc["schema_version"] == SCENE_SCHEMA_VERSION
    assert scene_proc["source_mode"] == "procedural_blocks"
    assert len(scene_proc["lanes"]) >= 16
    assert len(scene_proc["lane_dividers"]) >= 8
    assert len(scene_proc["stop_bars"]) >= 8
    assert len(scene_proc["turn_arrows"]) >= 8
    assert len(scene_proc["junctions"]) >= 4
    assert len(scene_proc["signals"]) >= 4
    assert sum(len(s["poles"]) for s in scene_proc["signals"]) >= 4
    assert sum(len(s["links"]) for s in scene_proc["signals"]) >= 8
    assert len(scene_proc["buildings"]) >= 2

    for junc in scene_proc["junctions"]:
        if len(junc["polygon"]) >= 3:
            assert len(junc["triangles"]) >= 3 and len(junc["triangles"]) % 3 == 0

    for bldg in scene_proc["buildings"]:
        assert bldg["height_m"] > 5.0
        assert len(bldg["footprint"]) >= 3
        assert len(bldg["triangles"]) >= 3 and len(bldg["triangles"]) % 3 == 0

    # Test real OSM <poly> XML mode including multipart #0/#1 deduplication and park extraction
    poly_path = tmp_path / "synthetic_buildings.poly.xml"
    poly_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<additional>
    <poly id="bldg_hq#0" type="building.commercial" shape="45,45 85,45 85,80 45,80 45,45"/>
    <poly id="bldg_hq#1" type="building.commercial" shape="45,45 85,45 85,80 45,80"/>
    <poly id="bldg_res" type="building.apartments" shape="48,92 88,92 88,118 68,118 68,105 48,105"/>
    <poly id="plaza_green" type="leisure.park" shape="92,48 116,48 116,76 92,76"/>
</additional>
""",
        encoding="utf-8",
    )
    topo_osm = _build_synthetic_topology(tmp_path / "osm_sub", poly_xml_path=poly_path)
    scene_osm = export_scene_3d(topo_osm)
    assert scene_osm["source_mode"] == "osm_poly"
    assert len(scene_osm["buildings"]) == 2  # bldg_hq#0 and #1 deduplicated cleanly
    assert len(scene_osm["parks"]) == 1

    # Fail-loud check on nonexistent poly_xml_path
    bad_topo = RoadTopology(
        network_id=topo_proc.network_id,
        net_xml_path=topo_proc.net_xml_path,
        poly_xml_path=str(tmp_path / "missing_buildings.poly.xml"),
        corridors=topo_proc.corridors,
        edge_to_corridor=topo_proc.edge_to_corridor,
        edge_offset_in_corridor=topo_proc.edge_offset_in_corridor,
        junctions=topo_proc.junctions,
        successor_corridors=topo_proc.successor_corridors,
        predecessor_corridors=topo_proc.predecessor_corridors,
    )
    with pytest.raises(ConfigValidationError):
        export_scene_3d(bad_topo)


def _record_synthetic_trace(
    topology: RoadTopology,
    policy_name: str,
    *,
    speed_scale: float = 1.0,
) -> dict[str, Any]:
    recorder = EpisodeTraceRecorder(
        scenario_id="grid2x2_rush",
        policy_name=policy_name,
        seed=3001,
        topology=topology,
        max_dbn_edges_per_step=16,
    )
    assert isinstance(recorder, TraceRecorderProtocol)

    sample_corr = next(iter(topology.corridors.keys()))
    sample_tls = next((jid for jid, j in topology.junctions.items() if j.tls_id is not None), "A0")

    for k in range(6):
        t = 0.5 * k
        v1_spd = (9.5 - 0.8 * k) * speed_scale
        v2_spd = (6.2 + 0.4 * k) * speed_scale
        v3_spd = (4.0 * speed_scale) if k < 3 else (8.0 * speed_scale)
        raw_vehs = (
            VehicleRawTelemetry(
                veh_id="veh_0",
                vtype="passenger_urban",
                center_xy=(20.0 + 4.5 * k * speed_scale, 3.2),
                yaw_ccw_rad=0.0,
                speed=v1_spd,
                accel=-0.6 if k == 2 else 0.4,
                road_id=sample_corr,
                lane_id=f"{sample_corr}_0",
                lane_index=0,
                lane_pos=20.0 + 4.5 * k,
                lane_length=130.0,
                route_edges=(sample_corr,),
                route_index=0,
                route_goal=sample_corr,
                length=4.5,
                width=1.85,
                brake_light=(k == 2),
                turn_signal=1 if k >= 3 else 0,
                leader_id=None,
                leader_gap=None,
                time_loss=0.3 * k,
                waiting_time=0.0,
                co2_mg_s=1800.0,
            ),
            VehicleRawTelemetry(
                veh_id="veh_1",
                vtype="delivery_van",
                center_xy=(8.0 + 3.2 * k * speed_scale, 3.2),
                yaw_ccw_rad=0.02,
                speed=v2_spd,
                accel=0.2,
                road_id=sample_corr,
                lane_id=f"{sample_corr}_0",
                lane_index=0,
                lane_pos=8.0 + 3.2 * k,
                lane_length=130.0,
                route_edges=(sample_corr,),
                route_index=0,
                route_goal=sample_corr,
                length=6.2,
                width=2.25,
                brake_light=False,
                turn_signal=2 if k >= 4 else 0,
                leader_id="veh_0",
                leader_gap=12.0,
                time_loss=0.4 * k,
                waiting_time=0.0,
                co2_mg_s=2400.0,
            ),
            VehicleRawTelemetry(
                veh_id="veh_2",
                vtype="passenger_cautious",
                center_xy=(138.0, 28.0 + 3.0 * k * speed_scale),
                yaw_ccw_rad=math.pi * 0.5,
                speed=v3_spd,
                accel=-1.2 if k < 3 else 1.1,
                road_id=sample_corr,
                lane_id=f"{sample_corr}_1",
                lane_index=1,
                lane_pos=28.0 + 3.0 * k,
                lane_length=130.0,
                route_edges=(sample_corr,),
                route_index=0,
                route_goal=sample_corr,
                length=4.5,
                width=1.82,
                brake_light=(k < 3),
                turn_signal=0,
                leader_id=None,
                leader_gap=None,
                time_loss=0.6 * k,
                waiting_time=0.5 if k < 2 else 0.0,
                co2_mg_s=1650.0,
            ),
        )
        gt = GroundTruthStepRecord(
            sim_time=t,
            raw_vehicles=raw_vehs,
            collisions=0,
            teleports=0,
            emergency_stops=0,
            emergency_brakes=0,
            near_misses_ttc=0,
            min_ttc_s=3.4,
            reroutes_applied=1 if k == 3 else 0,
            v2x_sent=2,
            v2x_delivered=2,
            step_co2_mg=5850.0,
            spillback_corridor_count=1 if k >= 4 else 0,
        )
        belief = SwarmBeliefState(
            sim_time=t,
            dt=0.5,
            vehicles={
                "veh_0": VehicleBelief(
                    veh_id="veh_0",
                    entered_at=0.0,
                    corridor_id=sample_corr,
                    edge_id=sample_corr,
                    in_junction_id=None,
                    lane_index=0,
                    length_m=4.5,
                    width_m=1.85,
                    mean_s_m=20.0 + 4.5 * k,
                    mean_v_mps=v1_spd,
                    cov_2x2=np.array([[1.44, 0.1], [0.1, 0.36]], dtype=np.float64),
                    accel_est_mps2=0.2,
                    obs_age_s=0.5,
                    wait_time_s=0.0,
                    turn_probs=(0.2, 0.65, 0.15),
                    free_eta_s=8.5,
                    eta_std_s=0.9,
                    tail_blockage_score=0.1,
                    leader_id=None,
                    noisy_leader_gap_m=None,
                    noisy_leader_speed_mps=None,
                    last_v2x_poll_time=t,
                ),
                "veh_1": VehicleBelief(
                    veh_id="veh_1",
                    entered_at=0.0,
                    corridor_id=sample_corr,
                    edge_id=sample_corr,
                    in_junction_id=None,
                    lane_index=0,
                    length_m=6.2,
                    width_m=2.25,
                    mean_s_m=8.0 + 3.2 * k,
                    mean_v_mps=v2_spd,
                    cov_2x2=np.array([[4.0, 0.2], [0.2, 0.64]], dtype=np.float64),
                    accel_est_mps2=0.2,
                    obs_age_s=1.0,
                    wait_time_s=0.0,
                    turn_probs=(0.5, 0.5),
                    free_eta_s=11.2,
                    eta_std_s=1.3,
                    tail_blockage_score=0.35,
                    leader_id="veh_0",
                    noisy_leader_gap_m=12.1,
                    noisy_leader_speed_mps=v1_spd,
                    last_v2x_poll_time=0.0,
                ),
            },
            corridors={
                sample_corr: CorridorBeliefStats(
                    corridor_id=sample_corr,
                    active_count=3,
                    effective_lanes=2,
                    storage_capacity=30,
                    occupancy_ratio=0.45 + 0.05 * k,
                    tail_blockage_prob=0.25 + 0.06 * k,
                    expected_inflow=1.8,
                    mean_speed_mps=7.2,
                )
            },
            open_lanes_by_edge={sample_corr: 2},
            tls_states={sample_tls: "GGrrGGrr" if k < 4 else "yyrrGGrr"},
        )
        edges = (
            DBNEdge("veh_0", "veh_1", RelationType.SAME_CORRIDOR, 0.88, 0.72, 0.41, 0.10, 0.85),
            DBNEdge("veh_0", "veh_2", RelationType.SAME_JUNCTION_CONFLICT, 0.76, 0.64, 0.55, 0.82, 0.20),
            DBNEdge("veh_1", "veh_2", RelationType.DOWNSTREAM_SPILLBACK, 0.61, 0.52, 0.48, 0.35, 0.60),
        )
        mem = DBNTemporalMemory(
            sim_time=t,
            veh_hidden={"veh_0": np.ones(8, dtype=np.float32)},
            corridor_hidden={sample_corr: np.zeros(8, dtype=np.float32)},
            edge_hidden={("veh_0", "veh_1"): np.ones(8, dtype=np.float32)},
        )
        snap = DBNStateSnapshot(
            sim_time=t,
            veh_ids=("veh_0", "veh_1", "veh_2"),
            precedence_scores={"veh_0": 1.4, "veh_1": 0.3, "veh_2": -0.5},
            edges=edges,
            edges_by_dst={"veh_1": (edges[0],), "veh_2": (edges[1], edges[2])},
            edges_by_src={"veh_0": (edges[0], edges[1]), "veh_1": (edges[2],)},
            delay_pred={"veh_0": 0.18, "veh_1": 0.62, "veh_2": 0.74},
            delay_post={"veh_0": 0.15, "veh_1": 0.58, "veh_2": 0.79},
            expected_delay_s={"veh_0": 2.1, "veh_1": 8.4, "veh_2": 12.6},
            cascade_score={"veh_0": 0.84, "veh_1": 0.49, "veh_2": 0.22},
            corridor_spillback_pred={sample_corr: 0.42 + 0.05 * k},
            raw_edge_probs={("veh_0", "veh_1"): 0.88, ("veh_0", "veh_2"): 0.76},
            update_latency_ms=0.95,
            inference_latency_ms=0.38,
            temporal_memory=mem,
        )
        act = SwarmControlAction(
            sim_time=t,
            target_speeds={"veh_0": 12.5, "veh_1": 10.0, "veh_2": 6.5},
            post_safety_accels={"veh_0": 0.4, "veh_1": 0.2, "veh_2": -0.8},
            modes={
                "veh_0": LongitudinalMode.GO_WAVE,
                "veh_1": LongitudinalMode.ECO_GLIDE,
                "veh_2": LongitudinalMode.YIELD,
            },
            tls_desired_greens=None,
            broadcast_edge_costs={sample_corr: 14.2},
            v2x_polls=("veh_1",),
        )
        recorder.record_step(gt, belief, snap, act, belief.tls_states)

    metrics = _make_episode_metrics("grid2x2_rush", policy_name, 3001, 18.4 if speed_scale >= 1.0 else 29.7)
    return recorder.finalize(metrics=metrics)


def test_episode_trace_recorder_protocol_and_invariants(tmp_path: Path) -> None:
    """Verify trace serialization, DBNTemporalMemory exclusion, and fail-loud invariant checks."""
    topo = _build_synthetic_topology(tmp_path)
    trace = _record_synthetic_trace(topo, "neural_dbn")
    assert trace["frame_count"] == 6
    assert len(trace["frames"]) == 6
    assert trace["frames"][0]["vehicles"][0]["pos_std"] == pytest.approx(1.2, abs=1e-3)
    assert len(trace["frames"][0]["dbn_edges"]) == 3

    trace_file = tmp_path / "episode_trace.json"
    rec = EpisodeTraceRecorder("s", "p", 1)
    # Test atomic save_json directly as well as full trace serialization
    empty_saved = rec.save_json(tmp_path / "empty_trace.json")
    assert empty_saved.is_file()
    trace_file.write_text(json.dumps(trace), encoding="utf-8")
    raw_txt = trace_file.read_text(encoding="utf-8")
    assert "temporal_memory" not in raw_txt
    assert "veh_hidden" not in raw_txt
    reloaded = load_trace_json(trace_file)
    assert reloaded["policy_name"] == "neural_dbn"

    # Non-monotonic timestamp must raise SimulationLifecycleError loudly
    gt_t1 = GroundTruthStepRecord(1.0, (), 0, 0, 0, 0, 0, 5.0, 0, 0, 0, 0.0)
    gt_t05 = GroundTruthStepRecord(0.5, (), 0, 0, 0, 0, 0, 5.0, 0, 0, 0, 0.0)
    empty_act = SwarmControlAction(1.0, {}, {}, {}, None, None)
    rec.record_step(gt_t1, None, None, empty_act, {})
    with pytest.raises(SimulationLifecycleError):
        rec.record_step(gt_t05, None, None, empty_act, {})


def test_write_city_3d_viewer_single_and_split_screen(tmp_path: Path) -> None:
    """Verify compressed Gzip+SoA 3D HTML viewer generation, round-trip fidelity, minimalist HUD selectors, and ``node --check``."""
    topo = _build_synthetic_topology(tmp_path)
    scene = export_scene_3d(topo)
    assert "focus_bounds" in scene
    primary_trace = _record_synthetic_trace(topo, "neural_dbn", speed_scale=1.0)
    baseline_trace = _record_synthetic_trace(topo, "actuated_signal", speed_scale=0.72)

    single_html_path = write_city_3d_viewer(scene, primary_trace, tmp_path / "replay_single_3d.html")
    split_html_path = write_city_3d_viewer(
        scene,
        primary_trace,
        tmp_path / "replay_split_3d.html",
        comparison_trace=baseline_trace,
    )

    required_selectors = (
        'id="viewport-grid"',
        'data-layout="split"',
        'data-layout="primary"',
        'data-layout="compare"',
        'id="pane-primary"',
        'id="canvas-primary"',
        'id="pane-compare"',
        'id="canvas-compare"',
        'id="top-hud"',
        'id="scenario-title"',
        'id="layout-group"',
        'id="camera-group"',
        'data-cam="orbit"',
        'data-cam="ortho"',
        'data-cam="chase"',
        'id="btn-export-png"',
        'id="scorecard-primary"',
        'id="scorecard-compare"',
        'id="title-primary"',
        'id="title-compare"',
        'id="badge-delta"',
        'id="kpi-p-delay"',
        'id="kpi-p-throughput"',
        'id="kpi-p-spill"',
        'id="kpi-p-safety"',
        'id="kpi-p-live"',
        'id="kpi-c-delay"',
        'id="kpi-c-throughput"',
        'id="kpi-c-spill"',
        'id="kpi-c-safety"',
        'id="kpi-c-live"',
        'id="vehicle-chip"',
        'id="chip-id"',
        'id="chip-mode"',
        'id="chip-speed"',
        'id="chip-delay"',
        'id="chip-close"',
        'id="playback-bar"',
        'id="btn-play"',
        'id="time-readout"',
        'id="timeline-slider"',
        'id="btn-speed"',
        'id="toggle-arcs"',
        'id="arcs-legend"',
        'id="toggle-buildings"',
        'id="loading-overlay"',
        'id="loading-status"',
        'id="error-banner"',
        'id="error-message"',
    )
    removed_clutter = (
        'id="side-drawer"',
        'id="slider-omega"',
        'id="sparkline-canvas"',
    )

    for html_p, expect_compare in ((single_html_path, False), (split_html_path, True)):
        assert html_p.is_file() and html_p.stat().st_size > 40_000
        text = html_p.read_text(encoding="utf-8")
        assert re.findall(r"https?://", text, flags=re.IGNORECASE) == []
        for sel in required_selectors:
            assert sel in text, f"Missing required minimalist UI selector {sel!r} in {html_p.name}"
        for old_sel in removed_clutter:
            assert old_sel not in text, f"Legacy clutter selector {old_sel!r} still present in {html_p.name}"

        # Verify metadata script block
        m_meta = re.search(
            r'<script id="swarm-payload-meta" type="application/json">\s*(.*?)\s*</script>',
            text,
            flags=re.DOTALL,
        )
        assert m_meta is not None
        meta_obj = json.loads(m_meta.group(1))
        assert meta_obj["format"] == PAYLOAD_FORMAT_VERSION
        assert meta_obj["network_id"] == "synthetic_grid2x2"
        assert meta_obj["primary_policy"] == "neural_dbn"
        assert meta_obj["has_comparison"] is expect_compare
        assert "focus_bounds" in meta_obj

        # Verify direct Python base64 + gzip.decompress round-trip fidelity of <script id="swarm-payload-gz">
        m_gz = re.search(
            r'<script id="swarm-payload-gz" type="application/octet-stream">\s*(.*?)\s*</script>',
            text,
            flags=re.DOTALL,
        )
        assert m_gz is not None
        raw_gz_bytes = base64.b64decode(m_gz.group(1).strip(), validate=True)
        decompressed_json = gzip.decompress(raw_gz_bytes).decode("utf-8")
        packed_payload = json.loads(decompressed_json)
        assert packed_payload["format"] == PAYLOAD_FORMAT_VERSION
        assert packed_payload["scene"]["network_id"] == "synthetic_grid2x2"
        assert "focus_bounds" in packed_payload["scene"]

        unpacked = unpack_viewer_payload(packed_payload)
        decoded_direct = decode_viewer_html_payload(text)
        for check_obj in (unpacked, decoded_direct):
            assert check_obj["scene"]["network_id"] == "synthetic_grid2x2"
            assert len(check_obj["scene"]["lanes"]) == len(scene["lanes"])
            assert len(check_obj["scene"]["buildings"]) == len(scene["buildings"])
            p_tr = check_obj["primary_trace"]
            assert p_tr["policy_name"] == "neural_dbn"
            assert p_tr["frame_count"] == 6
            assert len(p_tr["frames"]) == 6
            assert len(p_tr["frames"][0]["vehicles"]) == 3
            assert len(p_tr["frames"][0]["dbn_edges"]) == 3
            assert p_tr["frames"][0]["vehicles"][0]["id"] == "veh_0"
            assert p_tr["frames"][0]["vehicles"][0]["x"] == pytest.approx(20.0, abs=0.02)
            if expect_compare:
                c_tr = check_obj["comparison_trace"]
                assert c_tr is not None
                assert c_tr["policy_name"] == "actuated_signal"
                assert len(c_tr["frames"]) == 6
            else:
                assert check_obj["comparison_trace"] is None

    # Also verify >20-frame large-trace pure SoA packing/unpacking & weak-edge pruning
    large_frames = [
        dict(primary_trace["frames"][k % 6], t=0.5 * k)
        for k in range(25)
    ]
    large_frames[0] = dict(
        large_frames[0],
        dbn_edges=list(large_frames[0]["dbn_edges"])
        + [{"src": "veh_1", "dst": "veh_0", "rel": 0, "omega": 0.04, "kappa": 0.02, "gamma": 0.01, "p_conf": 0.0}],
    )
    large_trace = dict(primary_trace, frame_count=25, frames=large_frames)
    packed_large = pack_viewer_payload(scene, large_trace)
    assert "frames" not in packed_large["primary_trace"]
    assert packed_large["primary_trace"]["encoding"] == "soa_v2"
    unpacked_large = unpack_viewer_payload(packed_large)["primary_trace"]
    assert len(unpacked_large["frames"]) == 25
    # Weak edge (omega=0.04 < 0.12) pruned cleanly while all 3 strong edges preserved
    assert len(unpacked_large["frames"][0]["dbn_edges"]) == 3
    assert unpacked_large["frames"][4]["vehicles"][0]["x"] == pytest.approx(
        primary_trace["frames"][4]["vehicles"][0]["x"], abs=0.02
    )

    # Validate embedded JS syntax using Node.js (`node --check`)
    node_bin = shutil.which("node") or (
        "/opt/homebrew/bin/node" if Path("/opt/homebrew/bin/node").is_file() else None
    )
    assert node_bin is not None, "node binary required for JS syntax verification"
    split_txt = split_html_path.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>\s*(.*?)\s*</script>", split_txt, flags=re.DOTALL)
    assert len(scripts) == 1
    extracted_js = tmp_path / "embedded_viewer_bundle.js"
    extracted_js.write_text(scripts[0], encoding="utf-8")
    proc = subprocess.run([node_bin, "--check", str(extracted_js)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"node --check failed:\n{proc.stderr}"

    with pytest.raises(ConfigValidationError):
        write_city_3d_viewer(scene, {"frames": []}, tmp_path / "invalid.html")


def test_publication_plots_2d_and_no_cwd_mplconfig(tmp_path: Path) -> None:
    """Verify all 5 300-DPI publication plots, NaN fail-loud checks, and zero CWD ``.mplconfig``."""
    cwd_mpl = Path.cwd() / ".mplconfig"
    existed_before = cwd_mpl.exists()

    # Build paired EpisodeMetrics across ID + Zero-Shot 4x4 OOD + Zero-Shot Palo Alto OOD + Ablations
    episodes: list[EpisodeMetrics] = []
    policy_delays = {
        "fixed_signal": 42.0,
        "actuated_signal": 31.5,
        "fcfs_reservation": 34.0,
        "independent_belief": 28.5,
        "heuristic_radius_dbn": 24.5,
        "online_bic_dbn": 22.8,
        "neural_dbn": 16.4,
        "no_temporal_prior": 21.2,
        "static_cpd": 20.4,
        "fixed_route": 19.3,
    }
    for scen_idx, scen_id in enumerate(("grid3x3_rush_asymmetric", "grid4x4_incident", "osm_palo_alto_rush")):
        for pol, base_d in policy_delays.items():
            for s in range(3000, 3006):
                jitter = 0.7 * ((s - 3000) - 2.5) + 3.5 * scen_idx
                m = _make_episode_metrics(scen_id, pol, s, base_d + jitter)
                episodes.append(m)

    eval_report = {"episodes": episodes}

    rng = np.random.default_rng(42)
    y_true_id = (rng.uniform(size=160) > 0.65).astype(np.float64)
    y_prob_id_pre = np.clip(0.15 + 0.72 * y_true_id + rng.normal(0.0, 0.14, size=160), 0.01, 0.99)
    y_prob_id_post = np.clip(0.22 + 0.58 * y_true_id + rng.normal(0.0, 0.11, size=160), 0.02, 0.98)

    y_true_ood = (rng.uniform(size=160) > 0.60).astype(np.float64)
    y_prob_ood = np.clip(0.20 + 0.55 * y_true_ood + rng.normal(0.0, 0.15, size=160), 0.02, 0.98)

    predictive_report = {
        "splits": {
            "held_out_grid3x3_id": {
                "y_true": y_true_id.tolist(),
                "y_prob": y_prob_id_post.tolist(),
                "y_prob_uncalibrated": y_prob_id_pre.tolist(),
            },
            "zero_shot_palo_alto_ood": {
                "y_true": y_true_ood.tolist(),
                "y_prob": y_prob_ood.tolist(),
            },
        }
    }

    sizes = [16, 32, 64, 128, 256]
    scaling_report = {
        "swarm_sizes": sizes,
        "methods": {
            "neural_dbn": [
                {"n": n, "median_ms": 0.35 + 0.0085 * n, "iqr_ms": 0.08 + 0.001 * n, "p95_ms": 0.48 + 0.011 * n}
                for n in sizes
            ],
            "online_bic_dbn": [
                {"n": n, "median_ms": 0.45 + 0.016 * n, "iqr_ms": 0.12 + 0.002 * n, "p95_ms": 0.62 + 0.021 * n}
                for n in sizes
            ],
            "unconstrained_bic": [
                {"n": n, "median_ms": 0.12 + 0.00095 * (n**2), "iqr_ms": 0.05 + 0.0001 * (n**2), "p95_ms": 0.2 + 0.0012 * (n**2)}
                for n in sizes
            ],
        },
    }

    out_dir = tmp_path / "publication_figures"
    paths = render_publication_plots(
        eval_report,
        out_dir,
        predictive_report=predictive_report,
        scaling_report=scaling_report,
        dpi=300,
    )
    assert len(paths) == 5
    for p in paths:
        assert p.is_file()
        raw_bytes = p.read_bytes()
        assert raw_bytes[:8] == PNG_MAGIC
        assert len(raw_bytes) > 15_000

    # Also test scalar-only predictive calibration report schema
    scalar_pred_png = plot_calibration_and_roc_pr(
        {
            "splits": {
                "held_out_id": {"roc_auc": 0.912, "pr_auc": 0.845, "ece_pre": 0.094, "ece_post": 0.021},
                "zero_shot_ood": {"roc_auc": 0.874, "pr_auc": 0.791, "ece_pre": 0.112, "ece_post": 0.034},
            }
        },
        tmp_path / "scalar_cal.png",
    )
    assert scalar_pred_png.read_bytes()[:8] == PNG_MAGIC

    # Fail-loud NumericalInvariantError on NaN metric values
    bad_scaling = {"swarm_sizes": [16], "methods": {"neural_dbn": [{"n": 16, "median_ms": float("nan")}]}}
    with pytest.raises(NumericalInvariantError):
        plot_latency_scaling(bad_scaling, tmp_path / "bad_scale.png")

    with pytest.raises(ConfigValidationError):
        render_publication_plots(None, tmp_path / "empty_out")

    # Verify zero `.mplconfig` created in Path.cwd()
    if not existed_before:
        assert not cwd_mpl.exists(), "swarm.viz must NEVER create .mplconfig inside Path.cwd()!"

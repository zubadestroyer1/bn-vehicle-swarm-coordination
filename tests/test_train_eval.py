"""Comprehensive test suite for Wave 2B Slice 4 (``swarm.pipelines`` & ``swarm.cli``).

Verifies:
1. Boundary Lock C static AST causality firewall & zero-silent-fallback enforcement across ``swarm/**/*.py``.
2. Pre-registered seed-split (``TRAIN_SEEDS`` / ``VAL_SEEDS`` / ``TEST_SEEDS``) & ID/OOD anti-leakage locks.
3. Future-horizon ``[t, t + H_p]`` causal supervision label extraction (delay, tau, spillback, edges, ranking).
4. Unrolled multi-step ``STGraphTransformerDBN`` training, Boundary Lock B ``model.eval()`` L-BFGS temperature
   calibration, ``weights_only=True`` checkpoint integrity, and held-out ID/OOD predictive evaluation.
5. 10,000-resample paired percentile bootstrap CIs, exact Wilcoxon signed-rank tie/zero handling, & Holm-Bonferroni.
6. Multi-episode evaluator matrix, provenance recording, and 2D plot schema integration.
7. Warmed CPU DBN structure-update & LBP scaling benchmark across all 4 updaters.
8. ``swarm`` CLI subcommand dispatch and atomic ``error_report_<UTC>_pid<PID>.json`` crash diagnostics.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from swarm.cli import build_parser, main as cli_main
from swarm.config import (
    ControlConfig,
    DBNConfig,
    ExperimentConfig,
    ScenarioConfig,
    TEST_SEEDS,
    TRAIN_SEEDS,
    TrainConfig,
    VAL_SEEDS,
    file_sha256,
)
from swarm.contracts import (
    CORRIDOR_EDGE_FEATURE_DIM,
    CORRIDOR_FEATURE_DIM,
    CorridorBeliefStats,
    CorridorSpec,
    DepartedVehicleEvent,
    EDGE_FEATURE_DIM,
    GroundTruthStepRecord,
    JunctionSpec,
    NODE_FEATURE_DIM,
    ObservationBatch,
    RelationType,
    RoadTopology,
    SCHEMA_VERSION,
    SwarmBeliefState,
    SwarmControlAction,
    TurnOption,
    VehicleBelief,
    VehicleRawTelemetry,
)
from swarm.dbn.updater import load_dbn_checkpoint
from swarm.logging_utils import CheckpointIntegrityError, ConfigValidationError
from swarm.pipelines import (
    DEFAULT_BENCHMARK_METHODS,
    bootstrap_mean_ci,
    compute_episode_causal_labels,
    compute_paired_statistics,
    evaluate_experiment_matrix,
    evaluate_heldout_predictive,
    exact_wilcoxon_two_sided,
    holm_bonferroni_adjust,
    run_scaling_benchmark,
    train_neural_dbn,
    validate_and_hash_checkpoint,
    validate_eval_seeds,
    validate_rollout_split,
)
from swarm.viz import (
    plot_calibration_and_roc_pr,
    plot_latency_scaling,
    plot_paired_forest,
    plot_scenario_comparison,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SWARM_PKG_DIR = REPO_ROOT / "swarm"


def _handler_contains_raise(stmts: list[ast.stmt]) -> bool:
    """Return True if an except handler body contains a ``raise`` statement along its execution path."""
    for node in ast.walk(ast.Module(body=stmts, type_ignores=[])):
        if isinstance(node, ast.Raise):
            return True
    return False


def _collect_non_type_checking_imports(tree: ast.Module) -> list[tuple[int, str]]:
    """Collect imported module/symbol names outside ``if TYPE_CHECKING:`` guards."""
    found: list[tuple[int, str]] = []

    def _visit_block(stmts: list[ast.stmt], inside_type_checking: bool = False) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.If):
                test = stmt.test
                is_tc = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                    isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
                )
                _visit_block(stmt.body, inside_type_checking or is_tc)
                _visit_block(stmt.orelse, inside_type_checking)
                continue
            if inside_type_checking:
                continue
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    found.append((stmt.lineno, alias.name))
            elif isinstance(stmt, ast.ImportFrom):
                mod = stmt.module or ""
                found.append((stmt.lineno, mod))
                for alias in stmt.names:
                    found.append((stmt.lineno, f"{mod}.{alias.name}" if mod else alias.name))
            else:
                for child_field in ("body", "orelse", "finalbody"):
                    sub_body = getattr(stmt, child_field, None)
                    if isinstance(sub_body, list):
                        _visit_block(sub_body, inside_type_checking)
                if isinstance(stmt, ast.Try):
                    for h in stmt.handlers:
                        _visit_block(h.body, inside_type_checking)

    _visit_block(tree.body, False)
    return found


def test_boundary_lock_c_static_ast_firewall_and_zero_fallbacks() -> None:
    """Enforce Boundary Lock C across every Python module in ``swarm/``."""
    py_files = sorted(SWARM_PKG_DIR.rglob("*.py"))
    assert len(py_files) >= 20

    forbidden_firewall_tokens = (
        "libsumo",
        "traci",
        "VehicleRawTelemetry",
        "GroundTruthStepRecord",
        "pipelines.labels",
    )

    for py_path in py_files:
        rel = py_path.relative_to(REPO_ROOT).as_posix()
        src = py_path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(py_path))

        # 1. Zero nan_to_num, zero weights_only=False, mandatory weights_only=True on torch.load
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "nan_to_num":
                pytest.fail(f"Forbidden nan_to_num symbol in {rel}:{node.lineno}")
            if isinstance(node, ast.Attribute) and node.attr == "nan_to_num":
                pytest.fail(f"Forbidden .nan_to_num attribute in {rel}:{node.lineno}")

            if isinstance(node, ast.Call):
                is_torch_load = (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"
                )
                if is_torch_load:
                    kw_map = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
                    assert "weights_only" in kw_map, f"torch.load missing weights_only=True in {rel}:{node.lineno}"
                    w_val = kw_map["weights_only"]
                    assert isinstance(w_val, ast.Constant) and w_val.value is True, (
                        f"torch.load must pass literal weights_only=True in {rel}:{node.lineno}"
                    )

            # 2. Zero bare except, zero silent except (every except must re-raise), zero import traci in except
            if isinstance(node, ast.ExceptHandler):
                assert node.type is not None, f"Forbidden bare `except:` in {rel}:{node.lineno}"
                assert _handler_contains_raise(node.body), (
                    f"Silent exception handler without `raise` in {rel}:{node.lineno}"
                )
                for sub in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                    if isinstance(sub, ast.Import) and any(a.name == "traci" for a in sub.names):
                        pytest.fail(f"Forbidden `import traci` fallback inside except in {rel}:{sub.lineno}")

        # 3. Runtime privilege firewall on belief.py, swarm/dbn/*, and swarm/control/*
        is_firewalled = (
            rel == "swarm/env/belief.py"
            or rel.startswith("swarm/dbn/")
            or rel.startswith("swarm/control/")
        )
        if is_firewalled:
            for lineno, imp_target in _collect_non_type_checking_imports(tree):
                for bad_tok in forbidden_firewall_tokens:
                    if bad_tok in imp_target:
                        pytest.fail(
                            f"Causality/Privilege Firewall violation in {rel}:{lineno}: "
                            f"imported forbidden symbol/module {imp_target!r}"
                        )

            forbidden_runtime_symbols = {
                "libsumo",
                "traci",
                "VehicleRawTelemetry",
                "GroundTruthStepRecord",
                "EpisodeMetrics",
                "route_goal",
            }

            class _StripTypeChecking(ast.NodeTransformer):
                def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt] | None:
                    test = node.test
                    is_tc = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
                    )
                    if is_tc:
                        return [self.visit(s) for s in node.orelse] if node.orelse else None
                    return self.generic_visit(node)

            runtime_tree = _StripTypeChecking().visit(ast.parse(src, filename=str(py_path)))
            for node in ast.walk(runtime_tree):
                if isinstance(node, ast.Name) and node.id in forbidden_runtime_symbols:
                    pytest.fail(
                        f"Causality/Privilege Firewall runtime Name {node.id!r} in {rel}:{node.lineno}"
                    )
                if isinstance(node, ast.Attribute) and node.attr in forbidden_runtime_symbols:
                    pytest.fail(
                        f"Causality/Privilege Firewall runtime Attribute .{node.attr} in {rel}:{node.lineno}"
                    )


def _make_two_corridor_topology() -> RoadTopology:
    turn_c0_c1 = TurnOption(
        turn_idx=0,
        from_edge="E0",
        to_corridor="C1",
        to_edge="E1",
        angle_rad=0.0,
        tls_id="J0",
        tls_link_indices=(0,),
        junction_req_indices=(0,),
    )
    c0 = CorridorSpec(
        corridor_id="C0",
        edges=("E0",),
        edge_offsets_m=(0.0,),
        from_junction="J_IN",
        to_junction="J0",
        length_m=120.0,
        base_lanes=2,
        speed_limit_mps=15.0,
        turn_options=(turn_c0_c1,),
        is_entry_fringe=True,
        is_exit_fringe=False,
    )
    c1 = CorridorSpec(
        corridor_id="C1",
        edges=("E1",),
        edge_offsets_m=(0.0,),
        from_junction="J0",
        to_junction="J_OUT",
        length_m=120.0,
        base_lanes=2,
        speed_limit_mps=15.0,
        turn_options=(),
        is_entry_fringe=False,
        is_exit_fringe=True,
    )
    j_in = JunctionSpec(
        junction_id="J_IN",
        tls_id=None,
        coord_xy=(0.0, 0.0),
        incoming_corridors=(),
        outgoing_corridors=("C0",),
        num_tls_links=0,
        tls_link_foe_matrix=np.zeros((0, 0), dtype=bool),
        req_foe_matrix=np.zeros((0, 0), dtype=bool),
    )
    j0 = JunctionSpec(
        junction_id="J0",
        tls_id="J0",
        coord_xy=(120.0, 0.0),
        incoming_corridors=("C0",),
        outgoing_corridors=("C1",),
        num_tls_links=1,
        tls_link_foe_matrix=np.zeros((1, 1), dtype=bool),
        req_foe_matrix=np.zeros((1, 1), dtype=bool),
    )
    j_out = JunctionSpec(
        junction_id="J_OUT",
        tls_id=None,
        coord_xy=(240.0, 0.0),
        incoming_corridors=("C1",),
        outgoing_corridors=(),
        num_tls_links=0,
        tls_link_foe_matrix=np.zeros((0, 0), dtype=bool),
        req_foe_matrix=np.zeros((0, 0), dtype=bool),
    )
    return RoadTopology(
        network_id="mini_test",
        net_xml_path="configs/networks/grid3x3.net.xml",
        poly_xml_path=None,
        corridors={"C0": c0, "C1": c1},
        edge_to_corridor={"E0": "C0", "E1": "C1"},
        edge_offset_in_corridor={"E0": 0.0, "E1": 0.0},
        junctions={"J_IN": j_in, "J0": j0, "J_OUT": j_out},
        successor_corridors={"C0": ("C1",), "C1": ()},
        predecessor_corridors={"C0": (), "C1": ("C0",)},
    )


def test_seed_split_and_ood_anti_leakage_guards() -> None:
    """Verify strict seed-split and ID/OOD topology separation across train, val, and eval."""
    scen_id = ScenarioConfig(
        scenario_id="grid3x3_nominal",
        network_id="grid3x3",
        net_xml_path="configs/networks/grid3x3.net.xml",
        is_ood_regime=False,
    )
    scen_ood = ScenarioConfig(
        scenario_id="grid4x4_incident",
        network_id="grid4x4",
        net_xml_path="configs/networks/grid4x4.net.xml",
        is_ood_regime=True,
    )

    assert validate_rollout_split(scen_id, [TRAIN_SEEDS[0], TRAIN_SEEDS[1]], "train") == (
        TRAIN_SEEDS[0],
        TRAIN_SEEDS[1],
    )
    assert validate_rollout_split(scen_id, [VAL_SEEDS[0]], "val") == (VAL_SEEDS[0],)
    assert validate_rollout_split(scen_id, [TEST_SEEDS[0]], "eval_id") == (TEST_SEEDS[0],)
    assert validate_rollout_split(scen_ood, [TEST_SEEDS[0]], "eval_ood") == (TEST_SEEDS[0],)
    assert validate_eval_seeds([TEST_SEEDS[0], TEST_SEEDS[2]], require_test_seeds=True) == (
        TEST_SEEDS[0],
        TEST_SEEDS[2],
    )

    # Reject TEST_SEEDS in train or val
    with pytest.raises(ConfigValidationError):
        validate_rollout_split(scen_id, [TEST_SEEDS[0]], "train")
    with pytest.raises(ConfigValidationError):
        validate_rollout_split(scen_id, [TEST_SEEDS[0]], "val")

    # Reject OOD scenario in train or val even with valid TRAIN_SEEDS / VAL_SEEDS
    with pytest.raises(ConfigValidationError):
        validate_rollout_split(scen_ood, [TRAIN_SEEDS[0]], "train")
    with pytest.raises(ConfigValidationError):
        validate_rollout_split(scen_ood, [VAL_SEEDS[0]], "val")

    # Reject TRAIN_SEEDS or VAL_SEEDS in evaluation
    with pytest.raises(ConfigValidationError):
        validate_eval_seeds([TRAIN_SEEDS[0]], require_test_seeds=True)
    with pytest.raises(ConfigValidationError):
        validate_eval_seeds([VAL_SEEDS[0]], require_test_seeds=True)


def test_future_horizon_causal_label_extractor() -> None:
    """Verify offline future-horizon causal labels on a controlled car-following & spillback episode."""
    topo = _make_two_corridor_topology()
    belief_steps: list[SwarmBeliefState] = []
    gt_steps: list[GroundTruthStepRecord] = []
    cands_by_step: list[list[tuple[str, str, RelationType]]] = []

    # 6 ticks at dt=1.0s (horizon=4.0s):
    # v_lead (ahead on C0 at s=80m) brakes sharply at t=0..3, forcing v_follow (behind at s=55m) to slow down at t=1..4;
    # v_tail crawls at s=8m on C1 (pos <= 22m, speed <= 1.5 m/s) triggering C1 spillback!
    for step_k in range(6):
        sim_t = float(step_k)
        lead_v = 1.0 if step_k <= 3 else 12.0
        foll_v = 13.0 if step_k == 0 else (2.0 if step_k <= 4 else 12.0)
        lead_loss = float(min(step_k, 4) * 0.9)
        foll_loss = float(max(0, min(step_k - 1, 4)) * 0.85)

        cov_mat = np.array([[1.0, 0.0], [0.0, 0.5]], dtype=np.float64)
        vb_lead = VehicleBelief(
            veh_id="v_lead",
            entered_at=0.0,
            corridor_id="C0",
            edge_id="E0",
            in_junction_id=None,
            lane_index=0,
            length_m=4.5,
            width_m=1.8,
            mean_s_m=80.0 + step_k * 2.0,
            mean_v_mps=lead_v,
            cov_2x2=cov_mat.copy(),
            accel_est_mps2=-2.5 if step_k == 0 else 0.0,
            obs_age_s=0.0,
            wait_time_s=float(step_k),
            turn_probs=(1.0,),
            free_eta_s=2.5,
            eta_std_s=0.3,
            tail_blockage_score=0.0,
            leader_id=None,
            noisy_leader_gap_m=None,
            noisy_leader_speed_mps=None,
            last_v2x_poll_time=0.0,
        )
        vb_foll = VehicleBelief(
            veh_id="v_follow",
            entered_at=0.0,
            corridor_id="C0",
            edge_id="E0",
            in_junction_id=None,
            lane_index=0,
            length_m=4.5,
            width_m=1.8,
            mean_s_m=55.0 + step_k * 2.0,
            mean_v_mps=foll_v,
            cov_2x2=cov_mat.copy(),
            accel_est_mps2=-2.2 if step_k == 1 else 0.0,
            obs_age_s=0.0,
            wait_time_s=0.0,
            turn_probs=(1.0,),
            free_eta_s=4.5,
            eta_std_s=0.3,
            tail_blockage_score=0.0,
            leader_id="v_lead",
            noisy_leader_gap_m=18.0,
            noisy_leader_speed_mps=lead_v,
            last_v2x_poll_time=0.0,
        )
        vb_tail = VehicleBelief(
            veh_id="v_tail",
            entered_at=0.0,
            corridor_id="C1",
            edge_id="E1",
            in_junction_id=None,
            lane_index=0,
            length_m=4.5,
            width_m=1.8,
            mean_s_m=8.0 + step_k * 0.5,
            mean_v_mps=0.8,
            cov_2x2=cov_mat.copy(),
            accel_est_mps2=0.0,
            obs_age_s=0.0,
            wait_time_s=2.0,
            turn_probs=(),
            free_eta_s=7.5,
            eta_std_s=0.3,
            tail_blockage_score=0.9,
            leader_id=None,
            noisy_leader_gap_m=None,
            noisy_leader_speed_mps=None,
            last_v2x_poll_time=0.0,
        )

        c0_stats = CorridorBeliefStats("C0", 2, 2, 34, 0.15, 0.05, 0.5, 7.0)
        c1_stats = CorridorBeliefStats("C1", 1, 2, 34, 0.25, 0.85, 0.5, 1.0)
        belief_steps.append(
            SwarmBeliefState(
                sim_time=sim_t,
                dt=1.0,
                vehicles={"v_follow": vb_foll, "v_lead": vb_lead, "v_tail": vb_tail},
                corridors={"C0": c0_stats, "C1": c1_stats},
                open_lanes_by_edge={"E0": 2, "E1": 2},
                tls_states={"J0": "G"},
            )
        )

        gt_steps.append(
            GroundTruthStepRecord(
                sim_time=sim_t,
                raw_vehicles=(
                    VehicleRawTelemetry(
                        veh_id="v_lead",
                        vtype="passenger_urban",
                        center_xy=(80.0 + step_k * 2.0, 0.0),
                        yaw_ccw_rad=0.0,
                        speed=lead_v,
                        accel=-2.5 if step_k <= 1 else 0.0,
                        road_id="E0",
                        lane_id="E0_0",
                        lane_index=0,
                        lane_pos=80.0 + step_k * 2.0,
                        lane_length=120.0,
                        route_edges=("E0", "E1"),
                        route_index=0,
                        route_goal="E1",
                        length=4.5,
                        width=1.8,
                        brake_light=step_k <= 1,
                        turn_signal=0,
                        leader_id=None,
                        leader_gap=None,
                        time_loss=lead_loss,
                        waiting_time=float(step_k),
                        co2_mg_s=1500.0,
                        scheduled_depart=0.0,
                    ),
                    VehicleRawTelemetry(
                        veh_id="v_follow",
                        vtype="passenger_urban",
                        center_xy=(55.0 + step_k * 2.0, 0.0),
                        yaw_ccw_rad=0.0,
                        speed=foll_v,
                        accel=-2.2 if step_k == 1 else 0.0,
                        road_id="E0",
                        lane_id="E0_0",
                        lane_index=0,
                        lane_pos=55.0 + step_k * 2.0,
                        lane_length=120.0,
                        route_edges=("E0", "E1"),
                        route_index=0,
                        route_goal="E1",
                        length=4.5,
                        width=1.8,
                        brake_light=step_k == 1,
                        turn_signal=0,
                        leader_id="v_lead",
                        leader_gap=18.0,
                        time_loss=foll_loss,
                        waiting_time=0.0,
                        co2_mg_s=1400.0,
                        scheduled_depart=0.0,
                    ),
                    VehicleRawTelemetry(
                        veh_id="v_tail",
                        vtype="passenger_urban",
                        center_xy=(128.0 + step_k * 0.5, 0.0),
                        yaw_ccw_rad=0.0,
                        speed=0.8,
                        accel=0.0,
                        road_id="E1",
                        lane_id="E1_0",
                        lane_index=0,
                        lane_pos=8.0 + step_k * 0.5,
                        lane_length=120.0,
                        route_edges=("E0", "E1"),
                        route_index=1,
                        route_goal="E1",
                        length=4.5,
                        width=1.8,
                        brake_light=True,
                        turn_signal=0,
                        leader_id=None,
                        leader_gap=None,
                        time_loss=float(step_k * 0.95),
                        waiting_time=2.0,
                        co2_mg_s=1200.0,
                        scheduled_depart=0.0,
                    ),
                ),
                collisions=0,
                teleports=0,
                emergency_stops=0,
                emergency_brakes=0,
                near_misses_ttc=0,
                min_ttc_s=999.0,
                reroutes_applied=0,
                v2x_sent=0,
                v2x_delivered=0,
                step_co2_mg=4100.0,
                spillback_corridor_count=1,
            )
        )
        cands_by_step.append(
            [
                ("v_lead", "v_follow", RelationType.SAME_CORRIDOR),
                ("v_follow", "v_lead", RelationType.SAME_CORRIDOR),
                ("v_tail", "v_lead", RelationType.DOWNSTREAM_SPILLBACK),
            ]
        )

    labels = compute_episode_causal_labels(
        belief_steps=belief_steps,
        gt_steps=gt_steps,
        topology=topo,
        candidate_edges_by_step=cands_by_step,
        horizon_s=4.0,
        delay_threshold_s=1.2,
    )
    lbl0 = labels[0]
    assert lbl0.veh_ids == ("v_follow", "v_lead", "v_tail")
    assert np.all(lbl0.y_delay == 1.0)
    assert np.all(lbl0.tau_delay_s > 1.2)
    # C0 has no upstream tail crawl; C1 has v_tail at pos=8m <= 22m & speed=0.8 <= 1.5 m/s
    assert lbl0.corridor_ids == ("C0", "C1")
    assert float(lbl0.y_spillback[0]) == 0.0
    assert float(lbl0.y_spillback[1]) == 1.0
    # Directed edge v_lead -> v_follow is active (1.0), reverse v_follow -> v_lead is 0.0, spillback v_tail -> v_lead is 1.0
    assert float(lbl0.y_edge[0]) == 1.0
    assert float(lbl0.y_edge[1]) == 0.0
    assert float(lbl0.y_edge[2]) == 1.0
    assert lbl0.rank_pairs.shape[0] == 2
    assert lbl0.rank_pairs.shape[1] >= 1


def _write_synthetic_rollout_split(
    split_dir: Path,
    *,
    split: str,
    scenario_id: str,
    network_id: str,
    seeds: list[int],
    n_steps: int = 6,
) -> Path:
    """Create deterministic weights_only-safe rollout shards + SHA-256 manifest in ``split_dir``."""
    split_dir.mkdir(parents=True, exist_ok=True)
    corridor_ids = ("C0", "C1", "C2")
    corridor_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    shard_records: list[dict[str, Any]] = []

    for seed in seeds:
        rng = np.random.default_rng(seed * 31 + 7)
        steps: list[dict[str, Any]] = []
        for s_idx in range(n_steps):
            n_v = 6
            veh_ids = tuple(f"veh_{i}" for i in range(n_v))
            # Half delayed, half free-flowing with correlated kinematic features
            y_delay_np = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            if (s_idx + seed) % 2 == 1:
                y_delay_np = y_delay_np[::-1].copy()
            tau_np = y_delay_np * 3.2 + 0.1 * rng.uniform(0.0, 1.0, size=n_v).astype(np.float32)

            x_v = np.zeros((n_v, NODE_FEATURE_DIM), dtype=np.float32)
            x_v[:, 0] = np.linspace(0.15, 0.85, n_v)
            x_v[:, 3] = np.where(y_delay_np > 0.5, 0.18, 0.88)  # v_norm
            x_v[:, 13] = 0.08
            x_v += 0.03 * rng.standard_normal(x_v.shape).astype(np.float32)
            x_v[:, 13] = np.clip(x_v[:, 13], 0.01, 0.99)

            y_spill_np = np.array([1.0 if s_idx % 2 == 0 else 0.0, 0.0, 1.0], dtype=np.float32)
            x_c = np.zeros((3, CORRIDOR_FEATURE_DIM), dtype=np.float32)
            x_c[:, 0] = 0.6
            x_c[:, 5] = np.where(y_spill_np > 0.5, 0.75, 0.12)
            x_c[:, 6] = np.where(y_spill_np > 0.5, 0.82, 0.05)
            x_c[:, 8] = np.where(y_spill_np > 0.5, 0.22, 0.90)
            x_c[:, 9] = 1.0
            x_c[:, 10] = 0.08

            x_cc = np.ones((2, CORRIDOR_EDGE_FEATURE_DIM), dtype=np.float32) * 0.5

            # Directed candidate edges: (0->1), (1->2), (2->3), (3->4), (4->5), (1->0)
            edge_src = [0, 1, 2, 3, 4, 1]
            edge_dst = [1, 2, 3, 4, 5, 0]
            e_cand = len(edge_src)
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_relations = torch.tensor([0, 0, 1, 0, 2, 0], dtype=torch.long)

            y_edge_np = np.array(
                [
                    1.0 if (y_delay_np[s] > 0.5 and y_delay_np[d] > 0.5 and s < d) else 0.0
                    for s, d in zip(edge_src, edge_dst, strict=True)
                ],
                dtype=np.float32,
            )
            if float(y_edge_np.sum()) == 0.0:
                y_edge_np[0] = 1.0

            x_e = np.zeros((e_cand, EDGE_FEATURE_DIM), dtype=np.float32)
            for k, r_id in enumerate(edge_relations.tolist()):
                x_e[k, r_id] = 1.0
            x_e[:, 6] = np.where(y_edge_np > 0.5, 0.25, -0.40)
            x_e[:, 8] = np.where(y_edge_np > 0.5, 0.90, 0.15)
            x_e[:, 9] = 0.50
            x_e[:, 15] = 0.08

            topo_prec = np.linspace(5.0, -5.0, n_v, dtype=np.float32)
            speed_ev = np.where(y_delay_np > 0.5, 0.82, 0.18).astype(np.float32)

            steps.append(
                {
                    "sim_time": float(60.0 + s_idx * 0.5),
                    "veh_ids": veh_ids,
                    "veh_corridor_idx": torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
                    "veh_turn_src_idx": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                    "veh_turn_out_corridor_idx": torch.tensor([1, 1, 2, 2], dtype=torch.long),
                    "veh_turn_prob": torch.ones((4,), dtype=torch.float32),
                    "topo_precedence": torch.from_numpy(topo_prec),
                    "speed_evidence_pos": torch.from_numpy(speed_ev),
                    "x_veh": torch.from_numpy(x_v),
                    "x_corridor": torch.from_numpy(x_c),
                    "x_edge": torch.from_numpy(x_e),
                    "x_corridor_edge": torch.from_numpy(x_cc),
                    "edge_index": edge_index,
                    "edge_relations": edge_relations,
                    "p_conflict": torch.from_numpy(x_e[:, 8].copy()),
                    "p_merge": torch.from_numpy(x_e[:, 9].copy()),
                    "signed_path_gap_m": torch.from_numpy((x_e[:, 6] * 50.0).astype(np.float32)),
                    "y_delay": torch.from_numpy(y_delay_np),
                    "tau_delay_s": torch.from_numpy(tau_np),
                    "y_spillback": torch.from_numpy(y_spill_np),
                    "y_edge": torch.from_numpy(y_edge_np),
                    "rank_pairs": torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.long),
                }
            )

        shard_name = f"episode_{scenario_id}_seed{seed}.pt"
        shard_path = split_dir / shard_name
        torch.save(
            {
                "schema_version": SCHEMA_VERSION,
                "split": split,
                "scenario_id": scenario_id,
                "network_id": network_id,
                "seed": seed,
                "behavior_policy": "online_bic_dbn",
                "dt": 0.5,
                "corridor_ids": corridor_ids,
                "corridor_edge_index": corridor_edge_index,
                "steps": steps,
            },
            shard_path,
        )
        shard_records.append(
            {
                "shard_file": shard_name,
                "shard_sha256": file_sha256(shard_path),
                "scenario_id": scenario_id,
                "network_id": network_id,
                "seed": seed,
                "split": split,
                "num_steps": n_steps,
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "scenario_id": scenario_id,
        "network_id": network_id,
        "is_ood_regime": split == "eval_ood",
        "seeds": seeds,
        "shards": shard_records,
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return split_dir


def test_unrolled_neural_dbn_training_lock_b_calibration_and_heldout_predictive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify end-to-end BPTT training, Boundary Lock B L-BFGS calibration, ci_micro guard, & predictive plots."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / ".mpl_cache"))

    train_dir = _write_synthetic_rollout_split(
        tmp_path / "train",
        split="train",
        scenario_id="grid3x3_nominal",
        network_id="grid3x3",
        seeds=[TRAIN_SEEDS[0], TRAIN_SEEDS[1]],
        n_steps=6,
    )
    val_dir = _write_synthetic_rollout_split(
        tmp_path / "val",
        split="val",
        scenario_id="grid3x3_nominal",
        network_id="grid3x3",
        seeds=[VAL_SEEDS[0], VAL_SEEDS[1]],
        n_steps=6,
    )
    eval_id_dir = _write_synthetic_rollout_split(
        tmp_path / "eval_id",
        split="eval_id",
        scenario_id="grid3x3_nominal",
        network_id="grid3x3",
        seeds=[TEST_SEEDS[0], TEST_SEEDS[1]],
        n_steps=6,
    )
    eval_ood_dir = _write_synthetic_rollout_split(
        tmp_path / "eval_ood",
        split="eval_ood",
        scenario_id="grid4x4_incident",
        network_id="grid4x4",
        seeds=[TEST_SEEDS[0], TEST_SEEDS[1]],
        n_steps=6,
    )

    ckpt_path = tmp_path / "models" / "neural_dbn.pt"
    train_cfg = TrainConfig(
        model_preset="ci_micro",
        device="cpu",
        precision="fp32",
        epochs=6,
        batch_size=2,
        learning_rate=4e-3,
        early_stopping_patience=6,
        seed=123,
    )
    dbn_cfg = DBNConfig(model_preset="ci_micro")

    train_rep = train_neural_dbn(
        train_dir=train_dir,
        val_dir=val_dir,
        output_checkpoint_path=ckpt_path,
        train_cfg=train_cfg,
        dbn_cfg=dbn_cfg,
        report_path=tmp_path / "models" / "training_report.json",
    )

    assert ckpt_path.is_file()
    assert ckpt_path.with_name("neural_dbn.pt.sha256.json").is_file()
    assert (ckpt_path.parent / "checkpoint_meta.json").is_file()
    assert (ckpt_path.parent / "training_report.json").is_file()
    assert train_rep["final_train_loss"] < train_rep["initial_train_loss"]
    assert train_rep["temperatures"]["calibrated"] is True

    # Verify bf16 training mode + portable FP32 CPU checkpoint emission (#34)
    bf16_ckpt = tmp_path / "models_bf16" / "neural_dbn.pt"
    bf16_rep = train_neural_dbn(
        train_dir=train_dir,
        val_dir=val_dir,
        output_checkpoint_path=bf16_ckpt,
        train_cfg=TrainConfig(
            model_preset="ci_micro",
            device="cpu",
            precision="bf16",
            epochs=2,
            batch_size=1,
            seed=77,
        ),
        dbn_cfg=dbn_cfg,
    )
    assert bf16_rep["precision"] == "bf16"
    bf16_bundle = load_dbn_checkpoint(bf16_ckpt, expected_preset="ci_micro", require_calibrated=True, device="cpu")
    assert all(p.dtype == torch.float32 for p in bf16_bundle.model.parameters())

    bundle = load_dbn_checkpoint(ckpt_path, expected_preset="ci_micro", require_calibrated=True, device="cpu")
    assert bundle.temperatures.calibrated is True
    assert bundle.normalizer.fitted is True
    assert 0.1 <= bundle.temperatures.t_edge <= 10.0
    assert 0.1 <= bundle.temperatures.t_delay <= 10.0
    assert 0.1 <= bundle.temperatures.t_link <= 10.0

    # Enforce ci_micro anti-claim guard when allow_ci_micro=False
    with pytest.raises(CheckpointIntegrityError):
        validate_and_hash_checkpoint(ckpt_path, allow_ci_micro=False)
    with pytest.raises(CheckpointIntegrityError):
        evaluate_heldout_predictive(
            ckpt_path,
            {"eval_id": eval_id_dir},
            allow_ci_micro=False,
        )

    pred_out = tmp_path / "predictive_report.json"
    pred_rep = evaluate_heldout_predictive(
        ckpt_path,
        {"eval_id": eval_id_dir, "eval_ood": eval_ood_dir},
        output_path=pred_out,
        dbn_cfg=dbn_cfg,
        allow_ci_micro=True,
    )
    assert pred_out.is_file()
    for split_key in ("eval_id", "eval_ood"):
        edge_m = pred_rep["splits"][split_key]["edge"]
        assert 0.0 <= edge_m["roc_auc"] <= 1.0
        assert 0.0 <= edge_m["pr_auc"] <= 1.0
        assert 0.0 <= edge_m["ece_post"] <= 1.0

    cal_png = plot_calibration_and_roc_pr(pred_rep, tmp_path / "plots" / "calibration_roc_pr.png")
    assert cal_png.is_file() and cal_png.stat().st_size > 1000


def test_paired_bootstrap_wilcoxon_ties_and_holm_bonferroni() -> None:
    """Verify 10,000-resample paired bootstrap CIs, exact Wilcoxon zero/tie handling, & Holm-Bonferroni."""
    arr = [12.0, 14.0, 15.0, 13.0, 16.0, 11.0, 14.5, 13.5, 15.5, 12.5]
    mean_v, ci_lo, ci_hi = bootstrap_mean_ci(arr, n_resamples=10_000, rng=np.random.default_rng(42))
    assert ci_lo < mean_v < ci_hi

    # Identical vectors -> all zero differences -> p=1.0 without NaN
    stat_zero, p_zero = exact_wilcoxon_two_sided([0.0, 0.0, 0.0, 0.0, 0.0])
    assert stat_zero == 0.0 and p_zero == 1.0

    # Mixed ties and clear negative improvements across 15 paired seeds
    diffs_15 = [-2.5, -3.1, -1.8, 0.0, -2.2, -2.2, -1.9, -4.0, -0.8, -2.7, -3.3, -1.5, 0.2, -2.9, -2.1]
    _, p_sig = exact_wilcoxon_two_sided(diffs_15)
    assert 0.0 < p_sig < 0.01

    adj = holm_bonferroni_adjust([0.002, 0.025, 0.030])
    assert adj == pytest.approx([0.006, 0.050, 0.050])
    assert adj[0] <= adj[1] <= adj[2]

    records: list[dict[str, Any]] = []
    for idx, seed in enumerate(TEST_SEEDS[:10]):
        records.append(
            {
                "scenario_id": "grid3x3_nominal",
                "policy_name": "neural_dbn",
                "seed": seed,
                "cohort_mean_total_delay_s": 18.0 + 0.4 * idx,
                "throughput_veh_per_hr": 920.0 + 5.0 * idx,
            }
        )
        records.append(
            {
                "scenario_id": "grid3x3_nominal",
                "policy_name": "actuated_signal",
                "seed": seed,
                "cohort_mean_total_delay_s": 25.0 + 0.4 * idx,
                "throughput_veh_per_hr": 810.0 + 5.0 * idx,
            }
        )

    paired_stats = compute_paired_statistics(
        records,
        reference_policy="neural_dbn",
        metrics=("cohort_mean_total_delay_s", "throughput_veh_per_hr"),
        n_resamples=2_000,
    )
    comp_delay = paired_stats["scenarios"]["grid3x3_nominal"]["paired_comparisons"]["actuated_signal"][
        "cohort_mean_total_delay_s"
    ]
    assert comp_delay["mean_diff"] == pytest.approx(-7.0)
    assert comp_delay["ci95_high"] < 0.0
    assert comp_delay["wins"] == 10 and comp_delay["ties"] == 0 and comp_delay["losses"] == 0
    assert comp_delay["significant_holm_005"] is True


class _DeterministicMockAdapter:
    """Lightweight deterministic ``SimulationAdapterProtocol`` for fast unit-testing of evaluator matrix."""

    def __init__(self, scenario: ScenarioConfig, topology: RoadTopology, seed: int) -> None:
        self._scenario = scenario
        self._topology = topology
        self._seed = seed
        self._step = 0
        self._delay_accum = 0.0

    @property
    def topology(self) -> RoadTopology:
        return self._topology

    def reset(self, seed: int | None = None) -> tuple[ObservationBatch, GroundTruthStepRecord]:
        if seed is not None:
            self._seed = seed
        self._step = 0
        self._delay_accum = 0.0
        return self._emit(target_speed=12.0)

    def step(self, action: SwarmControlAction) -> tuple[ObservationBatch, GroundTruthStepRecord]:
        self._step += 1
        cmd_v = float(action.target_speeds.get("v0", 10.0))
        self._delay_accum += max(0.0, (15.0 - cmd_v) / 15.0) * self._scenario.step_length_s
        return self._emit(target_speed=cmd_v)

    def _emit(self, target_speed: float) -> tuple[ObservationBatch, GroundTruthStepRecord]:
        sim_t = float(self._step) * self._scenario.step_length_s
        dep = (
            (
                DepartedVehicleEvent(
                    veh_id="v0",
                    entry_sim_time=0.0,
                    entry_edge_id="E0",
                    entry_lane_index=0,
                    length_m=4.5,
                    width_m=1.8,
                ),
            )
            if self._step == 0
            else ()
        )
        arr = ("v0",) if self._step == self._scenario.total_steps else ()
        obs = ObservationBatch(
            sim_time=sim_t,
            dt=self._scenario.step_length_s,
            active_veh_ids=() if arr else ("v0",),
            departed=dep,
            arrived_veh_ids=arr,
            radar_obs=(),
            v2x_packets=(),
            open_lanes_by_edge={"E0": 2, "E1": 2},
            tls_states={"J0": "G"},
        )
        raw_vehs = (
            ()
            if arr
            else (
                VehicleRawTelemetry(
                    veh_id="v0",
                    vtype="passenger_urban",
                    center_xy=(min(115.0, sim_t * target_speed), 0.0),
                    yaw_ccw_rad=0.0,
                    speed=target_speed,
                    accel=0.0,
                    road_id="E0",
                    lane_id="E0_0",
                    lane_index=0,
                    lane_pos=min(115.0, sim_t * target_speed),
                    lane_length=120.0,
                    route_edges=("E0", "E1"),
                    route_index=0,
                    route_goal="E1",
                    length=4.5,
                    width=1.8,
                    brake_light=False,
                    turn_signal=0,
                    leader_id=None,
                    leader_gap=None,
                    time_loss=self._delay_accum + 0.01 * (self._seed % 7),
                    waiting_time=0.0,
                    co2_mg_s=1100.0,
                    scheduled_depart=0.0,
                ),
            )
        )
        gt = GroundTruthStepRecord(
            sim_time=sim_t,
            raw_vehicles=raw_vehs,
            collisions=0,
            teleports=0,
            emergency_stops=0,
            emergency_brakes=0,
            near_misses_ttc=0,
            min_ttc_s=999.0,
            reroutes_applied=0,
            v2x_sent=0,
            v2x_delivered=0,
            step_co2_mg=1100.0,
            spillback_corridor_count=0,
        )
        return obs, gt

    def close(self) -> None:
        return None


class _DeterministicMockController:
    def __init__(self, policy_name: str) -> None:
        self.policy_name = policy_name
        self.last_dbn_snapshot = None

    def reset(self) -> None:
        return None

    def act(self, obs: ObservationBatch) -> SwarmControlAction:
        speed = 14.2 if self.policy_name == "neural_dbn" else 9.5
        targets = {vid: speed for vid in obs.active_veh_ids}
        return SwarmControlAction(
            sim_time=obs.sim_time,
            target_speeds=targets,
            post_safety_accels={vid: 0.0 for vid in obs.active_veh_ids},
            modes={},
            tls_desired_greens=None,
            broadcast_edge_costs=None,
            reroute_hysteresis_s=3.0,
            v2x_polls=(),
            safety_clamp_count=0,
            zero_speed_deadzone_count=0,
            control_latency_ms=0.25,
        )


def test_evaluator_matrix_and_publication_plots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify multi-episode evaluation matrix, provenance artifacts, and 2D scenario/forest plots."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / ".mpl_cache"))

    # Train a tiny calibrated checkpoint for provenance validation
    tr_dir = _write_synthetic_rollout_split(
        tmp_path / "tr",
        split="train",
        scenario_id="grid3x3_nominal",
        network_id="grid3x3",
        seeds=[TRAIN_SEEDS[0]],
        n_steps=4,
    )
    va_dir = _write_synthetic_rollout_split(
        tmp_path / "va",
        split="val",
        scenario_id="grid3x3_nominal",
        network_id="grid3x3",
        seeds=[VAL_SEEDS[0]],
        n_steps=4,
    )
    ckpt_file = tmp_path / "ckpt.pt"
    train_neural_dbn(
        tr_dir,
        va_dir,
        ckpt_file,
        train_cfg=TrainConfig(model_preset="ci_micro", epochs=2, batch_size=1),
        dbn_cfg=DBNConfig(model_preset="ci_micro"),
    )

    topo = _make_two_corridor_topology()
    mini_scen = ScenarioConfig(
        scenario_id="grid3x3_nominal",
        network_id="grid3x3",
        net_xml_path="configs/networks/grid3x3.net.xml",
        step_length_s=1.0,
        warmup_duration_s=0.0,
        cohort_duration_s=4.0,
        drain_duration_s=2.0,
    )
    eval_dir = tmp_path / "eval_matrix"
    summary = evaluate_experiment_matrix(
        scenarios=[mini_scen],
        policies=["neural_dbn", "actuated_signal"],
        seeds=TEST_SEEDS[:4],
        output_dir=eval_dir,
        checkpoint_path=ckpt_file,
        dbn_cfg=DBNConfig(model_preset="ci_micro"),
        control_cfg=ControlConfig(),
        max_workers=1,
        bootstrap_resamples=1_000,
        allow_ci_micro=True,
        topology_loader=lambda _s: topo,
        adapter_factory=lambda s, t, seed: _DeterministicMockAdapter(s, t, seed),
        controller_factory=lambda pol, _t, _c, _ck: _DeterministicMockController(pol),
    )

    assert (eval_dir / "episodes.csv").is_file()
    assert (eval_dir / "episodes.json").is_file()
    assert (eval_dir / "provenance.json").is_file()
    assert (eval_dir / "summary.json").is_file()
    assert summary["num_episodes"] == 8

    p_scen = plot_scenario_comparison(summary, tmp_path / "scen_comp.png")
    p_forest = plot_paired_forest(summary, tmp_path / "forest.png")
    assert p_scen.is_file() and p_forest.is_file()


def test_cpu_latency_scaling_benchmark_and_plot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify warmed CPU scaling benchmark across all 4 DBN updaters and latency plot generation."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / ".mpl_cache"))

    out_json = tmp_path / "scaling.json"
    rep = run_scaling_benchmark(
        output_path=out_json,
        swarm_sizes=(6, 12),
        methods=DEFAULT_BENCHMARK_METHODS,
        model_preset="ci_micro",
        warmup_repeats=1,
        repeats=2,
        allow_ci_micro=True,
        seed=99,
    )
    assert out_json.is_file()
    for method in DEFAULT_BENCHMARK_METHODS:
        rows = rep["methods"][method]
        assert len(rows) == 2
        for r in rows:
            assert r["median_ms"] > 0.0
            assert r["iqr_ms"] >= 0.0
            assert r["p95_ms"] >= r["q25_ms"]

    png = plot_latency_scaling(rep, tmp_path / "scaling.png")
    assert png.is_file() and png.stat().st_size > 1000


def test_cli_subcommands_and_fail_loud_crash_report(tmp_path: Path) -> None:
    """Verify CLI subcommands, benchmark execution, and atomic fail-loud crash report emission."""
    parser = build_parser()
    subparsers_actions = [a for a in parser._actions if isinstance(a, type(parser._subparsers._group_actions[0]))]
    assert subparsers_actions
    registered = set(subparsers_actions[0].choices.keys())
    expected_cmds = {
        "build-nets",
        "collect",
        "train",
        "eval-predictive",
        "simulate",
        "evaluate",
        "benchmark",
        "render-3d",
        "demo",
    }
    assert expected_cmds.issubset(registered)

    err_dir = tmp_path / "crash_reports"
    bench_out = tmp_path / "cli_bench.json"
    rc = cli_main(
        [
            "--error-report-dir",
            str(err_dir),
            "benchmark",
            "--output",
            str(bench_out),
            "--sizes",
            "6",
            "--methods",
            "heuristic_radius_dbn",
            "--preset",
            "ci_micro",
            "--allow-ci-micro",
            "--warmup",
            "1",
            "--repeats",
            "2",
        ]
    )
    assert rc == 0
    assert bench_out.is_file()
    assert not err_dir.exists() or len(list(err_dir.glob("error_report_*.json"))) == 0

    # Trigger intentional fail-loud ConfigValidationError and verify atomic crash report in err_dir
    with pytest.raises(ConfigValidationError):
        cli_main(
            [
                "--error-report-dir",
                str(err_dir),
                "collect",
                "--scenario",
                "nonexistent_scenario_xyz",
                "--split",
                "train",
                "--output-dir",
                str(tmp_path / "bad_collect"),
            ]
        )

    reports = list(err_dir.glob("error_report_*_pid*.json"))
    assert len(reports) == 1
    crash_payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert "ConfigValidationError" in crash_payload["exception_type"]
    assert isinstance(crash_payload["traceback"], list) and len(crash_payload["traceback"]) > 0
    assert crash_payload["crash_context"]["command"] == "collect"

    # Verify CLI `demo --compare-policy` forwards checkpoint_path for ALL NEURAL_POLICY_NAMES (#39 item 3)
    import swarm.cli as cli_mod
    import swarm.env as env_mod
    import swarm.viz as viz_mod

    demo_calls: list[tuple[str, Path | None]] = []
    dummy_metrics = _DeterministicMockAdapter(_make_two_corridor_topology().corridors["C0"] if False else ScenarioConfig(
        scenario_id="grid3x3_nominal",
        network_id="grid3x3",
        net_xml_path="configs/networks/grid3x3.net.xml",
    ), _make_two_corridor_topology(), 3000)
    _ = dummy_metrics

    from swarm.contracts import EpisodeMetrics, NEURAL_POLICY_NAMES
    from swarm.dbn.updater import DBNCheckpointBundle, STGraphTransformerDBN
    from swarm.dbn.features import FeatureNormalizer
    from swarm.dbn.calibration import TemperatureScales
    from swarm.config import get_model_preset
    import swarm.dbn.updater as upd_mod

    fake_ckpt = tmp_path / "fake_neural.pt"
    fake_ckpt.write_bytes(b"placeholder")
    fake_ep_metrics = EpisodeMetrics(
        scenario_id="grid3x3_nominal",
        policy_name="neural_dbn",
        seed=3000,
        cohort_size=1,
        cohort_completed_count=1,
        cohort_unfinished_count=0,
        cohort_undeparted_count=0,
        cohort_completion_rate=1.0,
        cohort_mean_total_delay_s=2.0,
        cohort_p95_total_delay_s=2.0,
        cohort_mean_completed_delay_s=2.0,
        throughput_veh_per_hr=600.0,
        mean_stops_per_veh=0.0,
        spillback_corridor_seconds=0.0,
        mean_co2_g_per_veh=1.0,
        reroutes_per_veh=0.0,
        collision_count=0,
        teleport_count=0,
        emergency_stop_count=0,
        emergency_brake_count=0,
        emergency_brake_rate_per_veh_km=0.0,
        ttc_near_miss_count=0,
        min_ttc_s=999.0,
        safety_filter_clamp_count=0,
        safety_filter_clamp_rate=0.0,
        zero_speed_deadzone_count=0,
        v2x_polls_requested=0,
        v2x_packets_delivered=0,
        v2x_delivery_efficiency=1.0,
        mean_dbn_update_latency_ms=1.0,
        mean_dbn_inference_latency_ms=0.5,
        mean_control_latency_ms=1.5,
        edge_churn_hamming_rate=0.0,
        wall_time_s=0.1,
        sim_steps=10,
        steps_per_sec=100.0,
    )

    def _fake_load_ckpt(*_a: Any, **_kw: Any) -> DBNCheckpointBundle:
        m = STGraphTransformerDBN(preset=get_model_preset("ci_micro"))
        return DBNCheckpointBundle(
            model=m,
            normalizer=FeatureNormalizer.identity(),
            temperatures=TemperatureScales(1.0, 1.0, 1.0, calibrated=True),
            checkpoint_sha256="0" * 64,
            model_metadata=m.metadata_dict(),
            training_metadata={},
        )

    def _fake_run_ep(*, policy_name: str, checkpoint_path: Path | None = None, **_kw: Any) -> tuple[EpisodeMetrics, dict[str, Any]]:
        demo_calls.append((policy_name, checkpoint_path))
        return fake_ep_metrics, {"scenario_id": "grid3x3_nominal", "frames": []}

    with pytest.MonkeyPatch.context() as mp_ctx:
        mp_ctx.setattr(upd_mod, "load_dbn_checkpoint", _fake_load_ckpt)
        mp_ctx.setattr(cli_mod, "run_single_episode", _fake_run_ep)
        mp_ctx.setattr(env_mod, "load_or_build_topology", lambda _s: _make_two_corridor_topology())
        mp_ctx.setattr(viz_mod, "export_scene_3d", lambda _t: {"schema_version": 1})
        mp_ctx.setattr(viz_mod, "write_city_3d_viewer", lambda **_kw: tmp_path / "demo.html")

        for abl_pol in NEURAL_POLICY_NAMES:
            demo_calls.clear()
            rc_demo = cli_main(
                [
                    "--error-report-dir",
                    str(err_dir),
                    "demo",
                    "--scenario",
                    "grid3x3_nominal",
                    "--policy",
                    "neural_dbn",
                    "--compare-policy",
                    abl_pol,
                    "--checkpoint",
                    str(fake_ckpt),
                    "--allow-ci-micro",
                    "--output-html",
                    str(tmp_path / "demo.html"),
                ]
            )
            assert rc_demo == 0
            assert demo_calls == [("neural_dbn", fake_ckpt), (abl_pol, fake_ckpt)]

        demo_calls.clear()
        rc_base = cli_main(
            [
                "--error-report-dir",
                str(err_dir),
                "demo",
                "--scenario",
                "grid3x3_nominal",
                "--policy",
                "neural_dbn",
                "--compare-policy",
                "actuated_signal",
                "--checkpoint",
                str(fake_ckpt),
                "--allow-ci-micro",
                "--output-html",
                str(tmp_path / "demo.html"),
            ]
        )
        assert rc_base == 0
        assert demo_calls == [("neural_dbn", fake_ckpt), ("actuated_signal", None)]


def test_rank_pairs_loss_exactness_and_rollout_temporal_safety_invariants(tmp_path: Path) -> None:
    """Verify exact (M, 2) rank_pairs loss math (#39 item 1) & rollout t-1 snapshot + safe perturbation (#39 item 2)."""
    from types import SimpleNamespace

    from swarm.contracts import DBNStateSnapshot
    from swarm.dbn.features import extract_graph_features
    from swarm.pipelines.rollout_collector import (
        _apply_exploratory_perturbation,
        collect_single_rollout_episode,
    )
    from swarm.pipelines.trainer import compute_step_multitask_loss

    # 1. Exact M=3 non-symmetric rank_pairs margin ranking loss check
    topo = _make_two_corridor_topology()
    cov = np.eye(2, dtype=np.float64) * 0.5
    vehs = {
        f"v{i}": VehicleBelief(
            veh_id=f"v{i}",
            entered_at=0.0,
            corridor_id="C0",
            edge_id="E0",
            in_junction_id=None,
            lane_index=0,
            length_m=4.5,
            width_m=1.8,
            mean_s_m=10.0 + 12.0 * i,
            mean_v_mps=8.0,
            cov_2x2=cov.copy(),
            accel_est_mps2=0.0,
            obs_age_s=0.0,
            wait_time_s=0.0,
            turn_probs=(1.0,),
            free_eta_s=5.0,
            eta_std_s=0.2,
            tail_blockage_score=0.0,
            leader_id=None,
            noisy_leader_gap_m=None,
            noisy_leader_speed_mps=None,
            last_v2x_poll_time=0.0,
        )
        for i in range(6)
    }
    belief_6 = SwarmBeliefState(
        sim_time=10.0,
        dt=0.5,
        vehicles=vehs,
        corridors={
            "C0": CorridorBeliefStats("C0", 6, 2, 34, 0.2, 0.0, 0.5, 8.0),
            "C1": CorridorBeliefStats("C1", 0, 2, 34, 0.0, 0.0, 0.0, 15.0),
        },
        open_lanes_by_edge={"E0": 2, "E1": 2},
        tls_states={"J0": "G"},
    )
    fb_6 = extract_graph_features(belief_6, topo, DBNConfig(model_preset="ci_micro"))
    n_e = fb_6.candidate_graph.num_edges

    # Precedence scores: [2.0, 1.5, 0.5, 1.8, 0.2, 0.4]
    # Pairs: (0, 3) -> diff=0.2 -> relu(0.8)=0.8; (1, 4) -> diff=1.3 -> relu(-0.3)=0.0; (2, 5) -> diff=0.1 -> relu(0.9)=0.9
    # Exact mean loss_rank = (0.8 + 0.0 + 0.9) / 3 = 1.7 / 3
    prec_t = torch.tensor([2.0, 1.5, 0.5, 1.8, 0.2, 0.4], dtype=torch.float32)
    fake_step_out = SimpleNamespace(
        delay_pred_logits_scaled=torch.zeros((6,), dtype=torch.float32),
        delay_post_logits_scaled=torch.zeros((6,), dtype=torch.float32),
        expected_delay_s=torch.zeros((6,), dtype=torch.float32),
        edge_logits_scaled=torch.zeros((n_e,), dtype=torch.float32),
        omega_raw=torch.zeros((n_e,), dtype=torch.float32),
        corridor_spillback_logits_scaled=torch.zeros((2,), dtype=torch.float32),
        transformer_out=SimpleNamespace(precedence=prec_t),
    )
    lbl_m3 = {
        "y_delay": torch.zeros((6,), dtype=torch.float32),
        "tau_delay_s": torch.zeros((6,), dtype=torch.float32),
        "y_spillback": torch.zeros((2,), dtype=torch.float32),
        "y_edge": torch.zeros((n_e,), dtype=torch.float32),
        "rank_pairs": torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.long),
    }
    _, parts = compute_step_multitask_loss(
        step_out=fake_step_out,  # type: ignore[arg-type]
        feat_batch=fb_6,
        labels=lbl_m3,
        train_cfg=TrainConfig(model_preset="ci_micro"),
        prev_edge_omega_detached=None,
        device=torch.device("cpu"),
    )
    assert parts["rank"] == pytest.approx(1.7 / 3.0, abs=1e-6)

    # Reject transposed (2, 3) rank_pairs with ConfigValidationError
    bad_lbl = dict(lbl_m3, rank_pairs=torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long))
    with pytest.raises(ConfigValidationError):
        compute_step_multitask_loss(
            step_out=fake_step_out,  # type: ignore[arg-type]
            feat_batch=fb_6,
            labels=bad_lbl,
            train_cfg=TrainConfig(model_preset="ci_micro"),
            prev_edge_omega_detached=None,
            device=torch.device("cpu"),
        )

    # 2a. Verify _apply_exploratory_perturbation never exceeds target_speeds, preserves standstill, and syncs accels
    obs_6 = ObservationBatch(
        sim_time=10.0,
        dt=0.5,
        active_veh_ids=tuple(vehs),
        departed=(),
        arrived_veh_ids=(),
        radar_obs=(),
        v2x_packets=(),
        open_lanes_by_edge={"E0": 2, "E1": 2},
        tls_states={"J0": "G"},
    )
    base_speeds = {"v0": 0.0, "v1": 0.10, "v2": 9.0, "v3": 12.0, "v4": 6.5, "v5": 0.20}
    base_accels = {k: 0.0 for k in base_speeds}
    base_act = SwarmControlAction(
        sim_time=10.0,
        target_speeds=dict(base_speeds),
        post_safety_accels=dict(base_accels),
        modes={},
        tls_desired_greens=None,
        broadcast_edge_costs={"E0": 10.0, "E1": 10.0},
    )
    rng = np.random.default_rng(12345)
    for _ in range(50):
        pert_act = _apply_exploratory_perturbation(base_act, obs_6, belief_6, topo, rng)
        assert pert_act.target_speeds["v0"] == 0.0
        assert pert_act.target_speeds["v1"] == 0.10
        for vid, orig_v in base_speeds.items():
            new_v = pert_act.target_speeds[vid]
            assert 0.0 <= new_v <= orig_v
            if new_v != orig_v:
                expected_acc = float(np.clip((new_v - belief_6.vehicles[vid].mean_v_mps) / 0.5, -4.5, 2.6))
                assert pert_act.post_safety_accels[vid] == pytest.approx(expected_acc)

    # 2b. Verify rollout_collector passes step t-1 snapshot (NOT step t!) into feature_extractor_fn
    seen_prev_snap_times: list[float | None] = []

    class _SnapTrackingController:
        def __init__(self) -> None:
            self.policy_name = "heuristic_radius_dbn"
            self.last_dbn_snapshot: DBNStateSnapshot | None = None

        def reset(self) -> None:
            self.last_dbn_snapshot = None

        def act(self, obs: ObservationBatch) -> SwarmControlAction:
            self.last_dbn_snapshot = DBNStateSnapshot(
                sim_time=obs.sim_time,
                veh_ids=obs.active_veh_ids,
                precedence_scores={v: 0.0 for v in obs.active_veh_ids},
                edges=(),
                edges_by_dst={v: () for v in obs.active_veh_ids},
                edges_by_src={v: () for v in obs.active_veh_ids},
                delay_pred={v: 0.2 for v in obs.active_veh_ids},
                delay_post={v: 0.2 for v in obs.active_veh_ids},
                expected_delay_s={v: 0.5 for v in obs.active_veh_ids},
                cascade_score={v: 0.0 for v in obs.active_veh_ids},
                corridor_spillback_pred={"C0": 0.1, "C1": 0.1},
                raw_edge_probs={},
                update_latency_ms=0.1,
                inference_latency_ms=0.1,
            )
            return SwarmControlAction(
                sim_time=obs.sim_time,
                target_speeds={v: 10.0 for v in obs.active_veh_ids},
                post_safety_accels={v: 0.0 for v in obs.active_veh_ids},
                modes={},
                tls_desired_greens=None,
                broadcast_edge_costs=None,
            )

    class _SimpleTracker:
        def reset(self) -> None:
            return None

        def observe(self, obs: ObservationBatch) -> SwarmBeliefState:
            v_map = {
                vid: VehicleBelief(
                    veh_id=vid,
                    entered_at=0.0,
                    corridor_id="C0",
                    edge_id="E0",
                    in_junction_id=None,
                    lane_index=0,
                    length_m=4.5,
                    width_m=1.8,
                    mean_s_m=20.0,
                    mean_v_mps=10.0,
                    cov_2x2=np.eye(2, dtype=np.float64),
                    accel_est_mps2=0.0,
                    obs_age_s=0.0,
                    wait_time_s=0.0,
                    turn_probs=(1.0,),
                    free_eta_s=5.0,
                    eta_std_s=0.2,
                    tail_blockage_score=0.0,
                    leader_id=None,
                    noisy_leader_gap_m=None,
                    noisy_leader_speed_mps=None,
                    last_v2x_poll_time=0.0,
                )
                for vid in obs.active_veh_ids
            }
            return SwarmBeliefState(
                sim_time=obs.sim_time,
                dt=obs.dt,
                vehicles=v_map,
                corridors={
                    "C0": CorridorBeliefStats("C0", len(v_map), 2, 34, 0.1, 0.0, 0.1, 10.0),
                    "C1": CorridorBeliefStats("C1", 0, 2, 34, 0.0, 0.0, 0.0, 15.0),
                },
                open_lanes_by_edge={"E0": 2, "E1": 2},
                tls_states={"J0": "G"},
            )

        def predict(self, **_kw: Any) -> None:
            return None

    def _recording_extractor(
        belief: SwarmBeliefState,
        topology: RoadTopology,
        dbn_cfg: DBNConfig,
        prev_snapshot: DBNStateSnapshot | None = None,
    ) -> Any:
        seen_prev_snap_times.append(prev_snapshot.sim_time if prev_snapshot is not None else None)
        return extract_graph_features(belief, topology, dbn_cfg, prev_snapshot=prev_snapshot)

    mini_scen = ScenarioConfig(
        scenario_id="grid3x3_nominal",
        network_id="grid3x3",
        net_xml_path="configs/networks/grid3x3.net.xml",
        is_ood_regime=False,
        step_length_s=1.0,
        warmup_duration_s=0.0,
        cohort_duration_s=3.0,
        drain_duration_s=1.0,
    )
    collect_single_rollout_episode(
        exp_cfg=ExperimentConfig(scenario=mini_scen, dbn=DBNConfig(model_preset="ci_micro")),
        seed=TRAIN_SEEDS[8],  # seed % 10 == 8 -> exploratory=True!
        split="train",
        output_dir=tmp_path / "snap_order_check",
        topology_loader=lambda **_kw: topo,
        adapter_factory=lambda scenario, topology, seed, **_kw: _DeterministicMockAdapter(scenario, topology, seed),
        tracker_factory=lambda **_kw: _SimpleTracker(),
        controller_factory=lambda **_kw: _SnapTrackingController(),
        feature_extractor_fn=_recording_extractor,
    )
    assert seen_prev_snap_times == [None, 0.0, 1.0, 2.0]


def test_live_sumo_collect_train_and_simulate_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify full cross-slice live SUMO integration: collect -> portable shard -> train -> closed-loop simulate."""
    from dataclasses import replace as dc_replace

    from swarm.config import load_scenario_config
    from swarm.pipelines import collect_rollout_dataset, load_rollout_shards, run_single_episode

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / ".mpl_cache"))

    base_scen = load_scenario_config(REPO_ROOT / "configs" / "scenarios" / "grid3x3_nominal.json")
    short_scen = dc_replace(
        base_scen,
        warmup_duration_s=3.0,
        cohort_duration_s=10.0,
        drain_duration_s=4.0,
    )

    tr_dir = tmp_path / "live_train"
    va_dir = tmp_path / "live_val"
    collect_rollout_dataset(
        scenarios=[short_scen],
        seeds=[TRAIN_SEEDS[0]],
        split="train",
        output_dir=tr_dir,
        max_workers=1,
        behavior_policy_override="actuated_signal",
    )
    collect_rollout_dataset(
        scenarios=[short_scen],
        seeds=[VAL_SEEDS[0]],
        split="val",
        output_dir=va_dir,
        max_workers=1,
        behavior_policy_override="actuated_signal",
    )

    # Verify shards are 100% portable (relative paths + CPU detached tensors, #34)
    tr_manifest, tr_shards = load_rollout_shards(tr_dir, expected_split="train")
    assert len(tr_shards) == 1
    assert "/" not in tr_manifest["shards"][0]["shard_file"]
    assert len(tr_shards[0]["steps"]) > 0

    ckpt_path = tmp_path / "live_ckpt" / "neural_dbn.pt"
    train_neural_dbn(
        train_dir=tr_dir,
        val_dir=va_dir,
        output_checkpoint_path=ckpt_path,
        train_cfg=TrainConfig(model_preset="ci_micro", epochs=2, batch_size=2, seed=99),
        dbn_cfg=DBNConfig(model_preset="ci_micro"),
    )

    # Run closed-loop neural_dbn episode on TEST_SEEDS[0] with 3D trace recording
    exp_cfg = ExperimentConfig(
        scenario=short_scen,
        dbn=DBNConfig(),  # Deliberately default ("sota_local") to test automatic preset sync from checkpoint!
        control=ControlConfig(),
    )
    trace_file = tmp_path / "live_trace.json"
    metrics, trace_payload = run_single_episode(
        exp_cfg=exp_cfg,
        policy_name="neural_dbn",
        seed=TEST_SEEDS[0],
        checkpoint_path=ckpt_path.parent,  # Pass directory to verify neural_dbn.pt resolution (#34)
        record_trace=True,
        trace_output_path=trace_file,
        allow_ci_micro=True,
    )
    assert metrics.cohort_size > 0
    assert metrics.collision_count == 0
    assert metrics.teleport_count == 0
    assert metrics.checkpoint_sha256 is not None
    assert trace_payload is not None and trace_file.is_file()
    assert len(trace_payload["frames"]) == short_scen.total_steps


def test_test_split_named_splits_and_contiguous_window_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify split='test' anti-leakage, CLI --named-split predictive eval, and non-crossing contiguous windows."""
    from swarm.cli import main as cli_main
    from swarm.pipelines.rollout_collector import (
        _select_contiguous_window_indices,
        load_rollout_shards,
        validate_rollout_split,
    )
    from swarm.pipelines.trainer import RolloutSequenceDataset

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / ".mpl_cache"))

    id_scen = ScenarioConfig(
        scenario_id="grid3x3_nominal",
        network_id="grid3x3",
        net_xml_path="configs/networks/grid3x3.net.xml",
        is_ood_regime=False,
    )
    ood_scen = ScenarioConfig(
        scenario_id="osm_palo_alto_rush",
        network_id="palo_alto_clean",
        net_xml_path="configs/networks/palo_alto_clean.net.xml",
        is_ood_regime=True,
    )

    # split='test' allows TEST_SEEDS across both ID and OOD, but blocks TRAIN_SEEDS and VAL_SEEDS
    assert validate_rollout_split(id_scen, [TEST_SEEDS[0]], "test") == (TEST_SEEDS[0],)
    assert validate_rollout_split(ood_scen, [TEST_SEEDS[1]], "test") == (TEST_SEEDS[1],)
    with pytest.raises(ConfigValidationError):
        validate_rollout_split(id_scen, [TRAIN_SEEDS[0]], "test")
    with pytest.raises(ConfigValidationError):
        validate_rollout_split(ood_scen, [VAL_SEEDS[0]], "test")

    # Contiguous window selector produces non-overlapping windows inside [0, n_active - horizon_steps)
    sel = _select_contiguous_window_indices(
        200, max_windows_per_episode=3, window_len_steps=8, horizon_steps=20
    )
    assert len(sel) == 24
    assert max(sel) < 180

    # Train a quick ci_micro model and test CLI --named-split across 3 held-out splits,
    # while blocking train/val manifests from eval-predictive!
    tr_dir = _write_synthetic_rollout_split(
        tmp_path / "tr", split="train", scenario_id="grid3x3_nominal", network_id="grid3x3", seeds=[TRAIN_SEEDS[0]], n_steps=8
    )
    va_dir = _write_synthetic_rollout_split(
        tmp_path / "va", split="val", scenario_id="grid3x3_nominal", network_id="grid3x3", seeds=[VAL_SEEDS[0]], n_steps=8
    )
    t_id = _write_synthetic_rollout_split(
        tmp_path / "t_id", split="test", scenario_id="grid3x3_nominal", network_id="grid3x3", seeds=[TEST_SEEDS[0]], n_steps=8
    )
    t_g4 = _write_synthetic_rollout_split(
        tmp_path / "t_g4", split="test", scenario_id="grid4x4_incident", network_id="grid4x4", seeds=[TEST_SEEDS[1]], n_steps=8
    )
    t_pa = _write_synthetic_rollout_split(
        tmp_path / "t_pa", split="test", scenario_id="osm_palo_alto_rush", network_id="palo_alto_clean", seeds=[TEST_SEEDS[2]], n_steps=8
    )

    # Verify trainer rejects split='test' in train_dir or val_dir
    with pytest.raises(ConfigValidationError):
        train_neural_dbn(
            train_dir=t_id,
            val_dir=va_dir,
            output_checkpoint_path=tmp_path / "bad_ckpt.pt",
            train_cfg=TrainConfig(model_preset="ci_micro", epochs=1),
            dbn_cfg=DBNConfig(model_preset="ci_micro"),
        )

    ckpt_path = tmp_path / "ok_ckpt" / "neural_dbn.pt"
    train_neural_dbn(
        train_dir=tr_dir,
        val_dir=va_dir,
        output_checkpoint_path=ckpt_path,
        train_cfg=TrainConfig(model_preset="ci_micro", epochs=2, batch_size=2, seed=7),
        dbn_cfg=DBNConfig(model_preset="ci_micro"),
    )

    # Verify RolloutSequenceDataset never constructs windows crossing non-contiguous sim_time jumps
    _, raw_shards = load_rollout_shards(t_id, expected_split="test")
    gapped_shard = dict(raw_shards[0])
    gapped_steps = [dict(s) for s in gapped_shard["steps"]]
    for k in range(4, len(gapped_steps)):
        gapped_steps[k]["sim_time"] = float(gapped_steps[k]["sim_time"]) + 25.0
    gapped_shard["steps"] = gapped_steps
    gapped_ds = RolloutSequenceDataset([gapped_shard], dbn_cfg=DBNConfig(model_preset="ci_micro"), window_len=4, stride=2)
    for w_idx in range(len(gapped_ds)):
        win = gapped_ds.get_window(w_idx)
        times = [fb.sim_time for fb, _ in win]
        diffs = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        assert all(d <= 1.0 for d in diffs)

    pred_out = tmp_path / "multi_pred.json"
    plot_out = tmp_path / "multi_pred.png"
    rc = cli_main(
        [
            "--error-report-dir",
            str(tmp_path / "err"),
            "eval-predictive",
            "--checkpoint",
            str(ckpt_path),
            "--named-split",
            f"test_id_grid3x3={t_id}",
            f"test_ood_grid4x4_incident={t_g4}",
            f"test_ood_palo_alto={t_pa}",
            "--output",
            str(pred_out),
            "--plot-output",
            str(plot_out),
            "--allow-ci-micro",
        ]
    )
    assert rc == 0
    pred_json = json.loads(pred_out.read_text(encoding="utf-8"))
    assert set(pred_json["splits"]) == {
        "test_id_grid3x3",
        "test_ood_grid4x4_incident",
        "test_ood_palo_alto",
    }
    assert plot_out.is_file()

    # Passing a train split into --named-split must fail loudly
    with pytest.raises(ConfigValidationError):
        evaluate_heldout_predictive(
            checkpoint_path=ckpt_path,
            dataset_dirs={"leaked": tr_dir},
            allow_ci_micro=True,
        )

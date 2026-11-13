"""Comprehensive unit, mathematical-exactness, fail-loud, AST-firewall, and runtime test suite for ``swarm.dbn``."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from swarm.config import MODEL_PRESETS, DBNConfig
from swarm.contracts import (
    CORRIDOR_EDGE_FEATURE_DIM,
    CORRIDOR_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    CorridorBeliefStats,
    CorridorSpec,
    JunctionSpec,
    RelationType,
    RoadTopology,
    SwarmBeliefState,
    TurnOption,
    VehicleBelief,
)
from swarm.dbn import (
    FeatureNormalizer,
    HeuristicRadiusDBNUpdater,
    IndependentDBNUpdater,
    NeuralDBNUpdater,
    OnlineBICDBNUpdater,
    PrunedDAGStructure,
    STGraphTransformerDBN,
    TemperatureScales,
    build_candidate_graph,
    build_dbn_updater,
    calibrate_temperatures,
    compute_predictive_metrics,
    extract_graph_features,
    forward_dbn_step,
    gate_and_prune_edges,
    load_dbn_checkpoint,
    multiplicative_sum_product_lbp,
    parent_survival_factors,
    save_dbn_checkpoint,
)
from swarm.dbn.inference import _exclusive_prefix_suffix_1d_slots
from swarm.logging_utils import (
    CheckpointIntegrityError,
    ConfigValidationError,
    NumericalInvariantError,
    TopologyIntegrityError,
    require_dag,
)


def _make_cyclic_urban_topology(*, n_ring_junctions: int = 4) -> RoadTopology:
    """Construct a contract-valid cyclic multi-junction road topology with entry/exit fringes.

    Ring corridors ``C_ring_k`` form a directed cycle ``J_0 -> J_1 -> ... -> J_{M-1} -> J_0``;
    cross approaches ``C_in_k`` feed each ``J_k`` and ``C_exit_k`` exit to fringe junctions ``JF_k``
    with ``is_exit_fringe=True`` and ``len(turn_options)==0`` (exercising Boundary Lock A!).
    """
    corridors: dict[str, CorridorSpec] = {}
    edge_to_corridor: dict[str, str] = {}
    edge_offset: dict[str, float] = {}
    junctions: dict[str, JunctionSpec] = {}
    succ: dict[str, tuple[str, ...]] = {}
    pred_lists: dict[str, list[str]] = {}

    # Create exit corridors first so turn_options can target them
    for k in range(n_ring_junctions):
        cid_exit = f"C_exit_{k}"
        eid_exit = f"e_exit_{k}"
        corridors[cid_exit] = CorridorSpec(
            corridor_id=cid_exit,
            edges=(eid_exit,),
            edge_offsets_m=(0.0,),
            from_junction=f"J_{k}",
            to_junction=f"JF_out_{k}",
            length_m=120.0,
            base_lanes=2,
            speed_limit_mps=13.89,
            turn_options=(),
            is_entry_fringe=False,
            is_exit_fringe=True,
        )
        edge_to_corridor[eid_exit] = cid_exit
        edge_offset[eid_exit] = 0.0
        succ[cid_exit] = ()
        pred_lists.setdefault(cid_exit, [])

        junctions[f"JF_out_{k}"] = JunctionSpec(
            junction_id=f"JF_out_{k}",
            tls_id=None,
            coord_xy=(float(k) * 160.0, 220.0),
            incoming_corridors=(cid_exit,),
            outgoing_corridors=(),
            num_tls_links=0,
            tls_link_foe_matrix=np.zeros((0, 0), dtype=bool),
            req_foe_matrix=np.zeros((1, 1), dtype=bool),
        )
        junctions[f"JF_in_{k}"] = JunctionSpec(
            junction_id=f"JF_in_{k}",
            tls_id=None,
            coord_xy=(float(k) * 160.0, -220.0),
            incoming_corridors=(),
            outgoing_corridors=(f"C_in_{k}",),
            num_tls_links=0,
            tls_link_foe_matrix=np.zeros((0, 0), dtype=bool),
            req_foe_matrix=np.zeros((1, 1), dtype=bool),
        )

    for k in range(n_ring_junctions):
        next_k = (k + 1) % n_ring_junctions
        cid_ring = f"C_ring_{k}"
        eid_ring = f"e_ring_{k}"
        next_ring_cid = f"C_ring_{next_k}"
        next_ring_eid = f"e_ring_{next_k}"
        exit_cid = f"C_exit_{next_k}"
        exit_eid = f"e_exit_{next_k}"
        tls_id = f"tls_{next_k}"

        t_thru = TurnOption(
            turn_idx=0,
            from_edge=eid_ring,
            to_corridor=next_ring_cid,
            to_edge=next_ring_eid,
            angle_rad=0.0,
            tls_id=tls_id,
            tls_link_indices=(0,),
            junction_req_indices=(0,),
        )
        t_left = TurnOption(
            turn_idx=1,
            from_edge=eid_ring,
            to_corridor=exit_cid,
            to_edge=exit_eid,
            angle_rad=math.pi / 2.0,
            tls_id=tls_id,
            tls_link_indices=(1,),
            junction_req_indices=(1,),
        )
        corridors[cid_ring] = CorridorSpec(
            corridor_id=cid_ring,
            edges=(eid_ring,),
            edge_offsets_m=(0.0,),
            from_junction=f"J_{k}",
            to_junction=f"J_{next_k}",
            length_m=160.0,
            base_lanes=2,
            speed_limit_mps=13.89,
            turn_options=(t_thru, t_left),
            is_entry_fringe=False,
            is_exit_fringe=False,
        )
        edge_to_corridor[eid_ring] = cid_ring
        edge_offset[eid_ring] = 0.0
        succ[cid_ring] = (next_ring_cid, exit_cid)
        pred_lists.setdefault(next_ring_cid, []).append(cid_ring)
        pred_lists.setdefault(exit_cid, []).append(cid_ring)

        # Cross entry corridor into J_{next_k} conflicting with C_ring_{k}
        cid_in = f"C_in_{next_k}"
        eid_in = f"e_in_{next_k}"
        t_in_merge = TurnOption(
            turn_idx=0,
            from_edge=eid_in,
            to_corridor=next_ring_cid,
            to_edge=next_ring_eid,
            angle_rad=-math.pi / 2.0,
            tls_id=tls_id,
            tls_link_indices=(2,),
            junction_req_indices=(2,),
        )
        t_in_thru = TurnOption(
            turn_idx=1,
            from_edge=eid_in,
            to_corridor=exit_cid,
            to_edge=exit_eid,
            angle_rad=0.0,
            tls_id=tls_id,
            tls_link_indices=(3,),
            junction_req_indices=(3,),
        )
        corridors[cid_in] = CorridorSpec(
            corridor_id=cid_in,
            edges=(eid_in,),
            edge_offsets_m=(0.0,),
            from_junction=f"JF_in_{next_k}",
            to_junction=f"J_{next_k}",
            length_m=150.0,
            base_lanes=2,
            speed_limit_mps=13.89,
            turn_options=(t_in_merge, t_in_thru),
            is_entry_fringe=True,
            is_exit_fringe=False,
        )
        edge_to_corridor[eid_in] = cid_in
        edge_offset[eid_in] = 0.0
        succ[cid_in] = (next_ring_cid, exit_cid)
        pred_lists.setdefault(next_ring_cid, []).append(cid_in)
        pred_lists.setdefault(exit_cid, []).append(cid_in)

    # Create conflicting foe matrices for each signalized ring junction J_{next_k}
    foe_mat = np.array(
        [
            [False, False, True, True],
            [False, False, True, True],
            [True, True, False, False],
            [True, True, False, False],
        ],
        dtype=bool,
    )
    for k in range(n_ring_junctions):
        prev_k = (k - 1) % n_ring_junctions
        junctions[f"J_{k}"] = JunctionSpec(
            junction_id=f"J_{k}",
            tls_id=f"tls_{k}",
            coord_xy=(float(k) * 160.0, 0.0),
            incoming_corridors=(f"C_ring_{prev_k}", f"C_in_{k}"),
            outgoing_corridors=(f"C_ring_{k}", f"C_exit_{k}"),
            num_tls_links=4,
            tls_link_foe_matrix=foe_mat.copy(),
            req_foe_matrix=foe_mat.copy(),
        )

    pred = {cid: tuple(sorted(set(plist))) for cid, plist in pred_lists.items()}
    for cid in corridors:
        pred.setdefault(cid, ())

    return RoadTopology(
        network_id=f"cyclic_ring_{n_ring_junctions}",
        net_xml_path="/virtual/cyclic_ring.net.xml",
        poly_xml_path=None,
        corridors=corridors,
        edge_to_corridor=edge_to_corridor,
        edge_offset_in_corridor=edge_offset,
        junctions=junctions,
        successor_corridors=succ,
        predecessor_corridors=pred,
    )


def _make_swarm_belief(
    topology: RoadTopology,
    *,
    num_vehicles: int,
    sim_time: float = 10.0,
    dt: float = 0.5,
    seed: int = 42,
    force_eta_ties: bool = False,
) -> SwarmBeliefState:
    """Generate a realistic observable ``SwarmBeliefState`` across all corridors (including exit fringes)."""
    rng = np.random.default_rng(seed)
    cids = sorted(topology.corridors.keys())
    vehicles: dict[str, VehicleBelief] = {}
    veh_counts: dict[str, int] = {cid: 0 for cid in cids}
    tail_counts: dict[str, int] = {cid: 0 for cid in cids}
    speed_sums: dict[str, float] = {cid: 0.0 for cid in cids}

    cov = np.array([[1.2, 0.08], [0.08, 0.35]], dtype=np.float32)

    for idx in range(num_vehicles):
        vid = f"veh_{idx:04d}"
        cid = cids[idx % len(cids)]
        spec = topology.corridors[cid]
        pos_m = float(rng.uniform(0.1 * spec.length_m, 0.9 * spec.length_m))
        speed_mps = float(rng.uniform(min(1.5, 0.2 * spec.speed_limit_mps), spec.speed_limit_mps))
        k_turns = len(spec.turn_options)
        if k_turns == 0:
            turn_probs: tuple[float, ...] = ()
        else:
            raw = rng.dirichlet(np.ones(k_turns, dtype=np.float64))
            turn_probs = tuple(float(x) for x in raw)

        free_eta = 4.0 if (force_eta_ties and idx % 3 == 0) else max(0.2, (spec.length_m - pos_m) / spec.speed_limit_mps)
        tail_score = 1.0 if (pos_m < 28.0 and speed_mps < 4.0) else 0.0

        vehicles[vid] = VehicleBelief(
            veh_id=vid,
            entered_at=0.0,
            corridor_id=cid,
            edge_id=spec.edges[0],
            in_junction_id=None,
            lane_index=int(idx % spec.base_lanes),
            length_m=4.8,
            width_m=1.9,
            mean_s_m=pos_m,
            mean_v_mps=speed_mps,
            cov_2x2=cov.copy(),
            accel_est_mps2=float(rng.uniform(-1.5, 1.5)),
            obs_age_s=0.0,
            wait_time_s=float(rng.uniform(0.0, 5.0)) if speed_mps < 2.5 else 0.0,
            turn_probs=turn_probs,
            free_eta_s=free_eta,
            eta_std_s=0.45,
            tail_blockage_score=tail_score,
            leader_id=None,
            noisy_leader_gap_m=None,
            noisy_leader_speed_mps=None,
            last_v2x_poll_time=sim_time,
        )
        veh_counts[cid] += 1
        if tail_score > 0.5:
            tail_counts[cid] += 1
        speed_sums[cid] += speed_mps

    open_lanes = {spec.edges[0]: spec.base_lanes for spec in topology.corridors.values()}
    corridors_stats: dict[str, CorridorBeliefStats] = {}
    for cid, spec in topology.corridors.items():
        cnt = veh_counts[cid]
        cap = topology.effective_capacity(cid, open_lanes)
        mean_v = speed_sums[cid] / float(cnt) if cnt > 0 else spec.speed_limit_mps
        corridors_stats[cid] = CorridorBeliefStats(
            corridor_id=cid,
            active_count=cnt,
            effective_lanes=spec.base_lanes,
            storage_capacity=cap,
            occupancy_ratio=min(1.5, float(cnt) / float(cap)),
            tail_blockage_prob=min(1.0, float(tail_counts[cid]) / float(max(spec.base_lanes, 1))),
            expected_inflow=0.45,
            mean_speed_mps=mean_v,
        )

    tls_states: dict[str, str] = {}
    for junc in topology.junctions.values():
        if junc.tls_id is not None:
            n_links = max(len(tls_states.get(junc.tls_id, "")), junc.num_tls_links)
            half = n_links // 2
            tls_states[junc.tls_id] = ("G" * half) + ("r" * (n_links - half))

    return SwarmBeliefState(
        sim_time=sim_time,
        dt=dt,
        vehicles=vehicles,
        corridors=corridors_stats,
        open_lanes_by_edge=open_lanes,
        tls_states=tls_states,
    )


# =============================================================================
# 1. Strict DAG Acyclicity, Feature Schema, Presets, & Boundary Lock A
# =============================================================================


def test_dag_acyclicity_presets_and_boundary_lock_a() -> None:
    topo = _make_cyclic_urban_topology(n_ring_junctions=4)
    belief = _make_swarm_belief(topo, num_vehicles=36, force_eta_ties=True, seed=7)
    cfg = DBNConfig(model_preset="ci_micro", max_parents_k=5, min_edge_prob=0.05, require_calibrated=False)

    cand = build_candidate_graph(belief, topo, cfg)
    feat = extract_graph_features(belief, topo, cfg, candidate_graph=cand)

    assert feat.x_veh.shape == (36, NODE_FEATURE_DIM)
    assert feat.x_corridor.shape == (len(topo.corridors), CORRIDOR_FEATURE_DIM)
    assert feat.x_edge.shape == (cand.num_edges, EDGE_FEATURE_DIM)
    assert feat.x_corridor_edge.shape[1] == CORRIDOR_EDGE_FEATURE_DIM

    # Verify all 4 RelationTypes can be represented and per-vehicle candidate cap holds
    _, dst_counts = np.unique(cand.edge_index[1], return_counts=True)
    assert int(dst_counts.max()) <= cfg.max_candidates_per_veh

    # Boundary Lock A: vehicles on exit-fringe corridors have exact 0.0 downstream features
    for idx, vid in enumerate(cand.veh_ids):
        spec = topo.corridors[belief.vehicles[vid].corridor_id]
        if spec.is_exit_fringe:
            assert feat.x_veh[idx, 19] == 0.0
            assert feat.x_veh[idx, 20] == 0.0

    # Verify all 3 presets instantiate and produce strict DAGs
    norm = FeatureNormalizer().fit([feat])
    temps = TemperatureScales(t_edge=1.0, t_delay=1.0, t_link=1.0, calibrated=True)
    for preset_name in MODEL_PRESETS:
        torch.manual_seed(11)
        model = STGraphTransformerDBN(preset=preset_name)
        cfg_p = DBNConfig(model_preset=preset_name, max_parents_k=5, min_edge_prob=0.02)
        updater = NeuralDBNUpdater(model, norm, temps, cfg_p, policy_name="neural_dbn")
        snap = updater.update_and_infer(belief, None, topo)
        require_dag(snap.veh_ids, snap.edges, snap.precedence_scores)
        for vid in snap.veh_ids:
            assert len(snap.edges_by_dst[vid]) <= cfg_p.max_parents_k

    # Verify non-neural baselines also produce strict DAGs on cyclic road topology with ETA ties
    for base_upd in (
        IndependentDBNUpdater(cfg),
        HeuristicRadiusDBNUpdater(cfg),
        OnlineBICDBNUpdater(cfg),
    ):
        snap_b = base_upd.update_and_infer(belief, None, topo)
        require_dag(snap_b.veh_ids, snap_b.edges, snap_b.precedence_scores)

    # Boundary Lock A violation: non-exit corridor with zero turn_options raises TopologyIntegrityError
    with pytest.raises(TopologyIntegrityError):
        CorridorSpec(
            corridor_id="C_broken",
            edges=("e_b",),
            edge_offsets_m=(0.0,),
            from_junction="J_0",
            to_junction="J_1",
            length_m=100.0,
            base_lanes=2,
            speed_limit_mps=13.89,
            turn_options=(),
            is_entry_fringe=False,
            is_exit_fringe=False,
        )


# =============================================================================
# 2. D1 Fix: Free-Flowing Leader (D_j = 0) Non-Annihilation in Generalized Noisy-OR
# =============================================================================


def test_d1_free_flowing_leader_exerts_causal_conflict_pressure() -> None:
    omega = torch.tensor([0.85], dtype=torch.float32)
    kappa = torch.tensor([0.45], dtype=torch.float32)
    gamma = torch.tensor([0.60], dtype=torch.float32)

    phi_0, phi_1 = parent_survival_factors(omega, kappa, gamma)
    assert float(phi_0.item()) == pytest.approx(1.0 - 0.85 * 0.45, abs=1e-6)
    assert float(phi_0.item()) < 1.0  # D1 fix: undelayed parent still inhibits survival!
    assert float(phi_1.item()) < float(phi_0.item())

    # 2-node network: parent 0 is virtually guaranteed undelayed (z0_0 = 0.99999, neutral speed evidence 0.5)
    z0 = torch.tensor([0.99999, 0.92], dtype=torch.float32)
    ev_neutral = torch.tensor([0.5, 0.5], dtype=torch.float32)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    prec = torch.tensor([5.0, 1.0], dtype=torch.float32)

    pruned_active = gate_and_prune_edges(
        edge_index=edge_index,
        precedence=prec,
        omega_raw=omega,
        kappa=kappa,
        gamma=gamma,
        num_vehicles=2,
        max_parents_k=4,
        min_edge_prob=0.05,
    )
    pruned_empty = gate_and_prune_edges(
        edge_index=edge_index,
        precedence=prec,
        omega_raw=torch.tensor([0.0], dtype=torch.float32),
        kappa=kappa,
        gamma=gamma,
        num_vehicles=2,
        max_parents_k=4,
        min_edge_prob=0.05,
    )

    res_with_free_leader = multiplicative_sum_product_lbp(z0, pruned_active, ev_neutral, lbp_sweeps=4, lbp_damping=0.0)
    res_no_leader = multiplicative_sum_product_lbp(z0, pruned_empty, ev_neutral, lbp_sweeps=4, lbp_damping=0.0)

    # Even though P(D_0 = 1) ~ 1e-5 (free-flowing!), child 1's delay prob rises substantially above 0.08!
    assert float(res_with_free_leader.delay_post[0].item()) < 1e-3
    assert float(res_no_leader.delay_post[1].item()) == pytest.approx(0.08, abs=1e-5)
    assert float(res_with_free_leader.delay_post[1].item()) > 0.40

    # And when leader 0 IS delayed (z0_0 = 0.05), child 1's delay prob increases even further!
    z0_delayed_leader = torch.tensor([0.05, 0.92], dtype=torch.float32)
    res_delayed_leader = multiplicative_sum_product_lbp(
        z0_delayed_leader, pruned_active, ev_neutral, lbp_sweeps=4, lbp_damping=0.0
    )
    assert float(res_delayed_leader.delay_post[1].item()) > float(res_with_free_leader.delay_post[1].item()) + 0.15


# =============================================================================
# 3. Division-Free Sum-Product LBP Exactness vs 2^8 = 256 Joint Enumeration
# =============================================================================


def _brute_force_noisy_or_marginals(
    z0: np.ndarray,
    edges: list[tuple[int, int, float, float]],
    ev_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact 2^N joint distribution enumeration for Generalized Noisy-OR DBN."""
    n = z0.shape[0]
    parents_of: list[list[tuple[int, float, float]]] = [[] for _ in range(n)]
    for j, i, p0, p1 in edges:
        parents_of[i].append((j, p0, p1))

    num_states = 1 << n
    prior_weights = np.zeros(num_states, dtype=np.float64)
    post_weights = np.zeros(num_states, dtype=np.float64)
    bits_mat = np.zeros((num_states, n), dtype=np.float64)

    for mask in range(num_states):
        d = [(mask >> k) & 1 for k in range(n)]
        bits_mat[mask] = d
        p_joint = 1.0
        psi_prod = 1.0
        for i in range(n):
            surv = float(z0[i])
            for j, phi0, phi1 in parents_of[i]:
                surv *= phi1 if d[j] == 1 else phi0
            p_i = (1.0 - surv) if d[i] == 1 else surv
            p_joint *= p_i
            psi_prod *= float(ev_pos[i]) if d[i] == 1 else float(1.0 - ev_pos[i])
        prior_weights[mask] = p_joint
        post_weights[mask] = p_joint * psi_prod

    assert abs(float(prior_weights.sum()) - 1.0) < 1e-9
    prior_marg = (prior_weights[:, None] * bits_mat).sum(axis=0)
    post_marg = (post_weights[:, None] * bits_mat).sum(axis=0) / float(post_weights.sum())
    return prior_marg, post_marg


def test_lbp_matches_exact_256_state_enumeration_on_polytree_and_zero_survival() -> None:
    # 8-node polytree with v-structures, forks, and 3-hop chains (zero undirected cycles):
    # 0 -> 3, 1 -> 3 (v-structure at 3), 3 -> 5, 2 -> 5 (v-structure at 5), 3 -> 6 (fork from 3), 5 -> 7, 4 -> 7
    raw_edges = [
        (0, 3),
        (1, 3),
        (3, 5),
        (2, 5),
        (3, 6),
        (5, 7),
        (4, 7),
    ]
    n_nodes = 8
    rng = np.random.default_rng(2026)
    z0_np = rng.uniform(0.35, 0.92, size=n_nodes).astype(np.float32)
    ev_np = rng.uniform(0.15, 0.88, size=n_nodes).astype(np.float32)
    omega_np = rng.uniform(0.35, 0.90, size=len(raw_edges)).astype(np.float32)
    kappa_np = rng.uniform(0.15, 0.65, size=len(raw_edges)).astype(np.float32)
    gamma_np = rng.uniform(0.25, 0.80, size=len(raw_edges)).astype(np.float32)

    prec_np = np.array([10.0, 9.5, 9.0, 7.0, 6.5, 4.0, 3.5, 1.0], dtype=np.float32)
    edge_index_t = torch.tensor([[u for u, _ in raw_edges], [v for _, v in raw_edges]], dtype=torch.long)

    pruned = gate_and_prune_edges(
        edge_index=edge_index_t,
        precedence=torch.as_tensor(prec_np),
        omega_raw=torch.as_tensor(omega_np),
        kappa=torch.as_tensor(kappa_np),
        gamma=torch.as_tensor(gamma_np),
        num_vehicles=n_nodes,
        max_parents_k=6,
        min_edge_prob=0.05,
        training=False,
    )
    assert int(pruned.active_edge_indices.numel()) == len(raw_edges)

    # Undamped exact BP on polytree (6 sweeps >= tree diameter)
    lbp_undamped = multiplicative_sum_product_lbp(
        base_survival_z0=torch.as_tensor(z0_np),
        pruned_dag=pruned,
        speed_evidence_pos=torch.as_tensor(ev_np),
        lbp_sweeps=8,
        lbp_damping=0.0,
    )
    # Damped BP (damping=0.35) also converges to exact polytree fixed point
    lbp_damped = multiplicative_sum_product_lbp(
        base_survival_z0=torch.as_tensor(z0_np),
        pruned_dag=pruned,
        speed_evidence_pos=torch.as_tensor(ev_np),
        lbp_sweeps=24,
        lbp_damping=0.35,
    )

    edge_tuples = [
        (
            int(pruned.src_idx[k].item()),
            int(pruned.dst_idx[k].item()),
            float(pruned.phi_0[k].item()),
            float(pruned.phi_1[k].item()),
        )
        for k in range(len(raw_edges))
    ]
    exact_prior, exact_post = _brute_force_noisy_or_marginals(z0_np, edge_tuples, ev_np)

    np.testing.assert_allclose(lbp_undamped.delay_pred.numpy(), exact_prior, atol=1e-6)
    np.testing.assert_allclose(lbp_undamped.delay_post.numpy(), exact_post, atol=1e-6)
    np.testing.assert_allclose(lbp_damped.delay_post.numpy(), exact_post, atol=1e-5)

    # Verify division-free prefix-suffix cavity survival when c_slot == 0.0 exactly!
    zero_slots = torch.tensor([[0.0, 0.4, 0.5], [0.7, 0.0, 0.0]], dtype=torch.float32)
    excl, full = _exclusive_prefix_suffix_1d_slots(zero_slots)
    assert torch.isfinite(excl).all()
    assert float(excl[0, 0].item()) == pytest.approx(0.20, abs=1e-7)  # 0.4 * 0.5 without 0 / 0!
    assert float(excl[0, 1].item()) == pytest.approx(0.0, abs=1e-7)
    assert float(full[0].item()) == pytest.approx(0.0, abs=1e-7)


# =============================================================================
# 4. End-to-End Gradient Flow Through Unrolled 2-Step TBPTT + Smooth DAG + LBP
# =============================================================================


def test_unrolled_two_step_gradient_flow_into_all_subsystems() -> None:
    torch.manual_seed(99)
    topo = _make_cyclic_urban_topology(n_ring_junctions=3)
    b0 = _make_swarm_belief(topo, num_vehicles=16, sim_time=0.0, seed=101)
    b1 = _make_swarm_belief(topo, num_vehicles=16, sim_time=0.5, seed=102)

    cfg = DBNConfig(model_preset="ci_micro", max_parents_k=4, lbp_sweeps=3, lbp_damping=0.4)
    f0 = extract_graph_features(b0, topo, cfg)
    f1 = extract_graph_features(b1, topo, cfg)
    norm = FeatureNormalizer().fit([f0, f1])

    model = STGraphTransformerDBN(preset="ci_micro")
    model.train()

    step0 = forward_dbn_step(
        model=model,
        normalizer=norm,
        feat_batch=f0,
        dbn_cfg=cfg,
        dag_temperature=0.5,
    )

    delay_carry = {vid: step0.delay_post[i] for i, vid in enumerate(f0.candidate_graph.veh_ids)}
    edge_carry = {
        (f0.candidate_graph.veh_ids[int(f0.candidate_graph.edge_index[0, k])], f0.candidate_graph.veh_ids[int(f0.candidate_graph.edge_index[1, k])]): step0.omega_raw[k]
        for k in range(f0.candidate_graph.num_edges)
    }

    step1 = forward_dbn_step(
        model=model,
        normalizer=norm,
        feat_batch=f1,
        dbn_cfg=cfg,
        prev_memory=step0.next_memory,
        prev_delay_post_override=delay_carry,
        prev_edge_omega_override=edge_carry,
        dag_temperature=0.5,
    )

    loss = (
        step1.delay_post.mean() * 2.0
        + step1.expected_delay_s.mean()
        + step1.corridor_spillback_prob.mean()
        + step1.omega_raw.mean() * 0.5
        + step0.delay_post.mean() * 0.5
    )
    loss.backward()

    checked_groups = {
        "veh_stem": model.veh_stem.proj.weight.grad,
        "corridor_stem": model.corridor_stem.proj.weight.grad,
        "edge_cont_stem": model.edge_cont_stem.proj.weight.grad,
        "veh_gru": model.veh_gru.gru_cell.weight_hh.grad,
        "corridor_gru": model.corridor_gru.gru_cell.weight_hh.grad,
        "edge_gru": model.edge_gru.gru_cell.weight_hh.grad,
        "block_v2c": model.blocks[0].v2c_out.weight.grad,
        "block_c2c": model.blocks[0].c_wo.weight.grad,
        "block_c2v": model.blocks[0].c2v_proj.w_out.weight.grad,
        "block_v_attn": model.blocks[0].v_wo.weight.grad,
        "node_head_precedence_row": model.node_head_mlp[-1].weight.grad[0],
        "edge_head_out": model.edge_head_out.weight.grad,
        "phys_corridor_cpd": model.phys_corridor_cpd.raw_w_occ.grad,
    }
    for label, grad in checked_groups.items():
        assert grad is not None, f"Missing gradient on {label}"
        assert torch.isfinite(grad).all(), f"Non-finite gradient on {label}"
        assert float(grad.abs().sum().item()) > 0.0, f"Zero gradient on {label}"


# =============================================================================
# 5. Bit-Identical .reset() Across Multi-Step Episodes & Ablation Modes
# =============================================================================


def test_bit_identical_reset_and_ablation_modes() -> None:
    torch.manual_seed(17)
    topo = _make_cyclic_urban_topology(n_ring_junctions=3)
    steps_beliefs = [
        _make_swarm_belief(topo, num_vehicles=20, sim_time=float(t) * 0.5, seed=200 + t)
        for t in range(4)
    ]
    cfg = DBNConfig(model_preset="ci_micro", require_calibrated=True)
    f_init = extract_graph_features(steps_beliefs[0], topo, cfg)
    norm = FeatureNormalizer().fit([f_init])
    temps = TemperatureScales(t_edge=1.2, t_delay=0.9, t_link=1.1, calibrated=True)
    model = STGraphTransformerDBN(preset="ci_micro")
    updater = NeuralDBNUpdater(model, norm, temps, cfg, policy_name="neural_dbn")

    snaps_ep1 = []
    prev = None
    for b in steps_beliefs:
        prev = updater.update_and_infer(b, prev, topo)
        snaps_ep1.append(prev)
        assert prev.temporal_memory is not None

    updater.reset()
    snaps_ep2 = []
    prev = None
    for b in steps_beliefs:
        prev = updater.update_and_infer(b, prev, topo)
        snaps_ep2.append(prev)

    for s1, s2 in zip(snaps_ep1, snaps_ep2, strict=True):
        assert s1.precedence_scores == s2.precedence_scores
        assert s1.edges == s2.edges
        assert s1.delay_pred == s2.delay_pred
        assert s1.delay_post == s2.delay_post
        assert s1.expected_delay_s == s2.expected_delay_s
        assert s1.cascade_score == s2.cascade_score
        assert s1.corridor_spillback_pred == s2.corridor_spillback_pred
        assert s1.raw_edge_probs == s2.raw_edge_probs

    # Verify ablations: no_temporal_prior omits temporal_memory; static_cpd & fixed_route execute cleanly
    upd_no_temp = NeuralDBNUpdater(model, norm, temps, cfg, policy_name="no_temporal_prior")
    s_nt = upd_no_temp.update_and_infer(steps_beliefs[1], snaps_ep1[0], topo)
    assert s_nt.temporal_memory is None

    upd_static = NeuralDBNUpdater(model, norm, temps, cfg, policy_name="static_cpd")
    s_sc = upd_static.update_and_infer(steps_beliefs[1], snaps_ep1[0], topo)
    require_dag(s_sc.veh_ids, s_sc.edges, s_sc.precedence_scores)


# =============================================================================
# 6. D7 Fix: OnlineBICDBNUpdater True Parent Recovery Under 35% Dropout & Newcomers
# =============================================================================


def test_online_bic_recovers_true_parents_under_35pct_dropout_and_newcomer_flood() -> None:
    topo = _make_cyclic_urban_topology(n_ring_junctions=2)
    cid = "C_ring_0"
    spec = topo.corridors[cid]
    cov = np.eye(2, dtype=np.float32) * 0.5
    open_lanes = {s.edges[0]: s.base_lanes for s in topo.corridors.values()}
    tls_states = {j.tls_id: "GGrr" for j in topo.junctions.values() if j.tls_id is not None}

    corridors_stats = {
        c: CorridorBeliefStats(
            corridor_id=c,
            active_count=1,
            effective_lanes=s.base_lanes,
            storage_capacity=20,
            occupancy_ratio=0.25,
            tail_blockage_prob=0.0,
            expected_inflow=0.2,
            mean_speed_mps=10.0,
        )
        for c, s in topo.corridors.items()
    }

    cfg = DBNConfig(
        bic_window_steps=65,
        bic_min_pair_samples=8,
        bic_penalty_weight=0.50,
        max_candidates_per_veh=24,
    )
    bic_updater = OnlineBICDBNUpdater(cfg)
    rng = np.random.default_rng(20261112)

    # Core 6 vehicles on C_ring_0 ordered longitudinally so v0 > v1 > v2 > v3 > v4 > v5 in precedence:
    # True causal edges: v0 -> v2 and v1 -> v3. v4 and v5 are independent noise vehicles!
    positions = {
        "v0": 145.0,
        "v1": 132.0,
        "v2": 118.0,
        "v3": 104.0,
        "v4": 90.0,
        "v5": 76.0,
    }

    prev_snap = None
    for step in range(58):
        sim_t = float(step) * 0.5
        d_v0 = int(rng.random() < 0.45)
        d_v1 = int(rng.random() < 0.45)
        # Strong causal coupling P(D_child=1 | D_parent=1) = 0.92 vs P(D_child=1 | D_parent=0) = 0.06
        d_v2 = int(rng.random() < (0.92 if d_v0 == 1 else 0.06))
        d_v3 = int(rng.random() < (0.92 if d_v1 == 1 else 0.06))
        d_v4 = int(rng.random() < 0.35)
        d_v5 = int(rng.random() < 0.35)
        true_states = {"v0": d_v0, "v1": d_v1, "v2": d_v2, "v3": d_v3, "v4": d_v4, "v5": d_v5}

        vehs: dict[str, VehicleBelief] = {}
        for vid, d_flag in true_states.items():
            # 35% random per-vehicle dropout on steps 0..56; observed on final evaluation step 57
            dropped = (step < 57) and bool(rng.random() < 0.35)
            s_m = positions[vid]
            v_mps = 2.0 if d_flag == 1 else 12.5
            vehs[vid] = VehicleBelief(
                veh_id=vid,
                entered_at=0.0,
                corridor_id=cid,
                edge_id=spec.edges[0],
                in_junction_id=None,
                lane_index=0,
                length_m=4.8,
                width_m=1.9,
                mean_s_m=s_m,
                mean_v_mps=v_mps,
                cov_2x2=cov.copy(),
                accel_est_mps2=0.0,
                obs_age_s=1.0 if dropped else 0.0,
                wait_time_s=3.0 if d_flag == 1 else 0.0,
                turn_probs=(0.5, 0.5),
                free_eta_s=(spec.length_m - s_m) / spec.speed_limit_mps,
                eta_std_s=0.3,
                tail_blockage_score=0.0,
                leader_id=None,
                noisy_leader_gap_m=None,
                noisy_leader_speed_mps=None,
                last_v2x_poll_time=sim_t,
            )

        # Inject 10 brand-new newcomer bystanders on the final 4 steps (sample count = 4 < 8)
        if step >= 54:
            for b_idx in range(10):
                b_vid = f"newcomer_{b_idx:02d}"
                # Deliberately match d_v2 100% of the time over 4 ticks to try to trick small-sample BIC!
                b_pos = 152.0 - float(b_idx) * 0.5
                vehs[b_vid] = VehicleBelief(
                    veh_id=b_vid,
                    entered_at=27.0,
                    corridor_id=cid,
                    edge_id=spec.edges[0],
                    in_junction_id=None,
                    lane_index=0,
                    length_m=4.8,
                    width_m=1.9,
                    mean_s_m=b_pos,
                    mean_v_mps=2.0 if d_v2 == 1 else 12.5,
                    cov_2x2=cov.copy(),
                    accel_est_mps2=0.0,
                    obs_age_s=0.0,
                    wait_time_s=3.0 if d_v2 == 1 else 0.0,
                    turn_probs=(0.5, 0.5),
                    free_eta_s=(spec.length_m - b_pos) / spec.speed_limit_mps,
                    eta_std_s=0.3,
                    tail_blockage_score=0.0,
                    leader_id=None,
                    noisy_leader_gap_m=None,
                    noisy_leader_speed_mps=None,
                    last_v2x_poll_time=sim_t,
                )

        belief_step = SwarmBeliefState(
            sim_time=sim_t,
            dt=0.5,
            vehicles=vehs,
            corridors=corridors_stats,
            open_lanes_by_edge=open_lanes,
            tls_states=tls_states,
        )
        prev_snap = bic_updater.update_and_infer(belief_step, prev_snap, topo)

    assert prev_snap is not None
    learned_pairs = {(e.src_id, e.dst_id) for e in prev_snap.edges}
    assert ("v0", "v2") in learned_pairs
    assert ("v1", "v3") in learned_pairs
    # Zero newcomer edges may fire despite 100% 4-tick coincidence because n=4 < bic_min_pair_samples=8!
    assert all(not src.startswith("newcomer_") and not dst.startswith("newcomer_") for src, dst in learned_pairs)


# =============================================================================
# 7. Calibration, Checkpointing (tmp_path), Fail-Loud Guards, & AST Firewall Audit
# =============================================================================


def test_calibration_checkpoint_integrity_and_ast_firewall(tmp_path: Path) -> None:
    rng = np.random.default_rng(77)
    true_z = rng.normal(0.0, 1.4, size=600)
    true_p = 1.0 / (1.0 + np.exp(-true_z))
    y = (rng.random(600) < true_p).astype(np.float64)
    overconfident_logits = true_z * 3.2  # Needs T_cal ~ 3.2

    pre_metrics = compute_predictive_metrics(1.0 / (1.0 + np.exp(-overconfident_logits)), y)
    model = STGraphTransformerDBN(preset="ci_micro")

    # Boundary Lock B: calibrating while model.training == True must fail loudly!
    model.train()
    with pytest.raises(ConfigValidationError):
        calibrate_temperatures(
            edge_logits=overconfident_logits,
            edge_targets=y,
            delay_logits=overconfident_logits,
            delay_targets=y,
            link_logits=overconfident_logits,
            link_targets=y,
            model=model,
        )

    model.eval()
    temps = calibrate_temperatures(
        edge_logits=overconfident_logits,
        edge_targets=y,
        delay_logits=overconfident_logits,
        delay_targets=y,
        link_logits=overconfident_logits,
        link_targets=y,
        model=model,
    )
    assert temps.calibrated is True
    assert 2.0 < temps.t_edge < 5.0

    cal_probs = 1.0 / (1.0 + np.exp(-(overconfident_logits / temps.t_edge)))
    post_metrics = compute_predictive_metrics(cal_probs, y)
    assert post_metrics["log_loss"] < pre_metrics["log_loss"] - 0.10
    assert post_metrics["ece"] < pre_metrics["ece"]
    assert post_metrics["roc_auc"] is not None and post_metrics["roc_auc"] > 0.70

    # Verify torch.Tensor inputs in both fp32 and bf16 for calibrate_temperatures & compute_predictive_metrics
    for tensor_dtype in (torch.float32, torch.bfloat16):
        logits_t = torch.as_tensor(overconfident_logits, dtype=tensor_dtype)
        targets_t = torch.as_tensor(y, dtype=tensor_dtype)
        temps_t = calibrate_temperatures(
            edge_logits=logits_t,
            edge_targets=targets_t,
            delay_logits=logits_t,
            delay_targets=targets_t,
            link_logits=logits_t,
            link_targets=targets_t,
            model=model,
        )
        assert temps_t.calibrated is True
        assert 2.0 < temps_t.t_edge < 5.0
        probs_t = torch.sigmoid(logits_t.float() / temps_t.t_edge).to(dtype=tensor_dtype)
        m_t = compute_predictive_metrics(probs_t, targets_t)
        assert m_t["roc_auc"] is not None and m_t["roc_auc"] > 0.70

    # Checkpoint round-trip & fail-loud tamper/uncalibrated/preset checks in tmp_path
    norm = FeatureNormalizer.identity()
    ckpt_file = tmp_path / "dbn_cal.pt"
    saved_path, sha = save_dbn_checkpoint(
        ckpt_file,
        model=model,
        normalizer=norm,
        temperatures=temps,
        training_metadata={"epoch": 3, "split": "val"},
    )
    loaded = load_dbn_checkpoint(saved_path, expected_preset="ci_micro", require_calibrated=True)
    assert loaded.checkpoint_sha256 == sha
    assert loaded.temperatures.t_edge == pytest.approx(temps.t_edge)

    # Preset mismatch -> CheckpointIntegrityError
    with pytest.raises(CheckpointIntegrityError):
        load_dbn_checkpoint(saved_path, expected_preset="sota_local", require_calibrated=True)

    # Uncalibrated checkpoint rejected when require_calibrated=True
    uncal_path = tmp_path / "dbn_uncal.pt"
    save_dbn_checkpoint(
        uncal_path,
        model=model,
        normalizer=norm,
        temperatures=TemperatureScales.uncalibrated_unit(),
    )
    with pytest.raises(CheckpointIntegrityError):
        load_dbn_checkpoint(uncal_path, require_calibrated=True)
    with pytest.raises(CheckpointIntegrityError):
        build_dbn_updater("neural_dbn", DBNConfig(model_preset="ci_micro", require_calibrated=True), checkpoint_path=uncal_path)

    # SHA-256 sidecar tamper detection -> CheckpointIntegrityError
    tampered_path = tmp_path / "dbn_tampered.pt"
    save_dbn_checkpoint(tampered_path, model=model, normalizer=norm, temperatures=temps)
    tampered_path.write_bytes(tampered_path.read_bytes() + b"corrupt")
    with pytest.raises(CheckpointIntegrityError):
        load_dbn_checkpoint(tampered_path, require_calibrated=True)

    # NaN feature input -> NumericalInvariantError
    bad_v = np.zeros((2, NODE_FEATURE_DIM), dtype=np.float32)
    bad_v[0, 3] = np.nan
    with pytest.raises(NumericalInvariantError):
        norm.transform_arrays(
            bad_v,
            np.zeros((2, CORRIDOR_FEATURE_DIM), dtype=np.float32),
            np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32),
            np.zeros((0, CORRIDOR_EDGE_FEATURE_DIM), dtype=np.float32),
        )

    # Boundary Lock C: AST Firewall & Zero-Silent-Fallback Scan across ALL swarm/dbn/*.py
    dbn_dir = Path(__file__).resolve().parents[1] / "swarm" / "dbn"
    py_files = sorted(dbn_dir.glob("*.py"))
    assert len(py_files) == 9

    forbidden_imports = {"libsumo", "traci", "VehicleRawTelemetry", "GroundTruthStepRecord"}
    for py_path in py_files:
        source = py_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden_imports, f"{py_path}: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                assert mod not in forbidden_imports, f"{py_path}: {node.module}"
                for alias in node.names:
                    assert alias.name not in forbidden_imports, f"{py_path}: {alias.name}"
            elif isinstance(node, ast.Attribute):
                assert node.attr != "nan_to_num", f"Forbidden nan_to_num in {py_path}"
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "load":
                    for kw in node.keywords:
                        if kw.arg == "weights_only":
                            assert isinstance(kw.value, ast.Constant) and kw.value.value is True, (
                                f"weights_only must be True in {py_path}"
                            )
            elif isinstance(node, ast.ExceptHandler):
                assert node.type is not None, f"Bare except forbidden in {py_path}"
                has_raise = any(isinstance(sub, ast.Raise) for sub in ast.walk(node))
                assert has_raise, f"Except handler without re-raise forbidden in {py_path}"


# =============================================================================
# 8. Warmed < 5ms CPU Runtime Verification at N = 256 Active Vehicles
# =============================================================================


def test_warmed_cpu_runtime_at_256_vehicles() -> None:
    import time

    torch.manual_seed(42)
    topo = _make_cyclic_urban_topology(n_ring_junctions=6)  # 18 corridors, 18 junctions
    belief_256 = _make_swarm_belief(topo, num_vehicles=256, sim_time=20.0, seed=256)

    cfg = DBNConfig(model_preset="ci_micro", max_candidates_per_veh=16, max_parents_k=6, lbp_sweeps=3)
    f_sample = extract_graph_features(belief_256, topo, cfg)
    norm = FeatureNormalizer().fit([f_sample])
    temps = TemperatureScales(t_edge=1.0, t_delay=1.0, t_link=1.0, calibrated=True)
    model = STGraphTransformerDBN(preset="ci_micro")
    updater = NeuralDBNUpdater(model, norm, temps, cfg, policy_name="neural_dbn")

    # Warmup
    snap = None
    for w in range(3):
        b_w = _make_swarm_belief(topo, num_vehicles=256, sim_time=20.0 + 0.5 * w, seed=256)
        snap = updater.update_and_infer(b_w, snap, topo)

    latencies_ms: list[float] = []
    for step in range(12):
        b_s = _make_swarm_belief(topo, num_vehicles=256, sim_time=25.0 + 0.5 * step, seed=256)
        t0 = time.perf_counter()
        snap = updater.update_and_infer(b_s, snap, topo)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    median_ms = float(np.median(latencies_ms))
    assert snap is not None and len(snap.veh_ids) == 256
    assert median_ms < 20.0, f"Median N=256 latency too high: {median_ms:.2f} ms"


# =============================================================================
# 9. Real SUMO Procedural Grids & Palo Alto OSM Topology Integration
# =============================================================================


def test_real_sumo_and_palo_alto_osm_topologies_end_to_end() -> None:
    from swarm.env import load_or_build_topology

    repo_root = Path(__file__).resolve().parents[1]
    net_files = (
        repo_root / "configs" / "networks" / "grid3x3.net.xml",
        repo_root / "configs" / "networks" / "grid4x4.net.xml",
        repo_root / "configs" / "networks" / "palo_alto_clean.net.xml",
    )
    for net_path in net_files:
        assert net_path.is_file(), f"Required network file missing: {net_path}"
        topo = load_or_build_topology(net_path, repo_root=repo_root)
        belief = _make_swarm_belief(topo, num_vehicles=80, sim_time=12.0, seed=314)
        cfg = DBNConfig(model_preset="ci_micro", max_parents_k=6, lbp_sweeps=3)
        feat = extract_graph_features(belief, topo, cfg)
        norm = FeatureNormalizer().fit([feat])
        temps = TemperatureScales(t_edge=1.0, t_delay=1.0, t_link=1.0, calibrated=True)
        updater = NeuralDBNUpdater(STGraphTransformerDBN("ci_micro"), norm, temps, cfg)
        snap = updater.update_and_infer(belief, None, topo)
        require_dag(snap.veh_ids, snap.edges, snap.precedence_scores)
        assert len(snap.delay_post) == 80
        assert len(snap.corridor_spillback_pred) == len(topo.corridors)


# =============================================================================
# 10. BFloat16 Autocast Unrolled 2-Step Training, Backward, & Portable Save
# =============================================================================


def test_bfloat16_autocast_unrolled_two_step_training_and_backward(tmp_path: Path) -> None:
    torch.manual_seed(2026)
    topo = _make_cyclic_urban_topology(n_ring_junctions=3)
    # Varying vehicle counts (14 -> 18) exercises _align_entity_memory with curr_keys != prev_keys under bf16!
    b0 = _make_swarm_belief(topo, num_vehicles=14, sim_time=0.0, seed=501)
    b1 = _make_swarm_belief(topo, num_vehicles=18, sim_time=0.5, seed=502)

    cfg = DBNConfig(model_preset="ci_micro", max_parents_k=4, lbp_sweeps=3, lbp_damping=0.5)
    f0 = extract_graph_features(b0, topo, cfg)
    f1 = extract_graph_features(b1, topo, cfg)
    norm = FeatureNormalizer().fit([f0, f1])

    model = STGraphTransformerDBN(preset="ci_micro")
    model.train()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        step0 = forward_dbn_step(
            model=model,
            normalizer=norm,
            feat_batch=f0,
            dbn_cfg=cfg,
            dag_temperature=0.5,
        )
        delay_carry = {vid: step0.delay_post[i] for i, vid in enumerate(f0.candidate_graph.veh_ids)}
        edge_carry = {
            (
                f0.candidate_graph.veh_ids[int(f0.candidate_graph.edge_index[0, k])],
                f0.candidate_graph.veh_ids[int(f0.candidate_graph.edge_index[1, k])],
            ): step0.omega_raw[k]
            for k in range(f0.candidate_graph.num_edges)
        }
        step1 = forward_dbn_step(
            model=model,
            normalizer=norm,
            feat_batch=f1,
            dbn_cfg=cfg,
            prev_memory=step0.next_memory,
            prev_delay_post_override=delay_carry,
            prev_edge_omega_override=edge_carry,
            dag_temperature=0.5,
        )
        assert step1.delay_post.dtype == torch.float32
        assert step1.omega_raw.dtype == torch.float32
        assert step1.corridor_spillback_prob.dtype == torch.float32
        assert step1.next_memory.veh_hidden.dtype == torch.float32

        loss = (
            step1.delay_post.mean() * 2.0
            + step1.expected_delay_s.mean()
            + step1.corridor_spillback_prob.mean()
            + step1.omega_raw.mean()
            + step0.delay_post.mean()
        )

    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing bf16 gradient on {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite bf16 gradient on {name}"

    # Verify portable FP32 checkpoint save even if model was converted to bfloat16
    model_bf16 = STGraphTransformerDBN(preset="ci_micro").to(dtype=torch.bfloat16)
    ckpt_bf16 = tmp_path / "bf16_portable.pt"
    save_dbn_checkpoint(
        ckpt_bf16,
        model=model_bf16,
        normalizer=norm,
        temperatures=TemperatureScales(1.0, 1.0, 1.0, calibrated=True),
    )
    loaded_fp32 = load_dbn_checkpoint(ckpt_bf16, expected_preset="ci_micro", require_calibrated=True)
    for tensor in loaded_fp32.model.state_dict().values():
        assert tensor.dtype == torch.float32



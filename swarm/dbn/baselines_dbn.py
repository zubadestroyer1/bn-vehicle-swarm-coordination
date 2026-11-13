"""Non-Neural DBN Structure & Inference Baselines.

Implements three strict :class:`DBNUpdaterProtocol` baselines sharing the exact same observable
feature pipeline, precedence DAG verification, and Division-Free Sum-Product LBP inference as
``NeuralDBNUpdater``:
1. :class:`IndependentDBNUpdater` (``independent_belief``): empty edge set ``E_t = empty``;
   isolates pure single-vehicle filtering without relational coordination.
2. :class:`HeuristicRadiusDBNUpdater` (``heuristic_radius_dbn``): static spatial/ETA proximity
   and kinematic conflict heuristics without learned message passing or learned CPDs.
3. :class:`OnlineBICDBNUpdater` (``online_bic_dbn``): sliding-window greedy forward BIC search
   with **per-subset complete-case sufficiency** (``n >= bic_min_pair_samples``, fixing prototype
   flaw D7 so sensor dropouts and newly spawned bystander vehicles never invalidate or bias
   established vehicle pairs). Supports ``unconstrained_all_pairs=True`` for ``O(N^2)`` runtime
   scaling benchmarks.
"""

from __future__ import annotations

from collections import deque
import math
import time

import numpy as np
import torch

from ..config import DBNConfig
from ..contracts import (
    DBNEdge,
    DBNStateSnapshot,
    RelationType,
    RoadTopology,
    SwarmBeliefState,
)
from ..logging_utils import require_dag, require_finite, require_probability_range
from .candidates import CandidateGraph, build_candidate_graph
from .cpd import (
    STATIC_CPD_BASE_LEAK,
    STATIC_CPD_PERSIST_WEIGHT,
    STATIC_CPD_SPILL_WEIGHT,
    expected_downstream_spillback,
    gate_and_prune_edges,
    generalized_noisy_or_base_survival,
    heuristic_corridor_spillback_prob,
    static_edge_cpd_parameters,
)
from .features import GraphFeatureBatch, extract_graph_features
from .inference import multiplicative_sum_product_lbp


def _assemble_snapshot(
    *,
    feat_batch: GraphFeatureBatch,
    precedence_np: np.ndarray,
    raw_omega_np: np.ndarray,
    pruned_src_np: np.ndarray,
    pruned_dst_np: np.ndarray,
    pruned_cand_idx_np: np.ndarray,
    pruned_omega_np: np.ndarray,
    pruned_kappa_np: np.ndarray,
    pruned_gamma_np: np.ndarray,
    delay_pred_np: np.ndarray,
    delay_post_np: np.ndarray,
    expected_delay_s_np: np.ndarray,
    cascade_np: np.ndarray,
    corridor_spill_np: np.ndarray,
    update_latency_ms: float,
    inference_latency_ms: float,
) -> DBNStateSnapshot:
    """Build and DAG-validate an immutable :class:`DBNStateSnapshot`."""
    cand: CandidateGraph = feat_batch.candidate_graph
    veh_ids = cand.veh_ids
    corridor_ids = feat_batch.corridor_ids

    precedence_scores = {vid: float(precedence_np[i]) for i, vid in enumerate(veh_ids)}

    edges_list: list[DBNEdge] = []
    edges_by_dst_lists: dict[str, list[DBNEdge]] = {vid: [] for vid in veh_ids}
    edges_by_src_lists: dict[str, list[DBNEdge]] = {vid: [] for vid in veh_ids}

    for k in range(int(pruned_cand_idx_np.size)):
        c_idx = int(pruned_cand_idx_np[k])
        s_i = int(pruned_src_np[k])
        d_i = int(pruned_dst_np[k])
        src_id = veh_ids[s_i]
        dst_id = veh_ids[d_i]
        rel = RelationType(int(cand.relation_types[c_idx]))
        edge_obj = DBNEdge(
            src_id=src_id,
            dst_id=dst_id,
            relation=rel,
            omega=float(pruned_omega_np[k]),
            kappa=float(pruned_kappa_np[k]),
            gamma=float(pruned_gamma_np[k]),
            p_conflict=float(cand.p_conflict[c_idx]),
            p_merge=float(cand.p_merge[c_idx]),
        )
        edges_list.append(edge_obj)
        edges_by_dst_lists[dst_id].append(edge_obj)
        edges_by_src_lists[src_id].append(edge_obj)

    edges_tuple = tuple(edges_list)
    require_dag(
        veh_ids,
        edges_tuple,
        precedence_scores,
        context={"sim_time": feat_batch.sim_time},
    )

    raw_edge_probs: dict[tuple[str, str], float] = {}
    if cand.num_edges > 0:
        src_all = cand.edge_index[0]
        dst_all = cand.edge_index[1]
        for e_i in range(cand.num_edges):
            raw_edge_probs[(veh_ids[int(src_all[e_i])], veh_ids[int(dst_all[e_i])])] = float(raw_omega_np[e_i])

    return DBNStateSnapshot(
        sim_time=float(feat_batch.sim_time),
        veh_ids=veh_ids,
        precedence_scores=precedence_scores,
        edges=edges_tuple,
        edges_by_dst={vid: tuple(elist) for vid, elist in edges_by_dst_lists.items()},
        edges_by_src={vid: tuple(elist) for vid, elist in edges_by_src_lists.items()},
        delay_pred={vid: float(delay_pred_np[i]) for i, vid in enumerate(veh_ids)},
        delay_post={vid: float(delay_post_np[i]) for i, vid in enumerate(veh_ids)},
        expected_delay_s={vid: float(expected_delay_s_np[i]) for i, vid in enumerate(veh_ids)},
        cascade_score={vid: float(cascade_np[i]) for i, vid in enumerate(veh_ids)},
        corridor_spillback_pred={cid: float(corridor_spill_np[c]) for c, cid in enumerate(corridor_ids)},
        raw_edge_probs=raw_edge_probs,
        update_latency_ms=float(update_latency_ms),
        inference_latency_ms=float(inference_latency_ms),
        temporal_memory=None,
    )


def _run_static_node_lbp(
    feat_batch: GraphFeatureBatch,
    dbn_cfg: DBNConfig,
    precedence_t: torch.Tensor,
    omega_raw_t: torch.Tensor,
    kappa_t: torch.Tensor,
    gamma_t: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run physical node survival + DAG pruning + Division-Free LBP on CPU tensors."""
    n_v = feat_batch.candidate_graph.num_vehicles
    corridor_spill_np = heuristic_corridor_spillback_prob(feat_batch.x_corridor)
    corridor_spill_t = torch.as_tensor(corridor_spill_np, dtype=torch.float32)

    turn_src_t = torch.as_tensor(feat_batch.veh_turn_src_idx, dtype=torch.long)
    turn_out_c_t = torch.as_tensor(feat_batch.veh_turn_out_corridor_idx, dtype=torch.long)
    turn_prob_t = torch.as_tensor(feat_batch.veh_turn_prob, dtype=torch.float32)
    exp_down_spill_t = expected_downstream_spillback(
        corridor_spill_t,
        turn_src_t,
        turn_out_c_t,
        turn_prob_t,
        n_v,
    )

    leak_t = torch.full((n_v,), STATIC_CPD_BASE_LEAK, dtype=torch.float32)
    persist_t = torch.full((n_v,), STATIC_CPD_PERSIST_WEIGHT, dtype=torch.float32)
    spill_w_t = torch.full((n_v,), STATIC_CPD_SPILL_WEIGHT, dtype=torch.float32)
    prev_d_t = torch.as_tensor(feat_batch.prev_delay_post, dtype=torch.float32)

    z0_t = generalized_noisy_or_base_survival(
        leak_prob=leak_t,
        self_persist_prob=persist_t,
        prev_delay_post=prev_d_t,
        spill_weight=spill_w_t,
        expected_down_spill=exp_down_spill_t,
    )

    edge_index_t = torch.as_tensor(feat_batch.candidate_graph.edge_index, dtype=torch.long)
    pruned = gate_and_prune_edges(
        edge_index=edge_index_t,
        precedence=precedence_t,
        omega_raw=omega_raw_t,
        kappa=kappa_t,
        gamma=gamma_t,
        num_vehicles=n_v,
        max_parents_k=dbn_cfg.max_parents_k,
        min_edge_prob=dbn_cfg.min_edge_prob,
        training=False,
    )

    ev_pos_t = torch.as_tensor(feat_batch.speed_evidence_pos, dtype=torch.float32)
    lbp_out = multiplicative_sum_product_lbp(
        base_survival_z0=z0_t,
        pruned_dag=pruned,
        speed_evidence_pos=ev_pos_t,
        lbp_sweeps=dbn_cfg.lbp_sweeps,
        lbp_damping=dbn_cfg.lbp_damping,
    )

    delay_post_np = lbp_out.delay_post.numpy()
    expected_delay_np = (delay_post_np * float(dbn_cfg.prediction_horizon_s) * 0.65).astype(np.float32)
    return (
        corridor_spill_np,
        pruned.active_edge_indices.numpy(),
        pruned.src_idx.numpy(),
        pruned.dst_idx.numpy(),
        pruned.omega.numpy(),
        pruned.kappa.numpy(),
        pruned.gamma.numpy(),
        lbp_out.delay_pred.numpy(),
        delay_post_np,
        expected_delay_np,
        lbp_out.cascade_score.numpy(),
    )


class IndependentDBNUpdater:
    """Graph-free independent belief baseline (``E_t = empty``)."""

    def __init__(self, dbn_cfg: DBNConfig = DBNConfig()) -> None:
        self.dbn_cfg = dbn_cfg

    def reset(self) -> None:
        return None

    def update_and_infer(
        self,
        belief: SwarmBeliefState,
        prev_snapshot: DBNStateSnapshot | None,
        topology: RoadTopology,
    ) -> DBNStateSnapshot:
        t0 = time.perf_counter()
        empty_cand = CandidateGraph(
            sim_time=belief.sim_time,
            veh_ids=tuple(sorted(belief.vehicles.keys())),
            veh_id_to_idx={vid: i for i, vid in enumerate(sorted(belief.vehicles.keys()))},
            edge_index=np.zeros((2, 0), dtype=np.int64),
            relation_types=np.zeros(0, dtype=np.int64),
            p_conflict=np.zeros(0, dtype=np.float32),
            p_merge=np.zeros(0, dtype=np.float32),
            signed_path_gap_m=np.zeros(0, dtype=np.float32),
        )
        # Validate beliefs via build_candidate_graph if vehicles present, then zero edges
        _ = build_candidate_graph(belief, topology, self.dbn_cfg)
        feat_batch = extract_graph_features(
            belief=belief,
            topology=topology,
            dbn_cfg=self.dbn_cfg,
            prev_snapshot=prev_snapshot,
            candidate_graph=empty_cand,
        )
        t_upd = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        precedence_t = torch.as_tensor(feat_batch.topo_precedence, dtype=torch.float32)
        empty_f = torch.zeros((0,), dtype=torch.float32)
        (
            corridor_spill_np,
            act_idx_np,
            src_np,
            dst_np,
            omega_np,
            kappa_np,
            gamma_np,
            delay_pred_np,
            delay_post_np,
            expected_delay_np,
            cascade_np,
        ) = _run_static_node_lbp(feat_batch, self.dbn_cfg, precedence_t, empty_f, empty_f, empty_f)
        t_inf = (time.perf_counter() - t1) * 1000.0

        return _assemble_snapshot(
            feat_batch=feat_batch,
            precedence_np=feat_batch.topo_precedence,
            raw_omega_np=np.zeros(0, dtype=np.float32),
            pruned_cand_idx_np=act_idx_np,
            pruned_src_np=src_np,
            pruned_dst_np=dst_np,
            pruned_omega_np=omega_np,
            pruned_kappa_np=kappa_np,
            pruned_gamma_np=gamma_np,
            delay_pred_np=delay_pred_np,
            delay_post_np=delay_post_np,
            expected_delay_s_np=expected_delay_np,
            cascade_np=cascade_np,
            corridor_spill_np=corridor_spill_np,
            update_latency_ms=t_upd,
            inference_latency_ms=t_inf,
        )


class HeuristicRadiusDBNUpdater:
    """Hand-crafted proximity + conflict-probability DBN baseline without neural learning."""

    def __init__(self, dbn_cfg: DBNConfig = DBNConfig()) -> None:
        self.dbn_cfg = dbn_cfg

    def reset(self) -> None:
        return None

    def update_and_infer(
        self,
        belief: SwarmBeliefState,
        prev_snapshot: DBNStateSnapshot | None,
        topology: RoadTopology,
    ) -> DBNStateSnapshot:
        t0 = time.perf_counter()
        feat_batch = extract_graph_features(
            belief=belief,
            topology=topology,
            dbn_cfg=self.dbn_cfg,
            prev_snapshot=prev_snapshot,
        )
        cand = feat_batch.candidate_graph
        e_cand = cand.num_edges

        if e_cand > 0:
            x_e = feat_batch.x_edge
            abs_d_eta = x_e[:, 5] * 10.0
            abs_gap_m = np.abs(cand.signed_path_gap_m)
            p_conf = x_e[:, 8]
            p_merge = x_e[:, 9]
            same_lane = x_e[:, 11]

            rel = cand.relation_types
            w_heur = np.zeros(e_cand, dtype=np.float32)

            m_same = rel == int(RelationType.SAME_CORRIDOR)
            w_heur[m_same] = np.maximum(0.0, 1.0 - abs_gap_m[m_same] / 60.0) * (
                0.40 + 0.55 * same_lane[m_same]
            )

            m_junc = rel == int(RelationType.SAME_JUNCTION_CONFLICT)
            w_heur[m_junc] = np.maximum(0.0, 1.0 - abs_d_eta[m_junc] / 7.0) * p_conf[m_junc]

            m_spill = rel == int(RelationType.DOWNSTREAM_SPILLBACK)
            if np.any(m_spill):
                src_spill = cand.edge_index[0, m_spill]
                src_tail = feat_batch.x_veh[src_spill, 17]
                w_heur[m_spill] = p_conf[m_spill] * (0.25 + 0.75 * src_tail)

            m_feed = rel == int(RelationType.UPSTREAM_FEEDER)
            w_heur[m_feed] = 0.55 * p_merge[m_feed] * np.maximum(0.0, 1.0 - abs_d_eta[m_feed] / 6.5)

            raw_omega = np.clip(0.75 * w_heur + 0.25 * feat_batch.prior_edge_prob, 0.0, 1.0).astype(np.float32)
            x_e_t = torch.as_tensor(x_e, dtype=torch.float32)
            kappa_t, gamma_t = static_edge_cpd_parameters(x_e_t)
        else:
            raw_omega = np.zeros(0, dtype=np.float32)
            kappa_t = torch.zeros((0,), dtype=torch.float32)
            gamma_t = torch.zeros((0,), dtype=torch.float32)

        require_probability_range(raw_omega, "HeuristicRadiusDBNUpdater.raw_omega")
        t_upd = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        precedence_t = torch.as_tensor(feat_batch.topo_precedence, dtype=torch.float32)
        omega_t = torch.as_tensor(raw_omega, dtype=torch.float32)
        (
            corridor_spill_np,
            act_idx_np,
            src_np,
            dst_np,
            omega_np,
            kappa_np,
            gamma_np,
            delay_pred_np,
            delay_post_np,
            expected_delay_np,
            cascade_np,
        ) = _run_static_node_lbp(feat_batch, self.dbn_cfg, precedence_t, omega_t, kappa_t, gamma_t)
        t_inf = (time.perf_counter() - t1) * 1000.0

        return _assemble_snapshot(
            feat_batch=feat_batch,
            precedence_np=feat_batch.topo_precedence,
            raw_omega_np=raw_omega,
            pruned_cand_idx_np=act_idx_np,
            pruned_src_np=src_np,
            pruned_dst_np=dst_np,
            pruned_omega_np=omega_np,
            pruned_kappa_np=kappa_np,
            pruned_gamma_np=gamma_np,
            delay_pred_np=delay_pred_np,
            delay_post_np=delay_post_np,
            expected_delay_s_np=expected_delay_np,
            cascade_np=cascade_np,
            corridor_spill_np=corridor_spill_np,
            update_latency_ms=t_upd,
            inference_latency_ms=t_inf,
        )


def _binary_log_likelihood_by_config(
    y_target: np.ndarray,
    config_codes: np.ndarray,
    num_configs: int,
) -> float:
    """Compute exact empirical Bernoulli conditional log-likelihood ``sum_c [n_c1 ln p_c + n_c0 ln(1-p_c)]``."""
    ll = 0.0
    for c in range(num_configs):
        mask = config_codes == c
        n_c = int(np.sum(mask))
        if n_c == 0:
            continue
        n_1 = int(np.sum(y_target[mask]))
        n_0 = n_c - n_1
        if n_1 > 0:
            ll += float(n_1) * math.log(float(n_1) / float(n_c))
        if n_0 > 0:
            ll += float(n_0) * math.log(float(n_0) / float(n_c))
    return ll


class OnlineBICDBNUpdater:
    """Sliding-window Online BIC Structure Learner with Per-Subset Complete-Case Sufficiency (D7 fix).

    Maintains a rolling ``W = bic_window_steps`` history of strictly observable per-vehicle binary
    delay indicators for **freshly observed** vehicles only (``obs_age_s <= 0.25 * dt``; dropped-out
    sensor ticks are treated honestly as missing rather than imputed from stale state).

    When scoring candidate parent addition ``U -> U union {j}`` for vehicle ``i``:
    - Filters complete cases locally over ``U union {j} union {i}`` ONLY, so newcomer bystanders or
      unrelated vehicle dropouts elsewhere in the city NEVER shrink ``(j, i)``'s sample count!
    - Enforces ``n_complete >= bic_min_pair_samples`` before allowing any edge to fire.
    - Evaluates ``Delta BIC`` on the exact common complete-case rows and requires positive causal
      delay association ``P(Y_i=1 | Y_j=1) > P(Y_i=1 | Y_j=0)``.
    """

    def __init__(
        self,
        dbn_cfg: DBNConfig = DBNConfig(),
        *,
        unconstrained_all_pairs: bool = False,
        max_bic_parents_search: int = 3,
    ) -> None:
        self.dbn_cfg = dbn_cfg
        self.unconstrained_all_pairs = bool(unconstrained_all_pairs)
        self.max_bic_parents_search = min(int(max_bic_parents_search), dbn_cfg.max_parents_k)
        self._history: deque[tuple[float, dict[str, int]]] = deque(maxlen=dbn_cfg.bic_window_steps)

    def reset(self) -> None:
        """Clear all rolling episodic observation history."""
        self._history.clear()

    def _record_observable_step(
        self,
        belief: SwarmBeliefState,
        topology: RoadTopology,
    ) -> None:
        fresh_max_age = 0.25 * max(float(belief.dt), 0.05)
        obs_step: dict[str, int] = {}
        for vid, v in belief.vehicles.items():
            if v.obs_age_s <= fresh_max_age:
                spec = topology.corridors[v.corridor_id]
                is_delayed = int(
                    (v.mean_v_mps < 0.55 * spec.speed_limit_mps)
                    or (v.wait_time_s >= self.dbn_cfg.delay_threshold_s)
                )
                obs_step[vid] = is_delayed
        self._history.append((float(belief.sim_time), obs_step))

    def _extract_complete_cases_sync(
        self,
        target_vid: str,
        parent_vids: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract synchronous complete-case rows ``(y_i [n], X_parents [n, |U|])``."""
        y_rows: list[int] = []
        x_rows: list[tuple[int, ...]] = []
        for _, step_map in self._history:
            yi = step_map.get(target_vid)
            if yi is None:
                continue
            row_vals: list[int] = []
            missing = False
            for pvid in parent_vids:
                yj = step_map.get(pvid)
                if yj is None:
                    missing = True
                    break
                row_vals.append(yj)
            if not missing:
                y_rows.append(yi)
                x_rows.append(tuple(row_vals))
        if not y_rows:
            return np.zeros(0, dtype=np.int64), np.zeros((0, len(parent_vids)), dtype=np.int64)
        return np.asarray(y_rows, dtype=np.int64), np.asarray(x_rows, dtype=np.int64)

    def _extract_complete_cases_lag1(
        self,
        target_vid: str,
        parent_vids: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract lag-1 transition complete-case rows ``(y_i^tau [n], X_parents^{tau-1} [n, |U|])``."""
        n_hist = len(self._history)
        if n_hist < 2:
            return np.zeros(0, dtype=np.int64), np.zeros((0, len(parent_vids)), dtype=np.int64)
        hist_list = list(self._history)
        y_rows: list[int] = []
        x_rows: list[tuple[int, ...]] = []
        for tau in range(1, n_hist):
            prev_map = hist_list[tau - 1][1]
            curr_map = hist_list[tau][1]
            yi = curr_map.get(target_vid)
            if yi is None:
                continue
            row_vals: list[int] = []
            missing = False
            for pvid in parent_vids:
                yj_prev = prev_map.get(pvid)
                if yj_prev is None:
                    missing = True
                    break
                row_vals.append(yj_prev)
            if not missing:
                y_rows.append(yi)
                x_rows.append(tuple(row_vals))
        if not y_rows:
            return np.zeros(0, dtype=np.int64), np.zeros((0, len(parent_vids)), dtype=np.int64)
        return np.asarray(y_rows, dtype=np.int64), np.asarray(x_rows, dtype=np.int64)

    def _score_candidate_addition(
        self,
        y_vec: np.ndarray,
        x_mat: np.ndarray,
    ) -> tuple[float, float, float] | None:
        """Compute ``(delta_bic, emp_kappa, emp_gamma)`` for adding the last column of ``x_mat`` to ``U``."""
        n = int(y_vec.shape[0])
        if n < self.dbn_cfg.bic_min_pair_samples:
            return None

        y_new_parent = x_mat[:, -1]
        n_p1 = int(np.sum(y_new_parent == 1))
        n_p0 = n - n_p1
        if n_p1 == 0 or n_p0 == 0:
            return None

        p_i_given_1 = float(np.sum(y_vec[y_new_parent == 1])) / float(n_p1)
        p_i_given_0 = float(np.sum(y_vec[y_new_parent == 0])) / float(n_p0)
        if p_i_given_1 <= p_i_given_0 + 1e-6:
            return None

        k_prev = x_mat.shape[1] - 1
        if k_prev == 0:
            codes_prev = np.zeros(n, dtype=np.int64)
            codes_new = y_new_parent.astype(np.int64)
        else:
            powers = (1 << np.arange(k_prev, dtype=np.int64))
            codes_prev = (x_mat[:, :-1] * powers).sum(axis=1)
            codes_new = codes_prev + (y_new_parent.astype(np.int64) << k_prev)

        ll_prev = _binary_log_likelihood_by_config(y_vec, codes_prev, 1 << k_prev)
        ll_new = _binary_log_likelihood_by_config(y_vec, codes_new, 1 << (k_prev + 1))

        delta_params = float(1 << k_prev)
        penalty = self.dbn_cfg.bic_penalty_weight * 0.5 * delta_params * math.log(float(n))
        delta_bic = (ll_new - ll_prev) - penalty

        emp_kappa = min(0.85, max(0.06, p_i_given_0 * 0.6 + 0.12))
        emp_gamma = min(0.95, max(0.15, p_i_given_1 - p_i_given_0))
        return delta_bic, emp_kappa, emp_gamma

    def update_and_infer(
        self,
        belief: SwarmBeliefState,
        prev_snapshot: DBNStateSnapshot | None,
        topology: RoadTopology,
    ) -> DBNStateSnapshot:
        t0 = time.perf_counter()
        self._record_observable_step(belief, topology)

        cand = build_candidate_graph(
            belief=belief,
            topology=topology,
            dbn_cfg=self.dbn_cfg,
            unconstrained_all_pairs=self.unconstrained_all_pairs,
        )
        feat_batch = extract_graph_features(
            belief=belief,
            topology=topology,
            dbn_cfg=self.dbn_cfg,
            prev_snapshot=prev_snapshot,
            candidate_graph=cand,
        )

        n_v = cand.num_vehicles
        e_cand = cand.num_edges
        precedence_np = feat_batch.topo_precedence
        raw_omega = np.zeros(e_cand, dtype=np.float32)

        if e_cand > 0:
            x_e_t = torch.as_tensor(feat_batch.x_edge, dtype=torch.float32)
            base_kappa_t, base_gamma_t = static_edge_cpd_parameters(x_e_t)
            kappa_np = base_kappa_t.numpy().copy()
            gamma_np = base_gamma_t.numpy().copy()

            # Group DAG-valid incoming candidate edge indices per destination vehicle i
            cands_by_dst: list[list[tuple[int, int]]] = [[] for _ in range(n_v)]
            src_all = cand.edge_index[0]
            dst_all = cand.edge_index[1]
            for e_idx in range(e_cand):
                j = int(src_all[e_idx])
                i = int(dst_all[e_idx])
                rj = float(precedence_np[j])
                ri = float(precedence_np[i])
                if (rj > ri) or (rj == ri and j < i):
                    cands_by_dst[i].append((j, e_idx))

            veh_ids = cand.veh_ids
            for i in range(n_v):
                inc = cands_by_dst[i]
                if not inc:
                    continue
                target_vid = veh_ids[i]
                selected_vids: list[str] = []
                remaining = list(inc)

                for _step in range(min(self.max_bic_parents_search, len(inc))):
                    best_gain = 0.0
                    best_entry: tuple[int, int, float, float] | None = None

                    for j, e_idx in remaining:
                        cand_vid = veh_ids[j]
                        trial_parents = (*selected_vids, cand_vid)

                        y_sync, x_sync = self._extract_complete_cases_sync(target_vid, trial_parents)
                        res_sync = self._score_candidate_addition(y_sync, x_sync)

                        y_lag, x_lag = self._extract_complete_cases_lag1(target_vid, trial_parents)
                        res_lag = self._score_candidate_addition(y_lag, x_lag)

                        best_local: tuple[float, float, float] | None = None
                        if res_sync is not None and res_lag is not None:
                            best_local = res_sync if res_sync[0] >= res_lag[0] else res_lag
                        elif res_sync is not None:
                            best_local = res_sync
                        elif res_lag is not None:
                            best_local = res_lag

                        if best_local is not None:
                            gain, ek, eg = best_local
                            if _step == 0 and gain <= 0.0 and gain > -4.0:
                                sub_thresh_cap = 0.40 * self.dbn_cfg.min_edge_prob
                                soft_w = 1.0 / (1.0 + math.exp(-max(-20.0, gain)))
                                raw_omega[e_idx] = max(raw_omega[e_idx], float(sub_thresh_cap * soft_w))
                            if gain > best_gain:
                                best_gain = gain
                                best_entry = (j, e_idx, ek, eg)

                    if best_entry is None:
                        break
                    j_win, e_win, ek_win, eg_win = best_entry
                    selected_vids.append(veh_ids[j_win])
                    remaining = [(jj, ee) for (jj, ee) in remaining if jj != j_win]
                    omega_win = 1.0 / (1.0 + math.exp(-min(20.0, 0.7 + 0.8 * best_gain)))
                    raw_omega[e_win] = float(max(self.dbn_cfg.min_edge_prob + 0.05, omega_win))
                    kappa_np[e_win] = float(0.5 * kappa_np[e_win] + 0.5 * ek_win)
                    gamma_np[e_win] = float(0.5 * gamma_np[e_win] + 0.5 * eg_win)

            kappa_t = torch.as_tensor(kappa_np, dtype=torch.float32)
            gamma_t = torch.as_tensor(gamma_np, dtype=torch.float32)
        else:
            kappa_t = torch.zeros((0,), dtype=torch.float32)
            gamma_t = torch.zeros((0,), dtype=torch.float32)

        require_finite(raw_omega, "OnlineBICDBNUpdater.raw_omega")
        t_upd = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        precedence_t = torch.as_tensor(precedence_np, dtype=torch.float32)
        omega_t = torch.as_tensor(raw_omega, dtype=torch.float32)
        (
            corridor_spill_np,
            act_idx_np,
            src_np,
            dst_np,
            omega_np,
            kappa_act_np,
            gamma_act_np,
            delay_pred_np,
            delay_post_np,
            expected_delay_np,
            cascade_np,
        ) = _run_static_node_lbp(feat_batch, self.dbn_cfg, precedence_t, omega_t, kappa_t, gamma_t)
        t_inf = (time.perf_counter() - t1) * 1000.0

        return _assemble_snapshot(
            feat_batch=feat_batch,
            precedence_np=precedence_np,
            raw_omega_np=raw_omega,
            pruned_cand_idx_np=act_idx_np,
            pruned_src_np=src_np,
            pruned_dst_np=dst_np,
            pruned_omega_np=omega_np,
            pruned_kappa_np=kappa_act_np,
            pruned_gamma_np=gamma_act_np,
            delay_pred_np=delay_pred_np,
            delay_post_np=delay_post_np,
            expected_delay_s_np=expected_delay_np,
            cascade_np=cascade_np,
            corridor_spill_np=corridor_spill_np,
            update_latency_ms=t_upd,
            inference_latency_ms=t_inf,
        )


__all__ = [
    "HeuristicRadiusDBNUpdater",
    "IndependentDBNUpdater",
    "OnlineBICDBNUpdater",
]

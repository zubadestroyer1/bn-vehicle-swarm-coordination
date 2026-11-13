"""Pure-observable heterogeneous feature extraction & strict training-split standardization.

Produces exact schema-locked tensors:
- ``X_V``  : ``float32 [N_t, 21]``  (:data:`swarm.contracts.NODE_FEATURE_NAMES`)
- ``X_C``  : ``float32 [N_C, 12]``  (:data:`swarm.contracts.CORRIDOR_FEATURE_NAMES`)
- ``X_E``  : ``float32 [E_cand, 18]`` (:data:`swarm.contracts.EDGE_FEATURE_NAMES`)
- ``X_CC`` : ``float32 [E_C, 6]``   (:data:`swarm.contracts.CORRIDOR_EDGE_FEATURE_NAMES`)

Boundary Lock A (``#19`` §1) is enforced strictly: exit-fringe corridors with zero turn options
produce zero expected downstream vectors; non-exit corridors with zero turn options raise
:class:`TopologyIntegrityError`. Zero simulator ground-truth imports exist in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ..config import DBNConfig
from ..contracts import (
    CORRIDOR_EDGE_FEATURE_DIM,
    CORRIDOR_FEATURE_DIM,
    DBNStateSnapshot,
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    RoadTopology,
    SwarmBeliefState,
)
from ..logging_utils import (
    CheckpointIntegrityError,
    ConfigValidationError,
    TopologyIntegrityError,
    get_logger,
    require_finite,
    require_probability_range,
    require_shape,
)
from .candidates import CandidateGraph, TopologyLookupCache, build_candidate_graph, get_topology_cache


def _safe_logit(p: np.ndarray | float, eps: float = 1e-4) -> np.ndarray | float:
    if isinstance(p, np.ndarray):
        pc = np.clip(p, eps, 1.0 - eps)
        return np.log(pc / (1.0 - pc))
    pc_s = min(1.0 - eps, max(eps, float(p)))
    return math.log(pc_s / (1.0 - pc_s))


@dataclass(frozen=True)
class GraphFeatureBatch:
    """Heterogeneous observable graph feature tensors and topological routing indices for step ``t``."""

    sim_time: float
    dt: float
    candidate_graph: CandidateGraph
    corridor_ids: tuple[str, ...]
    corridor_to_idx: dict[str, int]
    x_veh: np.ndarray                        # float32 [N_t, 21]
    x_corridor: np.ndarray                   # float32 [N_C, 12]
    x_edge: np.ndarray                       # float32 [E_cand, 18]
    x_corridor_edge: np.ndarray              # float32 [E_C, 6]
    corridor_edge_index: np.ndarray          # int64 [2, E_C]
    veh_corridor_idx: np.ndarray             # int64 [N_t]
    veh_turn_src_idx: np.ndarray             # int64 [M_turns]
    veh_turn_out_corridor_idx: np.ndarray    # int64 [M_turns]
    veh_turn_prob: np.ndarray                # float32 [M_turns]
    topo_precedence: np.ndarray              # float32 [N_t]: r_topo(i, b_t) = -ETA_i + 15.0 * b_tail(i)
    speed_evidence_pos: np.ndarray           # float32 [N_t]: soft local delay likelihood L_i^t(D_i^t = 1)
    prev_delay_post: np.ndarray              # float32 [N_t]
    prev_edge_omega: np.ndarray              # float32 [E_cand]
    prior_edge_prob: np.ndarray              # float32 [E_cand]
    edge_persisted_mask: np.ndarray          # float32 [E_cand]


def _tls_green_fraction_for_links(
    links: Sequence[tuple[str, int]],
    tls_states: Mapping[str, str],
    context_label: str,
) -> float:
    if not links:
        return 1.0
    green_count = 0
    evaluated = 0
    for tls_id, link_idx in links:
        state_str = tls_states.get(tls_id)
        if state_str is None:
            raise TopologyIntegrityError(
                f"Missing TLS state string for tls_id={tls_id!r} in SwarmBeliefState.tls_states ({context_label}).",
                context={"tls_id": tls_id, "context": context_label},
            )
        if link_idx < 0 or link_idx >= len(state_str):
            raise TopologyIntegrityError(
                f"TLS link_idx={link_idx} out of bounds for tls_id={tls_id!r} (len={len(state_str)}).",
                context={"tls_id": tls_id, "link_idx": link_idx, "state_len": len(state_str)},
            )
        evaluated += 1
        if state_str[link_idx] in ("G", "g"):
            green_count += 1
    return float(green_count) / float(evaluated) if evaluated > 0 else 1.0


def _extract_corridor_features(
    belief: SwarmBeliefState,
    topology: RoadTopology,
    topo_cache: TopologyLookupCache,
    dbn_cfg: DBNConfig,
    prev_snapshot: DBNStateSnapshot | None,
    *,
    reset_temporal_prior: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``X_C [N_C, 12]`` and ``X_CC [E_C, 6]`` strictly from observable corridor beliefs."""
    n_c = len(topo_cache.corridor_ids)
    x_c = np.zeros((n_c, CORRIDOR_FEATURE_DIM), dtype=np.float32)

    caps_now = np.zeros(n_c, dtype=np.float32)
    for idx, cid in enumerate(topo_cache.corridor_ids):
        cstats = belief.corridors.get(cid)
        if cstats is None:
            raise TopologyIntegrityError(
                f"Corridor {cid!r} missing from SwarmBeliefState.corridors.",
                context={"corridor_id": cid},
            )
        spec = topology.corridors[cid]
        if prev_snapshot is not None and not reset_temporal_prior and cid in prev_snapshot.corridor_spillback_pred:
            prev_q = float(prev_snapshot.corridor_spillback_pred[cid])
        else:
            prev_q = dbn_cfg.prior_base_prob

        tls_green = _tls_green_fraction_for_links(
            topo_cache.corridor_tls_links[idx],
            belief.tls_states,
            f"corridor:{cid}",
        )
        base_lanes = float(max(spec.base_lanes, 1))
        cap_eff = float(max(cstats.storage_capacity, 1))
        caps_now[idx] = cap_eff

        x_c[idx, 0] = topo_cache.corridor_static_base[idx, 0]
        x_c[idx, 1] = topo_cache.corridor_static_base[idx, 1]
        x_c[idx, 2] = float(cstats.effective_lanes) / 4.0
        x_c[idx, 3] = float(cstats.effective_lanes) / base_lanes
        x_c[idx, 4] = min(cap_eff, 40.0) / 20.0
        x_c[idx, 5] = float(cstats.occupancy_ratio)
        x_c[idx, 6] = float(cstats.tail_blockage_prob)
        x_c[idx, 7] = min(max(0.0, float(cstats.expected_inflow)), 5.0) / 2.0
        x_c[idx, 8] = float(cstats.mean_speed_mps) / max(spec.speed_limit_mps, 1.0)
        x_c[idx, 9] = tls_green
        x_c[idx, 10] = prev_q
        x_c[idx, 11] = topo_cache.corridor_static_base[idx, 3]

    require_finite(x_c, "X_C")
    require_probability_range(x_c[:, 6], "CorridorBeliefStats.tail_blockage_prob")
    require_probability_range(x_c[:, 10], "prev_snapshot.corridor_spillback_pred")

    e_c = int(topo_cache.corridor_edge_index.shape[1])
    x_cc = np.zeros((e_c, CORRIDOR_EDGE_FEATURE_DIM), dtype=np.float32)
    if e_c > 0:
        x_cc[:, 0:4] = topo_cache.corridor_edge_static[:, 0:4]
        src_c = topo_cache.corridor_edge_index[0]
        dst_c = topo_cache.corridor_edge_index[1]
        x_cc[:, 5] = np.minimum(3.0, caps_now[dst_c] / np.maximum(caps_now[src_c], 1.0))
        for e_idx, (tls_id, link_indices) in enumerate(topo_cache.corridor_edge_tls_info):
            if tls_id is None or not link_indices:
                x_cc[e_idx, 4] = 1.0
            else:
                x_cc[e_idx, 4] = _tls_green_fraction_for_links(
                    [(tls_id, li) for li in link_indices],
                    belief.tls_states,
                    f"corridor_edge:{e_idx}",
                )

    require_shape(x_c, (n_c, CORRIDOR_FEATURE_DIM), "X_C")
    require_finite(x_c, "X_C")
    require_shape(x_cc, (e_c, CORRIDOR_EDGE_FEATURE_DIM), "X_CC")
    require_finite(x_cc, "X_CC")
    return x_c, x_cc


def extract_graph_features(
    belief: SwarmBeliefState,
    topology: RoadTopology,
    dbn_cfg: DBNConfig = DBNConfig(),
    prev_snapshot: DBNStateSnapshot | None = None,
    candidate_graph: CandidateGraph | None = None,
    *,
    reset_temporal_prior: bool = False,
) -> GraphFeatureBatch:
    """Extract schema-validated observable heterogeneous features for ``STGraphTransformerDBN``."""
    topo_cache = get_topology_cache(topology)
    cand = (
        candidate_graph
        if candidate_graph is not None
        else build_candidate_graph(belief, topology, dbn_cfg)
    )

    x_c, x_cc = _extract_corridor_features(
        belief=belief,
        topology=topology,
        topo_cache=topo_cache,
        dbn_cfg=dbn_cfg,
        prev_snapshot=prev_snapshot,
        reset_temporal_prior=reset_temporal_prior,
    )

    n_v = cand.num_vehicles
    x_v = np.zeros((n_v, NODE_FEATURE_DIM), dtype=np.float32)
    veh_corridor_idx = np.zeros(n_v, dtype=np.int64)
    topo_precedence = np.zeros(n_v, dtype=np.float32)
    speed_evidence_pos = np.zeros(n_v, dtype=np.float32)
    prev_delay_arr = np.zeros(n_v, dtype=np.float32)

    pos_vars = np.zeros(n_v, dtype=np.float32)
    etas = np.zeros(n_v, dtype=np.float32)
    speeds = np.zeros(n_v, dtype=np.float32)
    ages = np.zeros(n_v, dtype=np.float32)
    norm_ents = np.zeros(n_v, dtype=np.float32)
    lanes = np.zeros(n_v, dtype=np.int64)

    turn_src_list: list[int] = []
    turn_out_c_list: list[int] = []
    turn_prob_list: list[float] = []

    occ_by_c = x_c[:, 5]
    tail_by_c = x_c[:, 6]

    for i, vid in enumerate(cand.veh_ids):
        v = belief.vehicles[vid]
        spec = topology.corridors[v.corridor_id]
        c_idx = topo_cache.corridor_to_idx[v.corridor_id]
        veh_corridor_idx[i] = c_idx

        l_corr = max(spec.length_m, 20.0)
        rem_dist = 0.0 if v.in_junction_id is not None else max(0.0, spec.length_m - v.mean_s_m)
        clamped_s = min(max(0.0, v.mean_s_m), spec.length_m)
        pos_var = max(0.0, float(v.cov_2x2[0, 0]))
        vel_var = max(0.0, float(v.cov_2x2[1, 1]))
        pos_std = math.sqrt(pos_var)
        vel_std = math.sqrt(vel_var)
        obs_age = max(0.0, float(v.obs_age_s))

        k_e = len(spec.turn_options)
        if k_e == 0:
            if not spec.is_exit_fringe:
                raise TopologyIntegrityError(
                    f"Boundary Lock A violation on corridor {v.corridor_id!r}.",
                    context={"veh_id": vid, "corridor_id": v.corridor_id},
                )
            pi_left, pi_thru, pi_right, norm_ent = 0.0, 1.0, 0.0, 0.0
            exp_down_occ, exp_down_tail = 0.0, 0.0
        else:
            probs = np.asarray(v.turn_probs, dtype=np.float32)
            left_m, thru_m, right_m = topo_cache.turn_angle_bins[v.corridor_id]
            pi_left = float(probs[left_m].sum())
            pi_thru = float(probs[thru_m].sum())
            pi_right = float(probs[right_m].sum())
            nz = probs[probs > 1e-12]
            raw_ent = float(-(nz * np.log(nz)).sum()) if nz.size > 0 else 0.0
            norm_ent = raw_ent / max(math.log(k_e), 1.0) if k_e > 1 else 0.0

            target_c_indices = topo_cache.turn_target_corridor_indices[v.corridor_id]
            exp_down_occ = float(np.dot(probs, occ_by_c[target_c_indices]))
            exp_down_tail = float(np.dot(probs, tail_by_c[target_c_indices]))
            for out_c_idx, p_val in zip(target_c_indices.tolist(), probs.tolist(), strict=True):
                if p_val > 0.0:
                    turn_src_list.append(i)
                    turn_out_c_list.append(out_c_idx)
                    turn_prob_list.append(float(p_val))

        if prev_snapshot is not None and not reset_temporal_prior and vid in prev_snapshot.delay_post:
            prev_d = float(prev_snapshot.delay_post[vid])
        else:
            prev_d = dbn_cfg.prior_base_prob

        vmax = max(spec.speed_limit_mps, 1.0)
        free_eta = max(0.0, float(v.free_eta_s))
        wait_s = max(0.0, float(v.wait_time_s))

        # Soft current-tick kinematic delay evidence L_i^t(D_i^t = 1) (closes prototype 1-tick lag D3)
        slow_z = max(-25.0, min(25.0, (0.45 * vmax - v.mean_v_mps) / 1.2))
        slow_sig = 1.0 / (1.0 + math.exp(-slow_z))
        conf = 0.42 * math.exp(-obs_age / 3.0)
        ev_pos = 0.5 + conf * (2.0 * slow_sig - 1.0)

        # Global causal precedence anchor (#13 §1.2): r_topo(i, b_t) = -ETA_i^t + 15.0 * b_tail(i)
        r_topo = -free_eta + 15.0 * float(v.tail_blockage_score)

        x_v[i, 0] = rem_dist / l_corr
        x_v[i, 1] = topo_cache.corridor_static_base[c_idx, 0]
        x_v[i, 2] = clamped_s / l_corr
        x_v[i, 3] = v.mean_v_mps / vmax
        x_v[i, 4] = v.mean_v_mps / 15.0
        x_v[i, 5] = v.accel_est_mps2 / 4.0
        x_v[i, 6] = pos_std / 5.0
        x_v[i, 7] = vel_std / 2.0
        x_v[i, 8] = min(obs_age, 10.0) / 5.0
        x_v[i, 9] = pi_left
        x_v[i, 10] = pi_thru
        x_v[i, 11] = pi_right
        x_v[i, 12] = norm_ent
        x_v[i, 13] = prev_d
        x_v[i, 14] = min(free_eta, 20.0) / 15.0
        x_v[i, 15] = min(wait_s, 60.0) / 30.0
        x_v[i, 16] = occ_by_c[c_idx]
        x_v[i, 17] = tail_by_c[c_idx]
        x_v[i, 18] = float(v.lane_index) / float(max(spec.base_lanes, 1))
        x_v[i, 19] = exp_down_occ
        x_v[i, 20] = exp_down_tail

        topo_precedence[i] = r_topo
        speed_evidence_pos[i] = ev_pos
        prev_delay_arr[i] = prev_d
        pos_vars[i] = pos_var
        etas[i] = free_eta
        speeds[i] = v.mean_v_mps
        ages[i] = obs_age
        norm_ents[i] = norm_ent
        lanes[i] = v.lane_index

    e_cand = cand.num_edges
    x_e = np.zeros((e_cand, EDGE_FEATURE_DIM), dtype=np.float32)
    prev_omega_arr = np.full(e_cand, dbn_cfg.prior_base_prob, dtype=np.float32)
    prior_prob_arr = np.full(e_cand, dbn_cfg.prior_base_prob, dtype=np.float32)
    persisted_arr = np.zeros(e_cand, dtype=np.float32)

    if e_cand > 0:
        src = cand.edge_index[0]
        dst = cand.edge_index[1]
        rels = cand.relation_types

        x_e[np.arange(e_cand), rels] = 1.0
        d_eta = etas[dst] - etas[src]
        abs_d_eta = np.abs(d_eta)
        x_e[:, 4] = np.clip(d_eta / 10.0, -2.0, 2.0)
        x_e[:, 5] = np.minimum(2.0, abs_d_eta / 10.0)
        x_e[:, 6] = np.clip(cand.signed_path_gap_m / 80.0, -2.5, 2.5)
        x_e[:, 7] = (speeds[src] - speeds[dst]) / 15.0
        x_e[:, 8] = cand.p_conflict
        x_e[:, 9] = cand.p_merge
        x_e[:, 10] = np.maximum(0.0, 1.0 - abs_d_eta / 3.5)
        x_e[:, 11] = ((veh_corridor_idx[src] == veh_corridor_idx[dst]) & (lanes[src] == lanes[dst])).astype(np.float32)
        x_e[:, 12] = np.sqrt(pos_vars[src] + pos_vars[dst]) / 7.0
        x_e[:, 13] = np.minimum(3.0, (ages[src] + ages[dst]) / 6.0)
        x_e[:, 14] = norm_ents[src] + norm_ents[dst]

        if prev_snapshot is not None and not reset_temporal_prior and prev_snapshot.raw_edge_probs:
            raw_prev_map = prev_snapshot.raw_edge_probs
            veh_ids = cand.veh_ids
            stay_p = dbn_cfg.prior_stay_prob
            one_m_stay_base = (1.0 - stay_p) * dbn_cfg.prior_base_prob
            src_list_py = src.tolist()
            dst_list_py = dst.tolist()
            for k, (s_idx, d_idx) in enumerate(zip(src_list_py, dst_list_py)):
                w_prev = raw_prev_map.get((veh_ids[s_idx], veh_ids[d_idx]))
                if w_prev is not None:
                    prev_omega_arr[k] = w_prev
                    persisted_arr[k] = 1.0
                    prior_prob_arr[k] = one_m_stay_base + stay_p * w_prev
            require_probability_range(prev_omega_arr, "prev_snapshot.raw_edge_probs")

        prior_logits = _safe_logit(prior_prob_arr)
        x_e[:, 15] = prev_omega_arr
        x_e[:, 16] = np.asarray(prior_logits, dtype=np.float32) / 5.0
        x_e[:, 17] = persisted_arr

    require_probability_range(prev_delay_arr, "prev_snapshot.delay_post")
    require_shape(x_v, (n_v, NODE_FEATURE_DIM), "X_V")
    require_finite(x_v, "X_V")
    require_shape(x_e, (e_cand, EDGE_FEATURE_DIM), "X_E")
    require_finite(x_e, "X_E")
    require_finite(topo_precedence, "topo_precedence")
    require_probability_range(speed_evidence_pos, "speed_evidence_pos")

    return GraphFeatureBatch(
        sim_time=belief.sim_time,
        dt=belief.dt,
        candidate_graph=cand,
        corridor_ids=topo_cache.corridor_ids,
        corridor_to_idx=topo_cache.corridor_to_idx,
        x_veh=x_v,
        x_corridor=x_c,
        x_edge=x_e,
        x_corridor_edge=x_cc,
        corridor_edge_index=topo_cache.corridor_edge_index,
        veh_corridor_idx=veh_corridor_idx,
        veh_turn_src_idx=np.asarray(turn_src_list, dtype=np.int64),
        veh_turn_out_corridor_idx=np.asarray(turn_out_c_list, dtype=np.int64),
        veh_turn_prob=np.asarray(turn_prob_list, dtype=np.float32),
        topo_precedence=topo_precedence,
        speed_evidence_pos=speed_evidence_pos,
        prev_delay_post=prev_delay_arr,
        prev_edge_omega=prev_omega_arr,
        prior_edge_prob=prior_prob_arr,
        edge_persisted_mask=persisted_arr,
    )


class FeatureNormalizer:
    """Strict training-split z-score normalizer for ``(X_V, X_C, X_E, X_CC)``.

    Never fits on validation or test splits; fails loudly if ``transform`` is called before ``fit``.
    """

    def __init__(
        self,
        *,
        node_mean: np.ndarray | None = None,
        node_std: np.ndarray | None = None,
        corridor_mean: np.ndarray | None = None,
        corridor_std: np.ndarray | None = None,
        edge_mean: np.ndarray | None = None,
        edge_std: np.ndarray | None = None,
        corridor_edge_mean: np.ndarray | None = None,
        corridor_edge_std: np.ndarray | None = None,
        fitted: bool = False,
    ) -> None:
        self.node_mean = (
            np.asarray(node_mean, dtype=np.float32)
            if node_mean is not None
            else np.zeros(NODE_FEATURE_DIM, dtype=np.float32)
        )
        self.node_std = (
            np.asarray(node_std, dtype=np.float32)
            if node_std is not None
            else np.ones(NODE_FEATURE_DIM, dtype=np.float32)
        )
        self.corridor_mean = (
            np.asarray(corridor_mean, dtype=np.float32)
            if corridor_mean is not None
            else np.zeros(CORRIDOR_FEATURE_DIM, dtype=np.float32)
        )
        self.corridor_std = (
            np.asarray(corridor_std, dtype=np.float32)
            if corridor_std is not None
            else np.ones(CORRIDOR_FEATURE_DIM, dtype=np.float32)
        )
        self.edge_mean = (
            np.asarray(edge_mean, dtype=np.float32)
            if edge_mean is not None
            else np.zeros(EDGE_FEATURE_DIM, dtype=np.float32)
        )
        self.edge_std = (
            np.asarray(edge_std, dtype=np.float32)
            if edge_std is not None
            else np.ones(EDGE_FEATURE_DIM, dtype=np.float32)
        )
        self.corridor_edge_mean = (
            np.asarray(corridor_edge_mean, dtype=np.float32)
            if corridor_edge_mean is not None
            else np.zeros(CORRIDOR_EDGE_FEATURE_DIM, dtype=np.float32)
        )
        self.corridor_edge_std = (
            np.asarray(corridor_edge_std, dtype=np.float32)
            if corridor_edge_std is not None
            else np.ones(CORRIDOR_EDGE_FEATURE_DIM, dtype=np.float32)
        )
        self.fitted = bool(fitted)
        self._validate_internal()

    def _validate_internal(self) -> None:
        for arr, dim, name in (
            (self.node_mean, NODE_FEATURE_DIM, "node_mean"),
            (self.node_std, NODE_FEATURE_DIM, "node_std"),
            (self.corridor_mean, CORRIDOR_FEATURE_DIM, "corridor_mean"),
            (self.corridor_std, CORRIDOR_FEATURE_DIM, "corridor_std"),
            (self.edge_mean, EDGE_FEATURE_DIM, "edge_mean"),
            (self.edge_std, EDGE_FEATURE_DIM, "edge_std"),
            (self.corridor_edge_mean, CORRIDOR_EDGE_FEATURE_DIM, "corridor_edge_mean"),
            (self.corridor_edge_std, CORRIDOR_EDGE_FEATURE_DIM, "corridor_edge_std"),
        ):
            require_shape(arr, (dim,), f"FeatureNormalizer.{name}")
            require_finite(arr, f"FeatureNormalizer.{name}")
            if name.endswith("_std") and bool(np.any(arr <= 0.0)):
                raise CheckpointIntegrityError(
                    f"FeatureNormalizer.{name} contains non-positive standard deviations.",
                    context={"name": name, "min_std": float(arr.min())},
                )

    @classmethod
    def identity(cls) -> FeatureNormalizer:
        """Construct an explicit identity normalizer (mean=0, std=1, fitted=True) for unit testing."""
        return cls(fitted=True)

    def fit(self, batches: Sequence[GraphFeatureBatch], *, min_std: float = 0.05) -> FeatureNormalizer:
        """Fit standardization statistics strictly on training :class:`GraphFeatureBatch` instances."""
        if not batches:
            raise ConfigValidationError("FeatureNormalizer.fit requires at least one training GraphFeatureBatch.")
        if min_std <= 0.0:
            raise ConfigValidationError(f"min_std must be > 0, got {min_std}.")

        veh_rows = [b.x_veh for b in batches if b.x_veh.shape[0] > 0]
        corr_rows = [b.x_corridor for b in batches if b.x_corridor.shape[0] > 0]
        edge_rows = [b.x_edge for b in batches if b.x_edge.shape[0] > 0]
        cc_rows = [b.x_corridor_edge for b in batches if b.x_corridor_edge.shape[0] > 0]

        if not veh_rows or not corr_rows:
            raise ConfigValidationError(
                "FeatureNormalizer.fit received zero vehicle or corridor feature rows across training batches."
            )

        all_v = np.concatenate(veh_rows, axis=0)
        all_c = np.concatenate(corr_rows, axis=0)
        require_finite(all_v, "fit.all_v")
        require_finite(all_c, "fit.all_c")
        self.node_mean = all_v.mean(axis=0).astype(np.float32)
        self.node_std = np.maximum(all_v.std(axis=0), min_std).astype(np.float32)
        self.corridor_mean = all_c.mean(axis=0).astype(np.float32)
        self.corridor_std = np.maximum(all_c.std(axis=0), min_std).astype(np.float32)

        if edge_rows:
            all_e = np.concatenate(edge_rows, axis=0)
            require_finite(all_e, "fit.all_e")
            self.edge_mean = all_e.mean(axis=0).astype(np.float32)
            self.edge_std = np.maximum(all_e.std(axis=0), min_std).astype(np.float32)
            # Keep one-hot RelationType columns [0..3] uncentered/unit-scaled for clean categorical semantics.
            self.edge_mean[:4] = 0.0
            self.edge_std[:4] = 1.0
        else:
            self.edge_mean = np.zeros(EDGE_FEATURE_DIM, dtype=np.float32)
            self.edge_std = np.ones(EDGE_FEATURE_DIM, dtype=np.float32)

        if cc_rows:
            all_cc = np.concatenate(cc_rows, axis=0)
            require_finite(all_cc, "fit.all_cc")
            self.corridor_edge_mean = all_cc.mean(axis=0).astype(np.float32)
            self.corridor_edge_std = np.maximum(all_cc.std(axis=0), min_std).astype(np.float32)
        else:
            self.corridor_edge_mean = np.zeros(CORRIDOR_EDGE_FEATURE_DIM, dtype=np.float32)
            self.corridor_edge_std = np.ones(CORRIDOR_EDGE_FEATURE_DIM, dtype=np.float32)

        self.fitted = True
        self._validate_internal()
        return self

    def transform_arrays(
        self,
        x_veh: np.ndarray,
        x_corridor: np.ndarray,
        x_edge: np.ndarray,
        x_corridor_edge: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Standardize raw heterogeneous feature arrays, failing loudly if unfitted or non-finite."""
        if not self.fitted:
            raise CheckpointIntegrityError("FeatureNormalizer.transform_arrays called before fit().")
        require_shape(x_veh, (None, NODE_FEATURE_DIM), "transform.x_veh")
        require_shape(x_corridor, (None, CORRIDOR_FEATURE_DIM), "transform.x_corridor")
        require_shape(x_edge, (None, EDGE_FEATURE_DIM), "transform.x_edge")
        require_shape(x_corridor_edge, (None, CORRIDOR_EDGE_FEATURE_DIM), "transform.x_corridor_edge")
        require_finite(x_veh, "transform.x_veh")
        require_finite(x_corridor, "transform.x_corridor")
        require_finite(x_edge, "transform.x_edge")
        require_finite(x_corridor_edge, "transform.x_corridor_edge")

        z_v = ((x_veh - self.node_mean) / self.node_std).astype(np.float32)
        z_c = ((x_corridor - self.corridor_mean) / self.corridor_std).astype(np.float32)
        z_e = ((x_edge - self.edge_mean) / self.edge_std).astype(np.float32)
        z_cc = ((x_corridor_edge - self.corridor_edge_mean) / self.corridor_edge_std).astype(np.float32)

        require_finite(z_v, "transform.z_v")
        require_finite(z_c, "transform.z_c")
        require_finite(z_e, "transform.z_e")
        require_finite(z_cc, "transform.z_cc")
        return z_v, z_c, z_e, z_cc

    def to_dict(self) -> dict[str, Any]:
        """Serialize normalizer parameters to JSON-ready primitives."""
        self._validate_internal()
        return {
            "fitted": self.fitted,
            "node_mean": self.node_mean.tolist(),
            "node_std": self.node_std.tolist(),
            "corridor_mean": self.corridor_mean.tolist(),
            "corridor_std": self.corridor_std.tolist(),
            "edge_mean": self.edge_mean.tolist(),
            "edge_std": self.edge_std.tolist(),
            "corridor_edge_mean": self.corridor_edge_mean.tolist(),
            "corridor_edge_std": self.corridor_edge_std.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeatureNormalizer:
        """Strictly deserialize and validate normalizer metadata from a checkpoint dict."""
        required_keys = {
            "fitted",
            "node_mean",
            "node_std",
            "corridor_mean",
            "corridor_std",
            "edge_mean",
            "edge_std",
            "corridor_edge_mean",
            "corridor_edge_std",
        }
        missing = sorted(required_keys - set(data.keys()))
        if missing:
            raise CheckpointIntegrityError(
                f"FeatureNormalizer metadata missing required keys: {missing}.",
                context={"missing": missing},
            )
        if not isinstance(data["fitted"], bool):
            raise CheckpointIntegrityError("FeatureNormalizer.fitted must be a boolean.")
        try:
            norm = cls(
                node_mean=np.asarray(data["node_mean"], dtype=np.float32),
                node_std=np.asarray(data["node_std"], dtype=np.float32),
                corridor_mean=np.asarray(data["corridor_mean"], dtype=np.float32),
                corridor_std=np.asarray(data["corridor_std"], dtype=np.float32),
                edge_mean=np.asarray(data["edge_mean"], dtype=np.float32),
                edge_std=np.asarray(data["edge_std"], dtype=np.float32),
                corridor_edge_mean=np.asarray(data["corridor_edge_mean"], dtype=np.float32),
                corridor_edge_std=np.asarray(data["corridor_edge_std"], dtype=np.float32),
                fitted=bool(data["fitted"]),
            )
        except (ValueError, TypeError, CheckpointIntegrityError) as exc:
            get_logger("dbn.features").error("Invalid FeatureNormalizer arrays in checkpoint: %s", exc)
            raise CheckpointIntegrityError(
                f"Invalid FeatureNormalizer arrays in checkpoint: {exc}",
                context={"error": str(exc)},
            ) from exc
        if not norm.fitted:
            raise CheckpointIntegrityError("Checkpoint contains an unfitted FeatureNormalizer (fitted=False).")
        return norm


__all__ = [
    "FeatureNormalizer",
    "GraphFeatureBatch",
    "extract_graph_features",
]

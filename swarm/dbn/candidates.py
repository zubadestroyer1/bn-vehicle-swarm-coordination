"""Bucketed O(N_t) sparse topological candidate graph builder across urban road corridors.

Strictly firewall-safe: consumes only :class:`SwarmBeliefState`, :class:`RoadTopology`, and
:class:`DBNConfig`. Zero simulator ground-truth or ``libsumo``/``traci`` imports.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import math
from typing import Final

import numpy as np

from ..config import DBNConfig
from ..contracts import (
    CORRIDOR_EDGE_FEATURE_DIM,
    CorridorSpec,
    LEFT_TURN_THRESHOLD_RAD,
    RelationType,
    RoadTopology,
    SwarmBeliefState,
    VehicleBelief,
)
from ..logging_utils import (
    NumericalInvariantError,
    TopologyIntegrityError,
    require_finite,
    require_probability_simplex,
    require_psd_covariance,
)

SAME_CORRIDOR_MAX_GAP_M: Final[float] = 85.0
JUNCTION_CONFLICT_MAX_ETA_S: Final[float] = 12.0
SPILLBACK_TAIL_MAX_POS_M: Final[float] = 50.0
UPSTREAM_FEEDER_MAX_ETA_S: Final[float] = 8.0


@dataclass(frozen=True)
class TopologyLookupCache:
    """Precomputed static corridor indices, line-graph edges, and vectorized turn-conflict tables."""

    network_id: str
    corridor_ids: tuple[str, ...]
    corridor_to_idx: dict[str, int]
    corridor_static_base: np.ndarray          # float32 [N_C, 4]: len_norm, vmax_norm, base_lanes, fringe_ind
    corridor_edge_index: np.ndarray           # int64 [2, E_C]
    corridor_edge_static: np.ndarray          # float32 [E_C, 5]: sin, cos, is_thru, is_rev, base_cap_ratio
    corridor_edge_tls_info: tuple[tuple[str | None, tuple[int, ...]], ...]
    corridor_tls_links: tuple[tuple[tuple[str, int], ...], ...]
    pair_conflict_mats: dict[tuple[str, str], np.ndarray]   # (cid_a, cid_b) -> float32 [K_a, K_b]
    pair_merge_mats: dict[tuple[str, str], np.ndarray]      # (cid_a, cid_b) -> float32 [K_a, K_b]
    turn_target_corridor_indices: dict[str, np.ndarray]     # cid -> int64 [K_e]
    turn_angle_bins: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]  # cid -> bool masks (left, thru, right)
    max_turns_k: int
    conflict_pair_lookup: dict[tuple[int, int], int]        # (c_idx_a, c_idx_b) -> slot in padded_conflict_mats
    padded_conflict_mats: np.ndarray                        # float32 [M_pairs + 1, K_max_t, K_max_t] (slot 0 = zeros)
    padded_merge_mats: np.ndarray                           # float32 [M_pairs + 1, K_max_t, K_max_t] (slot 0 = zeros)


_TOPOLOGY_CACHE: dict[tuple[str, int, int], TopologyLookupCache] = {}


def get_topology_cache(topology: RoadTopology) -> TopologyLookupCache:
    """Return or build static topology lookup tables for ``topology``."""
    cache_key = (topology.network_id, len(topology.corridors), id(topology))
    cached = _TOPOLOGY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    topology.validate_integrity()
    corridor_ids = tuple(sorted(topology.corridors.keys()))
    corridor_to_idx = {cid: idx for idx, cid in enumerate(corridor_ids)}
    n_c = len(corridor_ids)

    corridor_static_base = np.zeros((n_c, 4), dtype=np.float32)
    corridor_tls_links: list[tuple[tuple[str, int], ...]] = []
    turn_target_corridor_indices: dict[str, np.ndarray] = {}
    turn_angle_bins: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    src_list: list[int] = []
    dst_list: list[int] = []
    edge_static_rows: list[tuple[float, float, float, float, float]] = []
    edge_tls_info: list[tuple[str | None, tuple[int, ...]]] = []

    base_caps = {
        cid: max(1, int(math.floor(spec.base_lanes * spec.length_m / 7.0)))
        for cid, spec in topology.corridors.items()
    }
    max_turns_k = 1

    for idx, cid in enumerate(corridor_ids):
        spec = topology.corridors[cid]
        k_e = len(spec.turn_options)
        if not spec.is_exit_fringe and k_e == 0:
            raise TopologyIntegrityError(
                f"Boundary Lock A violation on corridor {cid!r}: non-exit corridor has zero turn_options.",
                context={"corridor_id": cid},
            )
        max_turns_k = max(max_turns_k, k_e)
        corridor_static_base[idx, 0] = min(spec.length_m, 300.0) / 100.0
        corridor_static_base[idx, 1] = spec.speed_limit_mps / 15.0
        corridor_static_base[idx, 2] = float(spec.base_lanes)
        corridor_static_base[idx, 3] = 0.5 * float(spec.is_entry_fringe) - 0.5 * float(spec.is_exit_fringe)

        tls_pairs: list[tuple[str, int]] = []
        angles = np.array([opt.angle_rad for opt in spec.turn_options], dtype=np.float64)
        left_mask = angles > LEFT_TURN_THRESHOLD_RAD
        thru_mask = np.abs(angles) <= LEFT_TURN_THRESHOLD_RAD
        right_mask = angles < -LEFT_TURN_THRESHOLD_RAD
        turn_angle_bins[cid] = (left_mask, thru_mask, right_mask)
        turn_target_corridor_indices[cid] = np.array(
            [corridor_to_idx[opt.to_corridor] for opt in spec.turn_options],
            dtype=np.int64,
        )

        grouped_turns: dict[str, list[int]] = {}
        for t_pos, opt in enumerate(spec.turn_options):
            if opt.tls_id is not None:
                for link_idx in opt.tls_link_indices:
                    tls_pairs.append((opt.tls_id, int(link_idx)))
            grouped_turns.setdefault(opt.to_corridor, []).append(t_pos)

        corridor_tls_links.append(tuple(dict.fromkeys(tls_pairs)))

        for to_cid in sorted(grouped_turns.keys()):
            dst_idx = corridor_to_idx[to_cid]
            rep_opt = spec.turn_options[grouped_turns[to_cid][0]]
            all_tls_indices: list[int] = []
            rep_tls_id: str | None = None
            for t_pos in grouped_turns[to_cid]:
                t_opt = spec.turn_options[t_pos]
                if t_opt.tls_id is not None:
                    rep_tls_id = t_opt.tls_id
                    all_tls_indices.extend(int(li) for li in t_opt.tls_link_indices)
            dedup_tls = tuple(dict.fromkeys(all_tls_indices))
            sin_a = math.sin(rep_opt.angle_rad)
            cos_a = math.cos(rep_opt.angle_rad)
            is_thru = 1.0 if abs(rep_opt.angle_rad) <= LEFT_TURN_THRESHOLD_RAD else 0.0
            fwd_cap_ratio = min(3.0, float(base_caps[to_cid]) / float(base_caps[cid]))
            rev_cap_ratio = min(3.0, float(base_caps[cid]) / float(base_caps[to_cid]))

            src_list.append(idx)
            dst_list.append(dst_idx)
            edge_static_rows.append((sin_a, cos_a, is_thru, 0.0, fwd_cap_ratio))
            edge_tls_info.append((rep_tls_id, dedup_tls))

            src_list.append(dst_idx)
            dst_list.append(idx)
            edge_static_rows.append((-sin_a, cos_a, is_thru, 1.0, rev_cap_ratio))
            edge_tls_info.append((rep_tls_id, dedup_tls))

    if src_list:
        corridor_edge_index = np.array([src_list, dst_list], dtype=np.int64)
        corridor_edge_static = np.array(edge_static_rows, dtype=np.float32)
    else:
        corridor_edge_index = np.zeros((2, 0), dtype=np.int64)
        corridor_edge_static = np.zeros((0, CORRIDOR_EDGE_FEATURE_DIM - 1), dtype=np.float32)

    pair_conflict_mats: dict[tuple[str, str], np.ndarray] = {}
    pair_merge_mats: dict[tuple[str, str], np.ndarray] = {}
    conflict_pair_lookup: dict[tuple[int, int], int] = {}
    padded_c_list: list[np.ndarray] = [np.zeros((max_turns_k, max_turns_k), dtype=np.float32)]
    padded_m_list: list[np.ndarray] = [np.zeros((max_turns_k, max_turns_k), dtype=np.float32)]

    for junc_id, junc in topology.junctions.items():
        inc = junc.incoming_corridors
        for cid_a in inc:
            spec_a = topology.corridors[cid_a]
            k_a = len(spec_a.turn_options)
            if k_a == 0:
                continue
            idx_a = corridor_to_idx[cid_a]
            for cid_b in inc:
                spec_b = topology.corridors[cid_b]
                k_b = len(spec_b.turn_options)
                if k_b == 0:
                    continue
                idx_b = corridor_to_idx[cid_b]
                if (cid_a, cid_b) in pair_conflict_mats:
                    continue
                c_mat = np.zeros((k_a, k_b), dtype=np.float32)
                m_mat = np.zeros((k_a, k_b), dtype=np.float32)
                for ia, opt_a in enumerate(spec_a.turn_options):
                    for ib, opt_b in enumerate(spec_b.turn_options):
                        same_out = opt_a.to_corridor == opt_b.to_corridor
                        if same_out:
                            m_mat[ia, ib] = 1.0
                        if cid_a == cid_b:
                            c_mat[ia, ib] = 1.0 if ia == ib or same_out else 0.35
                        else:
                            is_foe = topology.turn_conflict(junc_id, opt_a, opt_b)
                            c_mat[ia, ib] = 1.0 if (is_foe or same_out) else 0.0
                pair_conflict_mats[(cid_a, cid_b)] = c_mat
                pair_merge_mats[(cid_a, cid_b)] = m_mat

                pad_c = np.zeros((max_turns_k, max_turns_k), dtype=np.float32)
                pad_m = np.zeros((max_turns_k, max_turns_k), dtype=np.float32)
                pad_c[:k_a, :k_b] = c_mat
                pad_m[:k_a, :k_b] = m_mat
                slot = len(padded_c_list)
                padded_c_list.append(pad_c)
                padded_m_list.append(pad_m)
                conflict_pair_lookup[(idx_a, idx_b)] = slot

    built = TopologyLookupCache(
        network_id=topology.network_id,
        corridor_ids=corridor_ids,
        corridor_to_idx=corridor_to_idx,
        corridor_static_base=corridor_static_base,
        corridor_edge_index=corridor_edge_index,
        corridor_edge_static=corridor_edge_static,
        corridor_edge_tls_info=tuple(edge_tls_info),
        corridor_tls_links=tuple(corridor_tls_links),
        pair_conflict_mats=pair_conflict_mats,
        pair_merge_mats=pair_merge_mats,
        turn_target_corridor_indices=turn_target_corridor_indices,
        turn_angle_bins=turn_angle_bins,
        max_turns_k=max_turns_k,
        conflict_pair_lookup=conflict_pair_lookup,
        padded_conflict_mats=np.stack(padded_c_list, axis=0),
        padded_merge_mats=np.stack(padded_m_list, axis=0),
    )
    if len(_TOPOLOGY_CACHE) >= 32:
        _TOPOLOGY_CACHE.clear()
    _TOPOLOGY_CACHE[cache_key] = built
    return built


@dataclass(frozen=True)
class CandidateGraph:
    """Sparse directed candidate interaction graph ``C_t`` over tracked vehicles at ``sim_time``."""

    sim_time: float
    veh_ids: tuple[str, ...]
    veh_id_to_idx: dict[str, int]
    edge_index: np.ndarray           # int64 [2, E_cand]: row 0 = src_idx (j), row 1 = dst_idx (i)
    relation_types: np.ndarray       # int64 [E_cand]: RelationType enum values
    p_conflict: np.ndarray           # float32 [E_cand]
    p_merge: np.ndarray              # float32 [E_cand]
    signed_path_gap_m: np.ndarray    # float32 [E_cand]

    @property
    def num_vehicles(self) -> int:
        return len(self.veh_ids)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


def _batch_validate_and_extract_vehicles(
    veh_ids: tuple[str, ...],
    belief: SwarmBeliefState,
    topology: RoadTopology,
    topo_cache: TopologyLookupCache,
) -> tuple[
    list[VehicleBelief],
    list[CorridorSpec],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Validate all vehicle beliefs in batched NumPy calls and return vectorized lookup arrays."""
    n_v = len(veh_ids)
    n_c = len(topo_cache.corridor_ids)
    k_max_t = topo_cache.max_turns_k

    vehs: list[VehicleBelief] = []
    specs: list[CorridorSpec] = []
    kin_mat = np.empty((n_v, 8), dtype=np.float64)
    cov_batch = np.empty((n_v, 2, 2), dtype=np.float64)
    s_arr = np.empty(n_v, dtype=np.float32)
    rem_arr = np.empty(n_v, dtype=np.float32)
    lane_arr = np.empty(n_v, dtype=np.int64)
    cid_idx_arr = np.empty(n_v, dtype=np.int64)
    in_junc_arr = np.zeros(n_v, dtype=bool)
    has_turns_arr = np.zeros(n_v, dtype=bool)
    pi_padded = np.zeros((n_v, k_max_t), dtype=np.float32)
    veh_out_prob = np.zeros((n_v, n_c), dtype=np.float32)

    turns_by_len: dict[int, list[tuple[float, ...]]] = {}

    for idx, vid in enumerate(veh_ids):
        v = belief.vehicles[vid]
        if v.veh_id != vid:
            raise TopologyIntegrityError(
                f"Vehicle key {vid!r} != VehicleBelief.veh_id {v.veh_id!r}.",
                context={"key": vid, "veh_id": v.veh_id},
            )
        spec = topology.corridors.get(v.corridor_id)
        if spec is None:
            raise TopologyIntegrityError(
                f"Vehicle {vid!r} references unknown corridor_id {v.corridor_id!r}.",
                context={"veh_id": vid, "corridor_id": v.corridor_id},
            )
        if v.in_junction_id is not None and v.in_junction_id not in topology.junctions:
            raise TopologyIntegrityError(
                f"Vehicle {vid!r} references unknown in_junction_id {v.in_junction_id!r}.",
                context={"veh_id": vid, "in_junction_id": v.in_junction_id},
            )

        k_e = len(spec.turn_options)
        if len(v.turn_probs) != k_e:
            raise TopologyIntegrityError(
                f"Vehicle {vid!r} on corridor {v.corridor_id!r} has len(turn_probs)={len(v.turn_probs)} != {k_e}.",
                context={"veh_id": vid, "corridor_id": v.corridor_id},
            )
        if k_e == 0 and not spec.is_exit_fringe:
            raise TopologyIntegrityError(
                f"Boundary Lock A violation: vehicle {vid!r} on non-exit corridor {v.corridor_id!r} with 0 turns.",
                context={"veh_id": vid, "corridor_id": v.corridor_id},
            )

        vehs.append(v)
        specs.append(spec)
        kin_mat[idx, 0] = v.mean_s_m
        kin_mat[idx, 1] = v.mean_v_mps
        kin_mat[idx, 2] = v.accel_est_mps2
        kin_mat[idx, 3] = v.obs_age_s
        kin_mat[idx, 4] = v.wait_time_s
        kin_mat[idx, 5] = v.free_eta_s
        kin_mat[idx, 6] = v.eta_std_s
        kin_mat[idx, 7] = v.tail_blockage_score
        cov_batch[idx] = v.cov_2x2

        s_arr[idx] = v.mean_s_m
        rem_arr[idx] = 0.0 if v.in_junction_id is not None else max(0.0, spec.length_m - v.mean_s_m)
        lane_arr[idx] = v.lane_index
        c_idx = topo_cache.corridor_to_idx[v.corridor_id]
        cid_idx_arr[idx] = c_idx
        in_junc_arr[idx] = v.in_junction_id is not None

        if k_e > 0:
            has_turns_arr[idx] = True
            turns_by_len.setdefault(k_e, []).append(v.turn_probs)
            pi_padded[idx, :k_e] = v.turn_probs
            target_c = topo_cache.turn_target_corridor_indices[v.corridor_id]
            for t_i in range(k_e):
                veh_out_prob[idx, target_c[t_i]] += float(v.turn_probs[t_i])

    # Batched fail-loud numerical, PSD covariance, and simplex assertions across all vehicles!
    require_finite(kin_mat, "VehicleBelief.kinematics_batch")
    require_psd_covariance(cov_batch, "VehicleBelief.cov_2x2_batch")
    for k_len, rows in turns_by_len.items():
        require_probability_simplex(np.asarray(rows, dtype=np.float32), f"VehicleBelief.turn_probs[k={k_len}]")

    return (
        vehs,
        specs,
        s_arr,
        rem_arr,
        lane_arr,
        cid_idx_arr,
        in_junc_arr,
        has_turns_arr,
        pi_padded,
        veh_out_prob,
    )


def _compute_vectorized_pair_metrics(
    src: np.ndarray,
    dst: np.ndarray,
    rel: np.ndarray,
    *,
    s_arr: np.ndarray,
    rem_arr: np.ndarray,
    lane_arr: np.ndarray,
    cid_idx_arr: np.ndarray,
    in_junc_arr: np.ndarray,
    has_turns_arr: np.ndarray,
    pi_padded: np.ndarray,
    veh_out_prob: np.ndarray,
    topo_cache: TopologyLookupCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized calculation of ``(p_conflict, p_merge, signed_path_gap_m)`` over all ``E_cand`` edges."""
    e_cand = int(src.shape[0])
    p_conf = np.zeros(e_cand, dtype=np.float32)
    p_merge = np.zeros(e_cand, dtype=np.float32)
    gap = np.zeros(e_cand, dtype=np.float32)

    # 1. SAME_CORRIDOR
    m_same = rel == int(RelationType.SAME_CORRIDOR)
    if np.any(m_same):
        sj = src[m_same]
        di = dst[m_same]
        gap[m_same] = s_arr[sj] - s_arr[di]
        same_l = lane_arr[sj] == lane_arr[di]
        p_conf[m_same] = np.where(same_l, 1.0, 0.35).astype(np.float32)
        both_t = has_turns_arr[sj] & has_turns_arr[di]
        dot_out = np.sum(veh_out_prob[sj] * veh_out_prob[di], axis=1)
        fallback_m = np.where(same_l, 1.0, 0.5).astype(np.float32)
        p_merge[m_same] = np.where(both_t, dot_out, fallback_m)

    # 2. SAME_JUNCTION_CONFLICT
    m_junc = rel == int(RelationType.SAME_JUNCTION_CONFLICT)
    if np.any(m_junc):
        sj = src[m_junc]
        di = dst[m_junc]
        gap[m_junc] = rem_arr[di] - rem_arr[sj]
        cj = cid_idx_arr[sj]
        ci = cid_idx_arr[di]
        lookup = topo_cache.conflict_pair_lookup
        slots = np.fromiter(
            (lookup.get((int(a), int(b)), 0) for a, b in zip(cj, ci)),
            dtype=np.int64,
            count=int(sj.shape[0]),
        )
        valid_mat = (slots > 0) & has_turns_arr[sj] & has_turns_arr[di]
        c_sub = np.zeros(sj.shape[0], dtype=np.float32)
        m_sub = np.zeros(sj.shape[0], dtype=np.float32)
        if np.any(valid_mat):
            sv = slots[valid_mat]
            pij = pi_padded[sj[valid_mat]]
            pii = pi_padded[di[valid_mat]]
            c_sub[valid_mat] = np.einsum("ek,ekm,em->e", pij, topo_cache.padded_conflict_mats[sv], pii)
            m_sub[valid_mat] = np.einsum("ek,ekm,em->e", pij, topo_cache.padded_merge_mats[sv], pii)
        if not np.all(valid_mat):
            inv = ~valid_mat
            either_junc = in_junc_arr[sj[inv]] | in_junc_arr[di[inv]]
            c_sub[inv] = np.where(either_junc, 0.65, 0.0).astype(np.float32)
            m_sub[inv] = np.where(either_junc, 0.25, 0.0).astype(np.float32)
        p_conf[m_junc] = c_sub
        p_merge[m_junc] = m_sub

    # 3. DOWNSTREAM_SPILLBACK
    m_spill = rel == int(RelationType.DOWNSTREAM_SPILLBACK)
    if np.any(m_spill):
        sj = src[m_spill]
        di = dst[m_spill]
        gap[m_spill] = rem_arr[di] + np.maximum(0.0, s_arr[sj])
        pt = veh_out_prob[di, cid_idx_arr[sj]]
        p_conf[m_spill] = pt
        p_merge[m_spill] = pt

    # 4. UPSTREAM_FEEDER
    m_feed = rel == int(RelationType.UPSTREAM_FEEDER)
    if np.any(m_feed):
        sj = src[m_feed]
        di = dst[m_feed]
        gap[m_feed] = -(rem_arr[sj] + np.maximum(0.0, s_arr[di]))
        pt = veh_out_prob[sj, cid_idx_arr[di]]
        p_conf[m_feed] = pt
        p_merge[m_feed] = pt

    return np.clip(p_conf, 0.0, 1.0), np.clip(p_merge, 0.0, 1.0), gap


def build_candidate_graph(
    belief: SwarmBeliefState,
    topology: RoadTopology,
    dbn_cfg: DBNConfig = DBNConfig(),
    *,
    unconstrained_all_pairs: bool = False,
) -> CandidateGraph:
    """Construct the sparse O(N_t) topological candidate graph ``C_t``."""
    require_finite(belief.sim_time, "SwarmBeliefState.sim_time")
    require_finite(belief.dt, "SwarmBeliefState.dt")
    if belief.dt <= 0.0:
        raise NumericalInvariantError(
            f"SwarmBeliefState.dt must be > 0, got {belief.dt}.",
            context={"dt": belief.dt},
        )

    topo_cache = get_topology_cache(topology)
    veh_ids = tuple(sorted(belief.vehicles.keys()))
    veh_id_to_idx = {vid: idx for idx, vid in enumerate(veh_ids)}
    n_v = len(veh_ids)

    if n_v == 0:
        return CandidateGraph(
            sim_time=belief.sim_time,
            veh_ids=veh_ids,
            veh_id_to_idx=veh_id_to_idx,
            edge_index=np.zeros((2, 0), dtype=np.int64),
            relation_types=np.zeros(0, dtype=np.int64),
            p_conflict=np.zeros(0, dtype=np.float32),
            p_merge=np.zeros(0, dtype=np.float32),
            signed_path_gap_m=np.zeros(0, dtype=np.float32),
        )

    (
        vehs,
        specs,
        s_arr,
        rem_arr,
        lane_arr,
        cid_idx_arr,
        in_junc_arr,
        has_turns_arr,
        pi_padded,
        veh_out_prob,
    ) = _batch_validate_and_extract_vehicles(veh_ids, belief, topology, topo_cache)

    if n_v == 1:
        return CandidateGraph(
            sim_time=belief.sim_time,
            veh_ids=veh_ids,
            veh_id_to_idx=veh_id_to_idx,
            edge_index=np.zeros((2, 0), dtype=np.int64),
            relation_types=np.zeros(0, dtype=np.int64),
            p_conflict=np.zeros(0, dtype=np.float32),
            p_merge=np.zeros(0, dtype=np.float32),
            signed_path_gap_m=np.zeros(0, dtype=np.float32),
        )

    by_corridor: dict[str, list[int]] = {}
    by_approach_junc: dict[str, list[int]] = {}
    by_upstream_tail: dict[str, list[int]] = {}
    by_feeder_junc: dict[str, list[int]] = {}

    for idx, v in enumerate(vehs):
        spec = specs[idx]
        by_corridor.setdefault(v.corridor_id, []).append(idx)
        target_junc = v.in_junction_id if v.in_junction_id is not None else spec.to_junction
        if v.in_junction_id is not None or v.free_eta_s <= JUNCTION_CONFLICT_MAX_ETA_S:
            by_approach_junc.setdefault(target_junc, []).append(idx)
        if v.in_junction_id is None and v.mean_s_m <= SPILLBACK_TAIL_MAX_POS_M:
            by_upstream_tail.setdefault(v.corridor_id, []).append(idx)
        if v.free_eta_s <= UPSTREAM_FEEDER_MAX_ETA_S and not spec.is_exit_fringe:
            by_feeder_junc.setdefault(spec.to_junction, []).append(idx)

    corridor_pos_sorted: dict[str, tuple[list[int], list[float]]] = {}
    for cid, indices in by_corridor.items():
        indices.sort(key=lambda k: (vehs[k].mean_s_m, k))
        corridor_pos_sorted[cid] = (indices, [vehs[k].mean_s_m for k in indices])

    junc_eta_sorted: dict[str, tuple[list[int], list[float]]] = {}
    for jid, indices in by_approach_junc.items():
        indices.sort(key=lambda k: (vehs[k].free_eta_s, k))
        junc_eta_sorted[jid] = (indices, [vehs[k].free_eta_s for k in indices])

    for indices in by_upstream_tail.values():
        indices.sort(key=lambda k: (vehs[k].mean_s_m, k))

    for indices in by_feeder_junc.values():
        indices.sort(key=lambda k: (vehs[k].free_eta_s, k))

    k_max_cand = dbn_cfg.max_candidates_per_veh
    q_same = max(1, round(0.20 * k_max_cand))
    q_junc = max(1, round(0.40 * k_max_cand))
    q_spill = max(1, round(0.20 * k_max_cand))
    q_feed = max(1, k_max_cand - q_same - q_junc - q_spill)

    src_out: list[int] = []
    dst_out: list[int] = []
    rel_out: list[int] = []

    for i in range(n_v):
        vi = vehs[i]
        spec_i = specs[i]
        cid_i = vi.corridor_id
        succ_i = topology.successor_corridors.get(cid_i, ())

        if unconstrained_all_pairs:
            succ_set = set(succ_i)
            for j in range(n_v):
                if j == i:
                    continue
                cid_j = vehs[j].corridor_id
                if cid_j == cid_i:
                    rel_val = int(RelationType.SAME_CORRIDOR)
                elif cid_j in succ_set:
                    rel_val = int(RelationType.DOWNSTREAM_SPILLBACK)
                elif cid_i in topology.successor_corridors.get(cid_j, ()):
                    rel_val = int(RelationType.UPSTREAM_FEEDER)
                else:
                    rel_val = int(RelationType.SAME_JUNCTION_CONFLICT)
                src_out.append(j)
                dst_out.append(i)
                rel_out.append(rel_val)
            continue

        chosen: dict[int, int] = {}

        # 1. SAME_CORRIDOR
        c_indices, c_positions = corridor_pos_sorted[cid_i]
        pos_i = vi.mean_s_m
        rank_i = bisect.bisect_left(c_positions, pos_i)
        same_cands: list[tuple[float, int]] = []
        left_ptr = rank_i - 1
        right_ptr = rank_i
        while (left_ptr >= 0 or right_ptr < len(c_indices)) and len(same_cands) < q_same + 4:
            if left_ptr >= 0:
                cand_l = c_indices[left_ptr]
                dist_l = abs(c_positions[left_ptr] - pos_i)
                if dist_l <= SAME_CORRIDOR_MAX_GAP_M and cand_l != i:
                    same_cands.append((dist_l, cand_l))
                elif dist_l > SAME_CORRIDOR_MAX_GAP_M:
                    left_ptr = -1
                left_ptr -= 1
            if right_ptr < len(c_indices):
                cand_r = c_indices[right_ptr]
                dist_r = abs(c_positions[right_ptr] - pos_i)
                if dist_r <= SAME_CORRIDOR_MAX_GAP_M and cand_r != i:
                    same_cands.append((dist_r, cand_r))
                elif dist_r > SAME_CORRIDOR_MAX_GAP_M:
                    right_ptr = len(c_indices)
                right_ptr += 1
        same_cands.sort()
        for _, j in same_cands[:q_same]:
            chosen[j] = int(RelationType.SAME_CORRIDOR)

        # 2. SAME_JUNCTION_CONFLICT
        target_junc = vi.in_junction_id if vi.in_junction_id is not None else spec_i.to_junction
        if target_junc in junc_eta_sorted and (vi.in_junction_id is not None or vi.free_eta_s <= JUNCTION_CONFLICT_MAX_ETA_S):
            j_indices, j_etas = junc_eta_sorted[target_junc]
            eta_i = vi.free_eta_s
            eta_rank = bisect.bisect_left(j_etas, eta_i)
            junc_cands: list[tuple[float, int]] = []
            l_ptr = eta_rank - 1
            r_ptr = eta_rank
            while (l_ptr >= 0 or r_ptr < len(j_indices)) and len(junc_cands) < q_junc * 2 + 4:
                if l_ptr >= 0:
                    cj = j_indices[l_ptr]
                    if cj != i and (vehs[cj].corridor_id != cid_i or vehs[cj].in_junction_id is not None):
                        junc_cands.append((abs(j_etas[l_ptr] - eta_i), cj))
                    l_ptr -= 1
                if r_ptr < len(j_indices):
                    cj = j_indices[r_ptr]
                    if cj != i and (vehs[cj].corridor_id != cid_i or vehs[cj].in_junction_id is not None):
                        junc_cands.append((abs(j_etas[r_ptr] - eta_i), cj))
                    r_ptr += 1
            junc_cands.sort()
            added_junc = 0
            for _, j in junc_cands:
                if j not in chosen:
                    chosen[j] = int(RelationType.SAME_JUNCTION_CONFLICT)
                    added_junc += 1
                    if added_junc >= q_junc:
                        break

        # 3. DOWNSTREAM_SPILLBACK
        if succ_i:
            spill_cands: list[tuple[float, int]] = []
            for out_cid in succ_i:
                if out_cid == cid_i:
                    continue
                for j in by_upstream_tail.get(out_cid, ())[:q_spill]:
                    if j != i and j not in chosen:
                        spill_cands.append((vehs[j].mean_s_m, j))
            spill_cands.sort()
            for _, j in spill_cands[:q_spill]:
                chosen[j] = int(RelationType.DOWNSTREAM_SPILLBACK)

        # 4. UPSTREAM_FEEDER
        remaining_budget = k_max_cand - len(chosen)
        if remaining_budget > 0 and spec_i.from_junction in by_feeder_junc:
            added_feed = 0
            for j in by_feeder_junc[spec_i.from_junction]:
                if j == i or j in chosen:
                    continue
                if cid_i in topology.successor_corridors.get(vehs[j].corridor_id, ()):
                    chosen[j] = int(RelationType.UPSTREAM_FEEDER)
                    added_feed += 1
                    if added_feed >= remaining_budget:
                        break

        ordered_cands = sorted(chosen.items(), key=lambda kv: (kv[1], kv[0]))[:k_max_cand]
        for j, rel_val in ordered_cands:
            src_out.append(j)
            dst_out.append(i)
            rel_out.append(rel_val)

    if src_out:
        src_arr = np.asarray(src_out, dtype=np.int64)
        dst_arr = np.asarray(dst_out, dtype=np.int64)
        relation_types = np.asarray(rel_out, dtype=np.int64)
        edge_index = np.stack([src_arr, dst_arr], axis=0)
        p_conflict, p_merge, signed_path_gap_m = _compute_vectorized_pair_metrics(
            src_arr,
            dst_arr,
            relation_types,
            s_arr=s_arr,
            rem_arr=rem_arr,
            lane_arr=lane_arr,
            cid_idx_arr=cid_idx_arr,
            in_junc_arr=in_junc_arr,
            has_turns_arr=has_turns_arr,
            pi_padded=pi_padded,
            veh_out_prob=veh_out_prob,
            topo_cache=topo_cache,
        )
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        relation_types = np.zeros(0, dtype=np.int64)
        p_conflict = np.zeros(0, dtype=np.float32)
        p_merge = np.zeros(0, dtype=np.float32)
        signed_path_gap_m = np.zeros(0, dtype=np.float32)

    return CandidateGraph(
        sim_time=belief.sim_time,
        veh_ids=veh_ids,
        veh_id_to_idx=veh_id_to_idx,
        edge_index=edge_index,
        relation_types=relation_types,
        p_conflict=p_conflict,
        p_merge=p_merge,
        signed_path_gap_m=signed_path_gap_m,
    )


__all__ = [
    "CandidateGraph",
    "JUNCTION_CONFLICT_MAX_ETA_S",
    "SAME_CORRIDOR_MAX_GAP_M",
    "SPILLBACK_TAIL_MAX_POS_M",
    "TopologyLookupCache",
    "UPSTREAM_FEEDER_MAX_ETA_S",
    "build_candidate_graph",
    "get_topology_cache",
]

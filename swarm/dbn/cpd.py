"""Causal Conditional Probability Distributions (CPDs) and Precedence-DAG Edge Gating.

Implements:
1. Temporal Markov edge persistence prior ``E_{t-1} -> E_t``.
2. Parameter-tied physical queue-conservation Corridor Spillback CPD ``Q_c^t``.
3. Boundary-Lock-A-compliant expected downstream corridor spillback per vehicle.
4. Generalized Causal Noisy-OR Vehicle Delay CPD fixing prototype flaw D1:
   ``phi_{j->i}(0) = 1 - omega * kappa < 1`` so undelayed conflicting leaders (``D_j = 0``)
   still exert physical right-of-way / headway delay pressure on follower ``i``, while
   delayed leaders (``D_j = 1``) exert stronger cascade pressure
   ``phi_{j->i}(1) = 1 - omega * (kappa + (1 - kappa) * gamma)``.
5. Boundary-Lock-B precedence DAG gating & top-``K_max`` parent pruning (smooth annealed gate
   during ``model.train()``, strict acyclic total-order indicator during ``model.eval()``).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..logging_utils import (
    DAGInvariantError,
    NumericalInvariantError,
    require_finite,
    require_probability_range,
    require_shape,
)

STATIC_CPD_BASE_LEAK: float = 0.05
STATIC_CPD_PERSIST_WEIGHT: float = 0.45
STATIC_CPD_SPILL_WEIGHT: float = 0.55


def temporal_edge_prior(
    prev_omega: Tensor | np.ndarray,
    persisted_mask: Tensor | np.ndarray,
    *,
    prior_stay_prob: float = 0.80,
    prior_base_prob: float = 0.10,
    check_invariants: bool = True,
) -> Tensor | np.ndarray:
    """Compute first-order Markov edge prior ``P(e_{j->i}^t = 1 | omega_{j->i}^{t-1})``.

    Parameters
    ----------
    prev_omega:
        Previous step raw edge probability ``[E_cand]`` in ``[0, 1]``.
    persisted_mask:
        Binary mask ``[E_cand]`` equal to ``1.0`` iff pair ``(j, i)`` existed in ``C_{t-1}``.
    prior_stay_prob:
        Markov persistence strength ``p_stay in [0, 1]``.
    prior_base_prob:
        Stationary base edge prior ``p_base in (0, 1)``.
    """
    if not (0.0 <= prior_stay_prob <= 1.0):
        raise NumericalInvariantError(
            f"prior_stay_prob must be in [0, 1], got {prior_stay_prob}.",
            context={"prior_stay_prob": prior_stay_prob},
        )
    if not (0.0 < prior_base_prob < 1.0):
        raise NumericalInvariantError(
            f"prior_base_prob must be in (0, 1), got {prior_base_prob}.",
            context={"prior_base_prob": prior_base_prob},
        )
    if check_invariants:
        require_probability_range(prev_omega, "temporal_edge_prior.prev_omega")
        require_probability_range(persisted_mask, "temporal_edge_prior.persisted_mask")

    persisted_prior = (1.0 - prior_stay_prob) * prior_base_prob + prior_stay_prob * prev_omega
    if isinstance(prev_omega, torch.Tensor):
        mask_t = torch.as_tensor(persisted_mask, dtype=prev_omega.dtype, device=prev_omega.device)
        base_t = torch.full_like(prev_omega, prior_base_prob)
        out = base_t + mask_t * (persisted_prior - base_t)
        if check_invariants:
            require_probability_range(out, "temporal_edge_prior.out")
        return out

    mask_np = np.asarray(persisted_mask, dtype=np.float32)
    out_np = np.float32(prior_base_prob) + mask_np * (
        np.asarray(persisted_prior, dtype=np.float32) - np.float32(prior_base_prob)
    )
    if check_invariants:
        require_probability_range(out_np, "temporal_edge_prior.out")
    return out_np


def _safe_logit_tensor(prob: Tensor, eps: float = 1e-4) -> Tensor:
    clamped = prob.clamp(min=eps, max=1.0 - eps)
    return torch.log(clamped / (1.0 - clamped))


class CorridorSpillbackCPD(nn.Module):
    """Parameter-tied physical queue-conservation CPD for corridor spillback ``Q_c^t``.

    Shared across all road corridors regardless of network size so the model generalizes to
    unseen corridor counts and lane-drop perturbations. Monotone physical sensitivities are
    parameterized via ``softplus`` to preserve conservation semantics.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(-1.35, dtype=torch.float32))
        self.raw_w_occ = nn.Parameter(torch.tensor(math.log(math.exp(3.2) - 1.0), dtype=torch.float32))
        self.raw_w_tail = nn.Parameter(torch.tensor(math.log(math.exp(2.8) - 1.0), dtype=torch.float32))
        self.raw_w_inflow = nn.Parameter(torch.tensor(math.log(math.exp(1.1) - 1.0), dtype=torch.float32))
        self.raw_w_green = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0), dtype=torch.float32))
        self.raw_w_speed = nn.Parameter(torch.tensor(math.log(math.exp(1.4) - 1.0), dtype=torch.float32))
        self.raw_w_lane = nn.Parameter(torch.tensor(math.log(math.exp(1.2) - 1.0), dtype=torch.float32))
        self.raw_w_prev = nn.Parameter(torch.tensor(math.log(math.exp(0.55) - 1.0), dtype=torch.float32))

    def forward(self, x_corridor_raw: Tensor, *, check_invariants: bool = True) -> Tensor:
        """Compute physical spillback logits ``[N_C]`` from unnormalized ``X_C [N_C, 12]``."""
        require_shape(x_corridor_raw, (None, 12), "CorridorSpillbackCPD.x_corridor_raw")
        if check_invariants:
            require_finite(x_corridor_raw, "CorridorSpillbackCPD.x_corridor_raw")
        if x_corridor_raw.shape[0] == 0:
            return x_corridor_raw.new_empty((0,))

        lane_ratio = x_corridor_raw[:, 3]
        occ_ratio = x_corridor_raw[:, 5]
        tail_prob = x_corridor_raw[:, 6]
        inflow_norm = x_corridor_raw[:, 7]
        speed_ratio = x_corridor_raw[:, 8]
        tls_green = x_corridor_raw[:, 9]
        prev_q = x_corridor_raw[:, 10]

        w_occ = F.softplus(self.raw_w_occ)
        w_tail = F.softplus(self.raw_w_tail)
        w_in = F.softplus(self.raw_w_inflow)
        w_green = F.softplus(self.raw_w_green)
        w_spd = F.softplus(self.raw_w_speed)
        w_lane = F.softplus(self.raw_w_lane)
        w_prev = F.softplus(self.raw_w_prev)

        logits = (
            self.bias
            + w_occ * (occ_ratio - 0.60)
            + w_tail * tail_prob
            + w_in * inflow_norm
            - w_green * (tls_green - 0.50)
            - w_spd * (speed_ratio - 0.50)
            - w_lane * (lane_ratio - 1.0)
            + w_prev * _safe_logit_tensor(prev_q)
        )
        if check_invariants:
            require_finite(logits, "CorridorSpillbackCPD.logits")
        return logits


def heuristic_corridor_spillback_prob(x_corridor_raw: np.ndarray) -> np.ndarray:
    """Closed-form physical queue-conservation spillback probability ``[N_C]`` for non-neural baselines."""
    require_shape(x_corridor_raw, (None, 12), "heuristic_corridor_spillback_prob.x_corridor_raw")
    require_finite(x_corridor_raw, "heuristic_corridor_spillback_prob.x_corridor_raw")
    n_c = x_corridor_raw.shape[0]
    if n_c == 0:
        return np.zeros(0, dtype=np.float32)

    lane_ratio = x_corridor_raw[:, 3]
    occ_ratio = x_corridor_raw[:, 5]
    tail_prob = x_corridor_raw[:, 6]
    inflow_norm = x_corridor_raw[:, 7]
    speed_ratio = x_corridor_raw[:, 8]
    tls_green = x_corridor_raw[:, 9]
    prev_q = np.clip(x_corridor_raw[:, 10], 1e-4, 1.0 - 1e-4)
    prev_logit = np.log(prev_q / (1.0 - prev_q))

    logits = (
        -1.35
        + 3.20 * (occ_ratio - 0.60)
        + 2.80 * tail_prob
        + 1.10 * inflow_norm
        - 1.00 * (tls_green - 0.50)
        - 1.40 * (speed_ratio - 0.50)
        - 1.20 * (lane_ratio - 1.0)
        + 0.55 * prev_logit
    )
    logits = np.clip(logits, -30.0, 30.0)
    probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    require_probability_range(probs, "heuristic_corridor_spillback_prob.probs")
    return probs


def expected_downstream_spillback(
    corridor_spillback_prob: Tensor,
    veh_turn_src_idx: Tensor,
    veh_turn_out_corridor_idx: Tensor,
    veh_turn_prob: Tensor,
    num_vehicles: int,
    *,
    check_invariants: bool = True,
) -> Tensor:
    """Compute turn-belief-weighted expected downstream spillback ``[N_t]`` (Boundary Lock A safe).

    Vehicles on exit-fringe corridors have zero turn entries in ``veh_turn_src_idx`` and
    therefore evaluate strictly to ``0.0``.
    """
    if check_invariants:
        require_probability_range(corridor_spillback_prob, "expected_downstream_spillback.corridor_spillback_prob")
    out = corridor_spillback_prob.new_zeros((num_vehicles,))
    if num_vehicles == 0 or veh_turn_src_idx.numel() == 0:
        return out
    weighted = veh_turn_prob.to(dtype=out.dtype) * corridor_spillback_prob.index_select(0, veh_turn_out_corridor_idx)
    out = out.index_add(0, veh_turn_src_idx, weighted)
    out = out.clamp(min=0.0, max=1.0)
    if check_invariants:
        require_probability_range(out, "expected_downstream_spillback.out")
    return out


def static_edge_cpd_parameters(x_edge_raw: Tensor) -> tuple[Tensor, Tensor]:
    """Compute fixed physical ``(kappa, gamma)`` parameters for the ``static_cpd`` ablation.

    Uses raw observable edge columns:
    - col 8: ``p_conflict``
    - col 9: ``p_merge``
    - col 10: ``simultaneity = max(0, 1 - |Delta ETA| / 3.5)``
    - col 11: ``same_lane_ind``
    """
    require_shape(x_edge_raw, (None, 18), "static_edge_cpd_parameters.x_edge_raw")
    if x_edge_raw.shape[0] == 0:
        empty = x_edge_raw.new_empty((0,))
        return empty, empty
    p_conflict = x_edge_raw[:, 8]
    p_merge = x_edge_raw[:, 9]
    simultaneity = x_edge_raw[:, 10]
    same_lane = x_edge_raw[:, 11]

    kappa = (0.12 + 0.42 * p_conflict * (0.35 + 0.65 * simultaneity) + 0.18 * same_lane).clamp(0.04, 0.90)
    gamma = (0.25 + 0.45 * p_merge + 0.20 * same_lane).clamp(0.08, 0.94)
    return kappa, gamma


def generalized_noisy_or_base_survival(
    leak_prob: Tensor,
    self_persist_prob: Tensor,
    prev_delay_post: Tensor,
    spill_weight: Tensor,
    expected_down_spill: Tensor,
    *,
    eps: float = 1e-6,
    check_invariants: bool = True,
) -> Tensor:
    """Compute parent-free survival probability ``z_{i,0}^t = P(D_i^t = 0 | Pa(i) = empty)``."""
    if check_invariants:
        for t_val, label in (
            (leak_prob, "leak_prob"),
            (self_persist_prob, "self_persist_prob"),
            (prev_delay_post, "prev_delay_post"),
            (spill_weight, "spill_weight"),
            (expected_down_spill, "expected_down_spill"),
        ):
            require_probability_range(t_val, f"base_survival.{label}")

    s_leak = (1.0 - leak_prob).clamp(min=eps, max=1.0)
    s_persist = (1.0 - self_persist_prob * prev_delay_post).clamp(min=eps, max=1.0)
    s_spill = (1.0 - spill_weight * expected_down_spill).clamp(min=eps, max=1.0)
    z0 = (s_leak * s_persist * s_spill).clamp(min=eps, max=1.0 - eps)
    if check_invariants:
        require_probability_range(z0, "base_survival.z0")
    return z0


def parent_survival_factors(
    omega: Tensor,
    kappa: Tensor,
    gamma: Tensor,
    *,
    eps: float = 1e-6,
    check_invariants: bool = True,
) -> tuple[Tensor, Tensor]:
    """Compute Generalized Causal Noisy-OR survival factors ``(phi_0, phi_1)`` for active edges.

    Fixes prototype flaw D1:
    - ``phi_0 = P(no inhibition from j | D_j^t = 0) = 1 - omega * kappa < 1`` whenever ``omega, kappa > 0``,
      so a free-flowing conflicting leader (``D_j^t = 0``) still exerts physical conflict delay pressure!
    - ``phi_1 = P(no inhibition from j | D_j^t = 1) = 1 - omega * (kappa + (1 - kappa) * gamma) <= phi_0``.
    """
    if check_invariants:
        require_probability_range(omega, "parent_survival_factors.omega")
        require_probability_range(kappa, "parent_survival_factors.kappa")
        require_probability_range(gamma, "parent_survival_factors.gamma")

    q0 = (omega * kappa).clamp(min=0.0, max=1.0 - eps)
    q1 = (omega * (kappa + (1.0 - kappa) * gamma)).clamp(min=0.0, max=1.0 - eps)
    phi_0 = (1.0 - q0).clamp(min=eps, max=1.0)
    phi_1 = (1.0 - q1).clamp(min=eps, max=1.0)
    return phi_0, phi_1


@dataclass(frozen=True)
class PrunedDAGStructure:
    """Pruned sparse parent structure in both compact edge-list and padded ``[N_t, K_max]`` slot form."""

    num_vehicles: int
    k_max: int
    active_edge_indices: Tensor     # int64 [E_act] indices into C_t
    src_idx: Tensor                 # int64 [E_act]
    dst_idx: Tensor                 # int64 [E_act]
    dst_rank: Tensor                # int64 [E_act] slot index in [0 .. K_max-1] within dst_idx
    omega: Tensor                   # float32 [E_act]
    kappa: Tensor                   # float32 [E_act]
    gamma: Tensor                   # float32 [E_act]
    phi_0: Tensor                   # float32 [E_act]
    phi_1: Tensor                   # float32 [E_act]
    parent_idx_padded: Tensor       # int64 [N_t, K_max] (0 where parent_mask == False)
    parent_edge_pos_padded: Tensor  # int64 [N_t, K_max] index into [0 .. E_act-1] (0 where False)
    parent_mask: Tensor             # bool [N_t, K_max]
    phi_0_padded: Tensor            # float32 [N_t, K_max] (1.0 where parent_mask == False)
    phi_1_padded: Tensor            # float32 [N_t, K_max] (1.0 where parent_mask == False)


def gate_and_prune_edges(
    edge_index: Tensor,
    precedence: Tensor,
    omega_raw: Tensor,
    kappa: Tensor,
    gamma: Tensor,
    *,
    num_vehicles: int,
    max_parents_k: int = 6,
    min_edge_prob: float = 0.05,
    training: bool = False,
    dag_temperature: float = 0.5,
    check_invariants: bool = True,
) -> PrunedDAGStructure:
    """Apply Boundary-Lock-B precedence DAG gating and deterministic top-``K_max`` parent selection.

    - In ``training=False`` (runtime & validation): enforces strict total-order DAG indicator
      ``(r_src > r_dst) | ((r_src == r_dst) & (src_idx < dst_idx))`` and threshold ``omega >= min_edge_prob``.
      Because ``veh_ids`` are lexicographically sorted, ``src_idx < dst_idx`` matches ``src_id < dst_id``
      in :func:`swarm.logging_utils.require_dag` identically.
    - In ``training=True``: applies smooth annealed precedence gate
      ``sigmoid((r_src - r_dst) / dag_temperature)`` so gradients propagate into ``precedence`` while
      still selecting at most ``max_parents_k`` parents per destination vehicle.
    """
    require_shape(edge_index, (2, None), "gate_and_prune_edges.edge_index")
    require_shape(precedence, (num_vehicles,), "gate_and_prune_edges.precedence")
    if check_invariants:
        require_finite(precedence, "gate_and_prune_edges.precedence")
        require_probability_range(omega_raw, "gate_and_prune_edges.omega_raw")
        require_probability_range(kappa, "gate_and_prune_edges.kappa")
        require_probability_range(gamma, "gate_and_prune_edges.gamma")

    if max_parents_k < 1:
        raise DAGInvariantError(
            f"max_parents_k must be >= 1, got {max_parents_k}.",
            context={"max_parents_k": max_parents_k},
        )
    if dag_temperature <= 0.0:
        raise NumericalInvariantError(
            f"dag_temperature must be > 0, got {dag_temperature}.",
            context={"dag_temperature": dag_temperature},
        )

    device = omega_raw.device
    dtype = omega_raw.dtype
    e_cand = int(edge_index.shape[1])

    if num_vehicles == 0 or e_cand == 0:
        empty_long = torch.zeros((0,), dtype=torch.long, device=device)
        empty_f = torch.zeros((0,), dtype=dtype, device=device)
        return PrunedDAGStructure(
            num_vehicles=num_vehicles,
            k_max=max_parents_k,
            active_edge_indices=empty_long,
            src_idx=empty_long,
            dst_idx=empty_long,
            dst_rank=empty_long,
            omega=empty_f,
            kappa=empty_f,
            gamma=empty_f,
            phi_0=empty_f,
            phi_1=empty_f,
            parent_idx_padded=torch.zeros((num_vehicles, max_parents_k), dtype=torch.long, device=device),
            parent_edge_pos_padded=torch.zeros((num_vehicles, max_parents_k), dtype=torch.long, device=device),
            parent_mask=torch.zeros((num_vehicles, max_parents_k), dtype=torch.bool, device=device),
            phi_0_padded=torch.ones((num_vehicles, max_parents_k), dtype=dtype, device=device),
            phi_1_padded=torch.ones((num_vehicles, max_parents_k), dtype=dtype, device=device),
        )

    src_all = edge_index[0]
    dst_all = edge_index[1]
    if check_invariants and bool((src_all == dst_all).any().item()):
        raise DAGInvariantError("Self-loop detected in candidate edge_index passed to gate_and_prune_edges.")

    r_src = precedence.index_select(0, src_all)
    r_dst = precedence.index_select(0, dst_all)

    if training:
        smooth_gate = torch.sigmoid((r_src - r_dst) / dag_temperature)
        gated_omega = omega_raw * smooth_gate
        eligible_mask = gated_omega > 1e-6
    else:
        strict_order = (r_src > r_dst) | ((r_src == r_dst) & (src_all < dst_all))
        eligible_mask = strict_order & (omega_raw >= min_edge_prob)
        gated_omega = torch.where(eligible_mask, omega_raw, torch.zeros_like(omega_raw))

    eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1)
    if eligible_indices.numel() == 0:
        empty_long = torch.zeros((0,), dtype=torch.long, device=device)
        empty_f = torch.zeros((0,), dtype=dtype, device=device)
        return PrunedDAGStructure(
            num_vehicles=num_vehicles,
            k_max=max_parents_k,
            active_edge_indices=empty_long,
            src_idx=empty_long,
            dst_idx=empty_long,
            dst_rank=empty_long,
            omega=empty_f,
            kappa=empty_f,
            gamma=empty_f,
            phi_0=empty_f,
            phi_1=empty_f,
            parent_idx_padded=torch.zeros((num_vehicles, max_parents_k), dtype=torch.long, device=device),
            parent_edge_pos_padded=torch.zeros((num_vehicles, max_parents_k), dtype=torch.long, device=device),
            parent_mask=torch.zeros((num_vehicles, max_parents_k), dtype=torch.bool, device=device),
            phi_0_padded=torch.ones((num_vehicles, max_parents_k), dtype=dtype, device=device),
            phi_1_padded=torch.ones((num_vehicles, max_parents_k), dtype=dtype, device=device),
        )

    # Deterministic per-destination top-K_max selection without Python loops over E_cand:
    # Pack (dst_idx [19b], ieee754_inv_omega [24b], src_idx [19b]) into a single 62-bit int64 key
    # so ONE radix argsort orders strictly by (dst_idx ASC, -gated_omega ASC, src_idx ASC) with zero
    # float64 casts and zero precision loss up to 524,287 vehicles (with 3-pass fallback above that).
    dst_el = dst_all.index_select(0, eligible_indices)
    src_el = src_all.index_select(0, eligible_indices)
    w_el = gated_omega.index_select(0, eligible_indices).detach().float()

    if num_vehicles < (1 << 19):
        inv_w_bits = (2.0 - w_el.clamp(min=0.0, max=1.0)).contiguous().view(torch.int32).to(torch.int64) - 0x3F800000
        sort_key_i64 = (dst_el << 43) + (inv_w_bits << 19) + src_el
        perm = torch.argsort(sort_key_i64, stable=True)
    else:
        p1 = torch.argsort(src_el, stable=True)
        p2 = torch.argsort(-w_el.index_select(0, p1), stable=True)
        p12 = p1.index_select(0, p2)
        p3 = torch.argsort(dst_el.index_select(0, p12), stable=True)
        perm = p12.index_select(0, p3)

    sorted_el_idx = eligible_indices.index_select(0, perm)
    sorted_dst = dst_el.index_select(0, perm)

    # Compute within-destination rank [0, 1, ..., deg(i)-1] in O(E_el) with zero bincount CUDA sync
    n_el = int(sorted_dst.numel())
    counts = torch.zeros((num_vehicles,), dtype=torch.long, device=device).index_add(
        0, sorted_dst, torch.ones_like(sorted_dst)
    )
    group_starts = torch.cumsum(counts, dim=0) - counts
    within_rank = torch.arange(n_el, dtype=torch.long, device=device) - group_starts.index_select(0, sorted_dst)

    keep_in_sorted = within_rank < max_parents_k
    active_edge_indices = sorted_el_idx[keep_in_sorted]
    active_ranks = within_rank[keep_in_sorted]

    src_act = src_all.index_select(0, active_edge_indices)
    dst_act = dst_all.index_select(0, active_edge_indices)
    omega_act = gated_omega.index_select(0, active_edge_indices)
    kappa_act = kappa.index_select(0, active_edge_indices)
    gamma_act = gamma.index_select(0, active_edge_indices)
    phi_0_act, phi_1_act = parent_survival_factors(
        omega_act, kappa_act, gamma_act, check_invariants=check_invariants
    )

    parent_idx_padded = torch.zeros((num_vehicles, max_parents_k), dtype=torch.long, device=device)
    parent_edge_pos_padded = torch.zeros((num_vehicles, max_parents_k), dtype=torch.long, device=device)
    parent_mask = torch.zeros((num_vehicles, max_parents_k), dtype=torch.bool, device=device)
    phi_0_padded = torch.ones((num_vehicles, max_parents_k), dtype=dtype, device=device)
    phi_1_padded = torch.ones((num_vehicles, max_parents_k), dtype=dtype, device=device)

    e_act = int(active_edge_indices.numel())
    if e_act > 0:
        edge_pos = torch.arange(e_act, dtype=torch.long, device=device)
        parent_idx_padded[dst_act, active_ranks] = src_act
        parent_edge_pos_padded[dst_act, active_ranks] = edge_pos
        parent_mask[dst_act, active_ranks] = True
        phi_0_padded = phi_0_padded.index_put((dst_act, active_ranks), phi_0_act)
        phi_1_padded = phi_1_padded.index_put((dst_act, active_ranks), phi_1_act)

    return PrunedDAGStructure(
        num_vehicles=num_vehicles,
        k_max=max_parents_k,
        active_edge_indices=active_edge_indices,
        src_idx=src_act,
        dst_idx=dst_act,
        dst_rank=active_ranks,
        omega=omega_act,
        kappa=kappa_act,
        gamma=gamma_act,
        phi_0=phi_0_act,
        phi_1=phi_1_act,
        parent_idx_padded=parent_idx_padded,
        parent_edge_pos_padded=parent_edge_pos_padded,
        parent_mask=parent_mask,
        phi_0_padded=phi_0_padded,
        phi_1_padded=phi_1_padded,
    )


__all__ = [
    "CorridorSpillbackCPD",
    "PrunedDAGStructure",
    "STATIC_CPD_BASE_LEAK",
    "STATIC_CPD_PERSIST_WEIGHT",
    "STATIC_CPD_SPILL_WEIGHT",
    "expected_downstream_spillback",
    "gate_and_prune_edges",
    "generalized_noisy_or_base_survival",
    "heuristic_corridor_spillback_prob",
    "parent_survival_factors",
    "static_edge_cpd_parameters",
    "temporal_edge_prior",
]

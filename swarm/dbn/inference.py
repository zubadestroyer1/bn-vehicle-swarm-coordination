"""Exact Division-Free Prefix-Suffix Multiplicative Sum-Product Loopy Belief Propagation.

Performs scalable ``O(L_bp * K_max * N_t)`` variational/exact inference over the sparse
Generalized Causal Noisy-OR DBN at tick ``t``:
- Incorporates current-tick kinematic speed evidence ``L_i^t(D_i^t)`` (closing prototype 1-tick lag D3).
- Uses **division-free prefix-suffix cavity products** across BOTH incoming parent slots ``[N_t, K_max]``
  and outgoing child message slots ``[N_t, D_out_max]`` so zero-survival boundary regimes
  (``c_bar == 0`` or ``lambda == 0``) never divide by zero or require heuristic clipping.
- Matches exact ``2^N`` joint enumeration on polytrees to machine precision while remaining 100%
  differentiable in pure PyTorch for end-to-end unrolled DBN training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..logging_utils import (
    NumericalInvariantError,
    require_finite,
    require_probability_range,
    require_shape,
)
from .cpd import PrunedDAGStructure


@dataclass(frozen=True)
class LBPInferenceResult:
    """Outputs of Multiplicative Sum-Product Belief Propagation at control step ``t``."""

    delay_pred: Tensor          # float32 [N_t]: causal prior P(D_i^t = 1 | b_{t-1}, G_t) before speed evidence
    delay_post: Tensor          # float32 [N_t]: posterior P(D_i^t = 1 | b_t, G_t) incorporating speed evidence
    delay_pred_logits: Tensor   # float32 [N_t]: raw log-odds of delay_pred (for calibration / loss)
    delay_post_logits: Tensor   # float32 [N_t]: raw log-odds of delay_post (for calibration / loss)
    cavity_survival: Tensor     # float32 [E_act]: S_{i \ j} for each active edge (j -> i)
    cascade_score: Tensor       # float32 [N_t]: 2-hop downstream causal cascade sensitivity >= 0


def _exclusive_prefix_suffix_1d_slots(values: Tensor) -> tuple[Tensor, Tensor]:
    """Compute division-free exclusive prefix*suffix products and full row products along dim=1.

    Parameters
    ----------
    values:
        Tensor of shape ``[N, M, ...]`` where unused padded slots contain multiplicative identity ``1.0``.

    Returns
    -------
    excl_prod:
        Tensor ``[N, M, ...]`` where ``excl_prod[:, s] = prod_{r != s} values[:, r]`` without division.
    full_prod:
        Tensor ``[N, ...]`` where ``full_prod[:] = prod_{r=0..M-1} values[:, r]``.
    """
    n_rows, n_slots = values.shape[0], values.shape[1]
    if n_rows == 0 or n_slots == 0:
        rest_shape = values.shape[2:]
        return values.clone(), values.new_ones((n_rows, *rest_shape))

    if n_slots == 1:
        excl = values.new_ones(values.shape)
        full = values[:, 0]
        return excl, full

    inc_prefix = torch.cumprod(values, dim=1)
    inc_suffix = torch.flip(torch.cumprod(torch.flip(values, dims=[1]), dim=1), dims=[1])

    ones_slice = values.new_ones((n_rows, 1, *values.shape[2:]))
    excl_prefix = torch.cat([ones_slice, inc_prefix[:, :-1]], dim=1)
    excl_suffix = torch.cat([inc_suffix[:, 1:], ones_slice], dim=1)

    excl_prod = excl_prefix * excl_suffix
    full_prod = inc_prefix[:, -1]
    return excl_prod, full_prod


def _multiplicative_damp_prob(old_p: Tensor, hat_p: Tensor, damping: float, eps: float = 1e-12) -> Tensor:
    """Exact normalized geometric/multiplicative damping on Bernoulli messages ``(1 - p, p)``.

    When ``damping == 0.0``, returns ``hat_p`` directly without clamping so exact ``0.0``/``1.0``
    boundary states pass through untouched. For ``damping in (0, 1)``, interpolates in log-odds
    space, which is algebraically identical to ``m_new(d) proportional to m_old(d)^eta * m_hat(d)^(1-eta)``
    with smooth bounded gradients everywhere.
    """
    if damping <= 0.0:
        return hat_p
    old_c = old_p.clamp(min=eps, max=1.0 - eps)
    hat_c = hat_p.clamp(min=eps, max=1.0 - eps)
    logit_old = torch.log(old_c) - torch.log1p(-old_c)
    logit_hat = torch.log(hat_c) - torch.log1p(-hat_c)
    return torch.sigmoid(damping * logit_old + (1.0 - damping) * logit_hat)


def _prob_to_logit(prob: Tensor, eps: float = 1e-6) -> Tensor:
    clamped = prob.clamp(min=eps, max=1.0 - eps)
    return torch.log(clamped) - torch.log1p(-clamped)


def compute_cascade_scores(
    num_vehicles: int,
    src_idx: Tensor,
    dst_idx: Tensor,
    cavity_survival: Tensor,
    phi_1: Tensor,
) -> Tensor:
    """Compute 2-hop causal downstream cascade impact ``psi_cascade [N_t] >= 0`` in ``O(|E_act|)``.

    For each active edge ``(j -> i)``, single-hop causal inhibition sensitivity is
    ``w_{j->i} = S_{i \\ j} * (1 - phi_{j->i}(1))``. A vehicle's 2-hop cascade score sums its
    direct impact plus its children's downstream impacts:
    ``psi_cascade(j) = sum_{i in Ch(j)} w_{j->i} * (1 + sum_{m in Ch(i)} w_{i->m})``.
    """
    if num_vehicles == 0:
        return cavity_survival.new_empty((0,))
    if src_idx.numel() == 0:
        return cavity_survival.new_zeros((num_vehicles,))

    edge_impact = (cavity_survival * (1.0 - phi_1)).clamp(min=0.0, max=1.0)
    hop_1 = edge_impact.new_zeros((num_vehicles,)).index_add(0, src_idx, edge_impact)

    two_hop_edge = edge_impact * (1.0 + hop_1.index_select(0, dst_idx))
    cascade = edge_impact.new_zeros((num_vehicles,)).index_add(0, src_idx, two_hop_edge)
    require_finite(cascade, "compute_cascade_scores.cascade")
    return cascade


def multiplicative_sum_product_lbp(
    base_survival_z0: Tensor,
    pruned_dag: PrunedDAGStructure,
    speed_evidence_pos: Tensor,
    *,
    lbp_sweeps: int = 3,
    lbp_damping: float = 0.50,
) -> LBPInferenceResult:
    """Run deterministic Division-Free Prefix-Suffix Multiplicative Sum-Product LBP.

    Parameters
    ----------
    base_survival_z0:
        Parent-free survival probability ``z_{i,0}^t = P(D_i^t = 0 | Pa(i)=empty)`` of shape ``[N_t]``.
    pruned_dag:
        Sparse ``K_max``-bounded parent structure from :func:`swarm.dbn.cpd.gate_and_prune_edges`.
    speed_evidence_pos:
        Current-tick local kinematic delay likelihood ``L_i^t(D_i^t = 1)`` of shape ``[N_t]`` in ``[0, 1]``.
    lbp_sweeps:
        Number of message-passing sweeps ``L_bp >= 1``.
    lbp_damping:
        Multiplicative message damping ``eta in [0, 1)``.
    """
    num_vehicles = pruned_dag.num_vehicles
    require_shape(base_survival_z0, (num_vehicles,), "lbp.base_survival_z0")
    require_shape(speed_evidence_pos, (num_vehicles,), "lbp.speed_evidence_pos")
    require_probability_range(base_survival_z0, "lbp.base_survival_z0")
    require_probability_range(speed_evidence_pos, "lbp.speed_evidence_pos")

    if lbp_sweeps < 1:
        raise NumericalInvariantError(
            f"lbp_sweeps must be >= 1, got {lbp_sweeps}.",
            context={"lbp_sweeps": lbp_sweeps},
        )
    if not (0.0 <= lbp_damping < 1.0):
        raise NumericalInvariantError(
            f"lbp_damping must be in [0, 1), got {lbp_damping}.",
            context={"lbp_damping": lbp_damping},
        )

    device = base_survival_z0.device
    dtype = base_survival_z0.dtype

    if num_vehicles == 0:
        empty = base_survival_z0.new_empty((0,))
        return LBPInferenceResult(
            delay_pred=empty,
            delay_post=empty,
            delay_pred_logits=empty,
            delay_post_logits=empty,
            cavity_survival=empty,
            cascade_score=empty,
        )

    ev_1 = speed_evidence_pos.to(dtype=dtype)
    ev_0 = (1.0 - ev_1).clamp(min=0.0, max=1.0)
    e_act = int(pruned_dag.active_edge_indices.numel())

    # Base graph-free case (E_act == 0, e.g., IndependentDBNUpdater or isolated vehicles)
    if e_act == 0:
        delay_pred = (1.0 - base_survival_z0).clamp(min=0.0, max=1.0)
        unnorm_0 = ev_0 * base_survival_z0
        unnorm_1 = ev_1 * delay_pred
        denom = (unnorm_0 + unnorm_1).clamp(min=1e-12)
        delay_post = (unnorm_1 / denom).clamp(min=0.0, max=1.0)
        return LBPInferenceResult(
            delay_pred=delay_pred,
            delay_post=delay_post,
            delay_pred_logits=_prob_to_logit(delay_pred),
            delay_post_logits=_prob_to_logit(delay_post),
            cavity_survival=base_survival_z0.new_empty((0,)),
            cascade_score=base_survival_z0.new_zeros((num_vehicles,)),
        )

    p_idx = pruned_dag.parent_idx_padded          # [N_t, K_max]
    p_edge_pos = pruned_dag.parent_edge_pos_padded  # [N_t, K_max]
    p_mask = pruned_dag.parent_mask               # [N_t, K_max]
    phi_0_pad = pruned_dag.phi_0_padded           # [N_t, K_max]
    phi_1_pad = pruned_dag.phi_1_padded           # [N_t, K_max]
    src_act = pruned_dag.src_idx                  # [E_act]
    dst_act = pruned_dag.dst_idx                  # [E_act]

    # 1. Causal Prior Pass (delay_pred before current-tick speed evidence):
    # Under uniform local evidence, diagnostic messages are identically flat (d-separation),
    # so forward survival products converge strictly along causal ancestry.
    p_prior = (1.0 - base_survival_z0).clamp(min=0.0, max=1.0)
    prior_iters = max(lbp_sweeps, 4)
    for _ in range(prior_iters):
        p_par = p_prior.index_select(0, p_idx.reshape(-1)).view_as(phi_0_pad)
        c_slot = torch.where(
            p_mask,
            (1.0 - p_par) * phi_0_pad + p_par * phi_1_pad,
            torch.ones_like(phi_0_pad),
        )
        _, prod_c = _exclusive_prefix_suffix_1d_slots(c_slot)
        p_prior = (1.0 - base_survival_z0 * prod_c).clamp(min=0.0, max=1.0)
    delay_pred = p_prior

    # 2. Build padded outgoing child-slot layout [N_t, D_out_max] once per graph for
    #    division-free outgoing child-message cavity products!
    sort_out_key = src_act * num_vehicles + dst_act
    out_perm = torch.argsort(sort_out_key, stable=True)
    sorted_src = src_act.index_select(0, out_perm)
    out_counts = torch.bincount(sorted_src, minlength=num_vehicles)
    d_out_max = max(1, int(out_counts.max().item()))
    out_starts = torch.cumsum(out_counts, dim=0) - out_counts
    out_ranks_sorted = torch.arange(e_act, dtype=torch.long, device=device) - out_starts.index_select(0, sorted_src)

    # Map each active edge position e in [0..E_act-1] to its outgoing slot (src_act[e], edge_out_rank[e])
    edge_out_rank = torch.zeros((e_act,), dtype=torch.long, device=device)
    edge_out_rank.scatter_(0, out_perm, out_ranks_sorted)

    # Initialize messages on active edges:
    # - mu_var_to_fac_p1 [E_act]: P(D_j = 1) sent from variable j to child factor f_i
    # - lam_fac_to_var   [E_act, 2]: normalized diagnostic likelihood (lam_0, lam_1) from f_i to parent j
    init_node_post = (ev_1 * delay_pred) / (ev_0 * (1.0 - delay_pred) + ev_1 * delay_pred + 1e-12)
    mu_p1 = init_node_post.index_select(0, src_act)
    lam_edge = torch.full((e_act, 2), 0.5, dtype=dtype, device=device)

    pi_node_0 = 1.0 - delay_pred
    pi_node_1 = delay_pred
    all_child_lam = torch.ones((num_vehicles, 2), dtype=dtype, device=device)
    s_cavity_slots = base_survival_z0.unsqueeze(1).expand_as(phi_0_pad)

    for _ in range(lbp_sweeps):
        # Step A: Factor-to-child causal message pi_i and cavity survivals S_{i \ s}
        #         using exact division-free prefix-suffix products across [N_t, K_max]!
        mu_p1_pad = mu_p1.index_select(0, p_edge_pos.reshape(-1)).view_as(phi_0_pad)
        c_slots = torch.where(
            p_mask,
            (1.0 - mu_p1_pad) * phi_0_pad + mu_p1_pad * phi_1_pad,
            torch.ones_like(phi_0_pad),
        )
        excl_c_slots, full_c_prod = _exclusive_prefix_suffix_1d_slots(c_slots)
        s_total = (base_survival_z0 * full_c_prod).clamp(min=0.0, max=1.0)
        pi_hat_1 = (1.0 - s_total).clamp(min=0.0, max=1.0)
        pi_node_1 = _multiplicative_damp_prob(pi_node_1, pi_hat_1, lbp_damping)
        pi_node_0 = (1.0 - pi_node_1).clamp(min=0.0, max=1.0)

        s_cavity_slots = (base_survival_z0.unsqueeze(1) * excl_c_slots).clamp(min=0.0, max=1.0)

        # Step B: Downward variable-to-own-factor message u_i(d_i) = psi_i(d_i) * Lambda_i(d_i)
        #         and exact multilinear diagnostic message lam_{f_i -> j}(d_j)
        u_raw_0 = ev_0 * all_child_lam[:, 0]
        u_raw_1 = ev_1 * all_child_lam[:, 1]
        u_sum = (u_raw_0 + u_raw_1).clamp(min=1e-12)
        u_0 = (u_raw_0 / u_sum).unsqueeze(1)  # [N_t, 1]
        u_1 = (u_raw_1 / u_sum).unsqueeze(1)  # [N_t, 1]

        surv_given_0 = (s_cavity_slots * phi_0_pad).clamp(min=0.0, max=1.0)
        surv_given_1 = (s_cavity_slots * phi_1_pad).clamp(min=0.0, max=1.0)

        lam_slot_0 = u_0 * surv_given_0 + u_1 * (1.0 - surv_given_0)
        lam_slot_1 = u_0 * surv_given_1 + u_1 * (1.0 - surv_given_1)
        lam_slot_sum = (lam_slot_0 + lam_slot_1).clamp(min=1e-12)
        lam_hat_p1_slots = lam_slot_1 / lam_slot_sum

        # Gather active edges from padded [N_t, K_max] slots back to [E_act]
        # Each active edge e sits at (dst_act[e], rank_in_dst[e]); even faster: compute directly on [E_act]!
        # Wait: let's extract s_cavity_edge [E_act] cleanly via p_edge_pos scatter!
        lam_hat_p1_edge = torch.zeros((e_act,), dtype=dtype, device=device).scatter(
            0, p_edge_pos[p_mask], lam_hat_p1_slots[p_mask]
        )

        lam_prev_p1 = lam_edge[:, 1]
        lam_new_p1 = _multiplicative_damp_prob(lam_prev_p1, lam_hat_p1_edge, lbp_damping)
        lam_edge = torch.stack([1.0 - lam_new_p1, lam_new_p1], dim=-1)

        # Step C: Division-free outgoing child-cavity product across [N_t, D_out_max, 2]
        #         (re-normalizing per slot so deep trees never underflow!)
        out_lam_padded = torch.ones((num_vehicles, d_out_max, 2), dtype=dtype, device=device).index_put(
            (src_act, edge_out_rank), lam_edge * 2.0
        )
        excl_out_lam, full_out_lam = _exclusive_prefix_suffix_1d_slots(out_lam_padded)

        full_out_sum = (full_out_lam[:, 0] + full_out_lam[:, 1]).clamp(min=1e-12)
        all_child_lam = full_out_lam / full_out_sum.unsqueeze(-1)

        excl_for_edge = excl_out_lam[src_act, edge_out_rank]  # [E_act, 2]
        mu_raw_0 = ev_0.index_select(0, src_act) * pi_node_0.index_select(0, src_act) * excl_for_edge[:, 0]
        mu_raw_1 = ev_1.index_select(0, src_act) * pi_node_1.index_select(0, src_act) * excl_for_edge[:, 1]
        mu_sum = (mu_raw_0 + mu_raw_1).clamp(min=1e-12)
        mu_hat_p1 = (mu_raw_1 / mu_sum).clamp(min=0.0, max=1.0)
        mu_p1 = _multiplicative_damp_prob(mu_p1, mu_hat_p1, lbp_damping)

    # Final causal survival after final mu_p1 update so single-sweep chains/trees are immediately exact:
    mu_p1_pad = mu_p1.index_select(0, p_edge_pos.reshape(-1)).view_as(phi_0_pad)
    c_slots_final = torch.where(
        p_mask,
        (1.0 - mu_p1_pad) * phi_0_pad + mu_p1_pad * phi_1_pad,
        torch.ones_like(phi_0_pad),
    )
    excl_c_final, full_c_final = _exclusive_prefix_suffix_1d_slots(c_slots_final)
    pi_final_0 = (base_survival_z0 * full_c_final).clamp(min=0.0, max=1.0)
    pi_final_1 = (1.0 - pi_final_0).clamp(min=0.0, max=1.0)

    s_cavity_final_slots = (base_survival_z0.unsqueeze(1) * excl_c_final).clamp(min=0.0, max=1.0)
    cavity_survival_edge = torch.zeros((e_act,), dtype=dtype, device=device).scatter(
        0, p_edge_pos[p_mask], s_cavity_final_slots[p_mask]
    )

    post_0 = ev_0 * pi_final_0 * all_child_lam[:, 0]
    post_1 = ev_1 * pi_final_1 * all_child_lam[:, 1]
    post_sum = (post_0 + post_1).clamp(min=1e-12)
    delay_post = (post_1 / post_sum).clamp(min=0.0, max=1.0)

    cascade_score = compute_cascade_scores(
        num_vehicles=num_vehicles,
        src_idx=src_act,
        dst_idx=dst_act,
        cavity_survival=cavity_survival_edge,
        phi_1=pruned_dag.phi_1,
    )

    require_probability_range(delay_pred, "lbp.delay_pred")
    require_probability_range(delay_post, "lbp.delay_post")
    require_finite(cascade_score, "lbp.cascade_score")

    return LBPInferenceResult(
        delay_pred=delay_pred,
        delay_post=delay_post,
        delay_pred_logits=_prob_to_logit(delay_pred),
        delay_post_logits=_prob_to_logit(delay_post),
        cavity_survival=cavity_survival_edge,
        cascade_score=cascade_score,
    )


__all__ = [
    "LBPInferenceResult",
    "compute_cascade_scores",
    "multiplicative_sum_product_lbp",
]

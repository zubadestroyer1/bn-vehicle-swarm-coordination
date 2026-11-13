"""Unrolled differentiable Neural DBN trainer, Boundary Lock B L-BFGS calibration, & held-out predictive evaluator.

Implements:
1. Contiguous multi-step trajectory window unrolling (``T_burn`` burn-in + ``T_seq`` BPTT ticks)
   through :func:`swarm.dbn.updater.forward_dbn_step` so ``ContinuousDecayGRU`` vehicle/corridor
   states learn real temporal filtering across sensor & V2X dropouts without cross-episode leakage.
2. Multi-task causal loss:
   - Generalized Causal Noisy-OR predictive delay BCE (``delay_pred``)
   - Post-LBP posterior delay BCE (``delay_post``)
   - Class-balanced focal directed causal edge BCE (``edge_logits``)
   - Pairwise causal precedence margin ranking loss (``precedence``)
   - Continuous horizon delay Huber loss (``expected_delay_s``)
   - Corridor upstream-tail spillback BCE (``corridor_spillback_logits``)
   - Temporal edge smoothness penalty + sparse edge ``L_1`` regularization
3. Cloud + local compute readiness: explicit ``ComputeDevice`` (``cpu``, ``mps``, ``cuda``),
   ``PrecisionMode`` (``fp32``, ``bf16`` autocast), ``torch.compile``, and ``DistributedDataParallel``
   via ``torchrun`` environment variables (``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE``).
4. Boundary Lock B (``#19`` §2): validation selection & post-hoc bounded L-BFGS-B temperature
   calibration ``(T_cal_edge, T_cal_delay, T_cal_link)`` executed strictly in ``model.eval()``
   hard-DAG mode on ``VAL_SEEDS`` (``2000..2009``).
5. Held-out predictive structure & delay evaluation (``ROC-AUC``, ``PR-AUC``, ``Brier``,
   ``Log-Loss``, ``10-bin ECE`` pre- & post-``T_cal``) across ID & Zero-Shot OOD ``TEST_SEEDS``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict, dataclass
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import (
    DBNConfig,
    TEST_SEEDS,
    TRAIN_SEEDS,
    TrainConfig,
    VAL_SEEDS,
    config_sha256,
    get_model_preset,
    validate_compute_device,
)
from ..contracts import (
    CORRIDOR_EDGE_FEATURE_DIM,
    CORRIDOR_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    SCHEMA_VERSION,
)
from ..dbn.calibration import (
    TemperatureScales,
    calibrate_temperatures,
    compute_predictive_metrics,
)
from ..dbn.candidates import CandidateGraph
from ..dbn.cpd import temporal_edge_prior
from ..dbn.features import FeatureNormalizer, GraphFeatureBatch
from ..dbn.transformer import STGraphTransformerDBN, TorchTemporalMemory, TransformerStepOutputs
from ..dbn.updater import (
    DifferentiableStepOutput,
    PrecomputedStepInputs,
    forward_dbn_precomputed_step,
    forward_dbn_step,
    load_dbn_checkpoint,
    save_dbn_checkpoint,
)
from ..logging_utils import (
    CheckpointIntegrityError,
    ConfigValidationError,
    NumericalInvariantError,
    get_logger,
    require_finite,
    require_shape,
    update_crash_context,
)
from .rollout_collector import RolloutSplit, load_rollout_shards

_LOG = get_logger("pipelines.trainer")

_LOSS_COMPONENT_KEYS: tuple[str, ...] = (
    "total",
    "delay_pred",
    "delay_post",
    "edge_focal",
    "rank",
    "tau_huber",
    "spillback",
    "smooth",
    "l1",
)


def _flush_log_streams() -> None:
    """Flush all configured logger handlers and standard streams for live unbuffered progress."""
    for logger_name in ("swarm", ""):
        lg = logging.getLogger(logger_name)
        for handler in lg.handlers:
            handler.flush()
    sys.stdout.flush()
    sys.stderr.flush()


def _to_numpy_f32(val: Any, name: str) -> np.ndarray:
    if isinstance(val, Tensor):
        arr = val.detach().cpu().to(torch.float32).numpy()
    else:
        arr = np.asarray(val, dtype=np.float32)
    require_finite(arr, name)
    return arr


def _to_numpy_i64(val: Any) -> np.ndarray:
    if isinstance(val, Tensor):
        return val.detach().cpu().to(torch.int64).numpy()
    return np.asarray(val, dtype=np.int64)


def _resolve_checkpoint_output_file(output_checkpoint_path: Path | str) -> Path:
    out_p = Path(output_checkpoint_path).expanduser().resolve()
    if out_p.suffix == ".pt":
        return out_p
    return out_p / "neural_dbn.pt"


def _resolve_checkpoint_input_file(checkpoint_path: Path | str) -> Path:
    ckpt_p = Path(checkpoint_path).expanduser().resolve()
    if ckpt_p.is_dir() and (ckpt_p / "neural_dbn.pt").is_file():
        return ckpt_p / "neural_dbn.pt"
    return ckpt_p


def shard_step_to_feature_batch(
    shard: Mapping[str, Any],
    step_dict: Mapping[str, Any],
    dbn_cfg: DBNConfig = DBNConfig(),
) -> GraphFeatureBatch:
    """Reconstruct a validated :class:`GraphFeatureBatch` from a serialized rollout shard step."""
    veh_ids = tuple(str(v) for v in step_dict["veh_ids"])
    corridor_ids = tuple(str(c) for c in shard["corridor_ids"])
    n_v = len(veh_ids)
    n_c = len(corridor_ids)

    x_veh = _to_numpy_f32(
        step_dict["x_veh"] if "x_veh" in step_dict else step_dict["X_V"],
        "step.x_veh",
    )
    x_corridor = _to_numpy_f32(
        step_dict["x_corridor"] if "x_corridor" in step_dict else step_dict["X_C"],
        "step.x_corridor",
    )
    x_edge = _to_numpy_f32(
        step_dict["x_edge"] if "x_edge" in step_dict else step_dict["X_E"],
        "step.x_edge",
    )
    x_corridor_edge = _to_numpy_f32(
        step_dict["x_corridor_edge"] if "x_corridor_edge" in step_dict else step_dict["X_CC"],
        "step.x_corridor_edge",
    )

    require_shape(x_veh, (n_v, NODE_FEATURE_DIM), "step.x_veh")
    require_shape(x_corridor, (n_c, CORRIDOR_FEATURE_DIM), "step.x_corridor")
    require_shape(x_edge, (None, EDGE_FEATURE_DIM), "step.x_edge")
    require_shape(x_corridor_edge, (None, CORRIDOR_EDGE_FEATURE_DIM), "step.x_corridor_edge")

    e_cand = int(x_edge.shape[0])
    e_corr = int(x_corridor_edge.shape[0])

    edge_index = _to_numpy_i64(step_dict["edge_index"]).reshape(2, e_cand)
    corridor_edge_index = _to_numpy_i64(shard["corridor_edge_index"]).reshape(2, e_corr)

    if "edge_relations" in step_dict:
        relation_types = _to_numpy_i64(step_dict["edge_relations"]).reshape(e_cand)
    elif "relation_types" in step_dict:
        relation_types = _to_numpy_i64(step_dict["relation_types"]).reshape(e_cand)
    elif e_cand > 0:
        relation_types = np.argmax(x_edge[:, :4], axis=1).astype(np.int64)
    else:
        relation_types = np.zeros((0,), dtype=np.int64)

    p_conflict = (
        _to_numpy_f32(step_dict["p_conflict"], "step.p_conflict").reshape(e_cand)
        if "p_conflict" in step_dict
        else (x_edge[:, 8].copy() if e_cand > 0 else np.zeros((0,), dtype=np.float32))
    )
    p_merge = (
        _to_numpy_f32(step_dict["p_merge"], "step.p_merge").reshape(e_cand)
        if "p_merge" in step_dict
        else (x_edge[:, 9].copy() if e_cand > 0 else np.zeros((0,), dtype=np.float32))
    )
    signed_gap = (
        _to_numpy_f32(step_dict["signed_path_gap_m"], "step.signed_path_gap_m").reshape(e_cand)
        if "signed_path_gap_m" in step_dict
        else ((x_edge[:, 6] * 50.0).astype(np.float32) if e_cand > 0 else np.zeros((0,), dtype=np.float32))
    )

    veh_corridor_idx = _to_numpy_i64(step_dict["veh_corridor_idx"]).reshape(n_v)

    if (
        "veh_turn_src_idx" in step_dict
        and "veh_turn_out_corridor_idx" in step_dict
        and "veh_turn_prob" in step_dict
    ):
        veh_turn_src_idx = _to_numpy_i64(step_dict["veh_turn_src_idx"]).reshape(-1)
        veh_turn_out_corridor_idx = _to_numpy_i64(step_dict["veh_turn_out_corridor_idx"]).reshape(-1)
        veh_turn_prob = _to_numpy_f32(step_dict["veh_turn_prob"], "step.veh_turn_prob").reshape(-1)
    elif "veh_out_corridor_idx" in step_dict and "veh_out_turn_probs" in step_dict:
        out_c_mat = _to_numpy_i64(step_dict["veh_out_corridor_idx"])
        out_p_mat = _to_numpy_f32(step_dict["veh_out_turn_probs"], "step.veh_out_turn_probs")
        valid_mask = out_c_mat >= 0
        src_rows, _ = np.nonzero(valid_mask)
        veh_turn_src_idx = src_rows.astype(np.int64)
        veh_turn_out_corridor_idx = out_c_mat[valid_mask].astype(np.int64)
        veh_turn_prob = out_p_mat[valid_mask].astype(np.float32)
    else:
        veh_turn_src_idx = np.zeros((0,), dtype=np.int64)
        veh_turn_out_corridor_idx = np.zeros((0,), dtype=np.int64)
        veh_turn_prob = np.zeros((0,), dtype=np.float32)

    if "topo_precedence" in step_dict:
        topo_precedence = _to_numpy_f32(step_dict["topo_precedence"], "step.topo_precedence").reshape(n_v)
    else:
        free_eta = _to_numpy_f32(step_dict["free_eta_s"], "step.free_eta_s").reshape(n_v)
        tail_blk = _to_numpy_f32(step_dict["tail_blockage_score"], "step.tail_blockage_score").reshape(n_v)
        topo_precedence = (-free_eta + 15.0 * tail_blk).astype(np.float32)

    if "speed_evidence_pos" in step_dict:
        speed_evidence_pos = _to_numpy_f32(
            step_dict["speed_evidence_pos"], "step.speed_evidence_pos"
        ).reshape(n_v)
    elif n_v > 0:
        v_norm = x_veh[:, 3]
        obs_age = _to_numpy_f32(step_dict["obs_age_s"], "step.obs_age_s").reshape(n_v)
        sig = 1.0 / (1.0 + np.exp(-((0.45 - v_norm) / 0.10)))
        speed_evidence_pos = (0.5 + 0.42 * np.exp(-obs_age / 3.0) * (2.0 * sig - 1.0)).astype(np.float32)
    else:
        speed_evidence_pos = np.zeros((0,), dtype=np.float32)

    prev_delay_post = (
        np.clip(x_veh[:, 13], 1e-4, 1.0 - 1e-4).astype(np.float32)
        if n_v > 0
        else np.zeros((0,), dtype=np.float32)
    )
    prev_edge_omega = (
        np.clip(x_edge[:, 15], 0.0, 1.0).astype(np.float32)
        if e_cand > 0
        else np.zeros((0,), dtype=np.float32)
    )
    edge_persisted_mask = (
        np.clip(x_edge[:, 17], 0.0, 1.0).astype(np.float32)
        if e_cand > 0
        else np.zeros((0,), dtype=np.float32)
    )
    prior_edge_prob = np.asarray(
        temporal_edge_prior(
            prev_edge_omega,
            edge_persisted_mask,
            prior_stay_prob=dbn_cfg.prior_stay_prob,
            prior_base_prob=dbn_cfg.prior_base_prob,
        ),
        dtype=np.float32,
    )

    sim_time = float(step_dict["sim_time"])
    dt = float(shard.get("dt", 0.5))

    cand_graph = CandidateGraph(
        sim_time=sim_time,
        veh_ids=veh_ids,
        veh_id_to_idx={vid: idx for idx, vid in enumerate(veh_ids)},
        edge_index=edge_index,
        relation_types=relation_types,
        p_conflict=p_conflict,
        p_merge=p_merge,
        signed_path_gap_m=signed_gap,
    )

    return GraphFeatureBatch(
        sim_time=sim_time,
        dt=dt,
        candidate_graph=cand_graph,
        corridor_ids=corridor_ids,
        corridor_to_idx={cid: idx for idx, cid in enumerate(corridor_ids)},
        x_veh=x_veh,
        x_corridor=x_corridor,
        x_edge=x_edge,
        x_corridor_edge=x_corridor_edge,
        corridor_edge_index=corridor_edge_index,
        veh_corridor_idx=veh_corridor_idx,
        veh_turn_src_idx=veh_turn_src_idx,
        veh_turn_out_corridor_idx=veh_turn_out_corridor_idx,
        veh_turn_prob=veh_turn_prob,
        topo_precedence=topo_precedence,
        speed_evidence_pos=speed_evidence_pos,
        prev_delay_post=prev_delay_post,
        prev_edge_omega=prev_edge_omega,
        prior_edge_prob=prior_edge_prob,
        edge_persisted_mask=edge_persisted_mask,
    )


class RolloutSequenceDataset:
    """Contiguous single-episode trajectory window dataset for unrolled BPTT training."""

    def __init__(
        self,
        shards: Sequence[Mapping[str, Any]],
        *,
        dbn_cfg: DBNConfig,
        window_len: int,
        stride: int,
    ) -> None:
        if not shards:
            raise ConfigValidationError("RolloutSequenceDataset requires at least one shard.")
        if window_len < 1 or stride < 1:
            raise ConfigValidationError(
                f"Require window_len >= 1 and stride >= 1, got ({window_len}, {stride})."
            )

        self.episodes: list[list[tuple[GraphFeatureBatch, dict[str, Tensor]]]] = []
        self.windows: list[tuple[int, int, int]] = []

        for ep_idx, shard in enumerate(shards):
            raw_steps = shard.get("steps")
            if not isinstance(raw_steps, Sequence) or len(raw_steps) == 0:
                raise ConfigValidationError(
                    f"Shard at index {ep_idx} contains zero trajectory steps."
                )
            ep_items: list[tuple[GraphFeatureBatch, dict[str, Tensor]]] = []
            for s_dict in raw_steps:
                fb = shard_step_to_feature_batch(shard, s_dict, dbn_cfg)
                y_delay_t = torch.as_tensor(s_dict["y_delay"], dtype=torch.float32)
                raw_rp = torch.as_tensor(s_dict["rank_pairs"], dtype=torch.long)
                if raw_rp.numel() == 0:
                    rank_pairs_t = raw_rp.reshape(0, 2)
                else:
                    if raw_rp.ndim != 2 or int(raw_rp.shape[1]) != 2:
                        raise ConfigValidationError(
                            f"Shard[{ep_idx}] rank_pairs must have shape (M, 2), got {tuple(raw_rp.shape)}."
                        )
                    n_v_step = int(y_delay_t.numel())
                    if int(raw_rp.min().item()) < 0 or int(raw_rp.max().item()) >= n_v_step:
                        raise ConfigValidationError(
                            f"Shard[{ep_idx}] rank_pairs indices out of range [0, {n_v_step})."
                        )
                    rank_pairs_t = raw_rp
                lbls = {
                    "y_delay": y_delay_t,
                    "tau_delay_s": torch.as_tensor(s_dict["tau_delay_s"], dtype=torch.float32),
                    "y_spillback": torch.as_tensor(s_dict["y_spillback"], dtype=torch.float32),
                    "y_edge": torch.as_tensor(s_dict["y_edge"], dtype=torch.float32),
                    "rank_pairs": rank_pairs_t,
                }
                for lname, lten in lbls.items():
                    if lname != "rank_pairs":
                        require_finite(lten, f"shard[{ep_idx}].{lname}")
                ep_items.append((fb, lbls))
            self.episodes.append(ep_items)

            ep_len = len(ep_items)
            seg_starts = [0]
            for k in range(1, ep_len):
                fb_prev = ep_items[k - 1][0]
                fb_cur = ep_items[k][0]
                if (fb_cur.sim_time - fb_prev.sim_time) > (1.5 * max(fb_cur.dt, 0.05) + 1e-6):
                    seg_starts.append(k)
            seg_starts.append(ep_len)

            for seg_i in range(len(seg_starts) - 1):
                s0 = seg_starts[seg_i]
                s1 = seg_starts[seg_i + 1]
                seg_len = s1 - s0
                if seg_len <= 0:
                    continue
                if seg_len <= window_len:
                    self.windows.append((ep_idx, s0, s1))
                else:
                    starts = list(range(s0, s1 - window_len + 1, stride))
                    last_start = s1 - window_len
                    if starts[-1] != last_start:
                        starts.append(last_start)
                    for st in starts:
                        self.windows.append((ep_idx, st, st + window_len))

    def __len__(self) -> int:
        return len(self.windows)

    def all_feature_batches(self) -> list[GraphFeatureBatch]:
        return [fb for ep in self.episodes for (fb, _) in ep]

    def get_window(self, index: int) -> list[tuple[GraphFeatureBatch, dict[str, Tensor]]]:
        ep_idx, start_i, end_i = self.windows[index]
        return self.episodes[ep_idx][start_i:end_i]


def compute_step_multitask_loss(
    *,
    step_out: DifferentiableStepOutput,
    feat_batch: GraphFeatureBatch,
    labels: Mapping[str, Tensor],
    train_cfg: TrainConfig,
    prev_edge_omega_detached: Mapping[tuple[str, str], Tensor] | None,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    """Compute the 8-term multi-task causal loss in FP32 for one supervised unrolled step."""
    zero = torch.zeros((), dtype=torch.float32, device=device)

    y_delay = labels["y_delay"].to(device=device, dtype=torch.float32)
    tau_delay = labels["tau_delay_s"].to(device=device, dtype=torch.float32)
    y_spill = labels["y_spillback"].to(device=device, dtype=torch.float32)
    y_edge = labels["y_edge"].to(device=device, dtype=torch.float32)
    rank_pairs = labels["rank_pairs"].to(device=device, dtype=torch.long)

    n_v = int(y_delay.numel())
    n_c = int(y_spill.numel())
    e_cand = int(y_edge.numel())
    if rank_pairs.numel() == 0:
        rank_pairs = rank_pairs.reshape(0, 2)
    else:
        if rank_pairs.ndim != 2 or int(rank_pairs.shape[1]) != 2:
            raise ConfigValidationError(
                f"rank_pairs must have canonical shape (M, 2), got {tuple(rank_pairs.shape)}."
            )
        if int(rank_pairs.min().item()) < 0 or int(rank_pairs.max().item()) >= n_v:
            raise ConfigValidationError(
                f"rank_pairs indices out of range [0, {n_v})."
            )
    m_rank = int(rank_pairs.shape[0])

    # 1 & 2. Predictive Noisy-OR delay BCE + Post-LBP posterior delay BCE
    if n_v > 0:
        n_pos_d = float((y_delay > 0.5).sum().item())
        n_neg_d = float(n_v) - n_pos_d
        pos_w_d = torch.tensor(
            min(8.0, max(1.0, (n_neg_d + 1.0) / (n_pos_d + 1.0))),
            dtype=torch.float32,
            device=device,
        )
        loss_delay_pred = F.binary_cross_entropy_with_logits(
            step_out.delay_pred_logits_scaled.float(),
            y_delay,
            pos_weight=pos_w_d,
        )
        loss_delay_post = F.binary_cross_entropy_with_logits(
            step_out.delay_post_logits_scaled.float(),
            y_delay,
            pos_weight=pos_w_d,
        )
        huber_vec = F.huber_loss(
            step_out.expected_delay_s.float(),
            tau_delay,
            reduction="none",
            delta=2.0,
        )
        w_tau = torch.where(y_delay > 0.5, 1.0, 0.35)
        loss_tau = torch.mean(w_tau * huber_vec)
    else:
        loss_delay_pred = zero
        loss_delay_post = zero
        loss_tau = zero

    # 3 & 8. Class-balanced focal directed edge BCE + sparse edge L1 penalty
    if e_cand > 0:
        logits_e = step_out.edge_logits_scaled.float()
        bce_raw = F.binary_cross_entropy_with_logits(logits_e, y_edge, reduction="none")
        probs_e = torch.sigmoid(logits_e)
        p_t = y_edge * probs_e + (1.0 - y_edge) * (1.0 - probs_e)
        focal_mod = (1.0 - p_t).clamp(min=0.0).pow(float(train_cfg.focal_gamma))

        e_pos = float((y_edge > 0.5).sum().item())
        e_neg = float(e_cand) - e_pos
        pos_w_e = min(12.0, max(1.0, (e_neg + 1.0) / (e_pos + 1.0)))
        alpha_w = torch.where(y_edge > 0.5, pos_w_e, 1.0)

        loss_edge = torch.mean(alpha_w * focal_mod * bce_raw)
        loss_l1 = torch.mean(step_out.omega_raw.float())
    else:
        loss_edge = zero
        loss_l1 = zero

    # 4. Pairwise causal precedence margin ranking loss
    if m_rank > 0 and n_v >= 2:
        prec = step_out.transformer_out.precedence.float()
        r_j = prec[rank_pairs[:, 0]]
        r_i = prec[rank_pairs[:, 1]]
        loss_rank = torch.mean(F.relu(1.0 - (r_j - r_i)))
    else:
        loss_rank = zero

    # 6. Corridor upstream-tail spillback BCE
    if n_c > 0:
        c_pos = float((y_spill > 0.5).sum().item())
        c_neg = float(n_c) - c_pos
        pos_w_c = torch.tensor(
            min(8.0, max(1.0, (c_neg + 1.0) / (c_pos + 1.0))),
            dtype=torch.float32,
            device=device,
        )
        loss_spill = F.binary_cross_entropy_with_logits(
            step_out.corridor_spillback_logits_scaled.float(),
            y_spill,
            pos_weight=pos_w_c,
        )
    else:
        loss_spill = zero

    # 7. Temporal edge smoothness penalty across persisted candidate pairs
    if e_cand > 0 and prev_edge_omega_detached:
        cand = feat_batch.candidate_graph
        src_arr = cand.edge_index[0]
        dst_arr = cand.edge_index[1]
        matched_curr: list[Tensor] = []
        matched_prev: list[Tensor] = []
        om_f32 = step_out.omega_raw.float()
        for k in range(e_cand):
            pair = (cand.veh_ids[int(src_arr[k])], cand.veh_ids[int(dst_arr[k])])
            prev_val = prev_edge_omega_detached.get(pair)
            if prev_val is not None:
                matched_curr.append(om_f32[k])
                matched_prev.append(prev_val)
        if matched_curr:
            curr_vec = torch.stack(matched_curr, dim=0)
            prev_vec = torch.stack(matched_prev, dim=0)
            loss_smooth = torch.mean((curr_vec - prev_vec) ** 2)
        else:
            loss_smooth = zero
    else:
        loss_smooth = zero

    total_loss = (
        loss_delay_pred
        + float(train_cfg.loss_w_post_delay) * loss_delay_post
        + float(train_cfg.loss_w_edge) * loss_edge
        + float(train_cfg.loss_w_rank) * loss_rank
        + float(train_cfg.loss_w_tau) * loss_tau
        + float(train_cfg.loss_w_spillback) * loss_spill
        + float(train_cfg.loss_w_smooth) * loss_smooth
        + float(train_cfg.loss_w_l1) * loss_l1
    )
    require_finite(total_loss, "compute_step_multitask_loss.total_loss")

    parts = {
        "total": float(total_loss.detach().item()),
        "delay_pred": float(loss_delay_pred.detach().item()),
        "delay_post": float(loss_delay_post.detach().item()),
        "edge_focal": float(loss_edge.detach().item()),
        "rank": float(loss_rank.detach().item()),
        "tau_huber": float(loss_tau.detach().item()),
        "spillback": float(loss_spill.detach().item()),
        "smooth": float(loss_smooth.detach().item()),
        "l1": float(loss_l1.detach().item()),
    }
    return total_loss, parts


def _build_step_carries(
    step_out: DifferentiableStepOutput,
    feat_batch: GraphFeatureBatch,
    *,
    detach: bool,
) -> tuple[TorchTemporalMemory, dict[str, Tensor], dict[tuple[str, str], Tensor], dict[tuple[str, str], Tensor]]:
    cand = feat_batch.candidate_graph
    next_mem = step_out.next_memory.detach_state() if detach else step_out.next_memory

    d_post = step_out.delay_post.detach() if detach else step_out.delay_post
    delay_map = {vid: d_post[idx] for idx, vid in enumerate(cand.veh_ids)}

    om_raw = step_out.omega_raw.detach() if detach else step_out.omega_raw
    om_det = step_out.omega_raw.detach().float()
    edge_map: dict[tuple[str, str], Tensor] = {}
    edge_det_map: dict[tuple[str, str], Tensor] = {}
    if cand.num_edges > 0:
        src_arr = cand.edge_index[0]
        dst_arr = cand.edge_index[1]
        for k in range(cand.num_edges):
            pair = (cand.veh_ids[int(src_arr[k])], cand.veh_ids[int(dst_arr[k])])
            edge_map[pair] = om_raw[k]
            edge_det_map[pair] = om_det[k]

    return next_mem, delay_map, edge_map, edge_det_map


@dataclass(frozen=True)
class PrecomputedDeviceStep:
    """Pre-staged step inputs, supervised targets, and class-balance weights on compute device."""

    inputs: PrecomputedStepInputs
    y_delay: Tensor         # [N_t] float32 on device
    tau_delay: Tensor       # [N_t] float32 on device
    w_tau: Tensor           # [N_t] float32 on device
    y_spill: Tensor         # [N_C] float32 on device
    y_edge: Tensor          # [E_t] float32 on device
    alpha_w_edge: Tensor    # [E_t] float32 on device
    rank_pairs: Tensor      # [M, 2] int64 on device
    pos_w_d: Tensor         # [] float32 on device
    pos_w_c: Tensor         # [] float32 on device


def _compute_string_alignment_np(
    curr_keys: tuple[str, ...],
    prev_keys: tuple[str, ...],
) -> tuple[bool, np.ndarray, np.ndarray]:
    """Compute exact last-wins entity index alignment ``prev_keys -> curr_keys`` on CPU."""
    n_curr = len(curr_keys)
    n_prev = len(prev_keys)
    empty = np.zeros((0,), dtype=np.int64)
    if n_curr == 0 or n_prev == 0:
        return (n_curr == 0 and n_prev == 0), empty, empty
    if curr_keys == prev_keys and len(set(curr_keys)) == n_curr:
        ident = np.arange(n_curr, dtype=np.int64)
        return True, ident, ident
    prev_lookup = {k: i for i, k in enumerate(prev_keys)}
    c_list: list[int] = []
    p_list: list[int] = []
    for ci, k in enumerate(curr_keys):
        pi = prev_lookup.get(k)
        if pi is not None:
            c_list.append(ci)
            p_list.append(pi)
    if not c_list:
        return False, empty, empty
    c_arr = np.asarray(c_list, dtype=np.int64)
    p_arr = np.asarray(p_list, dtype=np.int64)
    return False, c_arr, p_arr


def _compute_edge_alignment_np(
    curr_veh_ids: tuple[str, ...],
    curr_edge_index: np.ndarray,
    prev_veh_ids: tuple[str, ...],
    prev_edge_index: np.ndarray,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """Compute O(E log E) last-wins candidate-pair alignment ``(src_id, dst_id)`` via integer codes."""
    e_curr = int(curr_edge_index.shape[1])
    e_prev = int(prev_edge_index.shape[1])
    empty = np.zeros((0,), dtype=np.int64)
    if e_curr == 0 or e_prev == 0:
        return (e_curr == 0 and e_prev == 0), empty, empty

    n_prev = len(prev_veh_ids)
    n_curr = len(curr_veh_ids)
    if n_prev == 0 or n_curr == 0:
        return False, empty, empty

    prev_vid_to_uid: dict[str, int] = {}
    prev_pos_to_uid = np.empty((n_prev,), dtype=np.int64)
    for idx, vid in enumerate(prev_veh_ids):
        uid = prev_vid_to_uid.setdefault(vid, len(prev_vid_to_uid))
        prev_pos_to_uid[idx] = uid
    n_uid = len(prev_vid_to_uid)

    su_p = prev_pos_to_uid[prev_edge_index[0]]
    du_p = prev_pos_to_uid[prev_edge_index[1]]
    prev_code = su_p * n_uid + du_p
    order_p = np.argsort(prev_code, kind="stable")
    prev_sorted = prev_code[order_p]

    curr_pos_to_uid = np.fromiter(
        (prev_vid_to_uid.get(vid, -1) for vid in curr_veh_ids),
        dtype=np.int64,
        count=n_curr,
    )
    su_c = curr_pos_to_uid[curr_edge_index[0]]
    du_c = curr_pos_to_uid[curr_edge_index[1]]
    cand_c = np.flatnonzero((su_c >= 0) & (du_c >= 0))
    if cand_c.size == 0:
        return False, empty, empty

    curr_code = su_c[cand_c] * n_uid + du_c[cand_c]
    ins_right = np.searchsorted(prev_sorted, curr_code, side="right")
    pos = np.maximum(0, ins_right - 1)
    hit = (ins_right > 0) & (prev_sorted[pos] == curr_code)
    if not np.any(hit):
        return False, empty, empty

    edge_curr_idx = cand_c[hit].astype(np.int64)
    edge_prev_idx = order_p[pos[hit]].astype(np.int64)
    edge_same = (
        e_curr == e_prev
        and edge_curr_idx.size == e_curr
        and bool(np.array_equal(edge_prev_idx, np.arange(e_curr, dtype=np.int64)))
    )
    return edge_same, edge_curr_idx, edge_prev_idx


def prestage_rollout_dataset(
    dataset: RolloutSequenceDataset,
    normalizer: FeatureNormalizer,
    device: torch.device,
    *,
    include_raw_edge: bool = False,
) -> list[list[PrecomputedDeviceStep]]:
    """Pre-normalize, compute static step-to-step index alignments, and stage all tensors on ``device``."""
    if not normalizer.fitted:
        raise ConfigValidationError("prestage_rollout_dataset requires a fitted FeatureNormalizer.")

    empty_long_dev = torch.zeros((0,), dtype=torch.long, device=device)
    prestaged_episodes: list[list[PrecomputedDeviceStep]] = []

    for ep in dataset.episodes:
        ep_out: list[PrecomputedDeviceStep] = []
        prev_fb: GraphFeatureBatch | None = None

        for step_idx, (fb, lbls) in enumerate(ep):
            is_contig = (
                step_idx > 0
                and prev_fb is not None
                and (fb.sim_time - prev_fb.sim_time) <= (1.5 * max(fb.dt, 0.05) + 1e-6)
            )
            z_v_np, z_c_np, z_e_np, z_cc_np = normalizer.transform_arrays(
                fb.x_veh,
                fb.x_corridor,
                fb.x_edge,
                fb.x_corridor_edge,
            )

            cand = fb.candidate_graph
            n_v = cand.num_vehicles
            n_c = len(fb.corridor_ids)
            e_cand = cand.num_edges

            if is_contig and prev_fb is not None:
                veh_same, v_c_np, v_p_np = _compute_string_alignment_np(
                    cand.veh_ids, prev_fb.candidate_graph.veh_ids
                )
                corr_same, c_c_np, c_p_np = _compute_string_alignment_np(
                    fb.corridor_ids, prev_fb.corridor_ids
                )
                edge_same, e_c_np, e_p_np = _compute_edge_alignment_np(
                    cand.veh_ids,
                    cand.edge_index,
                    prev_fb.candidate_graph.veh_ids,
                    prev_fb.candidate_graph.edge_index,
                )
                veh_curr_idx = (
                    torch.as_tensor(v_c_np, dtype=torch.long, device=device)
                    if v_c_np.size > 0
                    else empty_long_dev
                )
                veh_prev_idx = (
                    torch.as_tensor(v_p_np, dtype=torch.long, device=device)
                    if v_p_np.size > 0
                    else empty_long_dev
                )
                corr_curr_idx = (
                    torch.as_tensor(c_c_np, dtype=torch.long, device=device)
                    if c_c_np.size > 0
                    else empty_long_dev
                )
                corr_prev_idx = (
                    torch.as_tensor(c_p_np, dtype=torch.long, device=device)
                    if c_p_np.size > 0
                    else empty_long_dev
                )
                edge_curr_idx = (
                    torch.as_tensor(e_c_np, dtype=torch.long, device=device)
                    if e_c_np.size > 0
                    else empty_long_dev
                )
                edge_prev_idx = (
                    torch.as_tensor(e_p_np, dtype=torch.long, device=device)
                    if e_p_np.size > 0
                    else empty_long_dev
                )
            else:
                veh_same = False
                corr_same = False
                edge_same = False
                veh_curr_idx = empty_long_dev
                veh_prev_idx = empty_long_dev
                corr_curr_idx = empty_long_dev
                corr_prev_idx = empty_long_dev
                edge_curr_idx = empty_long_dev
                edge_prev_idx = empty_long_dev

            inputs = PrecomputedStepInputs(
                sim_time=fb.sim_time,
                dt=fb.dt,
                veh_ids=cand.veh_ids,
                corridor_ids=fb.corridor_ids,
                num_vehicles=n_v,
                num_corridors=n_c,
                num_edges=e_cand,
                z_veh=torch.as_tensor(z_v_np, dtype=torch.float32, device=device),
                z_corr=torch.as_tensor(z_c_np, dtype=torch.float32, device=device),
                z_edge=torch.as_tensor(z_e_np, dtype=torch.float32, device=device),
                z_cc=torch.as_tensor(z_cc_np, dtype=torch.float32, device=device),
                x_corr_raw=torch.as_tensor(fb.x_corridor, dtype=torch.float32, device=device),
                x_edge_raw=(
                    torch.as_tensor(fb.x_edge, dtype=torch.float32, device=device)
                    if include_raw_edge
                    else None
                ),
                edge_index=torch.as_tensor(cand.edge_index, dtype=torch.long, device=device),
                corr_edge_index=torch.as_tensor(fb.corridor_edge_index, dtype=torch.long, device=device),
                veh_corridor_idx=torch.as_tensor(fb.veh_corridor_idx, dtype=torch.long, device=device),
                veh_turn_src_idx=torch.as_tensor(fb.veh_turn_src_idx, dtype=torch.long, device=device),
                veh_turn_out_corridor_idx=torch.as_tensor(
                    fb.veh_turn_out_corridor_idx, dtype=torch.long, device=device
                ),
                veh_turn_prob=torch.as_tensor(fb.veh_turn_prob, dtype=torch.float32, device=device),
                topo_precedence=torch.as_tensor(fb.topo_precedence, dtype=torch.float32, device=device),
                speed_evidence_pos=torch.as_tensor(fb.speed_evidence_pos, dtype=torch.float32, device=device),
                base_prev_delay_post=torch.as_tensor(fb.prev_delay_post, dtype=torch.float32, device=device),
                base_prior_edge_prob=torch.as_tensor(fb.prior_edge_prob, dtype=torch.float32, device=device),
                is_contiguous_from_prev=is_contig,
                veh_same=veh_same,
                veh_curr_idx=veh_curr_idx,
                veh_prev_idx=veh_prev_idx,
                corr_same=corr_same,
                corr_curr_idx=corr_curr_idx,
                corr_prev_idx=corr_prev_idx,
                edge_same=edge_same,
                edge_curr_idx=edge_curr_idx,
                edge_prev_idx=edge_prev_idx,
            )

            y_d_cpu = lbls["y_delay"].to(dtype=torch.float32)
            tau_cpu = lbls["tau_delay_s"].to(dtype=torch.float32)
            y_s_cpu = lbls["y_spillback"].to(dtype=torch.float32)
            y_e_cpu = lbls["y_edge"].to(dtype=torch.float32)
            rp_cpu = lbls["rank_pairs"].to(dtype=torch.long)
            if rp_cpu.numel() == 0:
                rp_cpu = rp_cpu.reshape(0, 2)

            if n_v > 0:
                n_pos_d = float((y_d_cpu > 0.5).sum().item())
                n_neg_d = float(n_v) - n_pos_d
                pos_w_d_val = min(8.0, max(1.0, (n_neg_d + 1.0) / (n_pos_d + 1.0)))
                w_tau_cpu = torch.where(y_d_cpu > 0.5, 1.0, 0.35)
            else:
                pos_w_d_val = 1.0
                w_tau_cpu = torch.zeros((0,), dtype=torch.float32)

            if e_cand > 0:
                e_pos = float((y_e_cpu > 0.5).sum().item())
                e_neg = float(e_cand) - e_pos
                pos_w_e_val = min(12.0, max(1.0, (e_neg + 1.0) / (e_pos + 1.0)))
                alpha_w_cpu = torch.where(y_e_cpu > 0.5, pos_w_e_val, 1.0)
            else:
                alpha_w_cpu = torch.zeros((0,), dtype=torch.float32)

            if n_c > 0:
                c_pos = float((y_s_cpu > 0.5).sum().item())
                c_neg = float(n_c) - c_pos
                pos_w_c_val = min(8.0, max(1.0, (c_neg + 1.0) / (c_pos + 1.0)))
            else:
                pos_w_c_val = 1.0

            ep_out.append(
                PrecomputedDeviceStep(
                    inputs=inputs,
                    y_delay=y_d_cpu.to(device=device),
                    tau_delay=tau_cpu.to(device=device),
                    w_tau=w_tau_cpu.to(device=device, dtype=torch.float32),
                    y_spill=y_s_cpu.to(device=device),
                    y_edge=y_e_cpu.to(device=device),
                    alpha_w_edge=alpha_w_cpu.to(device=device, dtype=torch.float32),
                    rank_pairs=rp_cpu.to(device=device),
                    pos_w_d=torch.tensor(pos_w_d_val, dtype=torch.float32, device=device),
                    pos_w_c=torch.tensor(pos_w_c_val, dtype=torch.float32, device=device),
                )
            )
            prev_fb = fb

        prestaged_episodes.append(ep_out)

    return prestaged_episodes


def compute_precomputed_step_loss(
    *,
    step_out: DifferentiableStepOutput,
    pstep: PrecomputedDeviceStep,
    train_cfg: TrainConfig,
    prev_omega_det_vec: Tensor | None,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Compute the 8-term multi-task causal loss in FP32 with zero Python loops or CPU syncs."""
    zero = torch.zeros((), dtype=torch.float32, device=device)
    inp = pstep.inputs
    n_v = inp.num_vehicles
    n_c = inp.num_corridors
    e_cand = inp.num_edges
    m_rank = int(pstep.rank_pairs.shape[0])

    if n_v > 0:
        loss_delay_pred = F.binary_cross_entropy_with_logits(
            step_out.delay_pred_logits_scaled.float(),
            pstep.y_delay,
            pos_weight=pstep.pos_w_d,
        )
        loss_delay_post = F.binary_cross_entropy_with_logits(
            step_out.delay_post_logits_scaled.float(),
            pstep.y_delay,
            pos_weight=pstep.pos_w_d,
        )
        huber_vec = F.huber_loss(
            step_out.expected_delay_s.float(),
            pstep.tau_delay,
            reduction="none",
            delta=2.0,
        )
        loss_tau = torch.mean(pstep.w_tau * huber_vec)
    else:
        loss_delay_pred = zero
        loss_delay_post = zero
        loss_tau = zero

    if e_cand > 0:
        logits_e = step_out.edge_logits_scaled.float()
        bce_raw = F.binary_cross_entropy_with_logits(logits_e, pstep.y_edge, reduction="none")
        probs_e = torch.sigmoid(logits_e)
        p_t = pstep.y_edge * probs_e + (1.0 - pstep.y_edge) * (1.0 - probs_e)
        focal_mod = (1.0 - p_t).clamp(min=0.0).pow(float(train_cfg.focal_gamma))
        loss_edge = torch.mean(pstep.alpha_w_edge * focal_mod * bce_raw)
        loss_l1 = torch.mean(step_out.omega_raw.float())
    else:
        loss_edge = zero
        loss_l1 = zero

    if m_rank > 0 and n_v >= 2:
        prec = step_out.transformer_out.precedence.float()
        r_j = prec.index_select(0, pstep.rank_pairs[:, 0])
        r_i = prec.index_select(0, pstep.rank_pairs[:, 1])
        loss_rank = torch.mean(F.relu(1.0 - (r_j - r_i)))
    else:
        loss_rank = zero

    if n_c > 0:
        loss_spill = F.binary_cross_entropy_with_logits(
            step_out.corridor_spillback_logits_scaled.float(),
            pstep.y_spill,
            pos_weight=pstep.pos_w_c,
        )
    else:
        loss_spill = zero

    if e_cand > 0 and prev_omega_det_vec is not None and inp.edge_curr_idx.numel() > 0:
        om_f32 = step_out.omega_raw.float()
        if inp.edge_same and prev_omega_det_vec.shape[0] == e_cand:
            diff = om_f32 - prev_omega_det_vec
        else:
            diff = om_f32.index_select(0, inp.edge_curr_idx) - prev_omega_det_vec.index_select(0, inp.edge_prev_idx)
        loss_smooth = torch.mean(diff * diff)
    else:
        loss_smooth = zero

    total_loss = (
        loss_delay_pred
        + float(train_cfg.loss_w_post_delay) * loss_delay_post
        + float(train_cfg.loss_w_edge) * loss_edge
        + float(train_cfg.loss_w_rank) * loss_rank
        + float(train_cfg.loss_w_tau) * loss_tau
        + float(train_cfg.loss_w_spillback) * loss_spill
        + float(train_cfg.loss_w_smooth) * loss_smooth
        + float(train_cfg.loss_w_l1) * loss_l1
    )

    parts_vec = torch.stack(
        (
            total_loss.detach(),
            loss_delay_pred.detach(),
            loss_delay_post.detach(),
            loss_edge.detach(),
            loss_rank.detach(),
            loss_tau.detach(),
            loss_spill.detach(),
            loss_smooth.detach(),
            loss_l1.detach(),
        ),
        dim=0,
    )
    return total_loss, parts_vec


def _build_vector_carries(
    step_out: DifferentiableStepOutput,
    *,
    detach: bool,
) -> tuple[TorchTemporalMemory, Tensor, Tensor, Tensor]:
    """Extract contiguous 1D device carry tensors across steps with zero Python edge loops."""
    next_mem = step_out.next_memory.detach_state() if detach else step_out.next_memory
    d_post_vec = step_out.delay_post.detach() if detach else step_out.delay_post
    om_raw_vec = step_out.omega_raw.detach() if detach else step_out.omega_raw
    om_det_vec = step_out.omega_raw.detach().float()
    return next_mem, d_post_vec, om_raw_vec, om_det_vec


def _parts_vec_to_dict(parts_vec: Tensor) -> dict[str, float]:
    vals = parts_vec.detach().cpu().to(torch.float64).tolist()
    return {k: float(v) for k, v in zip(_LOSS_COMPONENT_KEYS, vals, strict=True)}


def _evaluate_validation_episodes(
    *,
    raw_model: STGraphTransformerDBN,
    normalizer: FeatureNormalizer,
    val_dataset: RolloutSequenceDataset,
    train_cfg: TrainConfig,
    dbn_cfg: DBNConfig,
    device: torch.device,
    temperatures: TemperatureScales | None = None,
    collect_logits: bool = False,
    prestaged_episodes: Sequence[Sequence[PrecomputedDeviceStep]] | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Run full-episode sequential evaluation strictly in ``model.eval()`` hard-DAG mode (Boundary Lock B)."""
    raw_model.eval()
    if raw_model.training:
        raise ConfigValidationError("Boundary Lock B violation: model remained in training mode during validation.")

    episodes_dev = (
        prestaged_episodes
        if prestaged_episodes is not None
        else prestage_rollout_dataset(val_dataset, normalizer, device)
    )
    temps = temperatures if temperatures is not None else TemperatureScales.uncalibrated_unit()
    loss_sums_vec = torch.zeros((len(_LOSS_COMPONENT_KEYS),), dtype=torch.float32, device=device)
    total_steps = 0

    edge_logits_all: list[Tensor] = []
    edge_targets_all: list[Tensor] = []
    delay_logits_all: list[Tensor] = []
    delay_post_logits_all: list[Tensor] = []
    delay_targets_all: list[Tensor] = []
    link_logits_all: list[Tensor] = []
    link_targets_all: list[Tensor] = []

    with torch.no_grad():
        for ep_steps in episodes_dev:
            prev_mem: TorchTemporalMemory | None = None
            prev_delay_vec: Tensor | None = None
            prev_omega_vec: Tensor | None = None
            prev_omega_det_vec: Tensor | None = None

            for pstep in ep_steps:
                if not pstep.inputs.is_contiguous_from_prev:
                    prev_mem = None
                    prev_delay_vec = None
                    prev_omega_vec = None
                    prev_omega_det_vec = None

                step_out = forward_dbn_precomputed_step(
                    model=raw_model,
                    step=pstep.inputs,
                    dbn_cfg=dbn_cfg,
                    temperatures=temps,
                    prev_memory=prev_mem,
                    prev_delay_vec=prev_delay_vec,
                    prev_omega_vec=prev_omega_vec,
                )
                _, parts_vec = compute_precomputed_step_loss(
                    step_out=step_out,
                    pstep=pstep,
                    train_cfg=train_cfg,
                    prev_omega_det_vec=prev_omega_det_vec,
                    device=device,
                )
                loss_sums_vec = loss_sums_vec + parts_vec
                total_steps += 1

                if collect_logits:
                    if pstep.inputs.num_edges > 0:
                        edge_logits_all.append(step_out.transformer_out.edge_logits.detach().float().reshape(-1))
                        edge_targets_all.append(pstep.y_edge.detach().float().reshape(-1))
                    if pstep.inputs.num_vehicles > 0:
                        delay_logits_all.append(step_out.lbp_result.delay_pred_logits.detach().float().reshape(-1))
                        delay_post_logits_all.append(step_out.lbp_result.delay_post_logits.detach().float().reshape(-1))
                        delay_targets_all.append(pstep.y_delay.detach().float().reshape(-1))
                    if pstep.inputs.num_corridors > 0:
                        link_logits_all.append(
                            step_out.transformer_out.corridor_spillback_logits.detach().float().reshape(-1)
                        )
                        link_targets_all.append(pstep.y_spill.detach().float().reshape(-1))

                prev_mem, prev_delay_vec, prev_omega_vec, prev_omega_det_vec = _build_vector_carries(
                    step_out, detach=True
                )

    require_finite(loss_sums_vec, "val.loss_sums_vec")
    denom = float(max(1, total_steps))
    mean_losses = _parts_vec_to_dict(loss_sums_vec / denom)

    collected: dict[str, np.ndarray] = {}
    if collect_logits:
        collected = {
            "edge_logits": (
                torch.cat(edge_logits_all, dim=0).cpu().numpy()
                if edge_logits_all
                else np.zeros((0,), dtype=np.float32)
            ),
            "edge_targets": (
                torch.cat(edge_targets_all, dim=0).cpu().numpy()
                if edge_targets_all
                else np.zeros((0,), dtype=np.float32)
            ),
            "delay_pred_logits": (
                torch.cat(delay_logits_all, dim=0).cpu().numpy()
                if delay_logits_all
                else np.zeros((0,), dtype=np.float32)
            ),
            "delay_post_logits": (
                torch.cat(delay_post_logits_all, dim=0).cpu().numpy()
                if delay_post_logits_all
                else np.zeros((0,), dtype=np.float32)
            ),
            "delay_targets": (
                torch.cat(delay_targets_all, dim=0).cpu().numpy()
                if delay_targets_all
                else np.zeros((0,), dtype=np.float32)
            ),
            "link_logits": (
                torch.cat(link_logits_all, dim=0).cpu().numpy()
                if link_logits_all
                else np.zeros((0,), dtype=np.float32)
            ),
            "link_targets": (
                torch.cat(link_targets_all, dim=0).cpu().numpy()
                if link_targets_all
                else np.zeros((0,), dtype=np.float32)
            ),
        }

    return mean_losses, collected


def _format_head_predictive_summary(
    raw_logits: np.ndarray,
    targets: np.ndarray,
    temperature: float,
    *,
    name: str,
) -> dict[str, Any]:
    """Compute pre- and post-calibration predictive metrics formatted for ``plots_2d.py``."""
    if raw_logits.size == 0 or targets.size == 0:
        raise ConfigValidationError(
            f"Cannot compute predictive metrics for empty head {name!r} (size=0).",
            context={"head": name},
        )
    logits_f64 = np.asarray(raw_logits, dtype=np.float64).reshape(-1)
    targets_f64 = np.asarray(targets, dtype=np.float64).reshape(-1)
    require_finite(logits_f64, f"{name}.raw_logits")
    require_finite(targets_f64, f"{name}.targets")

    probs_pre = 1.0 / (1.0 + np.exp(-np.clip(logits_f64, -40.0, 40.0)))
    probs_post = 1.0 / (1.0 + np.exp(-np.clip(logits_f64 / float(temperature), -40.0, 40.0)))

    pre_m = compute_predictive_metrics(probs_pre, targets_f64, n_bins=10, name=f"{name}.pre")
    post_m = compute_predictive_metrics(probs_post, targets_f64, n_bins=10, name=f"{name}.post")

    post_nonempty = [b for b in post_m["reliability_bins"] if int(b["count"]) > 0]
    pre_nonempty = [b for b in pre_m["reliability_bins"] if int(b["count"]) > 0]

    rel_dict: dict[str, Any] = {
        "bins": post_m["reliability_bins"],
        "pre_bins": pre_m["reliability_bins"],
        "bin_confidences": [float(b["mean_pred"]) for b in post_nonempty],
        "bin_accuracies": [float(b["empirical_pos_rate"]) for b in post_nonempty],
        "pre_bin_confidences": [float(b["mean_pred"]) for b in pre_nonempty],
        "pre_bin_accuracies": [float(b["empirical_pos_rate"]) for b in pre_nonempty],
    }

    summary: dict[str, Any] = {
        "n_samples": int(post_m["n_samples"]),
        "positive_rate": float(post_m["positive_rate"]),
        "temperature": float(temperature),
        "brier_score": float(post_m["brier_score"]),
        "brier_score_pre": float(pre_m["brier_score"]),
        "log_loss": float(post_m["log_loss"]),
        "log_loss_pre": float(pre_m["log_loss"]),
        "ece": float(post_m["ece"]),
        "ece_post": float(post_m["ece"]),
        "ece_pre": float(pre_m["ece"]),
        "mce": float(post_m["mce"]),
        "accuracy": float(post_m["accuracy"]),
        "reliability_bins": rel_dict,
        "roc_curve": post_m["roc_curve"],
        "pr_curve": post_m["pr_curve"],
    }
    if post_m["roc_auc"] is not None and post_m["pr_auc"] is not None:
        summary["roc_auc"] = float(post_m["roc_auc"])
        summary["pr_auc"] = float(post_m["pr_auc"])
    return summary


def _setup_distributed(train_cfg: TrainConfig, device_type: str) -> tuple[bool, int, int, torch.device]:
    if not train_cfg.distributed:
        return False, 0, 1, torch.device(device_type)

    if not dist.is_available():
        raise ConfigValidationError("TrainConfig.distributed=True, but torch.distributed is unavailable.")

    if not dist.is_initialized():
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            raise ConfigValidationError(
                "TrainConfig.distributed=True requires torchrun environment variables RANK and WORLD_SIZE.",
                context={"env_keys": sorted(k for k in os.environ if "RANK" in k or "WORLD" in k)},
            )
        backend = "nccl" if device_type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
    else:
        dev = torch.device(device_type)
    return True, rank, world_size, dev


def _cosine_warmup_lr(step_idx: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if total_steps <= 1:
        return base_lr
    if warmup_steps > 0 and step_idx < warmup_steps:
        return base_lr * float(step_idx + 1) / float(warmup_steps)
    denom = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step_idx - warmup_steps) / float(denom)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (0.05 + 0.95 * cosine)


def train_neural_dbn(
    train_dir: Path | str,
    val_dir: Path | str,
    output_checkpoint_path: Path | str,
    *,
    train_cfg: TrainConfig = TrainConfig(),
    dbn_cfg: DBNConfig = DBNConfig(),
    report_path: Path | str | None = None,
) -> dict[str, Any]:
    """Train ``STGraphTransformerDBN`` via unrolled BPTT, calibrate ``T_cal`` on ``VAL_SEEDS``, and save checkpoint."""
    device_name, precision = validate_compute_device(train_cfg.device, train_cfg.precision)
    preset_spec = get_model_preset(train_cfg.model_preset)
    is_ddp, rank, world_size, device = _setup_distributed(train_cfg, device_name)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    update_crash_context(
        command="train",
        model_preset=preset_spec.name,
        device=str(device),
        precision=precision,
        seed=train_cfg.seed,
    )

    torch.manual_seed(train_cfg.seed + rank)
    np.random.seed(train_cfg.seed + rank)

    _, train_shards = load_rollout_shards(train_dir, expected_split="train")
    _, val_shards = load_rollout_shards(val_dir, expected_split="val")

    train_seeds = {int(s["seed"]) for s in train_shards}
    val_seeds = {int(s["seed"]) for s in val_shards}
    if (train_seeds & set(TEST_SEEDS)) or (val_seeds & set(TEST_SEEDS)):
        raise ConfigValidationError(
            "Anti-leakage lock triggered: TEST_SEEDS detected inside train/val shards."
        )
    if not train_seeds.issubset(set(TRAIN_SEEDS)):
        raise ConfigValidationError(
            f"Train shards contain seeds outside TRAIN_SEEDS: {sorted(train_seeds - set(TRAIN_SEEDS))}."
        )
    if not val_seeds.issubset(set(VAL_SEEDS)):
        raise ConfigValidationError(
            f"Validation shards contain seeds outside VAL_SEEDS: {sorted(val_seeds - set(VAL_SEEDS))}."
        )
    if train_seeds & val_seeds:
        raise ConfigValidationError(
            f"Overlapping seeds between train and val splits: {sorted(train_seeds & val_seeds)}."
        )

    window_len = int(preset_spec.t_burn + preset_spec.t_seq)
    stride = max(1, int(preset_spec.t_seq) // 2)

    train_dataset = RolloutSequenceDataset(
        train_shards,
        dbn_cfg=dbn_cfg,
        window_len=window_len,
        stride=stride,
    )
    val_dataset = RolloutSequenceDataset(
        val_shards,
        dbn_cfg=dbn_cfg,
        window_len=window_len,
        stride=max(1, int(preset_spec.t_seq)),
    )

    # Fit FeatureNormalizer strictly on training feature batches ONLY
    normalizer = FeatureNormalizer().fit(train_dataset.all_feature_batches())

    t_prestage_start = time.perf_counter()
    train_prestaged = prestage_rollout_dataset(train_dataset, normalizer, device)
    val_prestaged = prestage_rollout_dataset(val_dataset, normalizer, device)
    prestage_wall_s = float(time.perf_counter() - t_prestage_start)

    raw_model = STGraphTransformerDBN(
        preset=preset_spec,
        prediction_horizon_s=dbn_cfg.prediction_horizon_s,
    ).to(device)

    use_bf16 = precision == "bf16"
    autocast_device_type = device.type if device.type in ("cuda", "cpu") else "cpu"

    forward_model: Any = raw_model
    if train_cfg.compile_model:
        forward_model = torch.compile(forward_model)
    if is_ddp:
        ddp_kwargs: dict[str, Any] = {"find_unused_parameters": True}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [device.index]
        forward_model = torch.nn.parallel.DistributedDataParallel(forward_model, **ddp_kwargs)

    optim_target_params = list(raw_model.parameters())
    optimizer = torch.optim.AdamW(
        optim_target_params,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    num_windows = len(train_dataset)
    windows_per_rank = max(1, math.ceil(num_windows / float(world_size)))
    batches_per_epoch = max(1, math.ceil(windows_per_rank / float(train_cfg.batch_size)))
    total_optim_steps = max(1, train_cfg.epochs * batches_per_epoch)
    warmup_steps = int(round(total_optim_steps * train_cfg.warmup_fraction))
    unit_temps = TemperatureScales.uncalibrated_unit()

    if rank == 0:
        _LOG.info(
            "Pre-staged %d train episodes (%d windows) + %d val episodes on %s in %.2fs.",
            len(train_prestaged),
            num_windows,
            len(val_prestaged),
            device,
            prestage_wall_s,
        )
        _flush_log_streams()

    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = {k: v.detach().float().cpu().clone() for k, v in raw_model.state_dict().items()}
    epochs_no_improve = 0
    optim_step_idx = 0
    epoch_history: list[dict[str, Any]] = []
    t_train_start = time.perf_counter()

    for epoch in range(1, train_cfg.epochs + 1):
        t_epoch_start = time.perf_counter()
        raw_model.train()
        if train_cfg.epochs > 1:
            prog = float(epoch - 1) / float(train_cfg.epochs - 1)
        else:
            prog = 1.0
        dag_temp = float(
            train_cfg.dag_temp_start
            * ((train_cfg.dag_temp_end / train_cfg.dag_temp_start) ** prog)
        )

        epoch_rng = np.random.default_rng(train_cfg.seed * 10007 + epoch)
        perm = epoch_rng.permutation(num_windows).tolist()
        if is_ddp:
            # Pad deterministically so every DDP rank executes identical batch counts
            padded_len = windows_per_rank * world_size
            while len(perm) < padded_len:
                perm.extend(perm[: padded_len - len(perm)])
            rank_indices = perm[rank::world_size]
        else:
            rank_indices = perm

        epoch_train_losses_vec = torch.zeros((len(_LOSS_COMPONENT_KEYS),), dtype=torch.float32, device=device)
        epoch_sup_windows = 0
        last_grad_norm = 0.0

        for b_start in range(0, len(rank_indices), train_cfg.batch_size):
            batch_win_indices = rank_indices[b_start : b_start + train_cfg.batch_size]
            lr_now = _cosine_warmup_lr(
                optim_step_idx,
                total_optim_steps,
                warmup_steps,
                train_cfg.learning_rate,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            optimizer.zero_grad(set_to_none=True)
            batch_scale = 1.0 / float(len(batch_win_indices))

            for win_idx in batch_win_indices:
                ep_idx, start_i, end_i = train_dataset.windows[win_idx]
                window = train_prestaged[ep_idx][start_i:end_i]
                w_len = len(window)
                eff_burn = 0 if w_len <= preset_spec.t_burn else min(preset_spec.t_burn, w_len - 1)
                n_sup = max(1, w_len - eff_burn)

                prev_mem: TorchTemporalMemory | None = None
                prev_delay_vec: Tensor | None = None
                prev_omega_vec: Tensor | None = None
                prev_omega_det_vec: Tensor | None = None

                # Burn-in steps warm up ContinuousDecayGRU recurrent hidden states without backprop
                if eff_burn > 0:
                    with torch.no_grad(), torch.autocast(
                        device_type=autocast_device_type,
                        dtype=torch.bfloat16,
                        enabled=use_bf16,
                    ):
                        for s_i in range(eff_burn):
                            pstep_burn = window[s_i]
                            out_burn = forward_dbn_precomputed_step(
                                model=forward_model,
                                step=pstep_burn.inputs,
                                dbn_cfg=dbn_cfg,
                                temperatures=unit_temps,
                                prev_memory=prev_mem,
                                prev_delay_vec=prev_delay_vec,
                                prev_omega_vec=prev_omega_vec,
                                dag_temperature=dag_temp,
                            )
                            prev_mem, prev_delay_vec, prev_omega_vec, prev_omega_det_vec = _build_vector_carries(
                                out_burn, detach=True
                            )

                # Supervised unrolled BPTT steps
                win_loss_accum = torch.zeros((), dtype=torch.float32, device=device)
                win_parts_accum = torch.zeros((len(_LOSS_COMPONENT_KEYS),), dtype=torch.float32, device=device)
                for s_i in range(eff_burn, w_len):
                    pstep_sup = window[s_i]
                    with torch.autocast(
                        device_type=autocast_device_type,
                        dtype=torch.bfloat16,
                        enabled=use_bf16,
                    ):
                        out_sup = forward_dbn_precomputed_step(
                            model=forward_model,
                            step=pstep_sup.inputs,
                            dbn_cfg=dbn_cfg,
                            temperatures=unit_temps,
                            prev_memory=prev_mem,
                            prev_delay_vec=prev_delay_vec,
                            prev_omega_vec=prev_omega_vec,
                            dag_temperature=dag_temp,
                        )
                    step_loss, step_parts_vec = compute_precomputed_step_loss(
                        step_out=out_sup,
                        pstep=pstep_sup,
                        train_cfg=train_cfg,
                        prev_omega_det_vec=prev_omega_det_vec,
                        device=device,
                    )
                    win_loss_accum = win_loss_accum + step_loss
                    win_parts_accum = win_parts_accum + step_parts_vec
                    prev_mem, prev_delay_vec, prev_omega_vec, prev_omega_det_vec = _build_vector_carries(
                        out_sup, detach=False
                    )

                mean_win_loss = win_loss_accum / float(n_sup)
                require_finite(mean_win_loss, "train.mean_win_loss")
                (mean_win_loss * batch_scale).backward()

                epoch_train_losses_vec = epoch_train_losses_vec + (win_parts_accum / float(n_sup))
                epoch_sup_windows += 1

            grad_norm_t = torch.nn.utils.clip_grad_norm_(
                optim_target_params,
                max_norm=train_cfg.grad_clip_norm,
            )
            last_grad_norm = float(grad_norm_t.detach().item() if isinstance(grad_norm_t, Tensor) else grad_norm_t)
            require_finite(last_grad_norm, "train.grad_norm")
            optimizer.step()
            optim_step_idx += 1

        train_denom = float(max(1, epoch_sup_windows))
        mean_train_vec = epoch_train_losses_vec / train_denom
        if is_ddp:
            dist.all_reduce(mean_train_vec, op=dist.ReduceOp.AVG)
        require_finite(mean_train_vec, f"train.epoch_{epoch}.loss_components")
        train_summary = _parts_vec_to_dict(mean_train_vec)

        val_summary, _ = _evaluate_validation_episodes(
            raw_model=raw_model,
            normalizer=normalizer,
            val_dataset=val_dataset,
            train_cfg=train_cfg,
            dbn_cfg=dbn_cfg,
            device=device,
            collect_logits=False,
            prestaged_episodes=val_prestaged,
        )
        val_total = float(val_summary["total"])
        require_finite(val_total, f"val.epoch_{epoch}.total_loss")

        improved = val_total < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_total
            best_epoch = epoch
            best_state_dict = {k: v.detach().float().cpu().clone() for k, v in raw_model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epoch_wall_s = float(time.perf_counter() - t_epoch_start)
        epoch_rec = {
            "epoch": epoch,
            "wall_time_s": epoch_wall_s,
            "dag_temperature": dag_temp,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_norm": last_grad_norm,
            "train_loss": train_summary["total"],
            "val_loss": val_total,
            "train_components": train_summary,
            "val_components": val_summary,
            "is_best": improved,
        }
        epoch_history.append(epoch_rec)

        if rank == 0:
            _LOG.info(
                "Epoch %d/%d [%.2fs]: train_loss=%.4f val_loss=%.4f (best=%.4f @ ep %d) dag_T=%.3f grad=%.3f",
                epoch,
                train_cfg.epochs,
                epoch_wall_s,
                train_summary["total"],
                val_total,
                best_val_loss,
                best_epoch,
                dag_temp,
                last_grad_norm,
            )
            _flush_log_streams()

        if epochs_no_improve >= train_cfg.early_stopping_patience:
            if rank == 0:
                _LOG.info(
                    "Early stopping triggered at epoch %d (patience=%d, best_epoch=%d).",
                    epoch,
                    train_cfg.early_stopping_patience,
                    best_epoch,
                )
                _flush_log_streams()
            break

    # Restore best validation weights in FP32 & calibrate post-hoc temperatures strictly in model.eval() (Boundary Lock B)
    raw_model.to(dtype=torch.float32)
    raw_model.load_state_dict(copy.deepcopy(best_state_dict), strict=True)
    raw_model.eval()

    _, val_logits = _evaluate_validation_episodes(
        raw_model=raw_model,
        normalizer=normalizer,
        val_dataset=val_dataset,
        train_cfg=train_cfg,
        dbn_cfg=dbn_cfg,
        device=device,
        collect_logits=True,
        prestaged_episodes=val_prestaged,
    )

    if val_logits["edge_logits"].size == 0 or val_logits["delay_pred_logits"].size == 0:
        raise ConfigValidationError(
            "Validation dataset produced empty edge or delay logits; cannot fit temperature calibration.",
            context={
                "edge_samples": int(val_logits["edge_logits"].size),
                "delay_samples": int(val_logits["delay_pred_logits"].size),
            },
        )

    temperatures = calibrate_temperatures(
        edge_logits=val_logits["edge_logits"],
        edge_targets=val_logits["edge_targets"],
        delay_logits=val_logits["delay_pred_logits"],
        delay_targets=val_logits["delay_targets"],
        link_logits=val_logits["link_logits"],
        link_targets=val_logits["link_targets"],
        model=raw_model,
    )

    val_predictive = {
        "edge": _format_head_predictive_summary(
            val_logits["edge_logits"],
            val_logits["edge_targets"],
            temperatures.t_edge,
            name="val.edge",
        ),
        "delay_pred": _format_head_predictive_summary(
            val_logits["delay_pred_logits"],
            val_logits["delay_targets"],
            temperatures.t_delay,
            name="val.delay_pred",
        ),
        "delay_post": _format_head_predictive_summary(
            val_logits["delay_post_logits"],
            val_logits["delay_targets"],
            temperatures.t_delay,
            name="val.delay_post",
        ),
        "spillback": _format_head_predictive_summary(
            val_logits["link_logits"],
            val_logits["link_targets"],
            temperatures.t_link,
            name="val.spillback",
        ),
    }

    wall_time_s = float(time.perf_counter() - t_train_start)
    ckpt_file = _resolve_checkpoint_output_file(output_checkpoint_path)

    training_metadata: dict[str, Any] = {
        "train_config": asdict(train_cfg),
        "dbn_config": asdict(dbn_cfg),
        "train_config_sha256": config_sha256(train_cfg),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "epochs_completed": len(epoch_history),
        "train_seeds": sorted(train_seeds),
        "val_seeds": sorted(val_seeds),
        "wall_time_s": wall_time_s,
    }

    ckpt_sha = ""
    meta_file = ckpt_file.parent / "checkpoint_meta.json"
    target_report_path = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else (ckpt_file.parent / "training_report.json")
    )

    if rank == 0:
        saved_path, ckpt_sha = save_dbn_checkpoint(
            ckpt_file,
            model=raw_model,
            normalizer=normalizer,
            temperatures=temperatures,
            training_metadata=training_metadata,
        )
        ckpt_file = saved_path
        meta_file = ckpt_file.parent / "checkpoint_meta.json"
        meta_payload = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_file": ckpt_file.name,
            "checkpoint_sha256": ckpt_sha,
            "model_preset": preset_spec.name,
            "model_metadata": raw_model.metadata_dict(),
            "is_calibrated": bool(temperatures.calibrated),
            "temperatures": temperatures.to_dict(),
            "training_metadata": training_metadata,
        }
        tmp_meta = meta_file.with_name(f".{meta_file.name}.pid{os.getpid()}.tmp")
        tmp_meta.write_text(json.dumps(meta_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_meta, meta_file)

    if is_ddp:
        dist.barrier()
        if rank != 0:
            bundle_sync = load_dbn_checkpoint(ckpt_file, require_calibrated=True, device="cpu")
            ckpt_sha = bundle_sync.checkpoint_sha256

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_path": str(ckpt_file),
        "checkpoint_meta_path": str(meta_file),
        "training_report_path": str(target_report_path),
        "checkpoint_sha256": ckpt_sha,
        "model_preset": preset_spec.name,
        "device": str(device),
        "precision": precision,
        "compile_model": bool(train_cfg.compile_model),
        "distributed": bool(is_ddp),
        "world_size": int(world_size),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "initial_train_loss": float(epoch_history[0]["train_loss"]),
        "final_train_loss": float(epoch_history[-1]["train_loss"]),
        "temperatures": temperatures.to_dict(),
        "val_predictive": val_predictive,
        "epochs": epoch_history,
        "wall_time_s": wall_time_s,
    }

    if rank == 0:
        target_report_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_rep = target_report_path.with_name(f".{target_report_path.name}.pid{os.getpid()}.tmp")
        tmp_rep.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_rep, target_report_path)

    return report


def evaluate_heldout_predictive(
    checkpoint_path: Path | str,
    dataset_dirs: Mapping[str, Path | str] | Path | str,
    output_path: Path | str | None = None,
    *,
    dbn_cfg: DBNConfig = DBNConfig(),
    allow_ci_micro: bool = False,
    require_test_seeds: bool = True,
) -> dict[str, Any]:
    """Evaluate calibrated ``STGraphTransformerDBN`` predictive metrics across Held-Out ID & Zero-Shot OOD rollouts."""
    ckpt_file = _resolve_checkpoint_input_file(checkpoint_path)
    bundle = load_dbn_checkpoint(
        ckpt_file,
        require_calibrated=dbn_cfg.require_calibrated,
        device=dbn_cfg.inference_device,
    )
    if bundle.model.preset_name == "ci_micro" and not allow_ci_micro:
        raise CheckpointIntegrityError(
            "Held-out predictive evaluation blocked for model_preset='ci_micro' "
            "(pass allow_ci_micro=True only in unit tests).",
            context={"checkpoint_path": str(ckpt_file), "model_preset": bundle.model.preset_name},
        )

    if isinstance(dataset_dirs, (Path, str)):
        split_map: dict[str, Path] = {"held_out": Path(dataset_dirs)}
    elif isinstance(dataset_dirs, Mapping) and dataset_dirs:
        split_map = {str(k): Path(v) for k, v in dataset_dirs.items()}
    else:
        raise ConfigValidationError("dataset_dirs must be a non-empty Mapping or path.")

    device, _ = validate_compute_device(dbn_cfg.inference_device, "fp32")
    dev = torch.device(device)
    model = bundle.model.to(dev)
    model.eval()

    splits_out: dict[str, dict[str, Any]] = {}
    for split_label, split_dir in split_map.items():
        expected_split: RolloutSplit | None = (
            split_label  # type: ignore[assignment]
            if split_label in ("train", "val", "eval_id", "eval_ood", "test")
            else None
        )
        manifest, shards = load_rollout_shards(split_dir, expected_split=expected_split)
        shard_seeds = {int(s["seed"]) for s in shards}
        if require_test_seeds:
            actual_split = manifest.get("split")
            if actual_split in ("train", "val"):
                raise ConfigValidationError(
                    f"Held-out predictive anti-leakage guard: split {split_label!r} loaded manifest "
                    f"with split={actual_split!r} (only 'eval_id', 'eval_ood', or 'test' allowed).",
                    context={"split": split_label, "manifest_split": actual_split},
                )
            illegal = sorted(shard_seeds - set(TEST_SEEDS))
            if illegal:
                raise ConfigValidationError(
                    f"Held-out predictive anti-leakage guard: split {split_label!r} contains "
                    f"non-TEST_SEEDS {illegal} (expected 3000..3014).",
                    context={"split": split_label, "illegal_seeds": illegal},
                )

        seq_ds = RolloutSequenceDataset(
            shards,
            dbn_cfg=dbn_cfg,
            window_len=8,
            stride=8,
        )
        _, raw_logits = _evaluate_validation_episodes(
            raw_model=model,
            normalizer=bundle.normalizer,
            val_dataset=seq_ds,
            train_cfg=TrainConfig(model_preset=bundle.model.preset_name),  # type: ignore[arg-type]
            dbn_cfg=dbn_cfg,
            device=dev,
            temperatures=bundle.temperatures,
            collect_logits=True,
        )

        splits_out[split_label] = {
            "num_episodes": len(shards),
            "seeds": sorted(shard_seeds),
            "edge": _format_head_predictive_summary(
                raw_logits["edge_logits"],
                raw_logits["edge_targets"],
                bundle.temperatures.t_edge,
                name=f"{split_label}.edge",
            ),
            "delay_pred": _format_head_predictive_summary(
                raw_logits["delay_pred_logits"],
                raw_logits["delay_targets"],
                bundle.temperatures.t_delay,
                name=f"{split_label}.delay_pred",
            ),
            "delay_post": _format_head_predictive_summary(
                raw_logits["delay_post_logits"],
                raw_logits["delay_targets"],
                bundle.temperatures.t_delay,
                name=f"{split_label}.delay_post",
            ),
            "spillback": _format_head_predictive_summary(
                raw_logits["link_logits"],
                raw_logits["link_targets"],
                bundle.temperatures.t_link,
                name=f"{split_label}.spillback",
            ),
        }

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_path": str(ckpt_file),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "model_preset": bundle.model.preset_name,
        "temperatures": bundle.temperatures.to_dict(),
        "splits": splits_out,
    }

    if output_path is not None:
        out_p = Path(output_path).expanduser().resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = out_p.with_name(f".{out_p.name}.pid{os.getpid()}.tmp")
        tmp_p.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_p, out_p)

    return report


__all__ = [
    "PrecomputedDeviceStep",
    "RolloutSequenceDataset",
    "compute_precomputed_step_loss",
    "compute_step_multitask_loss",
    "evaluate_heldout_predictive",
    "prestage_rollout_dataset",
    "shard_step_to_feature_batch",
    "train_neural_dbn",
]

"""Heterogeneous Spatio-Temporal Edge-Augmented Graph Transformer for Dynamic Bayesian Networks.

Pure-PyTorch implementation of ``STGraphTransformerDBN`` (zero external C++ GNN wheel
dependencies; portable across CPU, Apple MPS, and CUDA in FP32/BF16):
- ``RMSNorm`` (FP32 variance accumulation) & zero-bias ``SwiGLU`` FFNs with LayerScale.
- ``MultiScaleBasisEncoder`` combining learnable Gaussian RBF kernels and log-spaced Fourier scales.
- ``segment_softmax`` with smooth ``20 * tanh(s / 20)`` logit capping and out-of-place ``index_add``.
- ``ContinuousDecayGRU`` carrying recurrent latent representations for Vehicles, Corridors, and
  Candidate Edges across control ticks via ``DBNTemporalMemory`` / ``TorchTemporalMemory``.
- ``STGraphTransformerBlock`` coupling ``Vehicle -> Corridor`` pooling, ``Corridor <-> Corridor``
  line-graph message passing, ``Corridor -> Vehicle`` turn-belief-weighted bottleneck injection
  (strictly respecting Boundary Lock A), and node-first factorized ``Vehicle <-> Vehicle``
  edge-augmented multi-head attention.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..config import ModelPresetName, ModelPresetSpec, get_model_preset
from ..contracts import (
    CORRIDOR_EDGE_FEATURE_DIM,
    CORRIDOR_FEATURE_DIM,
    DBNTemporalMemory,
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
)
from ..logging_utils import (
    CheckpointIntegrityError,
    ConfigValidationError,
    require_finite,
    require_shape,
)
from .cpd import CorridorSpillbackCPD

NUM_RELATION_TYPES: int = 4
SPATIOTEMP_EDGE_SLICE: slice = slice(4, 8)  # signed_d_eta, abs_d_eta, signed_path_gap, rel_speed
SPATIOTEMP_EDGE_DIM: int = 4


def segment_softmax(raw_scores: Tensor, dst_idx: Tensor, num_nodes: int) -> Tensor:
    """Pure-PyTorch segment softmax over incoming edges grouped by ``dst_idx``.

    Applies smooth logit capping ``20 * tanh(s / 20)`` in FP32 so ``exp`` is strictly bounded in
    ``[2.06e-9, 4.85e8]`` without requiring ``scatter_reduce("amax")`` (which lacks complete
    backward/device support on MPS and older CPU backends).
    """
    if raw_scores.shape[0] == 0:
        return raw_scores.clone()
    capped = 20.0 * torch.tanh(raw_scores.float() / 20.0)
    weights = torch.exp(capped)
    denom_shape = (num_nodes, *raw_scores.shape[1:])
    denom = torch.zeros(denom_shape, dtype=torch.float32, device=raw_scores.device).index_add(
        0, dst_idx, weights
    )
    probs = weights / (denom.index_select(0, dst_idx) + 1e-12)
    return probs.to(dtype=raw_scores.dtype)


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization with FP32 variance stabilization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim < 1:
            raise ConfigValidationError(f"RMSNorm dim must be >= 1, got {dim}.")
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[0] == 0:
            return x
        x_f32 = x.float()
        inv_rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = x_f32 * inv_rms * self.weight
        return normed.to(dtype=x.dtype)


class SwiGLU(nn.Module):
    """Zero-bias SwiGLU feed-forward network (``W_out(SiLU(W_g x) * W_v x)``)."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int | None = None,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        out_dim = d_in if d_out is None else d_out
        self.w_gate_val = nn.Linear(d_in, 2 * d_hidden, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.w_out = nn.Linear(d_hidden, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[0] == 0:
            return x.new_empty((0, self.w_out.out_features))
        gate, val = self.w_gate_val(x).chunk(2, dim=-1)
        activated = F.silu(gate) * val
        return self.w_out(self.drop(activated))


class MultiScaleBasisEncoder(nn.Module):
    """Hybrid Gaussian RBF + Log-Frequency Fourier + Linear Skip encoder for continuous features."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        n_rbf: int = 8,
        n_fourier_freqs: int = 4,
        rbf_min: float = -2.5,
        rbf_max: float = 2.5,
    ) -> None:
        super().__init__()
        if in_dim < 1 or out_dim < 1:
            raise ConfigValidationError(f"Invalid MultiScaleBasisEncoder dims: in={in_dim}, out={out_dim}.")
        if n_rbf < 2 or n_fourier_freqs < 1:
            raise ConfigValidationError(f"Require n_rbf >= 2 and n_fourier_freqs >= 1, got {n_rbf}, {n_fourier_freqs}.")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_rbf = n_rbf
        self.n_fourier_freqs = n_fourier_freqs

        centers = torch.linspace(rbf_min, rbf_max, steps=n_rbf, dtype=torch.float32)
        step = float((rbf_max - rbf_min) / max(n_rbf - 1, 1))
        init_gamma = 1.0 / max(step * step, 1e-3)
        freqs = (2.0 ** torch.arange(n_fourier_freqs, dtype=torch.float32)) * math.pi

        self.register_buffer("rbf_centers", centers, persistent=True)
        self.register_buffer("fourier_freqs", freqs, persistent=True)
        self.log_rbf_gamma = nn.Parameter(torch.full((n_rbf,), math.log(init_gamma), dtype=torch.float32))

        expanded_dim = in_dim * (1 + n_rbf + 2 * n_fourier_freqs)
        self.proj = nn.Linear(expanded_dim, out_dim)
        self.norm = RMSNorm(out_dim)

    def forward(self, u: Tensor) -> Tensor:
        require_shape(u, (None, self.in_dim), "MultiScaleBasisEncoder.u")
        if u.shape[0] == 0:
            return u.new_empty((0, self.out_dim))

        u_clamped = u.clamp(min=-8.0, max=8.0)
        # RBF expansion: [N, in_dim, n_rbf]
        diff = u_clamped.unsqueeze(-1) - self.rbf_centers.to(dtype=u.dtype)
        gamma = torch.exp(self.log_rbf_gamma).clamp(min=1e-3, max=50.0).to(dtype=u.dtype)
        rbf_feat = torch.exp(-gamma * diff.pow(2)).reshape(u.shape[0], -1)

        # Multi-frequency Fourier expansion: [N, in_dim, n_fourier_freqs]
        angles = u_clamped.unsqueeze(-1) * self.fourier_freqs.to(dtype=u.dtype)
        sin_feat = torch.sin(angles).reshape(u.shape[0], -1)
        cos_feat = torch.cos(angles).reshape(u.shape[0], -1)

        concat = torch.cat([u_clamped, rbf_feat, sin_feat, cos_feat], dim=-1)
        return F.silu(self.norm(self.proj(concat)))


class ContinuousDecayGRU(nn.Module):
    """Continuous-time decayed GRU state updater handling variable ``dt`` and dynamic entity sets."""

    def __init__(self, dim: int, *, init_half_life_s: float = 4.0) -> None:
        super().__init__()
        self.dim = dim
        self.z_init = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        init_rate = math.log(2.0) / max(init_half_life_s, 0.2)
        self.raw_decay_rate = nn.Parameter(
            torch.full((dim,), math.log(math.exp(init_rate) - 1.0), dtype=torch.float32)
        )
        self.gru_cell = nn.GRUCell(dim, dim)

    def init_state(self, count: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if count == 0:
            return torch.empty((0, self.dim), device=device, dtype=torch.float32)
        return self.z_init.to(device=device, dtype=torch.float32).unsqueeze(0).expand(count, -1).clone()

    def decay(self, prev_h: Tensor, delta_t: float) -> Tensor:
        if prev_h.shape[0] == 0:
            return prev_h.float()
        prev_f32 = prev_h.float()
        dt_pos = max(0.0, float(delta_t))
        rate = F.softplus(self.raw_decay_rate.float()).to(device=prev_h.device)
        decay_factor = torch.exp(-rate * dt_pos).unsqueeze(0)
        z0 = self.z_init.to(device=prev_h.device, dtype=torch.float32).unsqueeze(0)
        return decay_factor * prev_f32 + (1.0 - decay_factor) * z0

    def forward(self, x_in: Tensor, prev_aligned_h: Tensor) -> Tensor:
        if x_in.shape[0] == 0:
            return torch.empty((0, self.dim), device=x_in.device, dtype=torch.float32)
        with torch.autocast(device_type=x_in.device.type, enabled=False):
            return self.gru_cell(x_in.float(), prev_aligned_h.float())


@dataclass(frozen=True)
class TorchTemporalMemory:
    """Live differentiable recurrent latent state across vehicles, corridors, and candidate edges."""

    sim_time: float
    veh_ids: tuple[str, ...]
    veh_hidden: Tensor                   # [N_prev, d_model]
    corridor_ids: tuple[str, ...]
    corridor_hidden: Tensor              # [N_C_prev, d_model]
    edge_keys: tuple[tuple[str, str], ...]
    edge_hidden: Tensor                  # [E_prev, d_edge]

    def detach_state(self) -> TorchTemporalMemory:
        """Detach hidden states at TBPTT window boundaries while preserving state values."""
        return TorchTemporalMemory(
            sim_time=self.sim_time,
            veh_ids=self.veh_ids,
            veh_hidden=self.veh_hidden.detach(),
            corridor_ids=self.corridor_ids,
            corridor_hidden=self.corridor_hidden.detach(),
            edge_keys=self.edge_keys,
            edge_hidden=self.edge_hidden.detach(),
        )

    def to_numpy_memory(self) -> DBNTemporalMemory:
        """Serialize recurrent tensors into contract-compliant :class:`DBNTemporalMemory`."""
        veh_np = self.veh_hidden.detach().cpu().to(torch.float32).numpy().copy()
        corr_np = self.corridor_hidden.detach().cpu().to(torch.float32).numpy().copy()
        edge_np = self.edge_hidden.detach().cpu().to(torch.float32).numpy().copy()
        return DBNTemporalMemory(
            sim_time=float(self.sim_time),
            veh_hidden=dict(zip(self.veh_ids, veh_np)),
            corridor_hidden=dict(zip(self.corridor_ids, corr_np)),
            edge_hidden=dict(zip(self.edge_keys, edge_np)),
        )

    @classmethod
    def from_numpy_memory(
        cls,
        mem: DBNTemporalMemory,
        *,
        d_model: int,
        d_edge: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> TorchTemporalMemory:
        """Reconstruct :class:`TorchTemporalMemory` from a snapshot's :class:`DBNTemporalMemory`."""
        veh_ids = tuple(mem.veh_hidden.keys())
        if veh_ids:
            v_arr = np.stack([np.asarray(mem.veh_hidden[vid], dtype=np.float32) for vid in veh_ids], axis=0)
            require_shape(v_arr, (len(veh_ids), d_model), "DBNTemporalMemory.veh_hidden")
            require_finite(v_arr, "DBNTemporalMemory.veh_hidden")
            veh_h = torch.as_tensor(v_arr, device=device, dtype=dtype)
        else:
            veh_h = torch.empty((0, d_model), device=device, dtype=dtype)

        corr_ids = tuple(mem.corridor_hidden.keys())
        if corr_ids:
            c_arr = np.stack([np.asarray(mem.corridor_hidden[cid], dtype=np.float32) for cid in corr_ids], axis=0)
            require_shape(c_arr, (len(corr_ids), d_model), "DBNTemporalMemory.corridor_hidden")
            require_finite(c_arr, "DBNTemporalMemory.corridor_hidden")
            corr_h = torch.as_tensor(c_arr, device=device, dtype=dtype)
        else:
            corr_h = torch.empty((0, d_model), device=device, dtype=dtype)

        edge_keys = tuple(mem.edge_hidden.keys())
        if edge_keys:
            e_arr = np.stack([np.asarray(mem.edge_hidden[ek], dtype=np.float32) for ek in edge_keys], axis=0)
            require_shape(e_arr, (len(edge_keys), d_edge), "DBNTemporalMemory.edge_hidden")
            require_finite(e_arr, "DBNTemporalMemory.edge_hidden")
            edge_h = torch.as_tensor(e_arr, device=device, dtype=dtype)
        else:
            edge_h = torch.empty((0, d_edge), device=device, dtype=dtype)

        return cls(
            sim_time=float(mem.sim_time),
            veh_ids=veh_ids,
            veh_hidden=veh_h,
            corridor_ids=corr_ids,
            corridor_hidden=corr_h,
            edge_keys=edge_keys,
            edge_hidden=edge_h,
        )


class STGraphTransformerBlock(nn.Module):
    """Single heterogeneous relational block coupling Vehicles, Corridors, and Directed Edges."""

    def __init__(
        self,
        d_model: int,
        d_edge: int,
        n_heads: int,
        ffn_mult: float,
        *,
        dropout: float = 0.0,
        layer_scale_init: float = 0.15,
        update_corridor_edges: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ConfigValidationError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        self.d_model = d_model
        self.d_edge = d_edge
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.inv_sqrt_dh = 1.0 / math.sqrt(self.d_head)
        self.update_corridor_edges = bool(update_corridor_edges)
        d_ffn_v = max(8, int(round(d_model * float(ffn_mult) / 8.0)) * 8)
        d_ffn_e = max(8, int(round(d_edge * 1.5 / 8.0)) * 8)

        # Stage A: Vehicle -> Corridor attention pooling
        self.norm_v2c_c = RMSNorm(d_model)
        self.norm_v2c_v = RMSNorm(d_model)
        self.v2c_q = nn.Linear(d_model, d_model, bias=False)
        self.v2c_k = nn.Linear(d_model, d_model, bias=False)
        self.v2c_v = nn.Linear(d_model, d_model, bias=False)
        self.v2c_out = nn.Linear(d_model, d_model, bias=False)
        self.ls_v2c = nn.Parameter(torch.full((d_model,), layer_scale_init, dtype=torch.float32))

        # Stage B: Corridor <-> Corridor line-graph edge-biased attention + FFN + corridor-edge update
        self.norm_c_attn = RMSNorm(d_model)
        self.norm_cc_attn = RMSNorm(d_edge)
        self.c_wq = nn.Linear(d_model, d_model, bias=False)
        self.c_wk = nn.Linear(d_model, d_model, bias=False)
        self.c_wv = nn.Linear(d_model, d_model, bias=False)
        self.cc_wk = nn.Linear(d_edge, d_model, bias=False)
        self.cc_wbias = nn.Linear(d_edge, n_heads, bias=False)
        self.cc_wval = nn.Linear(d_edge, d_model, bias=False)
        self.c_wo = nn.Linear(d_model, d_model, bias=False)
        self.ls_c_attn = nn.Parameter(torch.full((d_model,), layer_scale_init, dtype=torch.float32))

        self.norm_c_ffn = RMSNorm(d_model)
        self.ffn_c = SwiGLU(d_model, d_ffn_v, dropout=dropout)
        self.ls_c_ffn = nn.Parameter(torch.full((d_model,), layer_scale_init, dtype=torch.float32))

        if self.update_corridor_edges:
            self.norm_c_for_cc = RMSNorm(d_model)
            self.norm_cc_upd = RMSNorm(d_edge)
            self.wcc_src = nn.Linear(d_model, d_edge, bias=False)
            self.wcc_dst = nn.Linear(d_model, d_edge, bias=False)
            self.wcc_self = nn.Linear(d_edge, d_edge, bias=False)
            self.ffn_cc = SwiGLU(d_edge, d_ffn_e, dropout=dropout)
            self.ls_cc_upd = nn.Parameter(torch.full((d_edge,), layer_scale_init, dtype=torch.float32))

        # Stage C: Corridor -> Vehicle host + turn-belief-weighted downstream context injection
        self.norm_c2v = RMSNorm(d_model)
        self.c2v_proj = SwiGLU(2 * d_model, d_ffn_v, d_model, dropout=dropout)
        self.ls_c2v = nn.Parameter(torch.full((d_model,), layer_scale_init, dtype=torch.float32))

        # Stage D: Node-first factorized Edge-Augmented Vehicle <-> Vehicle Transformer + Edge co-update
        self.norm_v_attn = RMSNorm(d_model)
        self.norm_e_attn = RMSNorm(d_edge)
        self.v_wq = nn.Linear(d_model, d_model, bias=False)
        self.v_wk = nn.Linear(d_model, d_model, bias=False)
        self.v_wv = nn.Linear(d_model, d_model, bias=False)
        self.e_wk = nn.Linear(d_edge, d_model, bias=False)
        self.e_wbias = nn.Linear(d_edge, n_heads, bias=False)
        self.e_wval = nn.Linear(d_edge, d_model, bias=False)
        self.v_wo = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.ls_v_attn = nn.Parameter(torch.full((d_model,), layer_scale_init, dtype=torch.float32))

        self.norm_v_ffn = RMSNorm(d_model)
        self.ffn_v = SwiGLU(d_model, d_ffn_v, dropout=dropout)
        self.ls_v_ffn = nn.Parameter(torch.full((d_model,), layer_scale_init, dtype=torch.float32))

        self.norm_v_for_e = RMSNorm(d_model)
        self.norm_e_upd = RMSNorm(d_edge)
        self.we_src = nn.Linear(d_model, d_edge, bias=False)
        self.we_dst = nn.Linear(d_model, d_edge, bias=False)
        self.we_self = nn.Linear(d_edge, d_edge, bias=False)
        self.ffn_e = SwiGLU(d_edge, d_ffn_e, dropout=dropout)
        self.ls_e_upd = nn.Parameter(torch.full((d_edge,), layer_scale_init, dtype=torch.float32))

    def forward(
        self,
        h_v: Tensor,
        h_c: Tensor,
        h_e: Tensor,
        h_cc: Tensor,
        *,
        edge_index: Tensor,
        corridor_edge_index: Tensor,
        veh_corridor_idx: Tensor,
        veh_turn_src_idx: Tensor,
        veh_turn_out_corridor_idx: Tensor,
        veh_turn_prob: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        n_v = h_v.shape[0]
        n_c = h_c.shape[0]
        e_cand = h_e.shape[0]
        e_c = h_cc.shape[0]
        h = self.n_heads
        dh = self.d_head

        # Stage A: Vehicle -> Corridor attention pooling
        if n_v > 0 and n_c > 0:
            nc_pool = self.norm_v2c_c(h_c)
            nv_pool = self.norm_v2c_v(h_v)
            q_vc = self.v2c_q(nc_pool).index_select(0, veh_corridor_idx).view(n_v, h, dh)
            k_vc = self.v2c_k(nv_pool).view(n_v, h, dh)
            v_vc = self.v2c_v(nv_pool).view(n_v, h, dh)
            scores_vc = (q_vc * k_vc).sum(dim=-1) * self.inv_sqrt_dh
            alpha_vc = segment_softmax(scores_vc, veh_corridor_idx, n_c)
            msg_vc = alpha_vc.to(dtype=v_vc.dtype).unsqueeze(-1) * v_vc
            pooled_c = msg_vc.new_zeros((n_c, h, dh)).index_add(0, veh_corridor_idx, msg_vc)
            upd_v2c = self.v2c_out(pooled_c.reshape(n_c, self.d_model))
            h_c = h_c + (self.ls_v2c.to(dtype=upd_v2c.dtype) * upd_v2c).to(dtype=h_c.dtype)

        # Stage B: Corridor <-> Corridor line-graph message passing
        if n_c > 0:
            if e_c > 0:
                src_c = corridor_edge_index[0]
                dst_c = corridor_edge_index[1]
                nc_a = self.norm_c_attn(h_c)
                ncc_a = self.norm_cc_attn(h_cc)

                qc = self.c_wq(nc_a).view(n_c, h, dh).index_select(0, dst_c)
                kc = (self.c_wk(nc_a).index_select(0, src_c) + self.cc_wk(ncc_a)).view(e_c, h, dh)
                vc = (self.c_wv(nc_a).index_select(0, src_c) + self.cc_wval(ncc_a)).view(e_c, h, dh)
                sc = (qc * kc).sum(dim=-1) * self.inv_sqrt_dh + self.cc_wbias(ncc_a)
                alpha_cc = self.attn_drop(segment_softmax(sc, dst_c, n_c))
                msg_cc = alpha_cc.to(dtype=vc.dtype).unsqueeze(-1) * vc
                agg_c = msg_cc.new_zeros((n_c, h, dh)).index_add(0, dst_c, msg_cc)
                upd_c_attn = self.c_wo(agg_c.reshape(n_c, self.d_model))
                h_c = h_c + (self.ls_c_attn.to(dtype=upd_c_attn.dtype) * upd_c_attn).to(dtype=h_c.dtype)

            upd_c_ffn = self.ffn_c(self.norm_c_ffn(h_c))
            h_c = h_c + (self.ls_c_ffn.to(dtype=upd_c_ffn.dtype) * upd_c_ffn).to(dtype=h_c.dtype)

            if self.update_corridor_edges and e_c > 0:
                src_c = corridor_edge_index[0]
                dst_c = corridor_edge_index[1]
                nc_e = self.norm_c_for_cc(h_c)
                ncc_u = self.norm_cc_upd(h_cc)
                cc_fused = (
                    self.wcc_src(nc_e).index_select(0, src_c)
                    + self.wcc_dst(nc_e).index_select(0, dst_c)
                    + self.wcc_self(ncc_u)
                )
                upd_cc = self.ffn_cc(cc_fused)
                h_cc = h_cc + (self.ls_cc_upd.to(dtype=upd_cc.dtype) * upd_cc).to(dtype=h_cc.dtype)

        # Stage C: Corridor -> Vehicle host + turn-belief-weighted downstream context injection
        if n_v > 0 and n_c > 0:
            nc_inj = self.norm_c2v(h_c)
            host_ctx = nc_inj.index_select(0, veh_corridor_idx)
            if veh_turn_src_idx.numel() > 0:
                weighted_out = veh_turn_prob.to(dtype=nc_inj.dtype).unsqueeze(-1) * nc_inj.index_select(
                    0, veh_turn_out_corridor_idx
                )
                down_ctx = weighted_out.new_zeros((n_v, self.d_model)).index_add(
                    0, veh_turn_src_idx, weighted_out
                )
            else:
                down_ctx = host_ctx.new_zeros((n_v, self.d_model))
            upd_c2v = self.c2v_proj(torch.cat([host_ctx, down_ctx], dim=-1))
            h_v = h_v + (self.ls_c2v.to(dtype=upd_c2v.dtype) * upd_c2v).to(dtype=h_v.dtype)

        # Stage D: Edge-Augmented Vehicle <-> Vehicle Graph Transformer + factorized Edge co-update
        if n_v > 0:
            if e_cand > 0:
                src_v = edge_index[0]
                dst_v = edge_index[1]
                nv_a = self.norm_v_attn(h_v)
                ne_a = self.norm_e_attn(h_e)

                qv = self.v_wq(nv_a).view(n_v, h, dh).index_select(0, dst_v)
                kv = (self.v_wk(nv_a).index_select(0, src_v) + self.e_wk(ne_a)).view(e_cand, h, dh)
                vv = (self.v_wv(nv_a).index_select(0, src_v) + self.e_wval(ne_a)).view(e_cand, h, dh)
                sv = (qv * kv).sum(dim=-1) * self.inv_sqrt_dh + self.e_wbias(ne_a)
                alpha_v = self.attn_drop(segment_softmax(sv, dst_v, n_v))
                msg_v = alpha_v.to(dtype=vv.dtype).unsqueeze(-1) * vv
                agg_v = msg_v.new_zeros((n_v, h, dh)).index_add(0, dst_v, msg_v)
                upd_v_attn = self.v_wo(agg_v.reshape(n_v, self.d_model))
                h_v = h_v + (self.ls_v_attn.to(dtype=upd_v_attn.dtype) * upd_v_attn).to(dtype=h_v.dtype)

            upd_v_ffn = self.ffn_v(self.norm_v_ffn(h_v))
            h_v = h_v + (self.ls_v_ffn.to(dtype=upd_v_ffn.dtype) * upd_v_ffn).to(dtype=h_v.dtype)

            if e_cand > 0:
                src_v = edge_index[0]
                dst_v = edge_index[1]
                nv_e = self.norm_v_for_e(h_v)
                ne_u = self.norm_e_upd(h_e)
                # Node-first factorized projection: projects O(N_t) nodes BEFORE gathering onto O(E_cand) edges!
                e_fused = (
                    self.we_src(nv_e).index_select(0, src_v)
                    + self.we_dst(nv_e).index_select(0, dst_v)
                    + self.we_self(ne_u)
                )
                upd_e = self.ffn_e(e_fused)
                h_e = h_e + (self.ls_e_upd.to(dtype=upd_e.dtype) * upd_e).to(dtype=h_e.dtype)

        return h_v, h_c, h_e, h_cc


@dataclass(frozen=True)
class TransformerStepOutputs:
    """Amortized DBN structural, precedence, and CPD head outputs for one control tick."""

    precedence: Tensor                 # [N_t]: r_topo + delta_r
    delta_precedence: Tensor           # [N_t]: bounded learned residual in (-8, +8)
    leak_prob: Tensor                  # [N_t]: lambda_i^t in (0.01, 0.65)
    self_persist_prob: Tensor          # [N_t]: alpha_i^t in (0.05, 0.95)
    spill_weight: Tensor               # [N_t]: beta_{spill, i}^t in (0.05, 0.95)
    expected_delay_s: Tensor           # [N_t]: hat{tau}_i^t >= 0.0 (seconds)
    edge_logits: Tensor                # [E_cand]: uncalibrated edge existence logits
    kappa: Tensor                      # [E_cand]: base conflict inhibition in (0.02, 0.98)
    gamma: Tensor                      # [E_cand]: delay cascade amplification in (0.02, 0.98)
    corridor_spillback_logits: Tensor  # [N_C]: uncalibrated corridor spillback logits
    next_memory: TorchTemporalMemory


def _align_entity_memory(
    curr_keys: tuple[str, ...] | tuple[tuple[str, str], ...],
    prev_keys: tuple[str, ...] | tuple[tuple[str, str], ...],
    prev_hidden: Tensor,
    gru_module: ContinuousDecayGRU,
    *,
    delta_t: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Align previous recurrent hidden states to current entity order with continuous-time decay."""
    n_curr = len(curr_keys)
    if n_curr == 0:
        return gru_module.init_state(0, device=device, dtype=dtype)
    if len(prev_keys) == 0 or prev_hidden.shape[0] == 0:
        return gru_module.init_state(n_curr, device=device, dtype=dtype)
    if curr_keys == prev_keys:
        return gru_module.decay(prev_hidden.to(device=device, dtype=dtype), delta_t)

    aligned = gru_module.init_state(n_curr, device=device, dtype=dtype)

    prev_lookup = {k: idx for idx, k in enumerate(prev_keys)}
    curr_matched: list[int] = []
    prev_matched: list[int] = []
    for c_idx, key in enumerate(curr_keys):
        p_idx = prev_lookup.get(key)
        if p_idx is not None:
            curr_matched.append(c_idx)
            prev_matched.append(p_idx)

    if not curr_matched:
        return aligned

    curr_idx_t = torch.tensor(curr_matched, dtype=torch.long, device=device)
    prev_idx_t = torch.tensor(prev_matched, dtype=torch.long, device=device)
    decayed = gru_module.decay(prev_hidden.to(device=device, dtype=dtype).index_select(0, prev_idx_t), delta_t)
    return aligned.index_put((curr_idx_t,), decayed)


def _align_entity_memory_indexed(
    n_curr: int,
    prev_hidden: Tensor,
    gru_module: ContinuousDecayGRU,
    *,
    curr_idx: Tensor | None,
    prev_idx: Tensor | None,
    same_order: bool,
    delta_t: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Vectorized O(|matched|) recurrent state alignment using precomputed index tensors."""
    if n_curr == 0:
        return gru_module.init_state(0, device=device, dtype=dtype)
    if prev_hidden.shape[0] == 0:
        return gru_module.init_state(n_curr, device=device, dtype=dtype)
    if same_order and prev_hidden.shape[0] == n_curr:
        return gru_module.decay(prev_hidden.to(device=device, dtype=dtype), delta_t)
    aligned = gru_module.init_state(n_curr, device=device, dtype=dtype)
    if curr_idx is None or prev_idx is None or curr_idx.numel() == 0:
        return aligned
    decayed = gru_module.decay(prev_hidden.to(device=device, dtype=dtype).index_select(0, prev_idx), delta_t)
    return aligned.index_put((curr_idx,), decayed)


class STGraphTransformerDBN(nn.Module):
    """Publication-grade Heterogeneous Spatio-Temporal Edge-Augmented Graph Transformer for DBNs."""

    def __init__(
        self,
        preset: ModelPresetSpec | ModelPresetName = "sota_local",
        *,
        prediction_horizon_s: float = 6.0,
    ) -> None:
        super().__init__()
        spec = get_model_preset(preset) if isinstance(preset, str) else preset
        if prediction_horizon_s <= 0.0:
            raise ConfigValidationError(f"prediction_horizon_s must be > 0, got {prediction_horizon_s}.")

        self.preset_spec: ModelPresetSpec = spec
        self.preset_name: str = spec.name
        self.d_model: int = spec.d_model
        self.d_edge: int = spec.d_edge
        self.n_layers: int = spec.n_layers
        self.n_heads: int = spec.n_heads
        self.prediction_horizon_s: float = float(prediction_horizon_s)

        # 1. Multi-Scale Spatio-Temporal Stems
        self.veh_stem = MultiScaleBasisEncoder(
            NODE_FEATURE_DIM,
            spec.d_model,
            n_rbf=spec.n_rbf,
            n_fourier_freqs=spec.n_fourier_freqs,
        )
        self.corridor_stem = MultiScaleBasisEncoder(
            CORRIDOR_FEATURE_DIM,
            spec.d_model,
            n_rbf=spec.n_rbf,
            n_fourier_freqs=spec.n_fourier_freqs,
        )
        self.edge_linear_embed = nn.Linear(EDGE_FEATURE_DIM, spec.d_edge, bias=False)
        self.edge_cont_stem = MultiScaleBasisEncoder(
            SPATIOTEMP_EDGE_DIM,
            spec.d_edge,
            n_rbf=spec.n_rbf,
            n_fourier_freqs=spec.n_fourier_freqs,
        )
        self.edge_stem_norm = RMSNorm(spec.d_edge)
        self.corridor_edge_stem = MultiScaleBasisEncoder(
            CORRIDOR_EDGE_FEATURE_DIM,
            spec.d_edge,
            n_rbf=spec.n_rbf,
            n_fourier_freqs=spec.n_fourier_freqs,
        )

        # 2. Continuous-Time Decayed Temporal Recurrent Cells (dNRI-inspired latent Markov state)
        self.veh_gru = ContinuousDecayGRU(spec.d_model, init_half_life_s=4.0)
        self.corridor_gru = ContinuousDecayGRU(spec.d_model, init_half_life_s=8.0)
        self.edge_gru = ContinuousDecayGRU(spec.d_edge, init_half_life_s=3.0)

        # 3. Stacked Heterogeneous Spatio-Temporal Graph Transformer Blocks
        self.blocks = nn.ModuleList(
            [
                STGraphTransformerBlock(
                    d_model=spec.d_model,
                    d_edge=spec.d_edge,
                    n_heads=spec.n_heads,
                    ffn_mult=spec.ffn_mult,
                    dropout=spec.dropout,
                    update_corridor_edges=(layer_idx < spec.n_layers - 1),
                )
                for layer_idx in range(spec.n_layers)
            ]
        )

        # 4. Final Normalization & Amortized DBN Output Heads
        self.final_norm_v = RMSNorm(spec.d_model)
        self.final_norm_c = RMSNorm(spec.d_model)
        self.final_norm_e = RMSNorm(spec.d_edge)

        # Node heads: [delta_r, leak_logit, persist_logit, spill_logit, expected_delay_raw]
        self.node_head_mlp = nn.Sequential(
            nn.Linear(spec.d_model, spec.d_model),
            nn.SiLU(),
            nn.Linear(spec.d_model, 5),
        )
        # Initialize node head biases for physically sensible initial regime before training:
        # delta_r ~ 0, leak ~ 0.08, persist ~ 0.45, spill ~ 0.50, expected_delay ~ 1.0s
        with torch.no_grad():
            nn.init.normal_(self.node_head_mlp[-1].weight, mean=0.0, std=0.01)
            self.node_head_mlp[-1].bias.copy_(torch.tensor([0.0, -2.1, -0.2, 0.0, 0.5], dtype=torch.float32))

        # Node-first factorized Edge Head: combines h_V[src], h_V[dst], and h_E -> [edge_logit, kappa, gamma]
        self.edge_head_src = nn.Linear(spec.d_model, spec.d_edge, bias=False)
        self.edge_head_dst = nn.Linear(spec.d_model, spec.d_edge, bias=False)
        self.edge_head_e = nn.Linear(spec.d_edge, spec.d_edge)
        self.edge_head_out = nn.Linear(spec.d_edge, 3)
        with torch.no_grad():
            nn.init.normal_(self.edge_head_out.weight, mean=0.0, std=0.02)
            self.edge_head_out.bias.copy_(torch.tensor([0.0, -0.4, 0.0], dtype=torch.float32))

        # Learnable Markov edge prior log-odds blend weight (> 0 via softplus; init ~ 0.75)
        self.raw_prior_blend = nn.Parameter(
            torch.tensor(math.log(math.exp(0.75) - 1.0), dtype=torch.float32)
        )

        # Corridor Spillback CPD: physical queue-conservation base + multi-hop corridor transformer residual
        self.phys_corridor_cpd = CorridorSpillbackCPD()
        self.corridor_res_head = nn.Sequential(
            nn.Linear(spec.d_model, spec.d_model // 2),
            nn.SiLU(),
            nn.Linear(spec.d_model // 2, 1),
        )
        with torch.no_grad():
            nn.init.normal_(self.corridor_res_head[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.corridor_res_head[-1].bias)

    def forward(
        self,
        *,
        z_veh: Tensor,
        z_corridor: Tensor,
        z_edge: Tensor,
        z_corridor_edge: Tensor,
        x_corridor_raw: Tensor,
        edge_index: Tensor,
        corridor_edge_index: Tensor,
        veh_corridor_idx: Tensor,
        veh_turn_src_idx: Tensor,
        veh_turn_out_corridor_idx: Tensor,
        veh_turn_prob: Tensor,
        topo_precedence: Tensor,
        prior_edge_prob: Tensor,
        veh_ids: tuple[str, ...],
        corridor_ids: tuple[str, ...],
        sim_time: float,
        dt: float,
        prev_memory: TorchTemporalMemory | DBNTemporalMemory | None = None,
        veh_curr_idx: Tensor | None = None,
        veh_prev_idx: Tensor | None = None,
        veh_same: bool = False,
        corr_curr_idx: Tensor | None = None,
        corr_prev_idx: Tensor | None = None,
        corr_same: bool = False,
        edge_curr_idx: Tensor | None = None,
        edge_prev_idx: Tensor | None = None,
        edge_same: bool = False,
        skip_edge_keys: bool = False,
        check_invariants: bool = True,
    ) -> TransformerStepOutputs:
        """Execute one full spatio-temporal forward pass of ``STGraphTransformerDBN``."""
        n_v = z_veh.shape[0]
        n_c = z_corridor.shape[0]
        e_cand = z_edge.shape[0]
        e_c = z_corridor_edge.shape[0]

        require_shape(z_veh, (n_v, NODE_FEATURE_DIM), "STGraphTransformerDBN.z_veh")
        require_shape(z_corridor, (n_c, CORRIDOR_FEATURE_DIM), "STGraphTransformerDBN.z_corridor")
        require_shape(z_edge, (e_cand, EDGE_FEATURE_DIM), "STGraphTransformerDBN.z_edge")
        require_shape(z_corridor_edge, (e_c, CORRIDOR_EDGE_FEATURE_DIM), "STGraphTransformerDBN.z_corridor_edge")
        require_shape(x_corridor_raw, (n_c, CORRIDOR_FEATURE_DIM), "STGraphTransformerDBN.x_corridor_raw")
        require_shape(edge_index, (2, e_cand), "STGraphTransformerDBN.edge_index")
        require_shape(corridor_edge_index, (2, e_c), "STGraphTransformerDBN.corridor_edge_index")
        require_shape(topo_precedence, (n_v,), "STGraphTransformerDBN.topo_precedence")
        require_shape(prior_edge_prob, (e_cand,), "STGraphTransformerDBN.prior_edge_prob")

        device = z_veh.device
        dtype = z_veh.dtype

        if isinstance(prev_memory, DBNTemporalMemory):
            torch_prev: TorchTemporalMemory | None = TorchTemporalMemory.from_numpy_memory(
                prev_memory,
                d_model=self.d_model,
                d_edge=self.d_edge,
                device=device,
                dtype=dtype,
            )
        else:
            torch_prev = prev_memory

        delta_t = (
            max(0.0, float(sim_time - torch_prev.sim_time))
            if torch_prev is not None and sim_time >= torch_prev.sim_time
            else float(dt)
        )

        # Build canonical candidate edge keys (src_id, dst_id) only when online dict alignment needs them
        if e_cand > 0 and not skip_edge_keys:
            src_cpu = edge_index[0].detach().cpu().tolist()
            dst_cpu = edge_index[1].detach().cpu().tolist()
            edge_keys: tuple[tuple[str, str], ...] = tuple(
                (veh_ids[int(s)], veh_ids[int(d)]) for s, d in zip(src_cpu, dst_cpu, strict=True)
            )
        else:
            edge_keys = ()

        # 1. Encode continuous & relational stems
        stem_v = self.veh_stem(z_veh)
        stem_c = self.corridor_stem(z_corridor)
        if e_cand > 0:
            lin_part = self.edge_linear_embed(z_edge)
            st_part = self.edge_cont_stem(z_edge[:, SPATIOTEMP_EDGE_SLICE])
            stem_e = F.silu(self.edge_stem_norm(lin_part + st_part))
        else:
            stem_e = z_edge.new_empty((0, self.d_edge))
        stem_cc = self.corridor_edge_stem(z_corridor_edge)

        # 2. Temporal Recurrent State Alignment + GRU Update
        if torch_prev is None:
            prev_v_h = self.veh_gru.init_state(n_v, device=device, dtype=dtype)
            prev_c_h = self.corridor_gru.init_state(n_c, device=device, dtype=dtype)
            prev_e_h = self.edge_gru.init_state(e_cand, device=device, dtype=dtype)
        else:
            if veh_same or (veh_curr_idx is not None and veh_prev_idx is not None):
                prev_v_h = _align_entity_memory_indexed(
                    n_v,
                    torch_prev.veh_hidden,
                    self.veh_gru,
                    curr_idx=veh_curr_idx,
                    prev_idx=veh_prev_idx,
                    same_order=veh_same,
                    delta_t=delta_t,
                    device=device,
                    dtype=dtype,
                )
            else:
                prev_v_h = _align_entity_memory(
                    veh_ids,
                    torch_prev.veh_ids,
                    torch_prev.veh_hidden,
                    self.veh_gru,
                    delta_t=delta_t,
                    device=device,
                    dtype=dtype,
                )

            if corr_same or (corr_curr_idx is not None and corr_prev_idx is not None):
                prev_c_h = _align_entity_memory_indexed(
                    n_c,
                    torch_prev.corridor_hidden,
                    self.corridor_gru,
                    curr_idx=corr_curr_idx,
                    prev_idx=corr_prev_idx,
                    same_order=corr_same,
                    delta_t=delta_t,
                    device=device,
                    dtype=dtype,
                )
            else:
                prev_c_h = _align_entity_memory(
                    corridor_ids,
                    torch_prev.corridor_ids,
                    torch_prev.corridor_hidden,
                    self.corridor_gru,
                    delta_t=delta_t,
                    device=device,
                    dtype=dtype,
                )

            if edge_same or (edge_curr_idx is not None and edge_prev_idx is not None):
                prev_e_h = _align_entity_memory_indexed(
                    e_cand,
                    torch_prev.edge_hidden,
                    self.edge_gru,
                    curr_idx=edge_curr_idx,
                    prev_idx=edge_prev_idx,
                    same_order=edge_same,
                    delta_t=delta_t,
                    device=device,
                    dtype=dtype,
                )
            else:
                prev_e_h = _align_entity_memory(
                    edge_keys,
                    torch_prev.edge_keys,
                    torch_prev.edge_hidden,
                    self.edge_gru,
                    delta_t=delta_t,
                    device=device,
                    dtype=dtype,
                )

        mem_v_h = self.veh_gru(stem_v, prev_v_h)
        mem_c_h = self.corridor_gru(stem_c, prev_c_h)
        mem_e_h = self.edge_gru(stem_e, prev_e_h)

        # 3. Stacked Spatio-Temporal Graph Transformer Blocks
        h_v = mem_v_h
        h_c = mem_c_h
        h_e = mem_e_h
        h_cc = stem_cc
        for block in self.blocks:
            h_v, h_c, h_e, h_cc = block(
                h_v,
                h_c,
                h_e,
                h_cc,
                edge_index=edge_index,
                corridor_edge_index=corridor_edge_index,
                veh_corridor_idx=veh_corridor_idx,
                veh_turn_src_idx=veh_turn_src_idx,
                veh_turn_out_corridor_idx=veh_turn_out_corridor_idx,
                veh_turn_prob=veh_turn_prob,
            )

        nv_out = self.final_norm_v(h_v)
        nc_out = self.final_norm_c(h_c)
        ne_out = self.final_norm_e(h_e)

        # 4. Amortized Node Heads (promoted to FP32 so downstream log-odds/Noisy-OR/LBP are bf16-hazard-free)
        if n_v > 0:
            node_raw = self.node_head_mlp(nv_out).float()
            delta_r = 8.0 * torch.tanh(node_raw[:, 0] / 8.0)
            precedence = topo_precedence.float() + delta_r
            leak_prob = 0.01 + 0.64 * torch.sigmoid(node_raw[:, 1])
            self_persist_prob = 0.05 + 0.90 * torch.sigmoid(node_raw[:, 2])
            spill_weight = 0.05 + 0.90 * torch.sigmoid(node_raw[:, 3])
            expected_delay_s = F.softplus(node_raw[:, 4])
        else:
            empty_n = torch.empty((0,), device=device, dtype=torch.float32)
            delta_r = empty_n
            precedence = empty_n
            leak_prob = empty_n
            self_persist_prob = empty_n
            spill_weight = empty_n
            expected_delay_s = empty_n

        # 5. Amortized Pairwise Edge Heads + Markov Prior Blending (promoted to FP32)
        if e_cand > 0:
            src_v = edge_index[0]
            dst_v = edge_index[1]
            e_hidden = F.silu(
                self.edge_head_src(nv_out).index_select(0, src_v)
                + self.edge_head_dst(nv_out).index_select(0, dst_v)
                + self.edge_head_e(ne_out)
            )
            edge_raw = self.edge_head_out(e_hidden).float()
            prior_clamped = prior_edge_prob.float().clamp(min=1e-4, max=1.0 - 1e-4)
            prior_logit = torch.log(prior_clamped) - torch.log1p(-prior_clamped)
            prior_weight = F.softplus(self.raw_prior_blend.float())
            edge_logits = edge_raw[:, 0] + prior_weight * prior_logit
            kappa = 0.02 + 0.96 * torch.sigmoid(edge_raw[:, 1])
            gamma = 0.02 + 0.96 * torch.sigmoid(edge_raw[:, 2])
        else:
            empty_e = torch.empty((0,), device=device, dtype=torch.float32)
            edge_logits = empty_e
            kappa = empty_e
            gamma = empty_e

        # 6. Hybrid Physical + Neural Corridor Spillback Logits (promoted to FP32)
        if n_c > 0:
            with torch.autocast(device_type=device.type, enabled=False):
                phys_logits = self.phys_corridor_cpd(
                    x_corridor_raw.float(), check_invariants=check_invariants
                ).float()
            neural_res = self.corridor_res_head(nc_out).squeeze(-1).float()
            corridor_spillback_logits = phys_logits + neural_res
        else:
            corridor_spillback_logits = torch.empty((0,), device=device, dtype=torch.float32)

        # Post-forward numerical guards
        if check_invariants:
            require_finite(precedence, "STGraphTransformerDBN.precedence")
            require_finite(edge_logits, "STGraphTransformerDBN.edge_logits")
            require_finite(corridor_spillback_logits, "STGraphTransformerDBN.corridor_spillback_logits")
            require_finite(expected_delay_s, "STGraphTransformerDBN.expected_delay_s")

        # Update stored hidden states in FP32 with post-transformer contextualized residuals for next tick
        next_mem = TorchTemporalMemory(
            sim_time=float(sim_time),
            veh_ids=veh_ids,
            veh_hidden=0.5 * (mem_v_h.float() + nv_out.float()),
            corridor_ids=corridor_ids,
            corridor_hidden=0.5 * (mem_c_h.float() + nc_out.float()),
            edge_keys=edge_keys,
            edge_hidden=0.5 * (mem_e_h.float() + ne_out.float()),
        )

        return TransformerStepOutputs(
            precedence=precedence,
            delta_precedence=delta_r,
            leak_prob=leak_prob,
            self_persist_prob=self_persist_prob,
            spill_weight=spill_weight,
            expected_delay_s=expected_delay_s,
            edge_logits=edge_logits,
            kappa=kappa,
            gamma=gamma,
            corridor_spillback_logits=corridor_spillback_logits,
            next_memory=next_mem,
        )

    def metadata_dict(self) -> dict[str, int | float | str]:
        """Return architecture metadata verified on checkpoint load."""
        return {
            "preset_name": self.preset_name,
            "d_model": self.d_model,
            "d_edge": self.d_edge,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "ffn_mult": self.preset_spec.ffn_mult,
            "n_rbf": self.preset_spec.n_rbf,
            "n_fourier_freqs": self.preset_spec.n_fourier_freqs,
            "node_feature_dim": NODE_FEATURE_DIM,
            "corridor_feature_dim": CORRIDOR_FEATURE_DIM,
            "edge_feature_dim": EDGE_FEATURE_DIM,
            "corridor_edge_feature_dim": CORRIDOR_EDGE_FEATURE_DIM,
            "prediction_horizon_s": self.prediction_horizon_s,
        }

    def verify_metadata(self, meta: dict[str, object]) -> None:
        """Fail loudly if checkpoint architecture metadata disagrees with ``self``."""
        expected = self.metadata_dict()
        for key, exp_val in expected.items():
            if key not in meta:
                raise CheckpointIntegrityError(
                    f"Checkpoint model metadata missing required key {key!r}.",
                    context={"missing_key": key},
                )
            act_val = meta[key]
            if isinstance(exp_val, float):
                if not isinstance(act_val, (int, float)) or abs(float(act_val) - exp_val) > 1e-5:
                    raise CheckpointIntegrityError(
                        f"Checkpoint metadata mismatch on {key!r}: got {act_val!r}, expected {exp_val!r}.",
                        context={"key": key, "actual": act_val, "expected": exp_val},
                    )
            elif act_val != exp_val:
                raise CheckpointIntegrityError(
                    f"Checkpoint metadata mismatch on {key!r}: got {act_val!r}, expected {exp_val!r}.",
                    context={"key": key, "actual": act_val, "expected": exp_val},
                )


__all__ = [
    "ContinuousDecayGRU",
    "MultiScaleBasisEncoder",
    "RMSNorm",
    "STGraphTransformerBlock",
    "STGraphTransformerDBN",
    "SwiGLU",
    "TorchTemporalMemory",
    "TransformerStepOutputs",
    "segment_softmax",
]

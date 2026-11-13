"""NeuralDBNUpdater, Differentiable TBPTT Step Runner, Checkpoint Integrity, & Updater Factory.

Satisfies ``DBNUpdaterProtocol`` and Boundary Locks A, B, and C (``#19``):
- ``torch.load(..., weights_only=True)`` exclusively, with SHA-256 verification, schema/preset/dim
  checks, fitted-normalizer checks, and ``require_calibrated`` enforcement.
- Differentiable ``forward_dbn_step`` enabling unrolled multi-step end-to-end PyTorch training
  through ``STGraphTransformerDBN`` + ``ContinuousDecayGRU`` + ``GeneralizedNoisyOR`` +
  ``multiplicative_sum_product_lbp``.
- Runtime ``NeuralDBNUpdater`` supporting ``neural_dbn`` and ablations ``no_temporal_prior``,
  ``static_cpd``, and ``fixed_route`` with bit-identical ``.reset()`` semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..config import DBNConfig, validate_compute_device
from ..contracts import (
    DBNEdge,
    DBNStateSnapshot,
    DBNTemporalMemory,
    DBNUpdaterProtocol,
    NEURAL_POLICY_NAMES,
    RelationType,
    RoadTopology,
    SCHEMA_VERSION,
    SwarmBeliefState,
)
from ..logging_utils import (
    CheckpointIntegrityError,
    ConfigValidationError,
    get_logger,
    require_dag,
    require_finite,
    require_probability_range,
    update_crash_context,
)
from .baselines_dbn import HeuristicRadiusDBNUpdater, IndependentDBNUpdater, OnlineBICDBNUpdater
from .calibration import TemperatureScales
from .cpd import (
    PrunedDAGStructure,
    STATIC_CPD_BASE_LEAK,
    STATIC_CPD_PERSIST_WEIGHT,
    STATIC_CPD_SPILL_WEIGHT,
    expected_downstream_spillback,
    gate_and_prune_edges,
    generalized_noisy_or_base_survival,
    static_edge_cpd_parameters,
    temporal_edge_prior,
)
from .features import FeatureNormalizer, GraphFeatureBatch, extract_graph_features
from .inference import LBPInferenceResult, multiplicative_sum_product_lbp
from .transformer import STGraphTransformerDBN, TorchTemporalMemory, TransformerStepOutputs

_LOG = get_logger("dbn.updater")


def _compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _sidecar_sha256_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.name}.sha256.json")


def _primitive_copy(obj: Any) -> Any:
    """Convert nested metadata to pure JSON/weights_only-safe Python primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, float):
            require_finite(obj, "checkpoint_metadata_float")
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return _primitive_copy(obj.item())
    if isinstance(obj, np.ndarray):
        require_finite(obj, "checkpoint_metadata_ndarray")
        return obj.tolist()
    if isinstance(obj, Mapping):
        return {str(k): _primitive_copy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_primitive_copy(v) for v in obj]
    raise ConfigValidationError(
        f"Unsupported non-primitive type {type(obj).__name__} in checkpoint training_metadata.",
        context={"type": type(obj).__name__},
    )


@dataclass(frozen=True)
class DBNCheckpointBundle:
    """Validated checkpoint artifacts loaded strictly via ``weights_only=True``."""

    model: STGraphTransformerDBN
    normalizer: FeatureNormalizer
    temperatures: TemperatureScales
    checkpoint_sha256: str
    model_metadata: dict[str, Any]
    training_metadata: dict[str, Any]


def save_dbn_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: STGraphTransformerDBN,
    normalizer: FeatureNormalizer,
    temperatures: TemperatureScales,
    training_metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, str]:
    """Atomically write a calibrated/uncalibrated DBN checkpoint + SHA-256 manifest."""
    if not normalizer.fitted:
        raise CheckpointIntegrityError("Refusing to save checkpoint with unfitted FeatureNormalizer.")

    out_path = Path(checkpoint_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clean_state_dict: dict[str, Tensor] = {}
    for param_name, tensor in model.state_dict().items():
        require_finite(tensor, f"save_dbn_checkpoint.state_dict[{param_name}]")
        clean_state_dict[param_name] = tensor.detach().cpu().to(torch.float32).contiguous()

    meta_dict = model.metadata_dict()
    train_meta = _primitive_copy(dict(training_metadata) if training_metadata is not None else {})

    payload: dict[str, Any] = {
        "schema_version": int(SCHEMA_VERSION),
        "model_metadata": meta_dict,
        "state_dict": clean_state_dict,
        "normalizer": normalizer.to_dict(),
        "temperatures": temperatures.to_dict(),
        "training_metadata": train_meta,
    }

    tmp_ckpt = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_ckpt)
    os.replace(tmp_ckpt, out_path)

    sha256_hex = _compute_file_sha256(out_path)
    sidecar = _sidecar_sha256_path(out_path)
    tmp_sidecar = sidecar.with_name(f".{sidecar.name}.tmp.{os.getpid()}")
    sidecar_payload = {
        "schema_version": int(SCHEMA_VERSION),
        "checkpoint_file": out_path.name,
        "sha256": sha256_hex,
        "preset_name": model.preset_name,
        "calibrated": bool(temperatures.calibrated),
    }
    tmp_sidecar.write_text(json.dumps(sidecar_payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_sidecar, sidecar)

    _LOG.info(
        "Saved DBN checkpoint %s (preset=%s, calibrated=%s, sha256=%s)",
        out_path,
        model.preset_name,
        temperatures.calibrated,
        sha256_hex[:12],
    )
    return out_path, sha256_hex


def load_dbn_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_preset: str | None = None,
    require_calibrated: bool = True,
    device: str = "cpu",
) -> DBNCheckpointBundle:
    """Strictly validate and load a DBN checkpoint using ``torch.load(..., weights_only=True)``."""
    target_device, _ = validate_compute_device(device, "fp32")
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise CheckpointIntegrityError(
            f"DBN checkpoint file does not exist: {ckpt_path}",
            context={"checkpoint_path": str(ckpt_path)},
        )
    if ckpt_path.stat().st_size == 0:
        raise CheckpointIntegrityError(
            f"DBN checkpoint file is empty (0 bytes): {ckpt_path}",
            context={"checkpoint_path": str(ckpt_path)},
        )

    actual_sha256 = _compute_file_sha256(ckpt_path)
    sidecar = _sidecar_sha256_path(ckpt_path)
    if sidecar.is_file():
        try:
            sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _LOG.error("Corrupted SHA-256 sidecar %s: %s", sidecar, exc)
            raise CheckpointIntegrityError(
                f"Corrupted SHA-256 sidecar at {sidecar}: {exc}",
                context={"sidecar": str(sidecar)},
            ) from exc
        expected_sha = sidecar_data.get("sha256")
        if expected_sha != actual_sha256:
            raise CheckpointIntegrityError(
                f"Checkpoint SHA-256 mismatch for {ckpt_path}: actual={actual_sha256} != expected={expected_sha}.",
                context={
                    "checkpoint_path": str(ckpt_path),
                    "actual_sha256": actual_sha256,
                    "expected_sha256": expected_sha,
                },
            )

    try:
        raw_payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except (RuntimeError, EOFError, OSError, ValueError, pickle.UnpicklingError) as exc:
        _LOG.error("Failed loading checkpoint %s with weights_only=True: %s", ckpt_path, exc)
        raise CheckpointIntegrityError(
            f"Failed to deserialize checkpoint {ckpt_path} with weights_only=True: {exc}",
            context={"checkpoint_path": str(ckpt_path), "error": str(exc)},
        ) from exc

    if not isinstance(raw_payload, Mapping):
        raise CheckpointIntegrityError(
            f"Checkpoint root object must be a dict, got {type(raw_payload).__name__}.",
            context={"checkpoint_path": str(ckpt_path)},
        )

    required_top = {
        "schema_version",
        "model_metadata",
        "state_dict",
        "normalizer",
        "temperatures",
        "training_metadata",
    }
    missing_top = sorted(required_top - set(raw_payload.keys()))
    if missing_top:
        raise CheckpointIntegrityError(
            f"Checkpoint missing required top-level keys: {missing_top}.",
            context={"checkpoint_path": str(ckpt_path), "missing": missing_top},
        )

    if raw_payload["schema_version"] != SCHEMA_VERSION:
        raise CheckpointIntegrityError(
            f"Checkpoint schema_version={raw_payload['schema_version']!r} != runtime SCHEMA_VERSION={SCHEMA_VERSION}.",
            context={
                "checkpoint_schema_version": raw_payload["schema_version"],
                "expected_schema_version": SCHEMA_VERSION,
            },
        )

    model_meta = raw_payload["model_metadata"]
    if not isinstance(model_meta, Mapping) or "preset_name" not in model_meta:
        raise CheckpointIntegrityError("Invalid or missing model_metadata in checkpoint.")

    ckpt_preset = str(model_meta["preset_name"])
    if expected_preset is not None and ckpt_preset != expected_preset:
        raise CheckpointIntegrityError(
            f"Checkpoint preset {ckpt_preset!r} != requested model_preset {expected_preset!r}.",
            context={"checkpoint_preset": ckpt_preset, "expected_preset": expected_preset},
        )

    pred_h = float(model_meta.get("prediction_horizon_s", 6.0))
    try:
        model = STGraphTransformerDBN(preset=ckpt_preset, prediction_horizon_s=pred_h)  # type: ignore[arg-type]
    except ConfigValidationError as exc:
        _LOG.error("Invalid preset in checkpoint %s: %s", ckpt_path, exc)
        raise CheckpointIntegrityError(
            f"Invalid preset metadata in checkpoint {ckpt_path}: {exc}",
            context={"preset_name": ckpt_preset},
        ) from exc

    model.verify_metadata(dict(model_meta))

    state_dict = raw_payload["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise CheckpointIntegrityError("Checkpoint state_dict must be a mapping.")

    expected_sd = model.state_dict()
    missing_params = sorted(set(expected_sd.keys()) - set(state_dict.keys()))
    extra_params = sorted(set(state_dict.keys()) - set(expected_sd.keys()))
    if missing_params or extra_params:
        raise CheckpointIntegrityError(
            "Checkpoint state_dict keys do not match STGraphTransformerDBN parameters.",
            context={"missing_keys": missing_params[:16], "unexpected_keys": extra_params[:16]},
        )

    validated_sd: dict[str, Tensor] = {}
    for k, exp_t in expected_sd.items():
        cand_t = state_dict[k]
        if not isinstance(cand_t, Tensor):
            raise CheckpointIntegrityError(f"Checkpoint state_dict[{k!r}] is not a Tensor.")
        if tuple(cand_t.shape) != tuple(exp_t.shape):
            raise CheckpointIntegrityError(
                f"Shape mismatch in state_dict[{k!r}]: got {tuple(cand_t.shape)} vs expected {tuple(exp_t.shape)}.",
                context={"param": k, "actual": list(cand_t.shape), "expected": list(exp_t.shape)},
            )
        require_finite(cand_t, f"checkpoint.state_dict[{k}]")
        validated_sd[k] = cand_t

    model.load_state_dict(validated_sd, strict=True)
    normalizer = FeatureNormalizer.from_dict(raw_payload["normalizer"])
    temperatures = TemperatureScales.from_dict(raw_payload["temperatures"])

    if require_calibrated and not temperatures.calibrated:
        raise CheckpointIntegrityError(
            f"Checkpoint {ckpt_path} is uncalibrated (temperatures.calibrated=False), "
            "but require_calibrated=True was requested.",
            context={"checkpoint_path": str(ckpt_path)},
        )

    model.to(torch.device(target_device))
    model.eval()
    update_crash_context(checkpoint_sha256=actual_sha256)

    train_meta_raw = raw_payload["training_metadata"]
    train_meta = dict(train_meta_raw) if isinstance(train_meta_raw, Mapping) else {}

    return DBNCheckpointBundle(
        model=model,
        normalizer=normalizer,
        temperatures=temperatures,
        checkpoint_sha256=actual_sha256,
        model_metadata=dict(model_meta),
        training_metadata=train_meta,
    )


@dataclass(frozen=True)
class DifferentiableStepOutput:
    """Full differentiable forward + CPD + LBP tensors for one DBN step (used in training & runtime)."""

    transformer_out: TransformerStepOutputs
    edge_logits_scaled: Tensor             # [E_cand]: edge_logits / T_cal_edge
    omega_raw: Tensor                      # [E_cand]: calibrated raw edge probabilities
    corridor_spillback_logits_scaled: Tensor  # [N_C]: corridor_spillback_logits / T_cal_link
    corridor_spillback_prob: Tensor        # [N_C]: calibrated corridor spillback probabilities
    expected_down_spill: Tensor            # [N_t]: turn-weighted expected downstream spillback
    base_survival_z0: Tensor               # [N_t]: parent-free survival probability
    pruned_dag: PrunedDAGStructure
    lbp_result: LBPInferenceResult
    delay_pred_logits_scaled: Tensor       # [N_t]: delay_pred_logits / T_cal_delay
    delay_pred: Tensor                     # [N_t]: calibrated prior delay probabilities
    delay_post_logits_scaled: Tensor       # [N_t]: delay_post_logits / T_cal_delay
    delay_post: Tensor                     # [N_t]: calibrated posterior delay probabilities
    expected_delay_s: Tensor               # [N_t]: continuous horizon delay estimate >= 0
    cascade_score: Tensor                  # [N_t]: 2-hop downstream cascade sensitivity >= 0
    next_memory: TorchTemporalMemory


@dataclass(frozen=True)
class PrecomputedStepInputs:
    """Pre-staged device tensors and step-to-step alignment indices for zero-overhead unrolled BPTT."""

    sim_time: float
    dt: float
    veh_ids: tuple[str, ...]
    corridor_ids: tuple[str, ...]
    num_vehicles: int
    num_corridors: int
    num_edges: int
    z_veh: Tensor                        # [N_t, NODE_FEATURE_DIM] float32 on device
    z_corr: Tensor                       # [N_C, CORRIDOR_FEATURE_DIM] float32 on device
    z_edge: Tensor                       # [E_t, EDGE_FEATURE_DIM] float32 on device
    z_cc: Tensor                         # [E_C, CORRIDOR_EDGE_FEATURE_DIM] float32 on device
    x_corr_raw: Tensor                   # [N_C, CORRIDOR_FEATURE_DIM] float32 on device
    x_edge_raw: Tensor | None            # [E_t, EDGE_FEATURE_DIM] float32 on device when needed
    edge_index: Tensor                   # [2, E_t] int64 on device
    corr_edge_index: Tensor              # [2, E_C] int64 on device
    veh_corridor_idx: Tensor             # [N_t] int64 on device
    veh_turn_src_idx: Tensor             # [T_nz] int64 on device
    veh_turn_out_corridor_idx: Tensor    # [T_nz] int64 on device
    veh_turn_prob: Tensor                # [T_nz] float32 on device
    topo_precedence: Tensor              # [N_t] float32 on device
    speed_evidence_pos: Tensor           # [N_t] float32 on device
    base_prev_delay_post: Tensor         # [N_t] float32 on device
    base_prior_edge_prob: Tensor         # [E_t] float32 on device
    is_contiguous_from_prev: bool
    veh_same: bool
    veh_curr_idx: Tensor                 # [M_v] int64 on device
    veh_prev_idx: Tensor                 # [M_v] int64 on device
    corr_same: bool
    corr_curr_idx: Tensor                # [M_c] int64 on device
    corr_prev_idx: Tensor                # [M_c] int64 on device
    edge_same: bool
    edge_curr_idx: Tensor                # [M_e] int64 on device
    edge_prev_idx: Tensor                # [M_e] int64 on device


def _run_post_transformer_cpd_lbp(
    *,
    tf_out: TransformerStepOutputs,
    edge_index: Tensor,
    turn_src_idx: Tensor,
    turn_out_c_idx: Tensor,
    turn_prob: Tensor,
    prev_delay_t: Tensor,
    ev_pos: Tensor,
    x_edge_raw_t: Tensor | None,
    n_v: int,
    dbn_cfg: DBNConfig,
    temps: TemperatureScales,
    use_static_cpd: bool,
    training: bool,
    dag_temperature: float,
    dev: torch.device,
    check_invariants: bool = True,
) -> DifferentiableStepOutput:
    """Shared FP32 CPD parameterization, DAG pruning, and multiplicative sum-product LBP stage."""
    with torch.autocast(device_type=dev.type, enabled=False):
        edge_logits_scaled = tf_out.edge_logits.float() / float(temps.t_edge)
        omega_raw = torch.sigmoid(edge_logits_scaled)

        corr_logits_scaled = tf_out.corridor_spillback_logits.float() / float(temps.t_link)
        corr_spill_prob = torch.sigmoid(corr_logits_scaled)

        exp_down_spill = expected_downstream_spillback(
            corr_spill_prob,
            turn_src_idx,
            turn_out_c_idx,
            turn_prob.float(),
            n_v,
            check_invariants=check_invariants,
        )

        if use_static_cpd:
            if x_edge_raw_t is None:
                raise ConfigValidationError(
                    "use_static_cpd=True requires raw edge features x_edge_raw_t."
                )
            leak_t = torch.full((n_v,), STATIC_CPD_BASE_LEAK, dtype=torch.float32, device=dev)
            persist_t = torch.full((n_v,), STATIC_CPD_PERSIST_WEIGHT, dtype=torch.float32, device=dev)
            spill_w_t = torch.full((n_v,), STATIC_CPD_SPILL_WEIGHT, dtype=torch.float32, device=dev)
            kappa_t, gamma_t = static_edge_cpd_parameters(x_edge_raw_t.float())
        else:
            leak_t = tf_out.leak_prob.float()
            persist_t = tf_out.self_persist_prob.float()
            spill_w_t = tf_out.spill_weight.float()
            kappa_t = tf_out.kappa.float()
            gamma_t = tf_out.gamma.float()

        z0 = generalized_noisy_or_base_survival(
            leak_prob=leak_t,
            self_persist_prob=persist_t,
            prev_delay_post=prev_delay_t.float(),
            spill_weight=spill_w_t,
            expected_down_spill=exp_down_spill,
            check_invariants=check_invariants,
        )

        pruned = gate_and_prune_edges(
            edge_index=edge_index,
            precedence=tf_out.precedence.float(),
            omega_raw=omega_raw,
            kappa=kappa_t,
            gamma=gamma_t,
            num_vehicles=n_v,
            max_parents_k=dbn_cfg.max_parents_k,
            min_edge_prob=dbn_cfg.min_edge_prob,
            training=training,
            dag_temperature=dag_temperature,
            check_invariants=check_invariants,
        )

        lbp_res = multiplicative_sum_product_lbp(
            base_survival_z0=z0,
            pruned_dag=pruned,
            speed_evidence_pos=ev_pos.float(),
            lbp_sweeps=dbn_cfg.lbp_sweeps,
            lbp_damping=dbn_cfg.lbp_damping,
            check_invariants=check_invariants,
        )

        delay_pred_logits_scaled = lbp_res.delay_pred_logits / float(temps.t_delay)
        delay_post_logits_scaled = lbp_res.delay_post_logits / float(temps.t_delay)
        if abs(float(temps.t_delay) - 1.0) <= 1e-8:
            delay_pred_cal = lbp_res.delay_pred
            delay_post_cal = lbp_res.delay_post
        else:
            delay_pred_cal = torch.sigmoid(delay_pred_logits_scaled)
            delay_post_cal = torch.sigmoid(delay_post_logits_scaled)

        if check_invariants:
            require_probability_range(delay_pred_cal, "forward_dbn_step.delay_pred")
            require_probability_range(delay_post_cal, "forward_dbn_step.delay_post")

    return DifferentiableStepOutput(
        transformer_out=tf_out,
        edge_logits_scaled=edge_logits_scaled,
        omega_raw=omega_raw,
        corridor_spillback_logits_scaled=corr_logits_scaled,
        corridor_spillback_prob=corr_spill_prob,
        expected_down_spill=exp_down_spill,
        base_survival_z0=z0,
        pruned_dag=pruned,
        lbp_result=lbp_res,
        delay_pred_logits_scaled=delay_pred_logits_scaled,
        delay_pred=delay_pred_cal,
        delay_post_logits_scaled=delay_post_logits_scaled,
        delay_post=delay_post_cal,
        expected_delay_s=tf_out.expected_delay_s,
        cascade_score=lbp_res.cascade_score,
        next_memory=tf_out.next_memory,
    )


def forward_dbn_step(
    *,
    model: Any,
    normalizer: FeatureNormalizer,
    feat_batch: GraphFeatureBatch,
    dbn_cfg: DBNConfig = DBNConfig(),
    temperatures: TemperatureScales | None = None,
    prev_memory: TorchTemporalMemory | DBNTemporalMemory | None = None,
    prev_delay_post_override: Mapping[str, Tensor] | None = None,
    prev_edge_omega_override: Mapping[tuple[str, str], Tensor] | None = None,
    use_static_cpd: bool = False,
    reset_temporal_prior: bool = False,
    dag_temperature: float = 0.50,
    device: torch.device | str | None = None,
) -> DifferentiableStepOutput:
    """Execute one differentiable end-to-end ``STGraphTransformerDBN -> CPD -> LBP`` control step."""
    temps = temperatures if temperatures is not None else TemperatureScales.uncalibrated_unit()
    if device is None:
        dev = next(model.parameters()).device
    elif isinstance(device, str):
        dev = torch.device(device)
    else:
        dev = device

    z_v_np, z_c_np, z_e_np, z_cc_np = normalizer.transform_arrays(
        feat_batch.x_veh,
        feat_batch.x_corridor,
        feat_batch.x_edge,
        feat_batch.x_corridor_edge,
    )

    cand = feat_batch.candidate_graph
    n_v = cand.num_vehicles
    e_cand = cand.num_edges

    z_veh = torch.as_tensor(z_v_np, dtype=torch.float32, device=dev)
    z_corr = torch.as_tensor(z_c_np, dtype=torch.float32, device=dev)
    z_edge = torch.as_tensor(z_e_np, dtype=torch.float32, device=dev)
    z_cc = torch.as_tensor(z_cc_np, dtype=torch.float32, device=dev)
    x_corr_raw = torch.as_tensor(feat_batch.x_corridor, dtype=torch.float32, device=dev)

    edge_index = torch.as_tensor(cand.edge_index, dtype=torch.long, device=dev)
    corr_edge_index = torch.as_tensor(feat_batch.corridor_edge_index, dtype=torch.long, device=dev)
    veh_corr_idx = torch.as_tensor(feat_batch.veh_corridor_idx, dtype=torch.long, device=dev)
    turn_src_idx = torch.as_tensor(feat_batch.veh_turn_src_idx, dtype=torch.long, device=dev)
    turn_out_c_idx = torch.as_tensor(feat_batch.veh_turn_out_corridor_idx, dtype=torch.long, device=dev)
    turn_prob = torch.as_tensor(feat_batch.veh_turn_prob, dtype=torch.float32, device=dev)
    topo_prec = torch.as_tensor(feat_batch.topo_precedence, dtype=torch.float32, device=dev)
    ev_pos = torch.as_tensor(feat_batch.speed_evidence_pos, dtype=torch.float32, device=dev)

    # Differentiable temporal carries when unrolling multi-step sequences in TBPTT
    if reset_temporal_prior:
        prev_delay_t = torch.full((n_v,), float(dbn_cfg.prior_base_prob), dtype=torch.float32, device=dev)
        prior_edge_t = torch.full((e_cand,), float(dbn_cfg.prior_base_prob), dtype=torch.float32, device=dev)
        active_prev_mem: TorchTemporalMemory | DBNTemporalMemory | None = None
    else:
        active_prev_mem = prev_memory
        if prev_delay_post_override is not None and n_v > 0:
            base_d = torch.as_tensor(feat_batch.prev_delay_post, dtype=torch.float32, device=dev)
            matched_idx: list[int] = []
            matched_vals: list[Tensor] = []
            for idx, vid in enumerate(cand.veh_ids):
                ov = prev_delay_post_override.get(vid)
                if ov is not None:
                    matched_idx.append(idx)
                    matched_vals.append(ov.to(device=dev, dtype=torch.float32).reshape(()))
            if matched_idx:
                idx_t = torch.tensor(matched_idx, dtype=torch.long, device=dev)
                val_t = torch.stack(matched_vals, dim=0)
                prev_delay_t = base_d.index_put((idx_t,), val_t)
            else:
                prev_delay_t = base_d
        else:
            prev_delay_t = torch.as_tensor(feat_batch.prev_delay_post, dtype=torch.float32, device=dev)

        if prev_edge_omega_override is not None and e_cand > 0:
            prev_w = torch.full((e_cand,), float(dbn_cfg.prior_base_prob), dtype=torch.float32, device=dev)
            pers_m = torch.zeros((e_cand,), dtype=torch.float32, device=dev)
            src_cpu = cand.edge_index[0]
            dst_cpu = cand.edge_index[1]
            m_e_idx: list[int] = []
            m_e_vals: list[Tensor] = []
            for k in range(e_cand):
                pair = (cand.veh_ids[int(src_cpu[k])], cand.veh_ids[int(dst_cpu[k])])
                ov_w = prev_edge_omega_override.get(pair)
                if ov_w is not None:
                    m_e_idx.append(k)
                    m_e_vals.append(ov_w.to(device=dev, dtype=torch.float32).reshape(()))
            if m_e_idx:
                e_idx_t = torch.tensor(m_e_idx, dtype=torch.long, device=dev)
                e_val_t = torch.stack(m_e_vals, dim=0)
                prev_w = prev_w.index_put((e_idx_t,), e_val_t)
                pers_m = pers_m.index_put((e_idx_t,), torch.ones_like(e_val_t))
            prior_edge_t = temporal_edge_prior(
                prev_w,
                pers_m,
                prior_stay_prob=dbn_cfg.prior_stay_prob,
                prior_base_prob=dbn_cfg.prior_base_prob,
            )  # type: ignore[assignment]
        else:
            prior_edge_t = torch.as_tensor(feat_batch.prior_edge_prob, dtype=torch.float32, device=dev)

    tf_out = model(
        z_veh=z_veh,
        z_corridor=z_corr,
        z_edge=z_edge,
        z_corridor_edge=z_cc,
        x_corridor_raw=x_corr_raw,
        edge_index=edge_index,
        corridor_edge_index=corr_edge_index,
        veh_corridor_idx=veh_corr_idx,
        veh_turn_src_idx=turn_src_idx,
        veh_turn_out_corridor_idx=turn_out_c_idx,
        veh_turn_prob=turn_prob,
        topo_precedence=topo_prec,
        prior_edge_prob=prior_edge_t,
        veh_ids=cand.veh_ids,
        corridor_ids=feat_batch.corridor_ids,
        sim_time=feat_batch.sim_time,
        dt=feat_batch.dt,
        prev_memory=active_prev_mem,
    )

    x_edge_raw_t = (
        torch.as_tensor(feat_batch.x_edge, dtype=torch.float32, device=dev)
        if use_static_cpd
        else None
    )
    return _run_post_transformer_cpd_lbp(
        tf_out=tf_out,
        edge_index=edge_index,
        turn_src_idx=turn_src_idx,
        turn_out_c_idx=turn_out_c_idx,
        turn_prob=turn_prob,
        prev_delay_t=prev_delay_t,
        ev_pos=ev_pos,
        x_edge_raw_t=x_edge_raw_t,
        n_v=n_v,
        dbn_cfg=dbn_cfg,
        temps=temps,
        use_static_cpd=use_static_cpd,
        training=bool(model.training),
        dag_temperature=dag_temperature,
        dev=dev,
    )


def forward_dbn_precomputed_step(
    *,
    model: Any,
    step: PrecomputedStepInputs,
    dbn_cfg: DBNConfig = DBNConfig(),
    temperatures: TemperatureScales | None = None,
    prev_memory: TorchTemporalMemory | DBNTemporalMemory | None = None,
    prev_delay_vec: Tensor | None = None,
    prev_omega_vec: Tensor | None = None,
    use_static_cpd: bool = False,
    reset_temporal_prior: bool = False,
    dag_temperature: float = 0.50,
    check_invariants: bool = False,
) -> DifferentiableStepOutput:
    """Execute one zero-Python-loop differentiable DBN step on pre-staged device tensors."""
    temps = temperatures if temperatures is not None else TemperatureScales.uncalibrated_unit()
    dev = step.z_veh.device
    n_v = step.num_vehicles
    e_cand = step.num_edges

    if reset_temporal_prior:
        prev_delay_t = torch.full((n_v,), float(dbn_cfg.prior_base_prob), dtype=torch.float32, device=dev)
        prior_edge_t = torch.full((e_cand,), float(dbn_cfg.prior_base_prob), dtype=torch.float32, device=dev)
        active_prev_mem: TorchTemporalMemory | DBNTemporalMemory | None = None
    else:
        active_prev_mem = prev_memory
        if prev_delay_vec is not None and n_v > 0:
            if step.veh_same and prev_delay_vec.shape[0] == n_v:
                prev_delay_t = prev_delay_vec.to(device=dev, dtype=torch.float32)
            elif step.veh_curr_idx.numel() > 0:
                val_t = prev_delay_vec.to(device=dev, dtype=torch.float32).index_select(0, step.veh_prev_idx)
                prev_delay_t = step.base_prev_delay_post.index_put((step.veh_curr_idx,), val_t)
            else:
                prev_delay_t = step.base_prev_delay_post
        else:
            prev_delay_t = step.base_prev_delay_post

        if prev_omega_vec is not None and e_cand > 0:
            if step.edge_same and prev_omega_vec.shape[0] == e_cand:
                prev_w = prev_omega_vec.to(device=dev, dtype=torch.float32)
                pers_m = torch.ones((e_cand,), dtype=torch.float32, device=dev)
            else:
                prev_w = torch.full((e_cand,), float(dbn_cfg.prior_base_prob), dtype=torch.float32, device=dev)
                pers_m = torch.zeros((e_cand,), dtype=torch.float32, device=dev)
                if step.edge_curr_idx.numel() > 0:
                    e_val_t = prev_omega_vec.to(device=dev, dtype=torch.float32).index_select(0, step.edge_prev_idx)
                    prev_w = prev_w.index_put((step.edge_curr_idx,), e_val_t)
                    pers_m = pers_m.index_put((step.edge_curr_idx,), torch.ones_like(e_val_t))
            prior_edge_t = temporal_edge_prior(
                prev_w,
                pers_m,
                prior_stay_prob=dbn_cfg.prior_stay_prob,
                prior_base_prob=dbn_cfg.prior_base_prob,
                check_invariants=check_invariants,
            )  # type: ignore[assignment]
        else:
            prior_edge_t = step.base_prior_edge_prob

    tf_out = model(
        z_veh=step.z_veh,
        z_corridor=step.z_corr,
        z_edge=step.z_edge,
        z_corridor_edge=step.z_cc,
        x_corridor_raw=step.x_corr_raw,
        edge_index=step.edge_index,
        corridor_edge_index=step.corr_edge_index,
        veh_corridor_idx=step.veh_corridor_idx,
        veh_turn_src_idx=step.veh_turn_src_idx,
        veh_turn_out_corridor_idx=step.veh_turn_out_corridor_idx,
        veh_turn_prob=step.veh_turn_prob,
        topo_precedence=step.topo_precedence,
        prior_edge_prob=prior_edge_t,
        veh_ids=step.veh_ids,
        corridor_ids=step.corridor_ids,
        sim_time=step.sim_time,
        dt=step.dt,
        prev_memory=active_prev_mem,
        veh_curr_idx=step.veh_curr_idx,
        veh_prev_idx=step.veh_prev_idx,
        veh_same=step.veh_same,
        corr_curr_idx=step.corr_curr_idx,
        corr_prev_idx=step.corr_prev_idx,
        corr_same=step.corr_same,
        edge_curr_idx=step.edge_curr_idx,
        edge_prev_idx=step.edge_prev_idx,
        edge_same=step.edge_same,
        skip_edge_keys=True,
        check_invariants=check_invariants,
    )

    return _run_post_transformer_cpd_lbp(
        tf_out=tf_out,
        edge_index=step.edge_index,
        turn_src_idx=step.veh_turn_src_idx,
        turn_out_c_idx=step.veh_turn_out_corridor_idx,
        turn_prob=step.veh_turn_prob,
        prev_delay_t=prev_delay_t,
        ev_pos=step.speed_evidence_pos,
        x_edge_raw_t=step.x_edge_raw,
        n_v=n_v,
        dbn_cfg=dbn_cfg,
        temps=temps,
        use_static_cpd=use_static_cpd,
        training=bool(model.training),
        dag_temperature=dag_temperature,
        dev=dev,
        check_invariants=check_invariants,
    )


class NeuralDBNUpdater:
    """Real-time calibrated ``STGraphTransformerDBN`` updater implementing ``DBNUpdaterProtocol``."""

    def __init__(
        self,
        model: STGraphTransformerDBN,
        normalizer: FeatureNormalizer,
        temperatures: TemperatureScales,
        dbn_cfg: DBNConfig = DBNConfig(),
        *,
        policy_name: str = "neural_dbn",
    ) -> None:
        if policy_name not in NEURAL_POLICY_NAMES:
            raise ConfigValidationError(
                f"Invalid neural policy_name {policy_name!r}; expected one of {sorted(NEURAL_POLICY_NAMES)}.",
                context={"policy_name": policy_name},
            )
        if not normalizer.fitted:
            raise CheckpointIntegrityError("NeuralDBNUpdater requires a fitted FeatureNormalizer.")
        if dbn_cfg.require_calibrated and not temperatures.calibrated:
            raise CheckpointIntegrityError(
                "NeuralDBNUpdater requires calibrated TemperatureScales when dbn_cfg.require_calibrated=True."
            )

        target_dev, _ = validate_compute_device(dbn_cfg.inference_device, "fp32")
        self.device = torch.device(target_dev)
        self.model = model.to(self.device)
        self.model.eval()
        self.normalizer = normalizer
        self.temperatures = temperatures
        self.dbn_cfg = dbn_cfg
        self.policy_name = policy_name
        self.reset_temporal_prior = policy_name == "no_temporal_prior"
        self.use_static_cpd = policy_name == "static_cpd"
        self._cached_memory: TorchTemporalMemory | None = None

    def reset(self) -> None:
        """Reset all internal temporal recurrent states for bit-identical multi-episode execution."""
        self._cached_memory = None
        self.model.eval()

    def update_and_infer(
        self,
        belief: SwarmBeliefState,
        prev_snapshot: DBNStateSnapshot | None,
        topology: RoadTopology,
    ) -> DBNStateSnapshot:
        """Run candidate generation, ``STGraphTransformerDBN`` structure/CPD update, and LBP inference."""
        t0 = time.perf_counter()
        self.model.eval()

        feat_batch = extract_graph_features(
            belief=belief,
            topology=topology,
            dbn_cfg=self.dbn_cfg,
            prev_snapshot=prev_snapshot,
            reset_temporal_prior=self.reset_temporal_prior,
        )

        prev_mem: TorchTemporalMemory | DBNTemporalMemory | None
        if self.reset_temporal_prior or prev_snapshot is None:
            prev_mem = None
        elif (
            self._cached_memory is not None
            and prev_snapshot.temporal_memory is not None
            and abs(self._cached_memory.sim_time - prev_snapshot.temporal_memory.sim_time) <= 1e-9
        ):
            prev_mem = self._cached_memory
        else:
            prev_mem = prev_snapshot.temporal_memory

        with torch.inference_mode():
            # Time neural structure/CPD update vs LBP separately for runtime attribution
            step_out = forward_dbn_step(
                model=self.model,
                normalizer=self.normalizer,
                feat_batch=feat_batch,
                dbn_cfg=self.dbn_cfg,
                temperatures=self.temperatures,
                prev_memory=prev_mem,
                use_static_cpd=self.use_static_cpd,
                reset_temporal_prior=self.reset_temporal_prior,
                device=self.device,
            )

        t_total_ms = (time.perf_counter() - t0) * 1000.0
        t_assemble_0 = time.perf_counter()

        cand = feat_batch.candidate_graph
        veh_ids = cand.veh_ids
        corridor_ids = feat_batch.corridor_ids

        prec_np = step_out.transformer_out.precedence.detach().cpu().numpy()
        raw_omega_np = step_out.omega_raw.detach().cpu().numpy()
        act_idx_np = step_out.pruned_dag.active_edge_indices.detach().cpu().numpy()
        src_np = step_out.pruned_dag.src_idx.detach().cpu().numpy()
        dst_np = step_out.pruned_dag.dst_idx.detach().cpu().numpy()
        omega_np = step_out.pruned_dag.omega.detach().cpu().numpy()
        kappa_np = step_out.pruned_dag.kappa.detach().cpu().numpy()
        gamma_np = step_out.pruned_dag.gamma.detach().cpu().numpy()

        delay_pred_np = step_out.delay_pred.detach().cpu().numpy()
        delay_post_np = step_out.delay_post.detach().cpu().numpy()
        exp_delay_np = step_out.expected_delay_s.detach().cpu().numpy()
        cascade_np = step_out.cascade_score.detach().cpu().numpy()
        corr_spill_np = step_out.corridor_spillback_prob.detach().cpu().numpy()

        precedence_scores = {vid: float(prec_np[i]) for i, vid in enumerate(veh_ids)}
        edges_list: list[DBNEdge] = []
        edges_by_dst_lists: dict[str, list[DBNEdge]] = {vid: [] for vid in veh_ids}
        edges_by_src_lists: dict[str, list[DBNEdge]] = {vid: [] for vid in veh_ids}

        for k in range(int(act_idx_np.size)):
            c_idx = int(act_idx_np[k])
            s_i = int(src_np[k])
            d_i = int(dst_np[k])
            src_id = veh_ids[s_i]
            dst_id = veh_ids[d_i]
            edge_obj = DBNEdge(
                src_id=src_id,
                dst_id=dst_id,
                relation=RelationType(int(cand.relation_types[c_idx])),
                omega=float(omega_np[k]),
                kappa=float(kappa_np[k]),
                gamma=float(gamma_np[k]),
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
            context={"sim_time": feat_batch.sim_time, "policy_name": self.policy_name},
        )

        if cand.num_edges > 0:
            raw_edge_probs = dict(zip(step_out.next_memory.edge_keys, raw_omega_np.tolist()))
        else:
            raw_edge_probs = {}

        if self.reset_temporal_prior:
            self._cached_memory = None
            snapshot_mem: DBNTemporalMemory | None = None
        else:
            self._cached_memory = step_out.next_memory.detach_state()
            snapshot_mem = self._cached_memory.to_numpy_memory()

        t_assemble_ms = (time.perf_counter() - t_assemble_0) * 1000.0

        return DBNStateSnapshot(
            sim_time=float(feat_batch.sim_time),
            veh_ids=veh_ids,
            precedence_scores=precedence_scores,
            edges=edges_tuple,
            edges_by_dst={vid: tuple(elist) for vid, elist in edges_by_dst_lists.items()},
            edges_by_src={vid: tuple(elist) for vid, elist in edges_by_src_lists.items()},
            delay_pred={vid: float(delay_pred_np[i]) for i, vid in enumerate(veh_ids)},
            delay_post={vid: float(delay_post_np[i]) for i, vid in enumerate(veh_ids)},
            expected_delay_s={vid: float(exp_delay_np[i]) for i, vid in enumerate(veh_ids)},
            cascade_score={vid: float(cascade_np[i]) for i, vid in enumerate(veh_ids)},
            corridor_spillback_pred={cid: float(corr_spill_np[c]) for c, cid in enumerate(corridor_ids)},
            raw_edge_probs=raw_edge_probs,
            update_latency_ms=float(t_total_ms * 0.85),
            inference_latency_ms=float(t_total_ms * 0.15 + t_assemble_ms),
            temporal_memory=snapshot_mem,
        )


def build_dbn_updater(
    policy_name: str,
    dbn_cfg: DBNConfig = DBNConfig(),
    *,
    checkpoint_path: str | Path | None = None,
    bundle: DBNCheckpointBundle | None = None,
    unconstrained_all_pairs: bool = False,
) -> DBNUpdaterProtocol:
    """Instantiate a validated :class:`DBNUpdaterProtocol` for ``policy_name`` (zero silent fallback)."""
    if policy_name == "independent_belief":
        return IndependentDBNUpdater(dbn_cfg)
    if policy_name == "heuristic_radius_dbn":
        return HeuristicRadiusDBNUpdater(dbn_cfg)
    if policy_name == "online_bic_dbn":
        return OnlineBICDBNUpdater(dbn_cfg, unconstrained_all_pairs=unconstrained_all_pairs)

    if policy_name in NEURAL_POLICY_NAMES:
        if bundle is None:
            if checkpoint_path is None:
                raise CheckpointIntegrityError(
                    f"Neural policy {policy_name!r} requires an explicit checkpoint_path or DBNCheckpointBundle.",
                    context={"policy_name": policy_name},
                )
            bundle = load_dbn_checkpoint(
                checkpoint_path,
                expected_preset=dbn_cfg.model_preset,
                require_calibrated=dbn_cfg.require_calibrated,
                device=dbn_cfg.inference_device,
            )
        return NeuralDBNUpdater(
            model=bundle.model,
            normalizer=bundle.normalizer,
            temperatures=bundle.temperatures,
            dbn_cfg=dbn_cfg,
            policy_name=policy_name,
        )

    raise ConfigValidationError(
        f"Policy {policy_name!r} is not a DBN policy; valid DBN policies are "
        f"{sorted({'independent_belief', 'heuristic_radius_dbn', 'online_bic_dbn', *NEURAL_POLICY_NAMES})}.",
        context={"policy_name": policy_name},
    )


__all__ = [
    "DBNCheckpointBundle",
    "DifferentiableStepOutput",
    "NeuralDBNUpdater",
    "PrecomputedStepInputs",
    "build_dbn_updater",
    "forward_dbn_precomputed_step",
    "forward_dbn_step",
    "load_dbn_checkpoint",
    "save_dbn_checkpoint",
]

"""Immutable pre-registered splits, SOTA model presets, configuration schemas, and strict JSON I/O.

Zero silent defaults or implicit coercions are permitted:
- Unknown JSON keys, missing required fields, ``bool`` passed where ``int``/``float`` is expected,
  non-existent network XML files, unavailable ``SumoBackend`` / ``ComputeDevice`` / ``PrecisionMode``,
  or overlapping ``TRAIN_SEEDS`` / ``VAL_SEEDS`` / ``TEST_SEEDS`` raise :class:`ConfigValidationError`
  or :class:`BackendUnavailableError` immediately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import types
from typing import Any, Final, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

import torch

from .contracts import ComputeDevice, ModelPresetName, PrecisionMode, SumoBackend
from .logging_utils import (
    BackendUnavailableError,
    ConfigValidationError,
    get_logger,
    require_finite,
)

T = TypeVar("T")

# =============================================================================
# 1. Pre-Registered Immutable Seed Locks (`#14`, `#16`, `#20`)
# =============================================================================

TRAIN_SEEDS: Final[tuple[int, ...]] = tuple(range(1000, 1040))
VAL_SEEDS: Final[tuple[int, ...]] = tuple(range(2000, 2010))
TEST_SEEDS: Final[tuple[int, ...]] = tuple(range(3000, 3015))


def _require_strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(
            f"Field {name!r} requires strict int, got {type(value).__name__}={value!r}.",
            context={"field": name, "value": repr(value), "type": type(value).__name__},
        )
    return value


def _require_strict_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(
            f"Field {name!r} requires numeric float, got {type(value).__name__}={value!r}.",
            context={"field": name, "value": repr(value), "type": type(value).__name__},
        )
    out = float(value)
    require_finite(out, name)
    return out


def _require_strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigValidationError(
            f"Field {name!r} requires strict bool, got {type(value).__name__}={value!r}.",
            context={"field": name, "value": repr(value), "type": type(value).__name__},
        )
    return value


def _require_non_empty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(
            f"Field {name!r} requires a non-empty str, got {value!r}.",
            context={"field": name, "value": repr(value)},
        )
    return value


def validate_seed_splits(
    train_seeds: Sequence[int],
    val_seeds: Sequence[int],
    test_seeds: Sequence[int],
) -> None:
    """Ensure train, validation, and held-out test seed sets are non-empty, integer, and disjoint."""
    for split_name, seq in (("train_seeds", train_seeds), ("val_seeds", val_seeds), ("test_seeds", test_seeds)):
        if len(seq) == 0:
            raise ConfigValidationError(
                f"Seed split {split_name!r} cannot be empty.",
                context={"split": split_name},
            )
        for idx, s in enumerate(seq):
            _require_strict_int(s, f"{split_name}[{idx}]")
            if s < 0:
                raise ConfigValidationError(
                    f"Seed {split_name}[{idx}] must be >= 0, got {s}.",
                    context={"split": split_name, "seed": s},
                )
        if len(set(seq)) != len(seq):
            raise ConfigValidationError(
                f"Duplicate seeds detected inside {split_name!r}.",
                context={"split": split_name, "seeds": list(seq)},
            )

    train_set = set(train_seeds)
    val_set = set(val_seeds)
    test_set = set(test_seeds)

    overlap_tv = sorted(train_set & val_set)
    overlap_tt = sorted(train_set & test_set)
    overlap_vt = sorted(val_set & test_set)
    if overlap_tv or overlap_tt or overlap_vt:
        raise ConfigValidationError(
            "Data-leakage guard triggered: train/val/test seed splits overlap.",
            context={
                "train_val_overlap": overlap_tv,
                "train_test_overlap": overlap_tt,
                "val_test_overlap": overlap_vt,
            },
        )


# =============================================================================
# 2. ModelPreset Registry & Hardware/Backend Fail-Loud Validators (`#18` §4.2)
# =============================================================================


@dataclass(frozen=True)
class ModelPresetSpec:
    """Architecture & sequence-unrolling hyperparameters for ``STGraphTransformerDBN``."""

    name: ModelPresetName
    d_model: int
    d_edge: int
    n_layers: int
    n_heads: int
    ffn_mult: float
    dropout: float
    n_rbf: int
    n_fourier_freqs: int
    t_seq: int
    t_burn: int
    allow_eval_claims: bool

    def __post_init__(self) -> None:
        _require_strict_int(self.d_model, "ModelPresetSpec.d_model")
        _require_strict_int(self.d_edge, "ModelPresetSpec.d_edge")
        _require_strict_int(self.n_layers, "ModelPresetSpec.n_layers")
        _require_strict_int(self.n_heads, "ModelPresetSpec.n_heads")
        _require_strict_float(self.ffn_mult, "ModelPresetSpec.ffn_mult")
        _require_strict_float(self.dropout, "ModelPresetSpec.dropout")
        _require_strict_int(self.n_rbf, "ModelPresetSpec.n_rbf")
        _require_strict_int(self.n_fourier_freqs, "ModelPresetSpec.n_fourier_freqs")
        _require_strict_int(self.t_seq, "ModelPresetSpec.t_seq")
        _require_strict_int(self.t_burn, "ModelPresetSpec.t_burn")
        _require_strict_bool(self.allow_eval_claims, "ModelPresetSpec.allow_eval_claims")
        if self.d_model <= 0 or self.d_model % self.n_heads != 0:
            raise ConfigValidationError(
                f"d_model={self.d_model} must be positive and divisible by n_heads={self.n_heads}.",
                context={"d_model": self.d_model, "n_heads": self.n_heads},
            )
        if self.d_edge <= 0 or self.n_layers < 1 or self.t_seq < 1 or self.t_burn < 0:
            raise ConfigValidationError(
                f"Invalid preset dimensions on {self.name!r}.",
                context=asdict(self),
            )


MODEL_PRESETS: Final[dict[ModelPresetName, ModelPresetSpec]] = {
    "ci_micro": ModelPresetSpec(
        name="ci_micro",
        d_model=64,
        d_edge=32,
        n_layers=2,
        n_heads=4,
        ffn_mult=2.0,
        dropout=0.05,
        n_rbf=8,
        n_fourier_freqs=4,
        t_seq=4,
        t_burn=1,
        allow_eval_claims=False,
    ),
    "sota_local": ModelPresetSpec(
        name="sota_local",
        d_model=128,
        d_edge=64,
        n_layers=4,
        n_heads=8,
        ffn_mult=2.6667,
        dropout=0.10,
        n_rbf=16,
        n_fourier_freqs=6,
        t_seq=8,
        t_burn=2,
        allow_eval_claims=True,
    ),
    "sota_cloud_xl": ModelPresetSpec(
        name="sota_cloud_xl",
        d_model=256,
        d_edge=128,
        n_layers=8,
        n_heads=8,
        ffn_mult=2.6667,
        dropout=0.10,
        n_rbf=16,
        n_fourier_freqs=6,
        t_seq=16,
        t_burn=4,
        allow_eval_claims=True,
    ),
}


def get_model_preset(name: str) -> ModelPresetSpec:
    """Return the :class:`ModelPresetSpec` for ``name``, failing loudly if unknown."""
    if name not in MODEL_PRESETS:
        raise ConfigValidationError(
            f"Unknown model_preset {name!r}; valid presets are {sorted(MODEL_PRESETS)}.",
            context={"model_preset": name, "valid_presets": sorted(MODEL_PRESETS)},
        )
    return MODEL_PRESETS[name]  # type: ignore[index]


def validate_sumo_backend(backend: str) -> SumoBackend:
    """Verify ``backend`` is ``'libsumo'`` or ``'traci'`` and is installed on host (zero fallback)."""
    if backend not in ("libsumo", "traci"):
        raise ConfigValidationError(
            f"Invalid sumo_backend {backend!r}; expected 'libsumo' or 'traci'.",
            context={"sumo_backend": backend},
        )
    if importlib.util.find_spec(backend) is None:
        raise BackendUnavailableError(
            f"Requested sumo_backend={backend!r} is not installed in the active Python environment.",
            context={"sumo_backend": backend},
        )
    if importlib.util.find_spec("sumo") is None and not os.environ.get("SUMO_HOME"):
        raise BackendUnavailableError(
            f"Neither eclipse-sumo package nor SUMO_HOME is available for sumo_backend={backend!r}.",
            context={"sumo_backend": backend},
        )
    return backend  # type: ignore[return-value]


def validate_compute_device(
    device: str,
    precision: str = "fp32",
) -> tuple[ComputeDevice, PrecisionMode]:
    """Verify explicit PyTorch ``device`` and ``precision`` availability without silent fallback."""
    if device not in ("cpu", "mps", "cuda"):
        raise ConfigValidationError(
            f"Invalid compute device {device!r}; expected 'cpu', 'mps', or 'cuda'.",
            context={"device": device},
        )
    if precision not in ("fp32", "bf16"):
        raise ConfigValidationError(
            f"Invalid precision {precision!r}; expected 'fp32' or 'bf16'.",
            context={"precision": precision},
        )

    if device == "cuda" and not torch.cuda.is_available():
        raise BackendUnavailableError(
            "Requested compute device 'cuda', but torch.cuda.is_available() is False.",
            context={"device": device, "precision": precision},
        )
    if device == "mps" and not torch.backends.mps.is_available():
        raise BackendUnavailableError(
            "Requested compute device 'mps', but torch.backends.mps.is_available() is False.",
            context={"device": device, "precision": precision},
        )
    if precision == "bf16" and device == "cuda" and not torch.cuda.is_bf16_supported():
        raise BackendUnavailableError(
            "Requested precision 'bf16' on CUDA device that does not support bfloat16.",
            context={"device": device, "precision": precision},
        )

    return device, precision  # type: ignore[return-value]


# =============================================================================
# 3. Literature-Grounded Vehicle Classes & Config Dataclasses
# =============================================================================


@dataclass(frozen=True)
class VehicleTypeSpec:
    """Microscopic IDM + LC2013 vehicle class grounded in Treiber & Kesting (2013) / NGSIM."""

    vtype_id: str
    share: float
    length_m: float
    width_m: float
    min_gap_m: float
    tau_s: float
    accel_mps2: float
    decel_mps2: float
    emergency_decel_mps2: float
    max_speed_mps: float
    speed_dev: float
    car_follow_model: str = "IDM"
    lane_change_model: str = "LC2013"

    def __post_init__(self) -> None:
        _require_non_empty_str(self.vtype_id, "VehicleTypeSpec.vtype_id")
        share = _require_strict_float(self.share, f"VehicleTypeSpec[{self.vtype_id}].share")
        length_m = _require_strict_float(self.length_m, f"VehicleTypeSpec[{self.vtype_id}].length_m")
        width_m = _require_strict_float(self.width_m, f"VehicleTypeSpec[{self.vtype_id}].width_m")
        min_gap_m = _require_strict_float(self.min_gap_m, f"VehicleTypeSpec[{self.vtype_id}].min_gap_m")
        tau_s = _require_strict_float(self.tau_s, f"VehicleTypeSpec[{self.vtype_id}].tau_s")
        accel = _require_strict_float(self.accel_mps2, f"VehicleTypeSpec[{self.vtype_id}].accel_mps2")
        decel = _require_strict_float(self.decel_mps2, f"VehicleTypeSpec[{self.vtype_id}].decel_mps2")
        emerg = _require_strict_float(
            self.emergency_decel_mps2, f"VehicleTypeSpec[{self.vtype_id}].emergency_decel_mps2"
        )
        vmax = _require_strict_float(self.max_speed_mps, f"VehicleTypeSpec[{self.vtype_id}].max_speed_mps")
        sdev = _require_strict_float(self.speed_dev, f"VehicleTypeSpec[{self.vtype_id}].speed_dev")
        if share <= 0.0 or share > 1.0:
            raise ConfigValidationError(
                f"VehicleTypeSpec[{self.vtype_id}].share must be in (0, 1], got {share}.",
                context={"vtype_id": self.vtype_id, "share": share},
            )
        if min(length_m, width_m, min_gap_m, tau_s, accel, decel, emerg, vmax) <= 0.0 or sdev < 0.0 or emerg < decel:
            raise ConfigValidationError(
                f"Invalid physical parameters on VehicleTypeSpec[{self.vtype_id}].",
                context=asdict(self),
            )


DEFAULT_VEHICLE_TYPES: Final[tuple[VehicleTypeSpec, ...]] = (
    VehicleTypeSpec(
        vtype_id="passenger_urban",
        share=0.75,
        length_m=4.5,
        width_m=1.85,
        min_gap_m=2.0,
        tau_s=1.2,
        accel_mps2=2.0,
        decel_mps2=3.0,
        emergency_decel_mps2=8.0,
        max_speed_mps=15.0,
        speed_dev=0.08,
    ),
    VehicleTypeSpec(
        vtype_id="passenger_cautious",
        share=0.15,
        length_m=4.7,
        width_m=1.88,
        min_gap_m=2.5,
        tau_s=1.5,
        accel_mps2=1.6,
        decel_mps2=2.8,
        emergency_decel_mps2=8.0,
        max_speed_mps=13.9,
        speed_dev=0.06,
    ),
    VehicleTypeSpec(
        vtype_id="delivery_van",
        share=0.10,
        length_m=6.2,
        width_m=2.15,
        min_gap_m=2.5,
        tau_s=1.6,
        accel_mps2=1.3,
        decel_mps2=2.5,
        emergency_decel_mps2=7.0,
        max_speed_mps=12.5,
        speed_dev=0.05,
    ),
)


@dataclass(frozen=True)
class IncidentSpec:
    """Scheduled observable arterial lane-closure / speed-bottleneck event for OOD testing."""

    edge_ids: tuple[str, ...]
    start_time_s: float
    end_time_s: float
    closed_lane_indices: tuple[int, ...] = (0,)
    reduced_speed_mps: float | None = None

    def __post_init__(self) -> None:
        if not self.edge_ids:
            raise ConfigValidationError("IncidentSpec.edge_ids cannot be empty.")
        for eid in self.edge_ids:
            _require_non_empty_str(eid, "IncidentSpec.edge_ids[]")
        start = _require_strict_float(self.start_time_s, "IncidentSpec.start_time_s")
        end = _require_strict_float(self.end_time_s, "IncidentSpec.end_time_s")
        if start < 0.0 or end <= start:
            raise ConfigValidationError(
                f"IncidentSpec requires 0 <= start_time_s < end_time_s, got ({start}, {end}).",
                context={"start_time_s": start, "end_time_s": end},
            )
        for lidx in self.closed_lane_indices:
            if _require_strict_int(lidx, "IncidentSpec.closed_lane_indices[]") < 0:
                raise ConfigValidationError(
                    f"Negative lane index {lidx} in IncidentSpec.closed_lane_indices.",
                    context={"lane_index": lidx},
                )
        if self.reduced_speed_mps is not None:
            rs = _require_strict_float(self.reduced_speed_mps, "IncidentSpec.reduced_speed_mps")
            if rs <= 0.0:
                raise ConfigValidationError(
                    f"IncidentSpec.reduced_speed_mps must be > 0, got {rs}.",
                    context={"reduced_speed_mps": rs},
                )


@dataclass(frozen=True)
class ObservationConfig:
    """Partial-observability radar, turn-indicator, budgeted V2X, and Kalman filter parameters."""

    radar_range_m: float = 95.0
    pos_noise_std_m: float = 1.2
    speed_noise_std_mps: float = 0.45
    sensor_dropout_prob: float = 0.10
    turn_sensor_accuracy: float = 0.78
    v2x_pos_noise_std_m: float = 0.30
    v2x_speed_noise_std_mps: float = 0.12
    v2x_turn_accuracy: float = 0.97
    v2x_dropout_prob: float = 0.08
    v2x_max_delay_steps: int = 2
    v2x_budget_per_step: int = 8
    v2x_poll_cooldown_s: float = 1.0
    kalman_process_pos_var: float = 0.60
    kalman_process_vel_var: float = 1.10
    intent_forgetting_rate: float = 0.04

    def __post_init__(self) -> None:
        if _require_strict_float(self.radar_range_m, "ObservationConfig.radar_range_m") <= 0.0:
            raise ConfigValidationError("ObservationConfig.radar_range_m must be > 0.")
        for fname in (
            "pos_noise_std_m",
            "speed_noise_std_mps",
            "v2x_pos_noise_std_m",
            "v2x_speed_noise_std_mps",
            "kalman_process_pos_var",
            "kalman_process_vel_var",
        ):
            if _require_strict_float(getattr(self, fname), f"ObservationConfig.{fname}") <= 0.0:
                raise ConfigValidationError(f"ObservationConfig.{fname} must be > 0.")
        for pname in ("sensor_dropout_prob", "v2x_dropout_prob", "intent_forgetting_rate"):
            pval = _require_strict_float(getattr(self, pname), f"ObservationConfig.{pname}")
            if pval < 0.0 or pval >= 1.0:
                raise ConfigValidationError(f"ObservationConfig.{pname} must lie in [0, 1), got {pval}.")
        for acc_name in ("turn_sensor_accuracy", "v2x_turn_accuracy"):
            aval = _require_strict_float(getattr(self, acc_name), f"ObservationConfig.{acc_name}")
            if aval <= 0.0 or aval > 1.0:
                raise ConfigValidationError(f"ObservationConfig.{acc_name} must lie in (0, 1], got {aval}.")
        if _require_strict_int(self.v2x_max_delay_steps, "ObservationConfig.v2x_max_delay_steps") < 0:
            raise ConfigValidationError("ObservationConfig.v2x_max_delay_steps must be >= 0.")
        if _require_strict_int(self.v2x_budget_per_step, "ObservationConfig.v2x_budget_per_step") < 0:
            raise ConfigValidationError("ObservationConfig.v2x_budget_per_step must be >= 0.")
        if _require_strict_float(self.v2x_poll_cooldown_s, "ObservationConfig.v2x_poll_cooldown_s") < 0.0:
            raise ConfigValidationError("ObservationConfig.v2x_poll_cooldown_s must be >= 0.")


@dataclass(frozen=True)
class DBNConfig:
    """Hyperparameters for candidate graph construction, ``STGraphTransformerDBN``, LBP, and BIC."""

    model_preset: ModelPresetName = "sota_local"
    prediction_horizon_s: float = 6.0
    delay_threshold_s: float = 1.5
    max_candidates_per_veh: int = 20
    max_parents_k: int = 6
    min_edge_prob: float = 0.05
    prior_stay_prob: float = 0.80
    prior_base_prob: float = 0.10
    lbp_sweeps: int = 3
    lbp_damping: float = 0.50
    bic_window_steps: int = 60
    bic_min_pair_samples: int = 8
    bic_penalty_weight: float = 0.50
    inference_device: ComputeDevice = "cpu"
    require_calibrated: bool = True

    def __post_init__(self) -> None:
        get_model_preset(self.model_preset)
        if self.inference_device not in ("cpu", "mps", "cuda"):
            raise ConfigValidationError(
                f"Invalid DBNConfig.inference_device={self.inference_device!r}.",
                context={"inference_device": self.inference_device},
            )
        _require_strict_bool(self.require_calibrated, "DBNConfig.require_calibrated")
        if _require_strict_float(self.prediction_horizon_s, "DBNConfig.prediction_horizon_s") <= 0.0:
            raise ConfigValidationError("DBNConfig.prediction_horizon_s must be > 0.")
        if _require_strict_float(self.delay_threshold_s, "DBNConfig.delay_threshold_s") <= 0.0:
            raise ConfigValidationError("DBNConfig.delay_threshold_s must be > 0.")
        mc = _require_strict_int(self.max_candidates_per_veh, "DBNConfig.max_candidates_per_veh")
        mk = _require_strict_int(self.max_parents_k, "DBNConfig.max_parents_k")
        if mc < 1 or mk < 1 or mk > mc:
            raise ConfigValidationError(
                f"Require 1 <= max_parents_k ({mk}) <= max_candidates_per_veh ({mc}).",
                context={"max_parents_k": mk, "max_candidates_per_veh": mc},
            )
        for pname in ("min_edge_prob", "prior_stay_prob", "prior_base_prob", "lbp_damping"):
            pv = _require_strict_float(getattr(self, pname), f"DBNConfig.{pname}")
            if pv < 0.0 or pv >= 1.0:
                raise ConfigValidationError(f"DBNConfig.{pname} must lie in [0, 1), got {pv}.")
        if _require_strict_int(self.lbp_sweeps, "DBNConfig.lbp_sweeps") < 1:
            raise ConfigValidationError("DBNConfig.lbp_sweeps must be >= 1.")
        bw = _require_strict_int(self.bic_window_steps, "DBNConfig.bic_window_steps")
        bm = _require_strict_int(self.bic_min_pair_samples, "DBNConfig.bic_min_pair_samples")
        if bm < 4 or bw < bm:
            raise ConfigValidationError(
                f"Require 4 <= bic_min_pair_samples ({bm}) <= bic_window_steps ({bw})."
            )
        if _require_strict_float(self.bic_penalty_weight, "DBNConfig.bic_penalty_weight") <= 0.0:
            raise ConfigValidationError("DBNConfig.bic_penalty_weight must be > 0.")


@dataclass(frozen=True)
class ControlConfig:
    """Factored Damped Max-Sum, spillback backpressure, Eco-Glide, routing, and safety parameters."""

    maxsum_iterations: int = 5
    maxsum_damping: float = 0.50
    wait_weight: float = 0.08
    cascade_weight: float = 0.65
    spillback_threshold: float = 0.55
    spillback_penalty: float = 6.0
    conflict_penalty: float = 8.0
    platoon_harmony_weight: float = 1.2
    conflict_risk_threshold: float = 0.22
    tls_min_green_hold_s: float = 2.5
    tls_yellow_clearance_s: float = 3.0
    reroute_interval_s: float = 5.0
    reroute_min_stopline_dist_m: float = 35.0
    reroute_hysteresis_s: float = 3.0
    reroute_spill_weight: float = 18.0
    reroute_flux_weight: float = 8.0
    min_glide_speed_mps: float = 2.2
    comfort_accel_mps2: float = 2.0
    comfort_decel_mps2: float = 2.2
    max_decel_mps2: float = 6.5
    idm_time_headway_s: float = 1.3
    idm_min_gap_m: float = 2.0
    uncertainty_safety_beta: float = 1.5
    zero_speed_deadzone_mps: float = 0.15

    def __post_init__(self) -> None:
        if _require_strict_int(self.maxsum_iterations, "ControlConfig.maxsum_iterations") < 1:
            raise ConfigValidationError("ControlConfig.maxsum_iterations must be >= 1.")
        damp = _require_strict_float(self.maxsum_damping, "ControlConfig.maxsum_damping")
        if damp < 0.0 or damp >= 1.0:
            raise ConfigValidationError(f"ControlConfig.maxsum_damping must lie in [0, 1), got {damp}.")
        for prob_field in ("spillback_threshold", "conflict_risk_threshold"):
            pv = _require_strict_float(getattr(self, prob_field), f"ControlConfig.{prob_field}")
            if pv <= 0.0 or pv >= 1.0:
                raise ConfigValidationError(f"ControlConfig.{prob_field} must lie in (0, 1), got {pv}.")
        for nonneg_field in (
            "wait_weight",
            "cascade_weight",
            "spillback_penalty",
            "conflict_penalty",
            "platoon_harmony_weight",
            "tls_min_green_hold_s",
            "reroute_hysteresis_s",
            "reroute_spill_weight",
            "reroute_flux_weight",
            "uncertainty_safety_beta",
            "zero_speed_deadzone_mps",
        ):
            if _require_strict_float(getattr(self, nonneg_field), f"ControlConfig.{nonneg_field}") < 0.0:
                raise ConfigValidationError(f"ControlConfig.{nonneg_field} must be >= 0.")
        for pos_field in (
            "tls_yellow_clearance_s",
            "reroute_interval_s",
            "reroute_min_stopline_dist_m",
            "min_glide_speed_mps",
            "comfort_accel_mps2",
            "comfort_decel_mps2",
            "max_decel_mps2",
            "idm_time_headway_s",
            "idm_min_gap_m",
        ):
            if _require_strict_float(getattr(self, pos_field), f"ControlConfig.{pos_field}") <= 0.0:
                raise ConfigValidationError(f"ControlConfig.{pos_field} must be > 0.")
        if self.max_decel_mps2 < self.comfort_decel_mps2:
            raise ConfigValidationError("ControlConfig.max_decel_mps2 must be >= comfort_decel_mps2.")


@dataclass(frozen=True)
class TrainConfig:
    """End-to-end unrolled trajectory BPTT & post-hoc L-BFGS calibration configuration."""

    model_preset: ModelPresetName = "sota_local"
    device: ComputeDevice = "cpu"
    precision: PrecisionMode = "fp32"
    compile_model: bool = False
    distributed: bool = False
    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_fraction: float = 0.10
    grad_clip_norm: float = 1.0
    focal_gamma: float = 2.0
    loss_w_post_delay: float = 0.50
    loss_w_edge: float = 1.20
    loss_w_rank: float = 0.40
    loss_w_tau: float = 0.50
    loss_w_spillback: float = 1.00
    loss_w_smooth: float = 0.15
    loss_w_l1: float = 0.02
    dag_temp_start: float = 1.0
    dag_temp_end: float = 0.15
    early_stopping_patience: int = 8
    num_rollout_workers: int = 4
    seed: int = 2026

    def __post_init__(self) -> None:
        get_model_preset(self.model_preset)
        if self.device not in ("cpu", "mps", "cuda"):
            raise ConfigValidationError(f"Invalid TrainConfig.device={self.device!r}.")
        if self.precision not in ("fp32", "bf16"):
            raise ConfigValidationError(f"Invalid TrainConfig.precision={self.precision!r}.")
        _require_strict_bool(self.compile_model, "TrainConfig.compile_model")
        _require_strict_bool(self.distributed, "TrainConfig.distributed")
        if _require_strict_int(self.epochs, "TrainConfig.epochs") < 1:
            raise ConfigValidationError("TrainConfig.epochs must be >= 1.")
        if _require_strict_int(self.batch_size, "TrainConfig.batch_size") < 1:
            raise ConfigValidationError("TrainConfig.batch_size must be >= 1.")
        if _require_strict_float(self.learning_rate, "TrainConfig.learning_rate") <= 0.0:
            raise ConfigValidationError("TrainConfig.learning_rate must be > 0.")
        if _require_strict_float(self.weight_decay, "TrainConfig.weight_decay") < 0.0:
            raise ConfigValidationError("TrainConfig.weight_decay must be >= 0.")
        wf = _require_strict_float(self.warmup_fraction, "TrainConfig.warmup_fraction")
        if wf < 0.0 or wf >= 1.0:
            raise ConfigValidationError(f"TrainConfig.warmup_fraction must lie in [0, 1), got {wf}.")
        if _require_strict_float(self.grad_clip_norm, "TrainConfig.grad_clip_norm") <= 0.0:
            raise ConfigValidationError("TrainConfig.grad_clip_norm must be > 0.")
        if _require_strict_float(self.focal_gamma, "TrainConfig.focal_gamma") < 0.0:
            raise ConfigValidationError("TrainConfig.focal_gamma must be >= 0.")
        for wname in (
            "loss_w_post_delay",
            "loss_w_edge",
            "loss_w_rank",
            "loss_w_tau",
            "loss_w_spillback",
            "loss_w_smooth",
            "loss_w_l1",
        ):
            if _require_strict_float(getattr(self, wname), f"TrainConfig.{wname}") < 0.0:
                raise ConfigValidationError(f"TrainConfig.{wname} must be >= 0.")
        ts = _require_strict_float(self.dag_temp_start, "TrainConfig.dag_temp_start")
        te = _require_strict_float(self.dag_temp_end, "TrainConfig.dag_temp_end")
        if te <= 0.0 or ts < te:
            raise ConfigValidationError(f"Require 0 < dag_temp_end ({te}) <= dag_temp_start ({ts}).")
        if _require_strict_int(self.early_stopping_patience, "TrainConfig.early_stopping_patience") < 1:
            raise ConfigValidationError("TrainConfig.early_stopping_patience must be >= 1.")
        if _require_strict_int(self.num_rollout_workers, "TrainConfig.num_rollout_workers") < 1:
            raise ConfigValidationError("TrainConfig.num_rollout_workers must be >= 1.")
        if _require_strict_int(self.seed, "TrainConfig.seed") < 0:
            raise ConfigValidationError("TrainConfig.seed must be >= 0.")


@dataclass(frozen=True)
class ScenarioConfig:
    """Complete SUMO urban network, HCM demand regime, warmup/cohort/drain, and noise spec."""

    scenario_id: str
    network_id: str
    net_xml_path: str
    poly_xml_path: str | None = None
    sumo_backend: SumoBackend = "libsumo"
    step_length_s: float = 0.5
    warmup_duration_s: float = 60.0
    cohort_duration_s: float = 240.0
    drain_duration_s: float = 180.0
    demand_rate_veh_per_s: float = 1.2
    asymmetric_rush_wave: bool = False
    asymmetric_wave_fraction: float = 0.55
    min_od_euclidean_dist_m: float = 200.0
    is_ood_regime: bool = False
    incidents: tuple[IncidentSpec, ...] = ()
    vehicle_types: tuple[VehicleTypeSpec, ...] = DEFAULT_VEHICLE_TYPES
    observation: ObservationConfig = ObservationConfig()

    def __post_init__(self) -> None:
        _require_non_empty_str(self.scenario_id, "ScenarioConfig.scenario_id")
        _require_non_empty_str(self.network_id, "ScenarioConfig.network_id")
        _require_non_empty_str(self.net_xml_path, "ScenarioConfig.net_xml_path")
        if self.poly_xml_path is not None:
            _require_non_empty_str(self.poly_xml_path, "ScenarioConfig.poly_xml_path")
        if self.sumo_backend not in ("libsumo", "traci"):
            raise ConfigValidationError(
                f"Invalid ScenarioConfig.sumo_backend={self.sumo_backend!r}; expected 'libsumo' or 'traci'."
            )
        dt = _require_strict_float(self.step_length_s, "ScenarioConfig.step_length_s")
        warmup = _require_strict_float(self.warmup_duration_s, "ScenarioConfig.warmup_duration_s")
        cohort = _require_strict_float(self.cohort_duration_s, "ScenarioConfig.cohort_duration_s")
        drain = _require_strict_float(self.drain_duration_s, "ScenarioConfig.drain_duration_s")
        rate = _require_strict_float(self.demand_rate_veh_per_s, "ScenarioConfig.demand_rate_veh_per_s")
        if dt <= 0.0 or cohort <= 0.0 or warmup < 0.0 or drain < 0.0 or rate <= 0.0:
            raise ConfigValidationError("Invalid timing or demand parameters in ScenarioConfig.", context=asdict(self))
        _require_strict_bool(self.asymmetric_rush_wave, "ScenarioConfig.asymmetric_rush_wave")
        _require_strict_bool(self.is_ood_regime, "ScenarioConfig.is_ood_regime")
        awf = _require_strict_float(self.asymmetric_wave_fraction, "ScenarioConfig.asymmetric_wave_fraction")
        if awf < 0.0 or awf > 1.0:
            raise ConfigValidationError("ScenarioConfig.asymmetric_wave_fraction must lie in [0, 1].")
        if _require_strict_float(self.min_od_euclidean_dist_m, "ScenarioConfig.min_od_euclidean_dist_m") < 0.0:
            raise ConfigValidationError("ScenarioConfig.min_od_euclidean_dist_m must be >= 0.")
        if not self.vehicle_types:
            raise ConfigValidationError("ScenarioConfig.vehicle_types cannot be empty.")
        share_sum = sum(vt.share for vt in self.vehicle_types)
        if abs(share_sum - 1.0) > 1e-4:
            raise ConfigValidationError(
                f"ScenarioConfig.vehicle_types shares must sum to 1.0, got {share_sum}.",
                context={"share_sum": share_sum},
            )

    @property
    def cohort_end_s(self) -> float:
        """Simulation timestamp marking the end of scheduled cohort injections."""
        return self.warmup_duration_s + self.cohort_duration_s

    @property
    def total_duration_s(self) -> float:
        """Total simulated episode horizon including warmup, measured cohort, and drain."""
        return self.warmup_duration_s + self.cohort_duration_s + self.drain_duration_s

    @property
    def total_steps(self) -> int:
        """Exact integer number of simulation steps over ``total_duration_s``."""
        return int(round(self.total_duration_s / self.step_length_s))

    def resolve_paths(
        self,
        repo_root: Path | str | None = None,
        *,
        must_exist: bool = True,
    ) -> tuple[Path, Path | None]:
        """Resolve ``net_xml_path`` and ``poly_xml_path``, failing loudly if ``must_exist`` and missing."""
        base = Path(repo_root) if repo_root is not None else Path.cwd()
        net_p = Path(self.net_xml_path)
        if not net_p.is_absolute():
            net_p = (base / net_p).resolve()
        poly_p: Path | None = None
        if self.poly_xml_path is not None:
            poly_p = Path(self.poly_xml_path)
            if not poly_p.is_absolute():
                poly_p = (base / poly_p).resolve()

        if must_exist:
            if not net_p.is_file():
                raise ConfigValidationError(
                    f"Scenario {self.scenario_id!r} net_xml_path does not exist: {net_p}",
                    context={"scenario_id": self.scenario_id, "net_xml_path": str(net_p)},
                )
            if poly_p is not None and not poly_p.is_file():
                raise ConfigValidationError(
                    f"Scenario {self.scenario_id!r} poly_xml_path does not exist: {poly_p}",
                    context={"scenario_id": self.scenario_id, "poly_xml_path": str(poly_p)},
                )
        return net_p, poly_p


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration binding scenario, DBN, control, training, and splits."""

    scenario: ScenarioConfig
    experiment_id: str = "cs238_city_swarm"
    dbn: DBNConfig = DBNConfig()
    control: ControlConfig = ControlConfig()
    train: TrainConfig = TrainConfig()
    train_seeds: tuple[int, ...] = TRAIN_SEEDS
    val_seeds: tuple[int, ...] = VAL_SEEDS
    test_seeds: tuple[int, ...] = TEST_SEEDS
    max_parallel_workers: int = 8
    bootstrap_resamples: int = 10000

    def __post_init__(self) -> None:
        _require_non_empty_str(self.experiment_id, "ExperimentConfig.experiment_id")
        validate_seed_splits(self.train_seeds, self.val_seeds, self.test_seeds)
        if _require_strict_int(self.max_parallel_workers, "ExperimentConfig.max_parallel_workers") < 1:
            raise ConfigValidationError("ExperimentConfig.max_parallel_workers must be >= 1.")
        if _require_strict_int(self.bootstrap_resamples, "ExperimentConfig.bootstrap_resamples") < 100:
            raise ConfigValidationError("ExperimentConfig.bootstrap_resamples must be >= 100.")


# =============================================================================
# 4. Strict Fail-Loud Dataclass Deserializer, Serializer, & SHA-256 Provenance
# =============================================================================


def _parse_typed_value(target_type: Any, raw_val: Any, field_path: str) -> Any:
    """Strictly validate and construct ``raw_val`` according to ``target_type`` without coercion."""
    origin = get_origin(target_type)

    if origin in (Union, types.UnionType):
        args = get_args(target_type)
        if raw_val is None:
            if type(None) in args:
                return None
            raise ConfigValidationError(
                f"Field {field_path!r} cannot be null.",
                context={"field": field_path},
            )
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _parse_typed_value(non_none[0], raw_val, field_path)

    if origin is Literal:
        allowed = get_args(target_type)
        if raw_val not in allowed or (
            isinstance(raw_val, bool) and not any(isinstance(a, bool) for a in allowed)
        ):
            raise ConfigValidationError(
                f"Invalid value {raw_val!r} for {field_path!r}; allowed={list(allowed)}.",
                context={"field": field_path, "value": repr(raw_val), "allowed": list(allowed)},
            )
        return raw_val

    if origin is tuple:
        if not isinstance(raw_val, (list, tuple)):
            raise ConfigValidationError(
                f"Field {field_path!r} expects a JSON array/tuple, got {type(raw_val).__name__}.",
                context={"field": field_path, "type": type(raw_val).__name__},
            )
        t_args = get_args(target_type)
        if len(t_args) == 2 and t_args[1] is Ellipsis:
            elem_type = t_args[0]
            return tuple(_parse_typed_value(elem_type, item, f"{field_path}[{i}]") for i, item in enumerate(raw_val))
        if len(raw_val) != len(t_args):
            raise ConfigValidationError(
                f"Field {field_path!r} expects tuple of length {len(t_args)}, got {len(raw_val)}.",
                context={"field": field_path, "expected_len": len(t_args), "actual_len": len(raw_val)},
            )
        return tuple(
            _parse_typed_value(elem_t, item, f"{field_path}[{i}]")
            for i, (elem_t, item) in enumerate(zip(t_args, raw_val, strict=True))
        )

    if isinstance(target_type, type) and is_dataclass(target_type):
        return dataclass_from_strict_dict(target_type, raw_val, path_prefix=field_path)

    if target_type is bool:
        return _require_strict_bool(raw_val, field_path)
    if target_type is int:
        return _require_strict_int(raw_val, field_path)
    if target_type is float:
        return _require_strict_float(raw_val, field_path)
    if target_type is str:
        if not isinstance(raw_val, str):
            raise ConfigValidationError(
                f"Field {field_path!r} requires str, got {type(raw_val).__name__}.",
                context={"field": field_path, "type": type(raw_val).__name__},
            )
        return raw_val

    raise ConfigValidationError(
        f"Unsupported config schema annotation {target_type!r} at {field_path!r}.",
        context={"field": field_path, "annotation": repr(target_type)},
    )


def dataclass_from_strict_dict(
    cls: type[T],
    data: Any,
    *,
    path_prefix: str | None = None,
) -> T:
    """Deserialize ``cls`` from ``data`` rejecting unknown keys, missing keys, and type coercions."""
    prefix = path_prefix or cls.__name__
    if not is_dataclass(cls):
        raise ConfigValidationError(f"{cls!r} is not a dataclass.")
    if not isinstance(data, Mapping):
        raise ConfigValidationError(
            f"Expected JSON object (dict) for {prefix!r}, got {type(data).__name__}.",
            context={"field": prefix, "type": type(data).__name__},
        )

    cls_fields = {f.name: f for f in fields(cls)}
    unknown_keys = sorted(set(data.keys()) - set(cls_fields.keys()))
    if unknown_keys:
        raise ConfigValidationError(
            f"Unknown configuration key(s) in {prefix}: {unknown_keys}.",
            context={"object": prefix, "unknown_keys": unknown_keys, "allowed_keys": sorted(cls_fields)},
        )

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    missing_required: list[str] = []

    for fname, fobj in cls_fields.items():
        field_path = f"{prefix}.{fname}"
        if fname in data:
            kwargs[fname] = _parse_typed_value(hints[fname], data[fname], field_path)
        elif fobj.default is MISSING and fobj.default_factory is MISSING:
            missing_required.append(fname)

    if missing_required:
        raise ConfigValidationError(
            f"Missing required configuration field(s) in {prefix}: {missing_required}.",
            context={"object": prefix, "missing_fields": missing_required},
        )

    return cls(**kwargs)


def _read_strict_json(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise ConfigValidationError(
            f"Configuration JSON file does not exist: {target}",
            context={"path": str(target)},
        )
    text = target.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        get_logger("config").error("Malformed JSON in %s: %s", target, exc)
        raise ConfigValidationError(
            f"Malformed JSON in {target}: {exc}",
            context={"path": str(target), "lineno": exc.lineno, "colno": exc.colno},
        ) from exc
    if not isinstance(parsed, dict):
        raise ConfigValidationError(
            f"Top-level JSON in {target} must be an object, got {type(parsed).__name__}.",
            context={"path": str(target), "type": type(parsed).__name__},
        )
    return parsed


def load_scenario_config(
    path: Path | str,
    *,
    must_exist_network_files: bool = True,
    repo_root: Path | str | None = None,
) -> ScenarioConfig:
    """Load and strictly validate a :class:`ScenarioConfig` JSON file."""
    raw = _read_strict_json(path)
    cfg = dataclass_from_strict_dict(ScenarioConfig, raw)
    if must_exist_network_files:
        cfg.resolve_paths(repo_root=repo_root, must_exist=True)
    return cfg


def load_experiment_config(
    path: Path | str,
    *,
    must_exist_network_files: bool = True,
    repo_root: Path | str | None = None,
) -> ExperimentConfig:
    """Load and strictly validate an :class:`ExperimentConfig` JSON file."""
    raw = _read_strict_json(path)
    cfg = dataclass_from_strict_dict(ExperimentConfig, raw)
    if must_exist_network_files:
        cfg.scenario.resolve_paths(repo_root=repo_root, must_exist=True)
    return cfg


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Convert any config dataclass into a deterministic JSON-serializable dict."""
    if not is_dataclass(cfg) or isinstance(cfg, type):
        raise ConfigValidationError(f"Expected config dataclass instance, got {type(cfg).__name__}.")
    return asdict(cfg)


def save_config_json(cfg: Any, path: Path | str) -> Path:
    """Atomically write ``cfg`` as sorted, indented JSON to ``path``."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.parent / f".{out_path.name}.pid{os.getpid()}.tmp"
    payload = config_to_dict(cfg)
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, out_path)
    return out_path


def config_sha256(cfg: Any) -> str:
    """Return canonical SHA-256 hex digest of a configuration dataclass."""
    canonical = json.dumps(config_to_dict(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def file_sha256(path: Path | str) -> str:
    """Return SHA-256 hex digest of an existing file, failing loudly if missing."""
    target = Path(path)
    if not target.is_file():
        raise ConfigValidationError(
            f"Cannot compute SHA-256 of missing file: {target}",
            context={"path": str(target)},
        )
    hasher = hashlib.sha256()
    with target.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


__all__ = [
    "ControlConfig",
    "DBNConfig",
    "DEFAULT_VEHICLE_TYPES",
    "ExperimentConfig",
    "IncidentSpec",
    "MODEL_PRESETS",
    "ModelPresetSpec",
    "ObservationConfig",
    "ScenarioConfig",
    "TEST_SEEDS",
    "TRAIN_SEEDS",
    "TrainConfig",
    "VAL_SEEDS",
    "VehicleTypeSpec",
    "config_sha256",
    "config_to_dict",
    "dataclass_from_strict_dict",
    "file_sha256",
    "get_model_preset",
    "load_experiment_config",
    "load_scenario_config",
    "save_config_json",
    "validate_compute_device",
    "validate_seed_splits",
    "validate_sumo_backend",
]

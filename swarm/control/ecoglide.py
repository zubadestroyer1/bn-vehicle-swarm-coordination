"""Smooth Eco-Glide longitudinal speed profiler and comfortable deceleration trajectory generator.

Maps discrete :class:`~swarm.contracts.LongitudinalMode` assignments and :class:`ScheduleResult`
crossing reservations into continuous, physically smooth desired speeds ``[m/s]`` so vehicles
pace cleanly into conflict-free slots without hard stop-and-go shockwaves.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from ..config import ControlConfig
from ..contracts import LongitudinalMode, RoadTopology, SwarmBeliefState
from ..logging_utils import (
    NumericalInvariantError,
    TopologyIntegrityError,
    require_finite,
    require_psd_covariance,
)
from .scheduler import ScheduleResult


@dataclass(frozen=True)
class EcoGlideProfile:
    """Continuous desired longitudinal speeds ``[m/s]`` and realized coordination modes."""

    desired_speeds: dict[str, float]
    effective_modes: dict[str, LongitudinalMode]


def compute_ecoglide_speeds(
    belief: SwarmBeliefState,
    topology: RoadTopology,
    schedule: ScheduleResult,
    control_config: ControlConfig,
    base_modes: Mapping[str, LongitudinalMode] | None = None,
) -> EcoGlideProfile:
    """Compute smooth non-stopping glide or comfortable braking speeds for all tracked vehicles."""
    require_finite(belief.dt, "SwarmBeliefState.dt")
    dt = float(belief.dt)
    if dt <= 0.0:
        raise NumericalInvariantError(
            f"SwarmBeliefState.dt must be positive in Eco-Glide, got {dt}.",
            context={"dt": dt},
        )

    cfg = control_config
    a_comf = cfg.comfort_accel_mps2
    b_comf = cfg.comfort_decel_mps2
    v_dead = cfg.zero_speed_deadzone_mps

    desired_speeds: dict[str, float] = {}
    effective_modes: dict[str, LongitudinalMode] = {}

    for vid in sorted(belief.vehicles):
        veh = belief.vehicles[vid]
        corr = topology.corridors.get(veh.corridor_id)
        if corr is None:
            raise TopologyIntegrityError(
                f"Vehicle {vid!r} references unknown corridor {veh.corridor_id!r} in Eco-Glide.",
                context={"veh_id": vid, "corridor_id": veh.corridor_id},
            )
        require_finite(veh.mean_s_m, f"VehicleBelief[{vid}].mean_s_m")
        require_finite(veh.mean_v_mps, f"VehicleBelief[{vid}].mean_v_mps")
        require_psd_covariance(veh.cov_2x2, f"VehicleBelief[{vid}].cov_2x2")

        v_max = corr.speed_limit_mps
        v_min_glide = min(v_max, cfg.min_glide_speed_mps)
        v_cur = min(v_max, max(0.0, float(veh.mean_v_mps)))
        pos_std = math.sqrt(max(0.0, float(veh.cov_2x2[0, 0])))

        if veh.in_junction_id is not None:
            desired_speeds[vid] = v_max
            effective_modes[vid] = LongitudinalMode.GO_WAVE
            continue

        d_stop = max(0.0, corr.length_m - float(veh.mean_s_m) - 0.8 - 0.5 * pos_std)
        v_comf_stop = min(v_max, math.sqrt(2.0 * b_comf * d_stop))
        m_base = base_modes.get(vid, LongitudinalMode.CRUISE) if base_modes is not None else LongitudinalMode.CRUISE

        if vid in schedule.must_hold_stopline:
            if v_comf_stop < v_dead:
                v_des = 0.0
                eff_mode = LongitudinalMode.YIELD
            elif d_stop > 18.0 and v_comf_stop >= v_min_glide and m_base == LongitudinalMode.ECO_GLIDE:
                v_des = v_comf_stop
                eff_mode = LongitudinalMode.ECO_GLIDE
            else:
                v_des = v_comf_stop
                eff_mode = LongitudinalMode.YIELD
        elif vid in schedule.granted_vehicles:
            t_slot = schedule.scheduled_start_s.get(vid, float(veh.free_eta_s))
            if m_base == LongitudinalMode.GO_WAVE:
                v_des = v_max
                eff_mode = LongitudinalMode.GO_WAVE
            elif m_base == LongitudinalMode.ECO_GLIDE:
                if math.isfinite(t_slot) and t_slot > float(veh.free_eta_s) + 0.30:
                    v_pace = d_stop / max(t_slot, 0.5)
                    v_des = min(v_max, max(v_min_glide, v_pace))
                else:
                    v_des = min(v_max, max(v_min_glide, 0.78 * v_max))
                eff_mode = LongitudinalMode.ECO_GLIDE
            else:
                v_des = min(v_max, max(v_cur + a_comf * dt, 0.92 * v_max))
                eff_mode = LongitudinalMode.CRUISE
        else:
            # Deferred slot reservation: smooth Eco-Glide arrival pacing without full stop where feasible.
            t_slot = schedule.scheduled_start_s.get(vid, math.inf)
            if math.isfinite(t_slot) and t_slot > 0.40:
                v_kin_glide = 2.0 * d_stop / t_slot - v_cur
                v_avg_glide = d_stop / t_slot
                v_cand = v_kin_glide if v_kin_glide >= v_min_glide else v_avg_glide
                if v_cand >= v_min_glide and v_comf_stop >= v_min_glide:
                    v_des = min(v_max, v_comf_stop, max(v_min_glide, v_cand))
                    eff_mode = LongitudinalMode.ECO_GLIDE
                else:
                    v_des = min(v_min_glide, v_comf_stop)
                    if v_des < v_dead:
                        v_des = 0.0
                    eff_mode = LongitudinalMode.ECO_GLIDE if v_des >= v_min_glide else LongitudinalMode.YIELD
            else:
                v_des = 0.0 if v_comf_stop < v_dead else v_comf_stop
                eff_mode = LongitudinalMode.YIELD

        require_finite(v_des, f"EcoGlideProfile.desired_speeds[{vid}]")
        desired_speeds[vid] = v_des
        effective_modes[vid] = eff_mode

    return EcoGlideProfile(
        desired_speeds=desired_speeds,
        effective_modes=effective_modes,
    )


__all__ = [
    "EcoGlideProfile",
    "compute_ecoglide_speeds",
]

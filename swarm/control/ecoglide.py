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
    b_comf = cfg.comfort_decel_mps2
    v_dead = cfg.zero_speed_deadzone_mps

    lane_queues: dict[tuple[str, int], list[str]] = {}
    for q_vid, q_veh in belief.vehicles.items():
        if q_veh.in_junction_id is None:
            lane_queues.setdefault((q_veh.corridor_id, q_veh.lane_index), []).append(q_vid)
    same_lane_lead: dict[str, str] = {}
    for q_vids in lane_queues.values():
        if len(q_vids) > 1:
            ordered_q = sorted(q_vids, key=lambda item: (float(belief.vehicles[item].mean_s_m), item))
            for q_i in range(len(ordered_q) - 1):
                same_lane_lead[ordered_q[q_i]] = ordered_q[q_i + 1]

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

        lead_id = same_lane_lead.get(vid)
        if lead_id is not None and vid in schedule.must_hold_stopline and v_cur <= 0.75:
            lead_v_bel = belief.vehicles[lead_id]
            est_gap = (
                float(lead_v_bel.mean_s_m)
                - float(veh.mean_s_m)
                - 0.5 * (float(lead_v_bel.length_m) + float(veh.length_m))
            )
            if float(lead_v_bel.mean_v_mps) <= 0.75 and est_gap <= cfg.idm_min_gap_m + 3.50:
                desired_speeds[vid] = 0.0
                effective_modes[vid] = LongitudinalMode.YIELD
                continue

        d_stop = max(0.0, corr.length_m - float(veh.mean_s_m) - 0.6 - cfg.uncertainty_safety_beta * pos_std)
        v_comf_stop = min(v_max, math.sqrt(2.0 * b_comf * d_stop))
        m_base = base_modes.get(vid, LongitudinalMode.CRUISE) if base_modes is not None else LongitudinalMode.CRUISE
        to_junc = topology.junctions.get(corr.to_junction)
        is_signalized_approach = to_junc is not None and to_junc.tls_id is not None and to_junc.num_tls_links > 0

        if vid in schedule.must_hold_stopline:
            raw_dist_to_line = corr.length_m - float(veh.mean_s_m)
            if v_comf_stop < v_dead or d_stop <= 1.25 or (v_cur <= 0.85 and raw_dist_to_line <= 5.5):
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
                if (
                    not is_signalized_approach
                    and math.isfinite(t_slot)
                    and t_slot > float(veh.free_eta_s) + 0.30
                ):
                    v_pace = d_stop / max(t_slot, 0.5)
                    v_des = min(v_max, max(v_min_glide, v_pace))
                else:
                    v_des = min(v_max, max(v_min_glide, 0.96 * v_max))
                eff_mode = LongitudinalMode.ECO_GLIDE
            else:
                v_des = v_max
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
                if is_signalized_approach and d_stop > 35.0 and v_des >= 0.90 * v_max:
                    eff_mode = (
                        LongitudinalMode.CRUISE
                        if m_base == LongitudinalMode.YIELD
                        else m_base
                    )
                else:
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

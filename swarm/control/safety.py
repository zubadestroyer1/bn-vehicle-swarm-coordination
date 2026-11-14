"""Non-cheating observable longitudinal safety filter with uncertainty-inflated IDM and anti-creep deadzone.

Enforces longitudinal collision-avoidance, stop-line holding, physical acceleration limits
``[-max_decel_mps2, comfort_accel_mps2]``, and a zero-speed deadzone (``v < 0.15 m/s => 0``) using
STRICTLY filtered :class:`~swarm.contracts.SwarmBeliefState` and static :class:`~swarm.contracts.RoadTopology`
data (closing prototype defect D6 with zero privileged simulator ground-truth access).
"""

from __future__ import annotations

from collections.abc import Mapping, Set as AbstractSet
from dataclasses import dataclass
import math

from ..config import ControlConfig
from ..contracts import RoadTopology, SwarmBeliefState, VehicleBelief
from ..logging_utils import (
    NumericalInvariantError,
    TopologyIntegrityError,
    require_finite,
    require_psd_covariance,
)


@dataclass(frozen=True)
class SafetyFilterResult:
    """Post-safety speed targets, realized acceleration commands, and intervention counters."""

    target_speeds: dict[str, float]
    post_safety_accels: dict[str, float]
    safety_clamp_count: int
    zero_speed_deadzone_count: int


class ObservableSafetyFilter:
    """Physically bounded observable IDM + stop-line safety envelope over :class:`SwarmBeliefState`."""

    def __init__(self, control_config: ControlConfig) -> None:
        self._cfg = control_config

    @property
    def config(self) -> ControlConfig:
        """Return the immutable :class:`ControlConfig` governing this safety filter."""
        return self._cfg

    def apply(
        self,
        belief: SwarmBeliefState,
        topology: RoadTopology,
        desired_speeds: Mapping[str, float],
        must_hold_stopline: AbstractSet[str] = frozenset(),
    ) -> SafetyFilterResult:
        """Filter ``desired_speeds`` into physically safe speeds and post-safety accelerations.

        Parameters
        ----------
        belief:
            Current city-wide filtered belief state ``b_t``.
        topology:
            Static road and corridor geometry.
        desired_speeds:
            Planner/Eco-Glide desired longitudinal speeds ``[m/s]`` keyed by ``veh_id``.
        must_hold_stopline:
            Set of ``veh_id``s that are NOT granted immediate junction entry and must remain
            safely behind their corridor stop-line.
        """
        require_finite(belief.sim_time, "SwarmBeliefState.sim_time")
        require_finite(belief.dt, "SwarmBeliefState.dt")
        dt = float(belief.dt)
        if dt <= 0.0:
            raise NumericalInvariantError(
                f"SwarmBeliefState.dt must be positive, got {dt}.",
                context={"dt": dt, "sim_time": belief.sim_time},
            )

        cfg = self._cfg
        a_comf = cfg.comfort_accel_mps2
        b_comf = cfg.comfort_decel_mps2
        b_max = cfg.max_decel_mps2
        t_hw = cfg.idm_time_headway_s
        s_0 = cfg.idm_min_gap_m
        beta_unc = cfg.uncertainty_safety_beta
        v_dead = cfg.zero_speed_deadzone_mps
        two_sqrt_ab = 2.0 * math.sqrt(a_comf * b_comf)

        # Index corridor-lane queues from filtered beliefs so vehicles maintain safe spacing
        # even on ticks when local front-bumper radar experiences sensor dropout.
        lane_queues: dict[tuple[str, int], list[VehicleBelief]] = {}
        for veh in belief.vehicles.values():
            if veh.in_junction_id is None:
                key = (veh.corridor_id, veh.lane_index)
                lane_queues.setdefault(key, []).append(veh)
        belief_leaders: dict[str, VehicleBelief] = {}
        for queue in lane_queues.values():
            if len(queue) > 1:
                ordered = sorted(queue, key=lambda item: (item.mean_s_m, item.veh_id))
                for idx in range(len(ordered) - 1):
                    belief_leaders[ordered[idx].veh_id] = ordered[idx + 1]

        target_speeds: dict[str, float] = {}
        post_safety_accels: dict[str, float] = {}
        clamp_count = 0
        deadzone_count = 0

        for veh_id in sorted(belief.vehicles):
            veh = belief.vehicles[veh_id]
            corr = topology.corridors.get(veh.corridor_id)
            if corr is None:
                raise TopologyIntegrityError(
                    f"Vehicle {veh_id!r} references unknown corridor {veh.corridor_id!r}.",
                    context={"veh_id": veh_id, "corridor_id": veh.corridor_id},
                )

            require_finite(veh.mean_s_m, f"VehicleBelief[{veh_id}].mean_s_m")
            require_finite(veh.mean_v_mps, f"VehicleBelief[{veh_id}].mean_v_mps")
            require_finite(veh.length_m, f"VehicleBelief[{veh_id}].length_m")
            require_psd_covariance(veh.cov_2x2, f"VehicleBelief[{veh_id}].cov_2x2")

            v_max = corr.speed_limit_mps
            v_ego = min(v_max, max(0.0, float(veh.mean_v_mps)))
            ego_pos_var = max(0.0, float(veh.cov_2x2[0, 0]))
            ego_pos_std = math.sqrt(ego_pos_var)

            raw_desired = desired_speeds.get(veh_id, v_max)
            require_finite(raw_desired, f"desired_speeds[{veh_id}]")
            v_des = min(v_max, max(0.0, float(raw_desired)))

            a_req = min(a_comf, max(-b_max, (v_des - v_ego) / dt))
            v_safety_cap = v_max

            # 1. Observable front-leader uncertainty-inflated IDM + kinematic gap constraint.
            obs_gap_m, obs_lead_v, lead_pos_var = self._resolve_observable_leader(
                veh=veh,
                belief=belief,
                belief_leaders=belief_leaders,
                ego_pos_var=ego_pos_var,
            )
            if obs_gap_m is not None and obs_lead_v is not None:
                raw_u_margin = beta_unc * math.sqrt(max(0.0, ego_pos_var + lead_pos_var))
                u_margin = min(1.0, v_ego / 4.0) * raw_u_margin
                delta_v = v_ego - obs_lead_v
                s_star = s_0 + u_margin + max(
                    0.0,
                    0.85 * v_ego * t_hw + (v_ego * delta_v) / two_sqrt_ab,
                )
                net_clearance = max(0.0, obs_gap_m - s_0 - u_margin)
                # Hysteretic anti-creep standstill lock behind a stopped leader under 0.45 m/s radar
                # speed noise (closing D6 without micro-lurches) while releasing immediately once the
                # leader genuinely accelerates on green startup (obs_lead_v > 0.65 m/s or gap > s_0+1.2m).
                is_held_queue = veh_id in must_hold_stopline
                already_stopped_in_queue = (
                    is_held_queue and v_ego <= 0.75 and obs_lead_v <= 0.75 and obs_gap_m <= s_0 + 3.50
                )
                arriving_at_stopped_leader = obs_lead_v <= 0.25 and obs_gap_m <= s_0 + 0.55
                if already_stopped_in_queue or arriving_at_stopped_leader:
                    v_lead_cap = 0.0
                else:
                    v_kin_lead = math.sqrt(obs_lead_v * obs_lead_v + 2.0 * b_comf * net_clearance)
                    if obs_gap_m < s_star and delta_v > 0.0:
                        eff_gap_denom = max(0.25, obs_gap_m)
                        if v_des <= v_dead:
                            a_free = -b_comf
                        else:
                            v_ref = max(v_max, v_des, v_dead)
                            a_free = a_comf * (1.0 - (v_ego / v_ref) ** 4)
                        a_idm = a_free - a_comf * ((s_star / eff_gap_denom) ** 2)
                        v_idm_cap = max(0.0, v_ego + a_idm * dt)
                        v_lead_cap = min(v_kin_lead, v_idm_cap)
                    else:
                        v_lead_cap = v_kin_lead
                v_safety_cap = min(v_safety_cap, v_lead_cap)

            # 2. Observable stop-line hold envelope for ungranted / backpressure-held approaches.
            if veh_id in must_hold_stopline and veh.in_junction_id is None:
                raw_dist_to_line = corr.length_m - float(veh.mean_s_m)
                d_stop = max(0.0, raw_dist_to_line - 0.6 - beta_unc * ego_pos_std)
                if d_stop <= 1.25 or (v_ego <= 0.85 and raw_dist_to_line <= 5.5):
                    v_stop_max = 0.0
                else:
                    v_stop_max = math.sqrt(2.0 * b_comf * d_stop)
                v_safety_cap = min(v_safety_cap, v_stop_max)

            v_req_next = min(v_max, max(0.0, v_ego + a_req * dt))
            has_safety_clamp = v_safety_cap < v_des - 1e-9
            if has_safety_clamp:
                v_next_raw = min(v_des, max(0.0, max(v_ego - b_max * dt, v_safety_cap)))
            else:
                # Pass unthrottled desired speed to SUMO when no safety hazard is active so noisy
                # Kalman v_ego estimates never drag down physical platoon acceleration.
                v_next_raw = v_des

            # 3. Zero-speed deadzone (v < 0.15 m/s => 0) eliminating clipped-Kalman queue creep.
            if v_next_raw < v_dead:
                if v_next_raw > 1e-9 or (1e-3 < v_ego < v_dead and v_des > 0.0):
                    deadzone_count += 1
                v_safe = 0.0
                a_post = max(-b_max, min(0.0, -v_ego / dt))
            else:
                v_safe = v_next_raw
                a_post = min(a_comf, max(-b_max, (v_safe - v_ego) / dt))

            if has_safety_clamp and v_req_next - v_safe > 0.05:
                clamp_count += 1

            require_finite(v_safe, f"SafetyFilterResult.target_speeds[{veh_id}]")
            require_finite(a_post, f"SafetyFilterResult.post_safety_accels[{veh_id}]")
            target_speeds[veh_id] = v_safe
            post_safety_accels[veh_id] = a_post

        return SafetyFilterResult(
            target_speeds=target_speeds,
            post_safety_accels=post_safety_accels,
            safety_clamp_count=clamp_count,
            zero_speed_deadzone_count=deadzone_count,
        )

    @staticmethod
    def _resolve_observable_leader(
        veh: VehicleBelief,
        belief: SwarmBeliefState,
        belief_leaders: Mapping[str, VehicleBelief],
        ego_pos_var: float,
    ) -> tuple[float | None, float | None, float]:
        """Extract purely observable front-leader gap, speed, and longitudinal variance."""
        if veh.noisy_leader_gap_m is not None and float(veh.obs_age_s) <= 1e-6:
            require_finite(veh.noisy_leader_gap_m, f"VehicleBelief[{veh.veh_id}].noisy_leader_gap_m")
            gap_m = float(veh.noisy_leader_gap_m)
            lead_bel = belief.vehicles.get(veh.leader_id) if veh.leader_id is not None else None
            if veh.noisy_leader_speed_mps is not None:
                require_finite(
                    veh.noisy_leader_speed_mps,
                    f"VehicleBelief[{veh.veh_id}].noisy_leader_speed_mps",
                )
                raw_lead_v = max(0.0, float(veh.noisy_leader_speed_mps))
                if lead_bel is not None:
                    require_finite(lead_bel.mean_v_mps, f"VehicleBelief[{lead_bel.veh_id}].mean_v_mps")
                    filt_v = max(0.0, float(lead_bel.mean_v_mps))
                    if raw_lead_v <= 0.25 or filt_v <= 0.25:
                        lead_v = min(raw_lead_v, filt_v)
                    else:
                        lead_v = 0.5 * raw_lead_v + 0.5 * filt_v
                else:
                    lead_v = raw_lead_v
            elif lead_bel is not None:
                require_finite(lead_bel.mean_v_mps, f"VehicleBelief[{lead_bel.veh_id}].mean_v_mps")
                lead_v = max(0.0, float(lead_bel.mean_v_mps))
            else:
                lead_v = max(0.0, float(veh.mean_v_mps))

            if (
                lead_bel is not None
                and lead_bel.corridor_id == veh.corridor_id
                and lead_bel.lane_index == veh.lane_index
                and lead_bel.in_junction_id is None
            ):
                require_finite(lead_bel.mean_s_m, f"VehicleBelief[{lead_bel.veh_id}].mean_s_m")
                require_finite(lead_bel.length_m, f"VehicleBelief[{lead_bel.veh_id}].length_m")
                filtered_gap = (
                    float(lead_bel.mean_s_m)
                    - float(veh.mean_s_m)
                    - 0.5 * (float(lead_bel.length_m) + float(veh.length_m))
                )
                gap_m = 0.5 * gap_m + 0.5 * filtered_gap

            if lead_bel is not None:
                require_psd_covariance(lead_bel.cov_2x2, f"VehicleBelief[{lead_bel.veh_id}].cov_2x2")
                lead_var = max(0.0, float(lead_bel.cov_2x2[0, 0]))
            else:
                lead_var = max(0.25, ego_pos_var)
            return gap_m, lead_v, lead_var

        fallback_lead = belief_leaders.get(veh.veh_id)
        if fallback_lead is not None:
            require_finite(fallback_lead.mean_s_m, f"VehicleBelief[{fallback_lead.veh_id}].mean_s_m")
            require_finite(fallback_lead.mean_v_mps, f"VehicleBelief[{fallback_lead.veh_id}].mean_v_mps")
            require_finite(fallback_lead.length_m, f"VehicleBelief[{fallback_lead.veh_id}].length_m")
            require_psd_covariance(fallback_lead.cov_2x2, f"VehicleBelief[{fallback_lead.veh_id}].cov_2x2")
            bumper_gap = (
                float(fallback_lead.mean_s_m)
                - float(veh.mean_s_m)
                - 0.5 * (float(fallback_lead.length_m) + float(veh.length_m))
            )
            lead_v = max(0.0, float(fallback_lead.mean_v_mps))
            lead_var = max(0.0, float(fallback_lead.cov_2x2[0, 0]))
            return bumper_gap, lead_v, lead_var

        return None, None, 0.0


def select_actuated_target_speeds(
    belief: SwarmBeliefState,
    topology: RoadTopology,
    safety: SafetyFilterResult,
    control_config: ControlConfig,
    must_hold_stopline: AbstractSet[str],
    granted_vehicles: AbstractSet[str],
    previously_controlled: AbstractSet[str],
) -> tuple[dict[str, float], set[str]]:
    """Select active longitudinal speed overrides and release previously held vehicles via ``-1.0``.

    Emits non-negative ``setSpeed`` commands only when longitudinal gating is actively needed
    (front-of-lane predictive DBN spillback/storage stopline hold inside the approach pocket,
    unsignalized AIM slot pacing, or close-range emergency braking), and emits a single-tick
    ``-1.0`` release otherwise so SUMO ``LC2013`` cooperative lane-changing operates unimpeded.
    """
    lane_queues: dict[tuple[str, int], list[VehicleBelief]] = {}
    for q_veh in belief.vehicles.values():
        if q_veh.in_junction_id is None:
            lane_queues.setdefault((q_veh.corridor_id, q_veh.lane_index), []).append(q_veh)
    same_lane_lead_id: dict[str, str] = {}
    for q_list in lane_queues.values():
        if len(q_list) > 1:
            ordered = sorted(q_list, key=lambda item: (float(item.mean_s_m), item.veh_id))
            for idx in range(len(ordered) - 1):
                same_lane_lead_id[ordered[idx].veh_id] = ordered[idx + 1].veh_id

    out_speeds: dict[str, float] = {}
    now_controlled: set[str] = set()
    s_0 = control_config.idm_min_gap_m

    for vid in sorted(belief.vehicles):
        veh = belief.vehicles[vid]
        if veh.in_junction_id is not None:
            if vid in previously_controlled:
                out_speeds[vid] = -1.0
            continue

        corr = topology.corridors[veh.corridor_id]
        d_rem = max(0.0, float(corr.length_m) - float(veh.mean_s_m))
        in_final_pocket = (
            len(corr.edges) <= 1
            or veh.edge_id == corr.edges[-1]
            or d_rem <= max(36.0, float(corr.length_m) - float(corr.edge_offsets_m[-1]))
        )
        lead_id = same_lane_lead_id.get(vid)
        has_held_same_lane_leader = lead_id is not None and lead_id in must_hold_stopline

        is_unsignalized_approach = (
            not corr.is_exit_fringe
            and topology.junctions[corr.to_junction].tls_id is None
        )
        needs_stopline_hold = (
            vid in must_hold_stopline
            and in_final_pocket
            and not has_held_same_lane_leader
        )
        needs_unsig_pacing = (
            is_unsignalized_approach
            and vid not in granted_vehicles
            and in_final_pocket
            and not has_held_same_lane_leader
        )
        needs_emergency_clamp = (
            veh.noisy_leader_gap_m is not None
            and float(veh.noisy_leader_gap_m) < s_0 + 0.6
            and safety.post_safety_accels.get(vid, 0.0) <= -0.80 * control_config.max_decel_mps2
        )

        if needs_stopline_hold or needs_unsig_pacing or needs_emergency_clamp:
            out_speeds[vid] = float(safety.target_speeds[vid])
            now_controlled.add(vid)
        elif vid in previously_controlled:
            out_speeds[vid] = -1.0

    return out_speeds, now_controlled


__all__ = [
    "ObservableSafetyFilter",
    "SafetyFilterResult",
    "select_actuated_target_speeds",
]

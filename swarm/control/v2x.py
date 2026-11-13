"""Fair budgeted Value-of-Information (VoI) V2X polling selector.

Ranks the exact same active tracked vehicle pool across all V2X-enabled controllers under a
shared per-step packet budget ``B_comm = ObservationConfig.v2x_budget_per_step`` and per-vehicle
cooldown ``v2x_poll_cooldown_s`` (closing the 3.6x head-only polling asymmetry D7).
"""

from __future__ import annotations

import math

from ..config import ObservationConfig
from ..contracts import DBNStateSnapshot, RoadTopology, SwarmBeliefState, VehicleBelief
from ..logging_utils import (
    NumericalInvariantError,
    TopologyIntegrityError,
    require_finite,
    require_psd_covariance,
    require_probability_simplex,
)


def compute_vehicle_voi_score(
    veh: VehicleBelief,
    topology: RoadTopology,
    dbn_snapshot: DBNStateSnapshot | None = None,
) -> float:
    """Compute Value-of-Information priority score for querying ``veh`` over V2X."""
    corr = topology.corridors.get(veh.corridor_id)
    if corr is None:
        raise TopologyIntegrityError(
            f"Vehicle {veh.veh_id!r} references unknown corridor {veh.corridor_id!r} in V2X selector.",
            context={"veh_id": veh.veh_id, "corridor_id": veh.corridor_id},
        )

    require_psd_covariance(veh.cov_2x2, f"VehicleBelief[{veh.veh_id}].cov_2x2")
    require_finite(veh.obs_age_s, f"VehicleBelief[{veh.veh_id}].obs_age_s")
    require_finite(veh.free_eta_s, f"VehicleBelief[{veh.veh_id}].free_eta_s")

    if corr.is_exit_fringe and len(corr.turn_options) == 0:
        intent_entropy = 0.0
    else:
        if len(veh.turn_probs) != len(corr.turn_options):
            raise NumericalInvariantError(
                f"Vehicle {veh.veh_id!r} turn_probs len {len(veh.turn_probs)} != "
                f"{len(corr.turn_options)} in V2X selector.",
                context={"veh_id": veh.veh_id, "corridor_id": corr.corridor_id},
            )
        require_probability_simplex(veh.turn_probs, f"VehicleBelief[{veh.veh_id}].turn_probs")
        intent_entropy = 0.0
        if len(veh.turn_probs) > 1:
            for prob in veh.turn_probs:
                p_val = float(prob)
                if p_val > 1e-12:
                    intent_entropy -= p_val * math.log(p_val)

    pos_std = math.sqrt(max(0.0, float(veh.cov_2x2[0, 0])))
    age_s = max(0.0, float(veh.obs_age_s))
    eta_s = max(0.0, float(veh.free_eta_s))

    psi_cascade = 0.0
    if dbn_snapshot is not None and veh.veh_id in dbn_snapshot.cascade_score:
        raw_psi = float(dbn_snapshot.cascade_score[veh.veh_id])
        require_finite(raw_psi, f"DBNStateSnapshot.cascade_score[{veh.veh_id}]")
        psi_cascade = max(0.0, raw_psi)

    uncertainty_term = intent_entropy + 0.25 * pos_std + 0.30 * age_s + 0.05
    score = (uncertainty_term * (1.0 + psi_cascade)) / (1.0 + 0.20 * eta_s)
    require_finite(score, f"VoI[{veh.veh_id}]")
    return score


class VoIV2XSelector:
    """Stateful budgeted V2X query selector enforcing cooldowns and shared active-pool eligibility."""

    def __init__(self, obs_config: ObservationConfig) -> None:
        self._obs_cfg = obs_config
        self._last_polled_at: dict[str, float] = {}

    @property
    def budget_per_step(self) -> int:
        """Return the configured per-step V2X query cap ``B_comm``."""
        return self._obs_cfg.v2x_budget_per_step

    def reset(self) -> None:
        """Clear per-vehicle poll cooldown timestamps across episodes."""
        self._last_polled_at.clear()

    def select_polls(
        self,
        belief: SwarmBeliefState,
        topology: RoadTopology,
        dbn_snapshot: DBNStateSnapshot | None = None,
    ) -> tuple[str, ...]:
        """Select up to ``v2x_budget_per_step`` eligible active vehicle IDs ranked by VoI."""
        require_finite(belief.sim_time, "SwarmBeliefState.sim_time")
        now = float(belief.sim_time)

        # Prune departed vehicles from local cooldown registry.
        live_ids = set(belief.vehicles.keys())
        if len(self._last_polled_at) > 0:
            self._last_polled_at = {
                vid: ts for vid, ts in self._last_polled_at.items() if vid in live_ids
            }

        budget = self._obs_cfg.v2x_budget_per_step
        if budget <= 0 or not belief.vehicles:
            return ()

        cooldown = self._obs_cfg.v2x_poll_cooldown_s
        scored_eligible: list[tuple[float, str]] = []

        for vid in sorted(belief.vehicles):
            veh = belief.vehicles[vid]
            require_finite(veh.last_v2x_poll_time, f"VehicleBelief[{vid}].last_v2x_poll_time")
            last_poll = max(
                float(veh.last_v2x_poll_time),
                self._last_polled_at.get(vid, -math.inf),
            )
            if (now - last_poll) < cooldown - 1e-9:
                continue
            voi = compute_vehicle_voi_score(veh, topology, dbn_snapshot)
            scored_eligible.append((-voi, vid))

        if not scored_eligible:
            return ()

        scored_eligible.sort()
        selected = tuple(vid for _, vid in scored_eligible[:budget])
        for vid in selected:
            self._last_polled_at[vid] = now
        return selected


__all__ = [
    "VoIV2XSelector",
    "compute_vehicle_voi_score",
]

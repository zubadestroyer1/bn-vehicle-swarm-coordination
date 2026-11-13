"""Firewall-safe DBN dynamic route travel-time cost broadcaster.

Computes purely observable public edge travel-time estimates ``broadcast_edge_costs`` across
all road edges in :class:`~swarm.contracts.RoadTopology` without reading hidden vehicle
destinations ``route_goal`` or ``route_edges`` (enforcing ``#13`` §1.3). Returns ``None`` for
the ``fixed_route`` ablation and static-route signal/FCFS baselines.
"""

from __future__ import annotations

import math
from typing import Final

from ..config import ControlConfig
from ..contracts import DBNStateSnapshot, POLICY_NAMES, RoadTopology, SwarmBeliefState
from ..logging_utils import (
    ConfigValidationError,
    NumericalInvariantError,
    require_finite,
    require_probability_range,
)

STATIC_ROUTE_POLICIES: Final[frozenset[str]] = frozenset(
    {
        "fixed_signal",
        "actuated_signal",
        "fcfs_reservation",
        "fixed_route",
    }
)

_CLOSED_CORRIDOR_PENALTY_S: float = 600.0


def compute_broadcast_edge_costs(
    belief: SwarmBeliefState,
    dbn_snapshot: DBNStateSnapshot | None,
    topology: RoadTopology,
    control_config: ControlConfig,
    *,
    policy_name: str = "neural_dbn",
) -> dict[str, float] | None:
    """Compute public edge travel-time costs ``c_hat_e^DBN(t) [s]`` for onboard Dijkstra rerouting.

    Returns ``None`` when ``policy_name`` disables dynamic routing (``fixed_route``,
    ``fixed_signal``, ``actuated_signal``, ``fcfs_reservation``).
    """
    if policy_name not in POLICY_NAMES:
        raise ConfigValidationError(
            f"Unknown policy_name {policy_name!r} in compute_broadcast_edge_costs.",
            context={"policy_name": policy_name, "valid_policies": list(POLICY_NAMES)},
        )
    if policy_name in STATIC_ROUTE_POLICIES:
        return None

    veh_by_corridor: dict[str, list[str]] = {cid: [] for cid in topology.corridors}
    for vid, veh in belief.vehicles.items():
        if veh.corridor_id in veh_by_corridor and veh.in_junction_id is None:
            veh_by_corridor[veh.corridor_id].append(vid)

    cfg = control_config
    edge_costs: dict[str, float] = {}

    for cid in sorted(topology.corridors):
        corr = topology.corridors[cid]
        free_time_s = corr.length_m / corr.speed_limit_mps
        eff_lanes = topology.effective_lanes(cid, belief.open_lanes_by_edge)
        eff_cap = topology.effective_capacity(cid, belief.open_lanes_by_edge)

        if cid in belief.corridors:
            c_stats = belief.corridors[cid]
            occ_ratio = float(c_stats.occupancy_ratio)
            exp_inflow = float(c_stats.expected_inflow)
            tail_prob = float(c_stats.tail_blockage_prob)
            require_finite(occ_ratio, f"CorridorBeliefStats[{cid}].occupancy_ratio")
            require_finite(exp_inflow, f"CorridorBeliefStats[{cid}].expected_inflow")
            require_probability_range(tail_prob, f"CorridorBeliefStats[{cid}].tail_blockage_prob")
        else:
            occ_ratio = len(veh_by_corridor[cid]) / float(max(1, eff_cap))
            exp_inflow = 0.0
            tail_prob = 0.0

        if dbn_snapshot is not None and cid in dbn_snapshot.corridor_spillback_pred:
            q_spill = float(dbn_snapshot.corridor_spillback_pred[cid])
            require_probability_range(q_spill, f"corridor_spillback_pred[{cid}]")
        else:
            q_spill = tail_prob
        q_spill = max(0.0, min(1.0, q_spill))

        if eff_lanes <= 0:
            corridor_cost_s = free_time_s + _CLOSED_CORRIDOR_PENALTY_S
        else:
            delay_sum_s = 0.0
            if dbn_snapshot is not None:
                for vid in veh_by_corridor[cid]:
                    d_prob = float(
                        dbn_snapshot.delay_post.get(vid, dbn_snapshot.delay_pred.get(vid, 0.0))
                    )
                    tau_s = float(dbn_snapshot.expected_delay_s.get(vid, 0.0))
                    require_probability_range(d_prob, f"DBNStateSnapshot.delay_prob[{vid}]")
                    require_finite(tau_s, f"DBNStateSnapshot.expected_delay_s[{vid}]")
                    delay_sum_s += max(0.0, min(1.0, d_prob)) * max(0.0, tau_s)

            veh_delay_term = delay_sum_s / float(eff_lanes)
            flux_term = cfg.reroute_flux_weight * (
                max(0.0, occ_ratio) + max(0.0, exp_inflow) / float(max(1, eff_cap))
            )
            spill_term = cfg.reroute_spill_weight * q_spill
            lane_loss_term = (
                free_time_s * (float(corr.base_lanes) / float(eff_lanes) - 1.0)
                if eff_lanes < corr.base_lanes
                else 0.0
            )
            corridor_cost_s = free_time_s + veh_delay_term + flux_term + spill_term + lane_loss_term

        require_finite(corridor_cost_s, f"corridor_cost_s[{cid}]")
        if corridor_cost_s <= 0.0:
            raise NumericalInvariantError(
                f"Non-positive corridor travel cost {corridor_cost_s} on {cid!r}.",
                context={"corridor_id": cid, "corridor_cost_s": corridor_cost_s},
            )

        # Partition corridor cost proportionally across split trunk + turn-pocket constituent edges.
        n_edges = len(corr.edges)
        seg_lengths: list[float] = []
        for idx in range(n_edges):
            next_off = corr.edge_offsets_m[idx + 1] if idx + 1 < n_edges else corr.length_m
            seg_lengths.append(max(0.1, float(next_off - corr.edge_offsets_m[idx])))
        total_seg_len = sum(seg_lengths)

        for edge_id, seg_len in zip(corr.edges, seg_lengths, strict=True):
            e_cost = max(0.05, corridor_cost_s * (seg_len / total_seg_len))
            require_finite(e_cost, f"broadcast_edge_costs[{edge_id}]")
            edge_costs[edge_id] = e_cost

    return edge_costs


class DBNRouteCostBroadcaster:
    """Periodic route-cost broadcaster enforcing ``reroute_interval_s`` and ablation gates."""

    def __init__(self, control_config: ControlConfig, policy_name: str) -> None:
        if policy_name not in POLICY_NAMES:
            raise ConfigValidationError(
                f"Unknown policy_name {policy_name!r} in DBNRouteCostBroadcaster.",
                context={"policy_name": policy_name, "valid_policies": list(POLICY_NAMES)},
            )
        self._cfg = control_config
        self._policy_name = policy_name
        self._last_broadcast_time: float = -math.inf

    def reset(self) -> None:
        """Reset periodic broadcast clock at episode start."""
        self._last_broadcast_time = -math.inf

    def maybe_broadcast(
        self,
        belief: SwarmBeliefState,
        dbn_snapshot: DBNStateSnapshot | None,
        topology: RoadTopology,
    ) -> dict[str, float] | None:
        """Emit public ``broadcast_edge_costs`` on interval ticks, or ``None`` if disabled/between intervals."""
        require_finite(belief.sim_time, "SwarmBeliefState.sim_time")
        if self._policy_name in STATIC_ROUTE_POLICIES:
            return None

        now = float(belief.sim_time)
        if (now - self._last_broadcast_time) < self._cfg.reroute_interval_s - 1e-9:
            return None

        costs = compute_broadcast_edge_costs(
            belief=belief,
            dbn_snapshot=dbn_snapshot,
            topology=topology,
            control_config=self._cfg,
            policy_name=self._policy_name,
        )
        self._last_broadcast_time = now
        return costs


__all__ = [
    "DBNRouteCostBroadcaster",
    "STATIC_ROUTE_POLICIES",
    "compute_broadcast_edge_costs",
]

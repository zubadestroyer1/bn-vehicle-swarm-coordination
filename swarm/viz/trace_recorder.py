"""Compact, high-fidelity 3D episode trace recorder implementing :class:`TraceRecorderProtocol`.

Combines ground-truth 3D vehicle poses/signals (:class:`GroundTruthStepRecord`), live TLS link RYG
strings, filtered belief uncertainties & corridor congestion (:class:`SwarmBeliefState`), calibrated
3D Neural DBN arcs & cascade priorities (:class:`DBNStateSnapshot`), and longitudinal control modes
(:class:`SwarmControlAction`) into a self-contained JSON replay payload. Explicitly excludes latent
``DBNTemporalMemory`` tensors per ``#18`` §5.2.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
from typing import Any

from ..contracts import (
    DBNStateSnapshot,
    EpisodeMetrics,
    GroundTruthStepRecord,
    RoadTopology,
    SCHEMA_VERSION,
    SwarmBeliefState,
    SwarmControlAction,
)
from ..logging_utils import (
    ConfigValidationError,
    NumericalInvariantError,
    SimulationLifecycleError,
    get_logger,
    require_finite,
)


def _shannon_entropy(probs: tuple[float, ...]) -> float:
    """Return normalized Shannon entropy in ``[0, 1]`` for a categorical tuple."""
    k = len(probs)
    if k <= 1:
        return 0.0
    h = 0.0
    for p in probs:
        if p > 1e-12:
            h -= p * math.log(p)
    return min(1.0, max(0.0, h / math.log(k)))


class EpisodeTraceRecorder:
    """Step-by-step recorder producing rich 3D WebGL replay frames and HUD telemetry series."""

    def __init__(
        self,
        scenario_id: str = "unknown",
        policy_name: str = "unknown",
        seed: int = 0,
        *,
        topology: RoadTopology | None = None,
        max_dbn_edges_per_step: int = 120,
        subsample_every: int = 1,
    ) -> None:
        if isinstance(max_dbn_edges_per_step, bool) or max_dbn_edges_per_step < 0:
            raise ConfigValidationError(
                f"max_dbn_edges_per_step must be an int >= 0, got {max_dbn_edges_per_step!r}."
            )
        if isinstance(subsample_every, bool) or subsample_every < 1:
            raise ConfigValidationError(
                f"subsample_every must be an int >= 1, got {subsample_every!r}."
            )
        self.scenario_id = scenario_id
        self.policy_name = policy_name
        self.seed = int(seed)
        self.topology = topology
        self.max_dbn_edges_per_step = max_dbn_edges_per_step
        self.subsample_every = subsample_every

        self._frames: list[dict[str, Any]] = []
        self._step_counter: int = 0
        self._last_sim_time: float | None = None
        self._cumulative_collisions: int = 0
        self._cumulative_emerg_brakes: int = 0
        self._cumulative_reroutes: int = 0

    def reset(self) -> None:
        """Clear all recorded frames and reset cumulative counters for a new episode."""
        self._frames.clear()
        self._step_counter = 0
        self._last_sim_time = None
        self._cumulative_collisions = 0
        self._cumulative_emerg_brakes = 0
        self._cumulative_reroutes = 0

    def record_step(
        self,
        gt_record: GroundTruthStepRecord,
        belief: SwarmBeliefState | None,
        dbn_snapshot: DBNStateSnapshot | None,
        action: SwarmControlAction,
        tls_states: Mapping[str, str],
    ) -> None:
        """Record one simulation tick into the 3D replay stream."""
        sim_t = float(gt_record.sim_time)
        require_finite(sim_t, "gt_record.sim_time")
        if self._last_sim_time is not None and sim_t <= self._last_sim_time - 1e-9:
            raise SimulationLifecycleError(
                f"Non-monotonic sim_time in EpisodeTraceRecorder: {sim_t} after {self._last_sim_time}.",
                context={"sim_time": sim_t, "prev_sim_time": self._last_sim_time},
            )
        self._last_sim_time = sim_t

        self._cumulative_collisions += int(gt_record.collisions)
        self._cumulative_emerg_brakes += int(gt_record.emergency_brakes)
        self._cumulative_reroutes += int(gt_record.reroutes_applied)

        step_idx = self._step_counter
        self._step_counter += 1
        if step_idx % self.subsample_every != 0:
            return

        veh_beliefs = belief.vehicles if belief is not None else {}
        edge_to_corr = self.topology.edge_to_corridor if self.topology is not None else {}

        delay_post_map = dbn_snapshot.delay_post if dbn_snapshot is not None else {}
        delay_pred_map = dbn_snapshot.delay_pred if dbn_snapshot is not None else {}
        cascade_map = dbn_snapshot.cascade_score if dbn_snapshot is not None else {}
        prec_map = dbn_snapshot.precedence_scores if dbn_snapshot is not None else {}

        vehicles_frame: list[dict[str, Any]] = []
        active_ids_set: set[str] = set()
        speed_sum = 0.0
        halted_count = 0

        for rv in gt_record.raw_vehicles:
            vid = rv.veh_id
            active_ids_set.add(vid)
            cx, cy = float(rv.center_xy[0]), float(rv.center_xy[1])
            yaw = float(rv.yaw_ccw_rad)
            spd = float(rv.speed)
            acc = float(rv.accel)
            length = float(rv.length)
            width = float(rv.width)
            require_finite(cx, f"raw_vehicle[{vid}].x")
            require_finite(cy, f"raw_vehicle[{vid}].y")
            require_finite(yaw, f"raw_vehicle[{vid}].yaw")
            require_finite(spd, f"raw_vehicle[{vid}].speed")
            require_finite(acc, f"raw_vehicle[{vid}].accel")

            speed_sum += spd
            if spd < 0.5:
                halted_count += 1

            vb = veh_beliefs.get(vid)
            if vb is not None:
                var_s = float(vb.cov_2x2[0, 0])
                if var_s < -1e-6 or not math.isfinite(var_s):
                    raise NumericalInvariantError(
                        f"Invalid position variance {var_s} in VehicleBelief[{vid}].",
                        context={"veh_id": vid, "var_s": var_s},
                    )
                pos_std = math.sqrt(max(0.0, var_s))
                entropy = _shannon_entropy(vb.turn_probs)
                corr_id = vb.corridor_id
                obs_age = float(vb.obs_age_s)
            else:
                pos_std = 0.0
                entropy = 0.0
                corr_id = edge_to_corr.get(rv.road_id, rv.road_id)
                obs_age = 0.0

            mode_enum = action.modes.get(vid)
            mode_int = int(mode_enum) if mode_enum is not None else -1
            psi_val = float(cascade_map.get(vid, 0.0))
            d_post = float(delay_post_map.get(vid, 0.0))
            d_pred = float(delay_pred_map.get(vid, 0.0))
            r_score = float(prec_map.get(vid, 0.0))
            require_finite(psi_val, f"cascade_score[{vid}]")
            require_finite(d_post, f"delay_post[{vid}]")

            vehicles_frame.append(
                {
                    "id": vid,
                    "vtype": rv.vtype,
                    "x": round(cx, 3),
                    "y": round(cy, 3),
                    "yaw": round(yaw, 4),
                    "v": round(spd, 3),
                    "a": round(acc, 3),
                    "L": round(length, 2),
                    "W": round(width, 2),
                    "brake": bool(rv.brake_light),
                    "turn": int(rv.turn_signal),
                    "goal": rv.route_goal,
                    "corridor": corr_id,
                    "mode": mode_int,
                    "pos_std": round(pos_std, 3),
                    "entropy": round(entropy, 3),
                    "obs_age": round(obs_age, 2),
                    "d_pred": round(d_pred, 4),
                    "d_post": round(d_post, 4),
                    "psi": round(psi_val, 4),
                    "r_score": round(r_score, 3),
                }
            )

        # Calibrated 3D Neural DBN arcs (filtered to active vehicles in this frame, sorted by omega)
        edges_frame: list[dict[str, Any]] = []
        total_active_edges = 0
        if dbn_snapshot is not None and dbn_snapshot.edges:
            valid_edges = [
                e for e in dbn_snapshot.edges if e.src_id in active_ids_set and e.dst_id in active_ids_set
            ]
            total_active_edges = len(valid_edges)
            valid_edges.sort(key=lambda e: float(e.omega), reverse=True)
            if self.max_dbn_edges_per_step > 0:
                valid_edges = valid_edges[: self.max_dbn_edges_per_step]
            else:
                valid_edges = []

            for edge in valid_edges:
                omega = float(edge.omega)
                kappa = float(edge.kappa)
                gamma = float(edge.gamma)
                p_conf = float(edge.p_conflict)
                require_finite(omega, "dbn_edge.omega")
                require_finite(kappa, "dbn_edge.kappa")
                require_finite(gamma, "dbn_edge.gamma")
                edges_frame.append(
                    {
                        "src": edge.src_id,
                        "dst": edge.dst_id,
                        "rel": int(edge.relation),
                        "omega": round(omega, 4),
                        "kappa": round(kappa, 4),
                        "gamma": round(gamma, 4),
                        "p_conf": round(p_conf, 4),
                    }
                )

        # Corridor congestion & DBN spillback heat overlay
        corridor_heat: dict[str, dict[str, float]] = {}
        spillback_corridors_count = int(gt_record.spillback_corridor_count)
        spill_map = dbn_snapshot.corridor_spillback_pred if dbn_snapshot is not None else {}
        if belief is not None:
            for cid, cstats in belief.corridors.items():
                occ = float(cstats.occupancy_ratio)
                tail = float(cstats.tail_blockage_prob)
                q_pred = float(spill_map.get(cid, tail))
                require_finite(occ, f"corridor[{cid}].occ")
                require_finite(tail, f"corridor[{cid}].tail")
                require_finite(q_pred, f"corridor[{cid}].q_pred")
                if occ > 0.02 or tail > 0.03 or q_pred > 0.05:
                    corridor_heat[cid] = {
                        "occ": round(occ, 4),
                        "tail": round(tail, 4),
                        "spill": round(q_pred, 4),
                    }
        elif spill_map:
            for cid, q_val in spill_map.items():
                qf = float(q_val)
                require_finite(qf, f"corridor[{cid}].q_pred")
                if qf > 0.05:
                    corridor_heat[cid] = {"occ": 0.0, "tail": 0.0, "spill": round(qf, 4)}

        if spillback_corridors_count == 0 and corridor_heat:
            spillback_corridors_count = sum(1 for h in corridor_heat.values() if h["spill"] >= 0.55 or h["tail"] >= 0.55)

        n_veh = len(vehicles_frame)
        mean_speed = speed_sum / n_veh if n_veh > 0 else 0.0
        mean_psi = (
            sum(v["psi"] for v in vehicles_frame) / n_veh
            if n_veh > 0
            else 0.0
        )
        dbn_ms = (
            float(dbn_snapshot.update_latency_ms + dbn_snapshot.inference_latency_ms)
            if dbn_snapshot is not None
            else 0.0
        )

        self._frames.append(
            {
                "t": round(sim_t, 3),
                "vehicles": vehicles_frame,
                "dbn_edges": edges_frame,
                "corridor_heat": corridor_heat,
                "tls": {str(k): str(v) for k, v in tls_states.items()},
                "v2x_polls": [vid for vid in action.v2x_polls if vid in active_ids_set],
                "kpi": {
                    "active_veh": n_veh,
                    "halted_veh": halted_count,
                    "mean_speed_mps": round(mean_speed, 3),
                    "active_edges": total_active_edges,
                    "mean_psi": round(mean_psi, 4),
                    "spillback_corridors": spillback_corridors_count,
                    "cum_collisions": self._cumulative_collisions,
                    "cum_emerg_brakes": self._cumulative_emerg_brakes,
                    "cum_reroutes": self._cumulative_reroutes,
                    "dbn_ms": round(dbn_ms, 3),
                },
            }
        )

    def finalize(self, metrics: EpisodeMetrics | None = None) -> dict[str, Any]:
        """Return the complete episode 3D replay trace dictionary (safe for JSON serialization)."""
        scenario_id = metrics.scenario_id if metrics is not None else self.scenario_id
        policy_name = metrics.policy_name if metrics is not None else self.policy_name
        seed = int(metrics.seed) if metrics is not None else self.seed

        dt_est = 0.5
        if len(self._frames) >= 2:
            dt_est = round(max(0.05, self._frames[1]["t"] - self._frames[0]["t"]), 4)

        timeline = {
            "t": [f["t"] for f in self._frames],
            "active_veh": [f["kpi"]["active_veh"] for f in self._frames],
            "halted_veh": [f["kpi"]["halted_veh"] for f in self._frames],
            "mean_speed_mps": [f["kpi"]["mean_speed_mps"] for f in self._frames],
            "active_edges": [f["kpi"]["active_edges"] for f in self._frames],
            "spillback_corridors": [f["kpi"]["spillback_corridors"] for f in self._frames],
            "dbn_ms": [f["kpi"]["dbn_ms"] for f in self._frames],
        }

        return {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": scenario_id,
            "policy_name": policy_name,
            "seed": seed,
            "dt": dt_est,
            "frame_count": len(self._frames),
            "frames": list(self._frames),
            "timeline": timeline,
            "summary_metrics": metrics.to_dict() if metrics is not None else None,
        }

    def save_json(self, path: Path | str, metrics: EpisodeMetrics | None = None) -> Path:
        """Atomically serialize finalized 3D replay trace to ``path``."""
        payload = self.finalize(metrics=metrics)
        out_p = Path(path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = out_p.parent / f".{out_p.name}.pid{os.getpid()}.tmp"
        tmp_p.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_p, out_p)
        return out_p


def load_trace_json(path: Path | str) -> dict[str, Any]:
    """Load and validate an episode 3D trace JSON file."""
    target = Path(path)
    if not target.is_file():
        raise ConfigValidationError(
            f"Trace JSON file does not exist: {target}",
            context={"path": str(target)},
        )
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        get_logger("viz.trace").error("Malformed trace JSON %s: %s", target, exc)
        raise ConfigValidationError(
            f"Malformed trace JSON {target}: {exc}",
            context={"path": str(target)},
        ) from exc
    if not isinstance(data, dict) or "frames" not in data or not isinstance(data["frames"], list):
        raise ConfigValidationError(
            f"Invalid trace JSON schema in {target}; expected object with 'frames' list.",
            context={"path": str(target)},
        )
    return data


__all__ = [
    "EpisodeTraceRecorder",
    "load_trace_json",
]

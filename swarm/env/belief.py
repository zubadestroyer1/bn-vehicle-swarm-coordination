"""City-scale 2D Joseph-form Kalman belief filter, OOSM lag inflation, and Cat(K_e) turn belief.

Firewall guarantees (enforced by static AST inspection in ``tests/test_train_eval.py``):
- Zero imports of ``libsumo``, ``traci``, ``sumolib``, ``VehicleRawTelemetry``,
  ``GroundTruthStepRecord``, or ``EpisodeMetrics``.
- Track ``entered_at`` strictly records observable road-entry clock ``sim_time``; passing a
  timestamp mismatching ``obs_batch.sim_time`` raises :class:`~swarm.logging_utils.CausalityFirewallError`.
- Prediction strictly consumes ``post_safety_accels`` (``#13`` §3) with stopped-queue deadzone
  protection against clipped-Kalman creep (``#8`` §3.3a).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np

from ..config import ObservationConfig
from ..contracts import (
    CorridorBeliefStats,
    CorridorSpec,
    ObservationBatch,
    RoadTopology,
    SwarmBeliefState,
    VehicleBelief,
)
from ..logging_utils import (
    CausalityFirewallError,
    NumericalInvariantError,
    TopologyIntegrityError,
    require_finite,
    require_probability_range,
    require_probability_simplex,
    require_psd_covariance,
)


def _std_normal_cdf(z: float) -> float:
    """Exact standard normal CDF ``Phi(z)`` in ``[0, 1]`` using ``math.erf``."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _compute_free_eta_and_std(
    spec: CorridorSpec,
    in_junction_id: str | None,
    mean_s_m: float,
    mean_v_mps: float,
    cov_2x2: np.ndarray,
    obs_age_s: float,
    comfort_accel_mps2: float = 2.0,
) -> tuple[float, float]:
    """Compute analytical free-flow ETA to corridor stop-line and delta-method ``eta_std_s``."""
    if in_junction_id is not None:
        return 0.0, float(0.10 + 0.10 * obs_age_s)

    d_rem = max(0.0, spec.length_m - mean_s_m)
    if d_rem <= 1e-6:
        return 0.0, float(0.10 + 0.10 * obs_age_s)

    v0 = max(0.0, mean_v_mps)
    vmax = max(0.5, spec.speed_limit_mps)
    acc = max(0.2, comfort_accel_mps2)

    if v0 >= vmax:
        eta_s = d_rem / max(v0, 0.5)
    else:
        d_acc = (vmax * vmax - v0 * v0) / (2.0 * acc)
        if d_rem <= d_acc:
            eta_s = (math.sqrt(max(0.0, v0 * v0 + 2.0 * acc * d_rem)) - v0) / acc
        else:
            t_acc = (vmax - v0) / acc
            eta_s = t_acc + (d_rem - d_acc) / vmax

    var_s = max(0.0, float(cov_2x2[0, 0]))
    var_v = max(0.0, float(cov_2x2[1, 1]))
    v_eff = max(v0, 2.5)
    dt_ds_var = var_s / (v_eff * v_eff)
    dt_dv_var = ((d_rem / (v_eff * v_eff)) ** 2) * var_v
    eta_std = math.sqrt(dt_ds_var + dt_dv_var + (0.15 * obs_age_s) ** 2 + 1e-4)
    return float(eta_s), float(eta_std)


def _compute_tail_blockage_score(
    spec: CorridorSpec,
    in_junction_id: str | None,
    mean_s_m: float,
    mean_v_mps: float,
    cov_2x2: np.ndarray,
) -> float:
    """Compute smooth observable upstream-exit tail blockage score in ``[0, 1]``."""
    if in_junction_id is not None:
        return 1.0
    if spec.is_entry_fringe:
        return 0.0
    var_s = max(0.0, float(cov_2x2[0, 0]))
    var_v = max(0.0, float(cov_2x2[1, 1]))
    p_near_exit = _std_normal_cdf((22.0 - mean_s_m) / math.sqrt(var_s + 1.0))
    p_halted = _std_normal_cdf((2.0 - mean_v_mps) / math.sqrt(var_v + 0.25))
    return float(np.clip(p_near_exit * p_halted, 0.0, 1.0))


@dataclass
class _MutableVehicleTrack:
    veh_id: str
    entered_at: float
    corridor_id: str
    edge_id: str
    in_junction_id: str | None
    lane_index: int
    length_m: float
    width_m: float
    mean_s_m: float
    mean_v_mps: float
    cov_2x2: np.ndarray
    accel_est_mps2: float
    last_meas_time: float
    obs_age_s: float
    wait_time_s: float
    turn_probs: np.ndarray
    leader_id: str | None
    noisy_leader_gap_m: float | None
    noisy_leader_speed_mps: float | None
    last_v2x_poll_time: float


def _uniform_turn_prior(k_turns: int) -> np.ndarray:
    if k_turns <= 0:
        return np.zeros((0,), dtype=np.float64)
    return np.full((k_turns,), 1.0 / float(k_turns), dtype=np.float64)


class CityBeliefTracker:
    """Multi-corridor 2D Joseph-form Kalman + OOSM lag-inflated + Cat(K_e) turn belief tracker."""

    def __init__(
        self,
        topology: RoadTopology,
        observation_config: ObservationConfig | None = None,
        *,
        config: ObservationConfig | None = None,
        prediction_horizon_s: float = 6.0,
    ) -> None:
        if observation_config is not None and config is not None and observation_config != config:
            raise NumericalInvariantError(
                "Conflicting observation_config and config arguments passed to CityBeliefTracker."
            )
        self._topology = topology
        self._config = observation_config or config or ObservationConfig()
        require_finite(prediction_horizon_s, "CityBeliefTracker.prediction_horizon_s")
        if prediction_horizon_s <= 0.0:
            raise NumericalInvariantError(
                f"prediction_horizon_s must be > 0, got {prediction_horizon_s}.",
                context={"prediction_horizon_s": prediction_horizon_s},
            )
        self._prediction_horizon_s = float(prediction_horizon_s)
        self._tracks: dict[str, _MutableVehicleTrack] = {}
        self._last_sim_time: float = 0.0

    @property
    def topology(self) -> RoadTopology:
        return self._topology

    def reset(self) -> None:
        """Clear all tracked vehicles at episode boundary."""
        self._tracks.clear()
        self._last_sim_time = 0.0

    def _register_track(
        self,
        *,
        veh_id: str,
        sim_time: float,
        corridor_id: str,
        edge_id: str,
        in_junction_id: str | None,
        lane_index: int,
        length_m: float = 4.6,
        width_m: float = 1.85,
        init_pos_m: float | None = None,
    ) -> _MutableVehicleTrack:
        spec = self._topology.corridors.get(corridor_id)
        if spec is None:
            raise TopologyIntegrityError(
                f"Unknown corridor_id {corridor_id!r} during track registration for {veh_id!r}.",
                context={"veh_id": veh_id, "corridor_id": corridor_id},
            )
        s0 = float(init_pos_m) if init_pos_m is not None else float(
            self._topology.edge_offset_in_corridor.get(edge_id, 0.0)
        )
        s0 = float(np.clip(s0, 0.0, spec.length_m))
        v0 = 0.85 * spec.speed_limit_mps
        cov0 = np.array([[4.0, 0.0], [0.0, 4.0]], dtype=np.float64)
        k_turns = len(spec.turn_options)
        track = _MutableVehicleTrack(
            veh_id=veh_id,
            entered_at=float(sim_time),
            corridor_id=corridor_id,
            edge_id=edge_id,
            in_junction_id=in_junction_id,
            lane_index=int(lane_index),
            length_m=float(length_m),
            width_m=float(width_m),
            mean_s_m=s0,
            mean_v_mps=float(v0),
            cov_2x2=cov0,
            accel_est_mps2=0.0,
            last_meas_time=-1.0,
            obs_age_s=0.0,
            wait_time_s=0.0,
            turn_probs=_uniform_turn_prior(k_turns),
            leader_id=None,
            noisy_leader_gap_m=None,
            noisy_leader_speed_mps=None,
            last_v2x_poll_time=-1e9,
        )
        self._tracks[veh_id] = track
        return track

    def _transition_track_corridor(
        self,
        track: _MutableVehicleTrack,
        new_corridor_id: str,
        new_edge_id: str,
        in_junction_id: str | None,
        lane_index: int,
        hint_pos_m: float,
    ) -> CorridorSpec:
        spec = self._topology.corridors.get(new_corridor_id)
        if spec is None:
            raise TopologyIntegrityError(
                f"Observation references unknown corridor {new_corridor_id!r} for {track.veh_id!r}.",
                context={"veh_id": track.veh_id, "corridor_id": new_corridor_id},
            )
        track.corridor_id = new_corridor_id
        track.edge_id = new_edge_id
        track.in_junction_id = in_junction_id
        track.lane_index = int(lane_index)
        track.mean_s_m = float(np.clip(hint_pos_m, 0.0, spec.length_m))
        track.cov_2x2 = np.array(
            [[max(float(track.cov_2x2[0, 0]), 2.5), 0.0], [0.0, max(float(track.cov_2x2[1, 1]), 1.5)]],
            dtype=np.float64,
        )
        track.turn_probs = _uniform_turn_prior(len(spec.turn_options))
        track.leader_id = None
        track.noisy_leader_gap_m = None
        track.noisy_leader_speed_mps = None
        return spec

    def _apply_measurement(
        self,
        track: _MutableVehicleTrack,
        *,
        meas_time: float,
        now_time: float,
        corridor_id: str,
        edge_id: str,
        in_junction_id: str | None,
        lane_index: int,
        obs_pos_m: float,
        obs_speed_mps: float,
        pos_std: float,
        speed_std: float,
        turn_likelihood: tuple[float, ...],
        is_radar: bool,
        leader_id: str | None = None,
        noisy_leader_gap_m: float | None = None,
        noisy_leader_speed_mps: float | None = None,
    ) -> None:
        require_finite(meas_time, "measurement.meas_time")
        require_finite(obs_pos_m, "measurement.obs_pos_m")
        require_finite(obs_speed_mps, "measurement.obs_speed_mps")
        require_finite(pos_std, "measurement.pos_std")
        require_finite(speed_std, "measurement.speed_std")

        # Out-of-sequence packet older than most recently fused measurement is discarded cleanly
        if meas_time < track.last_meas_time - 1e-9:
            return

        lag = max(0.0, float(now_time - meas_time))

        if corridor_id != track.corridor_id:
            # Only transition corridor if measurement is at least as recent as current track state
            spec = self._transition_track_corridor(
                track, corridor_id, edge_id, in_junction_id, lane_index, obs_pos_m
            )
        else:
            spec = self._topology.corridors.get(corridor_id)
            if spec is None:
                raise TopologyIntegrityError(
                    f"Unknown corridor {corridor_id!r} for vehicle {track.veh_id!r}.",
                    context={"veh_id": track.veh_id, "corridor_id": corridor_id},
                )
            track.edge_id = edge_id
            if is_radar or lag <= 1e-6:
                track.in_junction_id = in_junction_id
            track.lane_index = int(lane_index)

        # Extrapolate OOSM observation forward by lag to current filter clock now_time
        z_pos = float(np.clip(obs_pos_m + max(0.0, obs_speed_mps) * lag, 0.0, spec.length_m))
        z_vel = max(0.0, float(obs_speed_mps))
        z = np.array([z_pos, z_vel], dtype=np.float64)

        f_lag = np.array([[1.0, lag], [0.0, 1.0]], dtype=np.float64)
        r_base = np.array([[pos_std * pos_std, 0.0], [0.0, speed_std * speed_std]], dtype=np.float64)
        # OOSM lag process-noise inflation ensures strictly larger R_eff for delayed packets (lag > 0)
        q_pos = self._config.kalman_process_pos_var
        q_vel = self._config.kalman_process_vel_var
        accel_var = q_vel
        q_lag = np.array(
            [
                [0.25 * accel_var * (lag**4) + q_pos * lag, 0.5 * accel_var * (lag**3)],
                [0.5 * accel_var * (lag**3), accel_var * (lag**2) + q_vel * lag],
            ],
            dtype=np.float64,
        )
        r_eff = f_lag @ r_base @ f_lag.T + q_lag

        prior_cov = track.cov_2x2
        prior_mean = np.array([track.mean_s_m, track.mean_v_mps], dtype=np.float64)
        prev_v = track.mean_v_mps
        prev_t = track.last_meas_time

        s_innov = prior_cov + r_eff
        # Solve S^T K^T = Sigma^T for K = Sigma S^{-1}
        kalman_gain = np.linalg.solve(s_innov.T, prior_cov.T).T
        updated_mean = prior_mean + kalman_gain @ (z - prior_mean)

        eye2 = np.eye(2, dtype=np.float64)
        imk = eye2 - kalman_gain
        # Exact Joseph-form covariance update guarantees symmetry & positive semi-definiteness
        updated_cov = imk @ prior_cov @ imk.T + kalman_gain @ r_eff @ kalman_gain.T
        updated_cov = 0.5 * (updated_cov + updated_cov.T)
        require_psd_covariance(updated_cov, f"VehicleBelief[{track.veh_id}].cov_2x2")

        vmax_cap = 1.5 * spec.speed_limit_mps
        new_s = float(np.clip(updated_mean[0], 0.0, spec.length_m))
        new_v = float(np.clip(updated_mean[1], 0.0, vmax_cap))
        if z_vel <= 0.05 and new_v < 0.12:
            new_v = 0.0

        if prev_t >= 0.0 and meas_time > prev_t + 1e-6:
            dt_obs = meas_time - prev_t
            raw_acc = (new_v - prev_v) / max(dt_obs, 0.25)
            track.accel_est_mps2 = float(np.clip(0.5 * track.accel_est_mps2 + 0.5 * raw_acc, -8.5, 5.0))

        track.mean_s_m = new_s
        track.mean_v_mps = new_v
        track.cov_2x2 = updated_cov
        track.last_meas_time = float(meas_time)
        track.obs_age_s = float(lag)

        # Degree-agnostic Cat(K_e) Bayesian turn update
        k_turns = len(spec.turn_options)
        if k_turns == 0:
            track.turn_probs = np.zeros((0,), dtype=np.float64)
        elif k_turns == 1:
            track.turn_probs = np.ones((1,), dtype=np.float64)
        elif len(turn_likelihood) == k_turns:
            like = np.asarray(turn_likelihood, dtype=np.float64)
            require_probability_simplex(like, f"turn_likelihood[{track.veh_id}]")
            forget = self._config.intent_forgetting_rate
            if track.turn_probs.shape != (k_turns,):
                prior_pi = _uniform_turn_prior(k_turns)
            else:
                prior_pi = (1.0 - forget) * track.turn_probs + (forget / float(k_turns))
            unnorm = prior_pi * like
            denom = float(unnorm.sum())
            if denom <= 1e-12:
                raise NumericalInvariantError(
                    f"Degenerate zero normalization in turn belief update for {track.veh_id!r}.",
                    context={"veh_id": track.veh_id, "prior": prior_pi.tolist(), "like": like.tolist()},
                )
            post_pi = unnorm / denom
            require_probability_simplex(post_pi, f"VehicleBelief[{track.veh_id}].turn_probs")
            track.turn_probs = post_pi

        if is_radar:
            track.leader_id = leader_id
            track.noisy_leader_gap_m = noisy_leader_gap_m
            track.noisy_leader_speed_mps = noisy_leader_speed_mps

    def observe(self, obs_batch: ObservationBatch) -> SwarmBeliefState:
        """Fuse entry/exit detectors, roadside radar, and delayed V2X packets into ``SwarmBeliefState``."""
        now = float(obs_batch.sim_time)
        require_finite(now, "obs_batch.sim_time")
        require_finite(obs_batch.dt, "obs_batch.dt")
        self._last_sim_time = now

        # 1. Remove arrived / departed-from-network vehicles
        for arrived_id in obs_batch.arrived_veh_ids:
            self._tracks.pop(arrived_id, None)

        active_set = set(obs_batch.active_veh_ids)
        for stale_id in [vid for vid in self._tracks if vid not in active_set]:
            del self._tracks[stale_id]

        # 2. Register newly entered vehicles (enforcing #13 §1 entry_sim_time == sim_time firewall!)
        for dep in obs_batch.departed:
            if abs(dep.entry_sim_time - now) > 1e-6:
                raise CausalityFirewallError(
                    f"DepartedVehicleEvent for {dep.veh_id!r} carries entry_sim_time={dep.entry_sim_time} "
                    f"!= obs_batch.sim_time={now} (hidden departDelay/requested_at leak prohibited).",
                    context={
                        "veh_id": dep.veh_id,
                        "entry_sim_time": dep.entry_sim_time,
                        "sim_time": now,
                    },
                )
            cid = self._topology.edge_to_corridor.get(dep.entry_edge_id)
            if cid is None:
                raise TopologyIntegrityError(
                    f"DepartedVehicleEvent for {dep.veh_id!r} references unknown entry_edge_id {dep.entry_edge_id!r}.",
                    context={"veh_id": dep.veh_id, "entry_edge_id": dep.entry_edge_id},
                )
            if dep.veh_id not in self._tracks and dep.veh_id in active_set:
                self._register_track(
                    veh_id=dep.veh_id,
                    sim_time=now,
                    corridor_id=cid,
                    edge_id=dep.entry_edge_id,
                    in_junction_id=None,
                    lane_index=dep.entry_lane_index,
                    length_m=dep.length_m,
                    width_m=dep.width_m,
                )

        # Ensure any synthetic test vehicle in radar_obs/v2x_packets without prior departure is registered
        for robs in obs_batch.radar_obs:
            if robs.veh_id in active_set and robs.veh_id not in self._tracks:
                self._register_track(
                    veh_id=robs.veh_id,
                    sim_time=now,
                    corridor_id=robs.corridor_id,
                    edge_id=robs.edge_id,
                    in_junction_id=robs.in_junction_id,
                    lane_index=robs.lane_index,
                    init_pos_m=robs.noisy_corridor_pos_m,
                )
        for vpkt in obs_batch.v2x_packets:
            if vpkt.veh_id in active_set and vpkt.veh_id not in self._tracks:
                self._register_track(
                    veh_id=vpkt.veh_id,
                    sim_time=now,
                    corridor_id=vpkt.corridor_id,
                    edge_id=vpkt.edge_id,
                    in_junction_id=None,
                    lane_index=vpkt.lane_index,
                    init_pos_m=vpkt.noisy_corridor_pos_m,
                )

        # Clear stale single-step radar leader observations for vehicles not seen by radar this step
        radar_seen_ids = {r.veh_id for r in obs_batch.radar_obs}
        for vid, trk in self._tracks.items():
            if vid not in radar_seen_ids:
                trk.leader_id = None
                trk.noisy_leader_gap_m = None
                trk.noisy_leader_speed_mps = None

        # 3. Sort all measurements chronologically by measurement timestamp (sent_time / obs_time)
        combined_events: list[tuple[float, int, object]] = []
        for idx, pkt in enumerate(obs_batch.v2x_packets):
            combined_events.append((float(pkt.sent_time), idx, pkt))
        base_idx = len(combined_events)
        for idx, robs in enumerate(obs_batch.radar_obs):
            combined_events.append((float(robs.obs_time), base_idx + idx, robs))
        combined_events.sort(key=lambda item: (item[0], item[1]))

        for _, _, ev in combined_events:
            if hasattr(ev, "in_junction_id"):
                # NoisyVehicleObservation
                robs = ev  # type: ignore[assignment]
                trk = self._tracks.get(robs.veh_id)
                if trk is None:
                    continue
                self._apply_measurement(
                    trk,
                    meas_time=robs.obs_time,
                    now_time=now,
                    corridor_id=robs.corridor_id,
                    edge_id=robs.edge_id,
                    in_junction_id=robs.in_junction_id,
                    lane_index=robs.lane_index,
                    obs_pos_m=robs.noisy_corridor_pos_m,
                    obs_speed_mps=robs.noisy_speed_mps,
                    pos_std=robs.pos_noise_std,
                    speed_std=robs.speed_noise_std,
                    turn_likelihood=robs.turn_likelihood,
                    is_radar=True,
                    leader_id=robs.leader_id,
                    noisy_leader_gap_m=robs.noisy_leader_gap_m,
                    noisy_leader_speed_mps=robs.noisy_leader_speed_mps,
                )
            else:
                # V2XPacket
                vpkt = ev  # type: ignore[assignment]
                trk = self._tracks.get(vpkt.veh_id)
                if trk is None:
                    continue
                self._apply_measurement(
                    trk,
                    meas_time=vpkt.sent_time,
                    now_time=now,
                    corridor_id=vpkt.corridor_id,
                    edge_id=vpkt.edge_id,
                    in_junction_id=trk.in_junction_id if vpkt.corridor_id == trk.corridor_id else None,
                    lane_index=vpkt.lane_index,
                    obs_pos_m=vpkt.noisy_corridor_pos_m,
                    obs_speed_mps=vpkt.noisy_speed_mps,
                    pos_std=vpkt.pos_noise_std,
                    speed_std=vpkt.speed_noise_std,
                    turn_likelihood=vpkt.turn_likelihood,
                    is_radar=False,
                )

        # 4. Build immutable VehicleBelief snapshot
        vehicles_out: dict[str, VehicleBelief] = {}
        vehs_by_corridor: dict[str, list[VehicleBelief]] = {cid: [] for cid in self._topology.corridors}

        for vid in sorted(self._tracks):
            trk = self._tracks[vid]
            spec = self._topology.corridors[trk.corridor_id]
            free_eta, eta_std = _compute_free_eta_and_std(
                spec, trk.in_junction_id, trk.mean_s_m, trk.mean_v_mps, trk.cov_2x2, trk.obs_age_s
            )
            b_tail = _compute_tail_blockage_score(
                spec, trk.in_junction_id, trk.mean_s_m, trk.mean_v_mps, trk.cov_2x2
            )
            require_probability_range(b_tail, f"VehicleBelief[{vid}].tail_blockage_score")
            cov_copy = trk.cov_2x2.copy()
            cov_copy.setflags(write=False)

            vb = VehicleBelief(
                veh_id=vid,
                entered_at=trk.entered_at,
                corridor_id=trk.corridor_id,
                edge_id=trk.edge_id,
                in_junction_id=trk.in_junction_id,
                lane_index=trk.lane_index,
                length_m=trk.length_m,
                width_m=trk.width_m,
                mean_s_m=trk.mean_s_m,
                mean_v_mps=trk.mean_v_mps,
                cov_2x2=cov_copy,
                accel_est_mps2=trk.accel_est_mps2,
                obs_age_s=trk.obs_age_s,
                wait_time_s=trk.wait_time_s,
                turn_probs=tuple(float(p) for p in trk.turn_probs.tolist()),
                free_eta_s=free_eta,
                eta_std_s=eta_std,
                tail_blockage_score=b_tail,
                leader_id=trk.leader_id,
                noisy_leader_gap_m=trk.noisy_leader_gap_m,
                noisy_leader_speed_mps=trk.noisy_leader_speed_mps,
                last_v2x_poll_time=trk.last_v2x_poll_time,
            )
            vehicles_out[vid] = vb
            vehs_by_corridor[trk.corridor_id].append(vb)

        # 5. Aggregate observable CorridorBeliefStats (expected_inflow strictly from upstream belief ETAs)
        inflow_accum: dict[str, float] = {cid: 0.0 for cid in self._topology.corridors}
        for cid, spec in self._topology.corridors.items():
            if not spec.turn_options:
                continue
            for vb in vehs_by_corridor[cid]:
                if vb.in_junction_id is None and vb.free_eta_s <= self._prediction_horizon_s:
                    for opt, prob in zip(spec.turn_options, vb.turn_probs, strict=True):
                        inflow_accum[opt.to_corridor] += float(prob)

        corridors_out: dict[str, CorridorBeliefStats] = {}
        for cid, spec in self._topology.corridors.items():
            c_vehs = vehs_by_corridor[cid]
            eff_lanes = self._topology.effective_lanes(cid, obs_batch.open_lanes_by_edge)
            cap = self._topology.effective_capacity(cid, obs_batch.open_lanes_by_edge)
            count = len(c_vehs)
            occ = float(np.clip(count / float(max(1, cap)), 0.0, 1.0))

            if spec.is_entry_fringe or not c_vehs:
                tail_prob = 0.0
            else:
                surv = 1.0
                for vb in c_vehs:
                    if vb.in_junction_id is None:
                        surv *= 1.0 - vb.tail_blockage_score
                tail_prob = float(np.clip(1.0 - surv, 0.0, 1.0))

            mean_spd = (
                float(sum(vb.mean_v_mps for vb in c_vehs) / count)
                if count > 0
                else float(spec.speed_limit_mps)
            )
            require_finite(mean_spd, f"CorridorBeliefStats[{cid}].mean_speed_mps")

            corridors_out[cid] = CorridorBeliefStats(
                corridor_id=cid,
                active_count=count,
                effective_lanes=eff_lanes,
                storage_capacity=cap,
                occupancy_ratio=occ,
                tail_blockage_prob=tail_prob,
                expected_inflow=float(inflow_accum[cid]),
                mean_speed_mps=mean_spd,
            )

        return SwarmBeliefState(
            sim_time=now,
            dt=float(obs_batch.dt),
            vehicles=vehicles_out,
            corridors=corridors_out,
            open_lanes_by_edge=dict(obs_batch.open_lanes_by_edge),
            tls_states=dict(obs_batch.tls_states),
        )

    def predict(
        self,
        dt: float,
        post_safety_accels: Mapping[str, float],
        v2x_polled_ids: tuple[str, ...] = (),
        sim_time: float | None = None,
    ) -> None:
        """Propagate 2D kinematic state & covariance forward using post-safety accelerations."""
        require_finite(dt, "CityBeliefTracker.predict.dt")
        if dt <= 0.0:
            raise NumericalInvariantError(
                f"CityBeliefTracker.predict requires dt > 0, got {dt}.",
                context={"dt": dt},
            )

        poll_time = float(sim_time) if sim_time is not None else float(self._last_sim_time)
        polled_set = set(v2x_polled_ids)

        f_mat = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        b_vec = np.array([0.5 * dt * dt, dt], dtype=np.float64)
        q_pos = self._config.kalman_process_pos_var
        q_vel = self._config.kalman_process_vel_var
        q_mat = np.outer(b_vec, b_vec) * q_vel + np.array(
            [[q_pos * dt, 0.0], [0.0, 0.25 * q_vel * dt]], dtype=np.float64
        )

        for vid, trk in self._tracks.items():
            if vid in polled_set:
                trk.last_v2x_poll_time = poll_time

            a_cmd = float(post_safety_accels.get(vid, 0.0))
            require_finite(a_cmd, f"post_safety_accels[{vid}]")
            spec = self._topology.corridors[trk.corridor_id]
            vmax_cap = 1.5 * spec.speed_limit_mps

            # Stopped-queue deadzone (#8 §3.3a / D6): prevent clipped-Kalman forward position creep
            if trk.mean_v_mps < 0.15 and a_cmd <= 0.0:
                new_v = 0.0
                new_s = trk.mean_s_m
                trk.accel_est_mps2 = 0.0
            else:
                new_v = float(np.clip(trk.mean_v_mps + a_cmd * dt, 0.0, vmax_cap))
                if new_v < 0.05 and a_cmd < 0.0:
                    new_v = 0.0
                avg_v = 0.5 * (trk.mean_v_mps + new_v)
                new_s = float(np.clip(trk.mean_s_m + avg_v * dt, 0.0, spec.length_m))
                trk.accel_est_mps2 = float(np.clip(0.6 * trk.accel_est_mps2 + 0.4 * a_cmd, -8.5, 5.0))

            pred_cov = f_mat @ trk.cov_2x2 @ f_mat.T + q_mat
            pred_cov = 0.5 * (pred_cov + pred_cov.T)
            require_psd_covariance(pred_cov, f"VehicleBelief[{vid}].pred_cov_2x2")

            trk.mean_s_m = new_s
            trk.mean_v_mps = new_v
            trk.cov_2x2 = pred_cov
            trk.obs_age_s += dt
            if new_v < 1.0:
                trk.wait_time_s += dt


__all__ = [
    "CityBeliefTracker",
]

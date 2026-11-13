"""Range-limited noisy roadside radar, turn-signal likelihoods, and delayed/dropped V2X channel.

Firewall note: This module does NOT import ``libsumo``, ``traci``, ``VehicleRawTelemetry``,
``GroundTruthStepRecord``, or ``EpisodeMetrics``. All random noise and dropout realizations are
keyed deterministically by ``(episode_seed, stream_id, step_idx, crc32(veh_id))`` so policies
evaluated on identical seeds experience paired observation noise on shared vehicle states.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import zlib

import numpy as np

from ..config import ObservationConfig
from ..contracts import (
    CorridorSpec,
    DepartedVehicleEvent,
    LEFT_TURN_THRESHOLD_RAD,
    NoisyVehicleObservation,
    ObservationBatch,
    RoadTopology,
    V2XPacket,
)
from ..logging_utils import require_finite, require_probability_simplex


@dataclass(frozen=True)
class ObservableSourceVehicle:
    """Minimal physical sensor-target input supplied by ``SumoSimulationAdapter`` before noise."""

    veh_id: str
    edge_id: str
    corridor_id: str
    in_junction_id: str | None
    lane_index: int
    corridor_pos_m: float
    speed_mps: float
    turn_signal: int
    true_turn_idx: int | None
    leader_id: str | None
    leader_gap_m: float | None
    leader_speed_mps: float | None


def _veh_crc32(veh_id: str) -> int:
    return int(zlib.crc32(veh_id.encode("utf-8")) & 0xFFFFFFFF)


def _sample_turn_likelihood(
    rng: np.random.Generator,
    spec: CorridorSpec,
    true_turn_idx: int | None,
    turn_signal: int,
    base_accuracy: float,
    *,
    is_v2x: bool,
) -> tuple[float, ...]:
    """Sample a calibrated categorical likelihood vector over ``spec.turn_options``."""
    k_turns = len(spec.turn_options)
    if k_turns == 0:
        return ()
    if k_turns == 1:
        return (1.0,)

    acc = float(base_accuracy)
    if not is_v2x and turn_signal != 0:
        # Active physical turn blinker boosts roadside camera turn-intent confidence
        acc = min(0.94, max(acc, 0.86))
    acc = float(np.clip(acc, 1.0 / k_turns + 1e-3 if k_turns > 1 else 1.0, 0.995))

    # Determine underlying target turn index to perturb with sensor confusion
    if true_turn_idx is not None and 0 <= true_turn_idx < k_turns:
        target_idx = true_turn_idx
    elif turn_signal != 0:
        # Match blinker direction (+1 left => angle > +20deg; -1 right => angle < -20deg)
        matching = [
            t.turn_idx
            for t in spec.turn_options
            if (turn_signal > 0 and t.angle_rad > LEFT_TURN_THRESHOLD_RAD)
            or (turn_signal < 0 and t.angle_rad < -LEFT_TURN_THRESHOLD_RAD)
        ]
        target_idx = matching[int(rng.integers(len(matching)))] if matching else int(rng.integers(k_turns))
    else:
        target_idx = int(rng.integers(k_turns))

    if float(rng.random()) < acc:
        observed_idx = target_idx
    else:
        other_indices = [idx for idx in range(k_turns) if idx != target_idx]
        observed_idx = other_indices[int(rng.integers(len(other_indices)))]

    off_prob = (1.0 - acc) / float(k_turns - 1)
    probs = [acc if idx == observed_idx else off_prob for idx in range(k_turns)]
    total = sum(probs)
    normalized = tuple(float(p / total) for p in probs)
    require_probability_simplex(normalized, f"turn_likelihood[{spec.corridor_id}]")
    return normalized


class ObservationChannel:
    """Simulates range-limited Gaussian radar, turn-blinker sensing, and delayed V2X packets."""

    def __init__(
        self,
        config: ObservationConfig,
        topology: RoadTopology,
        seed: int = 0,
    ) -> None:
        self._config = config
        self._topology = topology
        self._seed = int(seed)
        self._in_flight_v2x: list[tuple[int, V2XPacket]] = []
        self._last_poll_time: dict[str, float] = {}
        self.total_v2x_sent: int = 0
        self.total_v2x_delivered: int = 0
        self.total_v2x_dropped: int = 0

    def reset(self, seed: int) -> None:
        """Clear in-flight V2X packets and re-seed deterministic observation streams."""
        self._seed = int(seed)
        self._in_flight_v2x.clear()
        self._last_poll_time.clear()
        self.total_v2x_sent = 0
        self.total_v2x_delivered = 0
        self.total_v2x_dropped = 0

    def enqueue_v2x_polls(
        self,
        polled_veh_ids: Sequence[str],
        *,
        sim_time: float,
        step_idx: int,
        dt: float,
        source_by_id: Mapping[str, ObservableSourceVehicle],
    ) -> int:
        """Process budgeted V2X queries for step ``step_idx`` and queue non-dropped packets."""
        if self._config.v2x_budget_per_step <= 0 or not polled_veh_ids:
            return 0

        deduped = list(dict.fromkeys(polled_veh_ids))
        sent_this_step = 0

        for vid in deduped:
            if sent_this_step >= self._config.v2x_budget_per_step:
                break
            src = source_by_id.get(vid)
            if src is None:
                continue
            last_t = self._last_poll_time.get(vid, -1e9)
            if sim_time - last_t < self._config.v2x_poll_cooldown_s - 1e-9:
                continue

            self._last_poll_time[vid] = sim_time
            sent_this_step += 1
            self.total_v2x_sent += 1

            rng = np.random.default_rng(
                np.random.SeedSequence([self._seed, 303, step_idx, _veh_crc32(vid)])
            )
            if float(rng.random()) < self._config.v2x_dropout_prob:
                self.total_v2x_dropped += 1
                continue

            spec = self._topology.corridors[src.corridor_id]
            delay_steps = 1 + int(rng.integers(self._config.v2x_max_delay_steps + 1))
            due_step = step_idx + delay_steps
            recv_time = float(sim_time + delay_steps * dt)

            noisy_pos = float(
                np.clip(
                    src.corridor_pos_m + float(rng.normal(0.0, self._config.v2x_pos_noise_std_m)),
                    0.0,
                    spec.length_m,
                )
            )
            noisy_spd = max(
                0.0,
                float(src.speed_mps + float(rng.normal(0.0, self._config.v2x_speed_noise_std_mps))),
            )
            if src.speed_mps < 0.05 and noisy_spd < 0.20:
                noisy_spd = 0.0

            turn_like = _sample_turn_likelihood(
                rng,
                spec,
                src.true_turn_idx,
                src.turn_signal,
                self._config.v2x_turn_accuracy,
                is_v2x=True,
            )
            pkt = V2XPacket(
                veh_id=vid,
                sent_time=float(sim_time),
                received_time=recv_time,
                edge_id=src.edge_id,
                corridor_id=src.corridor_id,
                lane_index=src.lane_index,
                noisy_corridor_pos_m=noisy_pos,
                noisy_speed_mps=noisy_spd,
                pos_noise_std=self._config.v2x_pos_noise_std_m,
                speed_noise_std=self._config.v2x_speed_noise_std_mps,
                turn_likelihood=turn_like,
            )
            self._in_flight_v2x.append((due_step, pkt))

        return sent_this_step

    def step_observations(
        self,
        *,
        sim_time: float,
        step_idx: int,
        dt: float,
        active_veh_ids: Sequence[str],
        departed: Sequence[DepartedVehicleEvent],
        arrived_veh_ids: Sequence[str],
        source_by_id: Mapping[str, ObservableSourceVehicle],
        open_lanes_by_edge: Mapping[str, int],
        tls_states: Mapping[str, str],
    ) -> tuple[ObservationBatch, int]:
        """Deliver due V2X packets and sample range-limited roadside radar measurements."""
        require_finite(sim_time, "ObservationChannel.sim_time")
        require_finite(dt, "ObservationChannel.dt")

        active_set = set(active_veh_ids)
        # Prune departed/arrived poll history cleanly
        for stale_id in [k for k in self._last_poll_time if k not in active_set]:
            del self._last_poll_time[stale_id]

        delivered_packets: list[V2XPacket] = []
        remaining_flight: list[tuple[int, V2XPacket]] = []
        for due_step, pkt in self._in_flight_v2x:
            if due_step <= step_idx:
                if pkt.veh_id in active_set:
                    delivered_packets.append(pkt)
                    self.total_v2x_delivered += 1
            else:
                if pkt.veh_id in active_set:
                    remaining_flight.append((due_step, pkt))
        self._in_flight_v2x = remaining_flight

        radar_list: list[NoisyVehicleObservation] = []
        r_range = self._config.radar_range_m
        exit_tail_range = min(25.0, 0.5 * r_range)

        for vid in active_veh_ids:
            src = source_by_id[vid]
            spec = self._topology.corridors[src.corridor_id]
            dist_to_stopline = max(0.0, spec.length_m - src.corridor_pos_m)

            in_range = (
                src.in_junction_id is not None
                or dist_to_stopline <= r_range
                or (not spec.is_entry_fringe and src.corridor_pos_m <= exit_tail_range)
            )
            if not in_range:
                continue

            rng = np.random.default_rng(
                np.random.SeedSequence([self._seed, 202, step_idx, _veh_crc32(vid)])
            )
            if float(rng.random()) < self._config.sensor_dropout_prob:
                continue

            pos_std = self._config.pos_noise_std_m
            spd_std = self._config.speed_noise_std_mps
            noisy_pos = float(
                np.clip(src.corridor_pos_m + float(rng.normal(0.0, pos_std)), 0.0, spec.length_m)
            )
            noisy_spd = max(0.0, float(src.speed_mps + float(rng.normal(0.0, spd_std))))
            # Preserve exact stopped-vehicle zero-speed floor when true speed is zero & noise is downward
            if src.speed_mps < 0.05 and noisy_spd < 0.15:
                noisy_spd = 0.0

            turn_like = _sample_turn_likelihood(
                rng,
                spec,
                src.true_turn_idx,
                src.turn_signal,
                self._config.turn_sensor_accuracy,
                is_v2x=False,
            )

            noisy_gap: float | None = None
            noisy_lead_spd: float | None = None
            lead_id: str | None = None
            if (
                src.leader_id is not None
                and src.leader_gap_m is not None
                and src.leader_speed_mps is not None
                and src.leader_gap_m <= r_range
                and math.isfinite(src.leader_gap_m)
            ):
                lead_id = src.leader_id
                gap_sigma = math.sqrt(2.0) * 0.5 * pos_std
                noisy_gap = max(0.1, float(src.leader_gap_m + float(rng.normal(0.0, gap_sigma))))
                noisy_lead_spd = max(
                    0.0, float(src.leader_speed_mps + float(rng.normal(0.0, spd_std)))
                )

            radar_list.append(
                NoisyVehicleObservation(
                    veh_id=vid,
                    obs_time=float(sim_time),
                    edge_id=src.edge_id,
                    corridor_id=src.corridor_id,
                    in_junction_id=src.in_junction_id,
                    lane_index=src.lane_index,
                    noisy_corridor_pos_m=noisy_pos,
                    noisy_speed_mps=noisy_spd,
                    pos_noise_std=pos_std,
                    speed_noise_std=spd_std,
                    turn_likelihood=turn_like,
                    noisy_leader_gap_m=noisy_gap,
                    noisy_leader_speed_mps=noisy_lead_spd,
                    leader_id=lead_id,
                )
            )

        batch = ObservationBatch(
            sim_time=float(sim_time),
            dt=float(dt),
            active_veh_ids=tuple(active_veh_ids),
            departed=tuple(departed),
            arrived_veh_ids=tuple(arrived_veh_ids),
            radar_obs=tuple(radar_list),
            v2x_packets=tuple(delivered_packets),
            open_lanes_by_edge=dict(open_lanes_by_edge),
            tls_states=dict(tls_states),
        )
        return batch, len(delivered_packets)


__all__ = [
    "ObservableSourceVehicle",
    "ObservationChannel",
]

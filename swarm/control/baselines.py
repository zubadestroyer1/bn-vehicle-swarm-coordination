"""Baseline traffic controllers: Fixed-Time TLS, Gap-Actuated TLS, and Dresner-Stone FCFS AIM.

Implements :class:`~swarm.contracts.PolicyControllerProtocol` for:
1. ``FixedSignalController`` (``fixed_signal``): Native SUMO fixed-time TLS baseline
   (``tls_desired_greens=None``, ``broadcast_edge_costs=None``).
2. ``ActuatedSignalController`` (``actuated_signal``): Native SUMO gap-actuated TLS baseline
   (``tls_desired_greens=None``, ``broadcast_edge_costs=None``).
3. ``FCFSReservationController`` (``fcfs_reservation``): Multi-intersection Dresner-Stone
   Autonomous Intersection Management (AIM) First-Come-First-Served reservation baseline using
   the exact same filtered :class:`~swarm.contracts.SwarmBeliefState`,
   :class:`~swarm.control.safety.ObservableSafetyFilter`, Eco-Glide speed profiler, and
   shared ``B_comm`` V2X budget as the DBN coordinators.
"""

from __future__ import annotations

import importlib
import time

from ..config import ControlConfig, ObservationConfig
from ..contracts import (
    BeliefTrackerProtocol,
    DBNStateSnapshot,
    LongitudinalMode,
    ObservationBatch,
    RoadTopology,
    SwarmBeliefState,
    SwarmControlAction,
)
from ..logging_utils import (
    ConfigValidationError,
    NumericalInvariantError,
    require_finite,
)
from .ecoglide import compute_ecoglide_speeds
from .safety import ObservableSafetyFilter
from .scheduler import PriorityDAGScheduler, ScheduleResult
from .v2x import VoIV2XSelector


def _build_default_belief_tracker(
    topology: RoadTopology,
    observation_config: ObservationConfig,
    prediction_horizon_s: float = 6.0,
) -> BeliefTrackerProtocol:
    """Lazily instantiate ``swarm.env.CityBeliefTracker`` without eager cross-slice import cycles."""
    belief_mod = importlib.import_module("..env.belief", package=__package__)
    tracker_cls = getattr(belief_mod, "CityBeliefTracker", None)
    if tracker_cls is None:
        raise ConfigValidationError(
            "swarm.env.belief.CityBeliefTracker is unavailable for lazy controller initialization."
        )
    tracker: BeliefTrackerProtocol = tracker_cls(
        topology=topology,
        observation_config=observation_config,
        prediction_horizon_s=prediction_horizon_s,
    )
    return tracker


class _NativeSignalBaselineController:
    """Shared implementation for SUMO native fixed-time and gap-actuated signal baselines."""

    def __init__(
        self,
        policy_id: str,
        topology: RoadTopology,
        control_config: ControlConfig | None = None,
        *,
        belief_tracker: BeliefTrackerProtocol | None = None,
    ) -> None:
        if policy_id not in ("fixed_signal", "actuated_signal"):
            raise ConfigValidationError(
                f"Invalid signal baseline policy_id {policy_id!r}.",
                context={"policy_id": policy_id},
            )
        self._policy_name = policy_id
        self._topology = topology
        self._control_config = control_config or ControlConfig()
        self._belief_tracker = belief_tracker
        self._last_belief_state: SwarmBeliefState | None = None
        self._prev_sim_time: float = 0.0
        self._has_stepped: bool = False

    @property
    def policy_name(self) -> str:
        """Return canonical baseline policy identifier."""
        return self._policy_name

    @property
    def last_dbn_snapshot(self) -> DBNStateSnapshot | None:
        """Return ``None`` (signal baselines do not maintain a DBN graph)."""
        return None

    @property
    def last_belief_state(self) -> SwarmBeliefState | None:
        """Return latest filtered belief state if an optional tracker was attached, else ``None``."""
        return self._last_belief_state

    def reset(self) -> None:
        """Reset episode state and optional belief tracker."""
        if self._belief_tracker is not None:
            self._belief_tracker.reset()
        self._last_belief_state = None
        self._prev_sim_time = 0.0
        self._has_stepped = False

    def act(self, obs_batch: ObservationBatch) -> SwarmControlAction:
        """Emit a pass-through action allowing SUMO's native TLS program and IDM physics to run."""
        t0 = time.perf_counter()
        require_finite(obs_batch.sim_time, "ObservationBatch.sim_time")
        require_finite(obs_batch.dt, "ObservationBatch.dt")
        dt = float(obs_batch.dt)
        if dt <= 0.0:
            raise NumericalInvariantError(
                f"ObservationBatch.dt must be positive, got {dt}.",
                context={"dt": dt, "sim_time": obs_batch.sim_time},
            )

        now = float(obs_batch.sim_time)
        if self._has_stepped and now < self._prev_sim_time - 1e-9:
            raise NumericalInvariantError(
                f"Non-monotonic ObservationBatch.sim_time={now} < prev={self._prev_sim_time} "
                f"in {self._policy_name}; call reset() between episodes.",
                context={"sim_time": now, "prev_sim_time": self._prev_sim_time},
            )

        if self._belief_tracker is not None:
            if self._has_stepped:
                step_dt = now - self._prev_sim_time if now > self._prev_sim_time + 1e-9 else dt
                self._belief_tracker.predict(
                    dt=step_dt,
                    post_safety_accels={},
                    v2x_polled_ids=(),
                    sim_time=self._prev_sim_time,
                )
            self._last_belief_state = self._belief_tracker.observe(obs_batch)

        modes = {vid: LongitudinalMode.CRUISE for vid in sorted(obs_batch.active_veh_ids)}
        self._prev_sim_time = now
        self._has_stepped = True
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return SwarmControlAction(
            sim_time=now,
            target_speeds={},
            post_safety_accels={},
            modes=modes,
            tls_desired_greens=None,
            broadcast_edge_costs=None,
            reroute_hysteresis_s=self._control_config.reroute_hysteresis_s,
            v2x_polls=(),
            safety_clamp_count=0,
            zero_speed_deadzone_count=0,
            control_latency_ms=latency_ms,
        )


class FixedSignalController(_NativeSignalBaselineController):
    """SUMO native fixed-cycle multi-phase traffic light baseline (``fixed_signal``)."""

    def __init__(
        self,
        topology: RoadTopology,
        control_config: ControlConfig | None = None,
        *,
        belief_tracker: BeliefTrackerProtocol | None = None,
    ) -> None:
        super().__init__(
            policy_id="fixed_signal",
            topology=topology,
            control_config=control_config,
            belief_tracker=belief_tracker,
        )


class ActuatedSignalController(_NativeSignalBaselineController):
    """SUMO native gap-actuated traffic light baseline (``actuated_signal``)."""

    def __init__(
        self,
        topology: RoadTopology,
        control_config: ControlConfig | None = None,
        *,
        belief_tracker: BeliefTrackerProtocol | None = None,
    ) -> None:
        super().__init__(
            policy_id="actuated_signal",
            topology=topology,
            control_config=control_config,
            belief_tracker=belief_tracker,
        )


class FCFSReservationController:
    """Multi-intersection Dresner-Stone AIM First-Come-First-Served reservation controller."""

    def __init__(
        self,
        topology: RoadTopology,
        control_config: ControlConfig | None = None,
        observation_config: ObservationConfig | None = None,
        *,
        belief_tracker: BeliefTrackerProtocol | None = None,
        prediction_horizon_s: float = 6.0,
    ) -> None:
        self._topology = topology
        self._control_config = control_config or ControlConfig()
        self._obs_config = observation_config or ObservationConfig()
        self._belief_tracker: BeliefTrackerProtocol = (
            belief_tracker
            if belief_tracker is not None
            else _build_default_belief_tracker(
                topology=topology,
                observation_config=self._obs_config,
                prediction_horizon_s=prediction_horizon_s,
            )
        )
        self._scheduler = PriorityDAGScheduler(self._control_config)
        self._safety_filter = ObservableSafetyFilter(self._control_config)
        self._v2x_selector = VoIV2XSelector(self._obs_config)

        self._last_belief_state: SwarmBeliefState | None = None
        self._last_schedule: ScheduleResult | None = None
        self._prev_post_safety_accels: dict[str, float] = {}
        self._prev_v2x_polls: tuple[str, ...] = ()
        self._prev_sim_time: float = 0.0
        self._has_stepped: bool = False

    @property
    def policy_name(self) -> str:
        """Return canonical policy identifier ``'fcfs_reservation'``."""
        return "fcfs_reservation"

    @property
    def last_dbn_snapshot(self) -> DBNStateSnapshot | None:
        """Return ``None`` (FCFS AIM does not construct a DBN)."""
        return None

    @property
    def last_belief_state(self) -> SwarmBeliefState | None:
        """Return the filtered :class:`SwarmBeliefState` from the most recent ``act`` call."""
        return self._last_belief_state

    @property
    def last_schedule(self) -> ScheduleResult | None:
        """Return the decoded FCFS :class:`ScheduleResult` from the most recent ``act`` call."""
        return self._last_schedule

    def reset(self) -> None:
        """Reset belief filter, TLS minimum-green holds, and V2X cooldowns."""
        self._belief_tracker.reset()
        self._scheduler.reset()
        self._v2x_selector.reset()
        self._last_belief_state = None
        self._last_schedule = None
        self._prev_post_safety_accels = {}
        self._prev_v2x_polls = ()
        self._prev_sim_time = 0.0
        self._has_stepped = False

    def act(self, obs_batch: ObservationBatch) -> SwarmControlAction:
        """Advance belief filter and compute conflict-free FCFS AIM longitudinal & TLS actions."""
        t0 = time.perf_counter()
        require_finite(obs_batch.sim_time, "ObservationBatch.sim_time")
        require_finite(obs_batch.dt, "ObservationBatch.dt")
        dt = float(obs_batch.dt)
        if dt <= 0.0:
            raise NumericalInvariantError(
                f"ObservationBatch.dt must be positive, got {dt}.",
                context={"dt": dt, "sim_time": obs_batch.sim_time},
            )

        now = float(obs_batch.sim_time)
        if self._has_stepped:
            if now < self._prev_sim_time - 1e-9:
                raise NumericalInvariantError(
                    f"Non-monotonic ObservationBatch.sim_time={now} < prev={self._prev_sim_time} "
                    "in FCFSReservationController; call reset() between episodes.",
                    context={"sim_time": now, "prev_sim_time": self._prev_sim_time},
                )
            step_dt = now - self._prev_sim_time if now > self._prev_sim_time + 1e-9 else dt
            self._belief_tracker.predict(
                dt=step_dt,
                post_safety_accels=self._prev_post_safety_accels,
                v2x_polled_ids=self._prev_v2x_polls,
                sim_time=self._prev_sim_time,
            )

        belief = self._belief_tracker.observe(obs_batch)
        self._last_belief_state = belief

        # Pure Dresner-Stone AIM: First-Come-First-Served slot reservation without predictive DBN spillback gate.
        schedule = self._scheduler.decode(
            belief=belief,
            topology=self._topology,
            maxsum_result=None,
            dbn_snapshot=None,
            enable_spillback_gate=False,
        )
        self._last_schedule = schedule

        glide = compute_ecoglide_speeds(
            belief=belief,
            topology=self._topology,
            schedule=schedule,
            control_config=self._control_config,
            base_modes=None,
        )
        safety = self._safety_filter.apply(
            belief=belief,
            topology=self._topology,
            desired_speeds=glide.desired_speeds,
            must_hold_stopline=schedule.must_hold_stopline,
        )
        v2x_polls = self._v2x_selector.select_polls(
            belief=belief,
            topology=self._topology,
            dbn_snapshot=None,
        )

        v_dead = self._control_config.zero_speed_deadzone_mps
        modes: dict[str, LongitudinalMode] = {}
        for vid, mode in glide.effective_modes.items():
            veh = belief.vehicles[vid]
            if veh.in_junction_id is None and safety.target_speeds.get(vid, 0.0) <= v_dead:
                modes[vid] = LongitudinalMode.YIELD
            else:
                modes[vid] = mode

        self._prev_post_safety_accels = dict(safety.post_safety_accels)
        self._prev_v2x_polls = v2x_polls
        self._prev_sim_time = now
        self._has_stepped = True

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return SwarmControlAction(
            sim_time=now,
            target_speeds=safety.target_speeds,
            post_safety_accels=safety.post_safety_accels,
            modes=modes,
            tls_desired_greens=schedule.tls_desired_greens,
            broadcast_edge_costs=None,
            reroute_hysteresis_s=self._control_config.reroute_hysteresis_s,
            v2x_polls=v2x_polls,
            safety_clamp_count=safety.safety_clamp_count,
            zero_speed_deadzone_count=safety.zero_speed_deadzone_count,
            control_latency_ms=latency_ms,
        )


__all__ = [
    "ActuatedSignalController",
    "FCFSReservationController",
    "FixedSignalController",
]

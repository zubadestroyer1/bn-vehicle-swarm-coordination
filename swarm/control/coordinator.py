"""Factored DBN multi-intersection swarm coordinator and unified policy controller factory.

Implements :class:`~swarm.contracts.PolicyControllerProtocol` via :class:`FactoredSwarmCoordinator`
for all 7 DBN-driven policies and ablations:
- ``independent_belief`` (exact same coordinator over zero-edge independent DBN snapshot),
- ``heuristic_radius_dbn`` (unlearned proximity DBN structure),
- ``online_bic_dbn`` (sliding-window per-pair BIC score-based DBN updater),
- ``neural_dbn`` (learned calibrated ``STGraphTransformerDBN`` with temporal prior & dynamic routing),
- ``no_temporal_prior`` (neural DBN ablation removing Markov ``E_{t-1} -> E_t`` & GRU state carry),
- ``static_cpd`` (neural DBN ablation fixing constant non-amortized CPD weights),
- ``fixed_route`` (neural DBN ablation disabling ``broadcast_edge_costs`` dynamic routing).

Also provides :func:`build_policy_controller` to construct any of the 10 pre-registered
:data:`~swarm.contracts.POLICY_NAMES` with lazy cross-slice imports so unit tests run cleanly
without cross-slice import races.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import importlib
import inspect
from pathlib import Path
import time
from typing import Any, Final

from ..config import (
    ControlConfig,
    DBNConfig,
    ExperimentConfig,
    ObservationConfig,
    ScenarioConfig,
)
from ..contracts import (
    BeliefTrackerProtocol,
    DBNStateSnapshot,
    DBNUpdaterProtocol,
    LongitudinalMode,
    NEURAL_POLICY_NAMES,
    ObservationBatch,
    POLICY_NAMES,
    PolicyControllerProtocol,
    RoadTopology,
    SwarmBeliefState,
    SwarmControlAction,
)
from ..logging_utils import (
    CheckpointIntegrityError,
    ConfigValidationError,
    NumericalInvariantError,
    require_finite,
)
from .baselines import (
    ActuatedSignalController,
    FCFSReservationController,
    FixedSignalController,
    _build_default_belief_tracker,
)
from .ecoglide import compute_ecoglide_speeds
from .maxsum import MaxSumResult, run_damped_maxsum
from .routing import DBNRouteCostBroadcaster
from .safety import ObservableSafetyFilter, select_actuated_target_speeds
from .scheduler import PriorityDAGScheduler, ScheduleResult
from .v2x import VoIV2XSelector

DBN_COORDINATOR_POLICY_NAMES: Final[frozenset[str]] = frozenset(
    {
        "independent_belief",
        "heuristic_radius_dbn",
        "online_bic_dbn",
        "neural_dbn",
        "no_temporal_prior",
        "static_cpd",
        "fixed_route",
    }
)


def _bind_and_call(func: Callable[..., Any], available: Mapping[str, Any]) -> Any:
    """Invoke ``func`` by binding matching signature parameter names from ``available``."""
    sig = inspect.signature(func)
    kwargs: dict[str, Any] = {}
    accepts_var_kw = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )
    for pname, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if pname in available:
            kwargs[pname] = available[pname]
        elif param.default is inspect.Parameter.empty and not accepts_var_kw:
            raise ConfigValidationError(
                f"Cannot bind required parameter {pname!r} on {func.__qualname__}; "
                f"available keys={sorted(available)}.",
                context={"function": func.__qualname__, "missing_param": pname},
            )
    return func(**kwargs)


def _build_default_dbn_updater(
    policy_name: str,
    topology: RoadTopology,
    dbn_config: DBNConfig,
    checkpoint_path: Path | str | None,
    exp_config: ExperimentConfig | None,
    seed: int,
) -> DBNUpdaterProtocol:
    """Lazily resolve ``swarm.dbn.build_dbn_updater`` without eager cross-slice imports."""
    if policy_name in NEURAL_POLICY_NAMES and checkpoint_path is None:
        raise CheckpointIntegrityError(
            f"Neural DBN policy {policy_name!r} requires a non-null checkpoint_path when dbn_updater is None.",
            context={"policy_name": policy_name},
        )

    dbn_pkg = importlib.import_module("..dbn", package=__package__)
    builder_fn = getattr(dbn_pkg, "build_dbn_updater", None)
    if builder_fn is None:
        updater_mod = importlib.import_module("..dbn.updater", package=__package__)
        builder_fn = getattr(updater_mod, "build_dbn_updater", None)
    if not callable(builder_fn):
        raise ConfigValidationError(
            "swarm.dbn.build_dbn_updater is unavailable for lazy DBN updater construction.",
            context={"policy_name": policy_name},
        )

    ckpt = Path(checkpoint_path) if checkpoint_path is not None else None
    updater: DBNUpdaterProtocol = _bind_and_call(
        builder_fn,
        {
            "policy_name": policy_name,
            "policy": policy_name,
            "topology": topology,
            "dbn_config": dbn_config,
            "dbn_cfg": dbn_config,
            "config": exp_config if exp_config is not None else dbn_config,
            "exp_cfg": exp_config,
            "checkpoint_path": ckpt,
            "checkpoint": ckpt,
            "device": dbn_config.inference_device,
            "seed": seed,
        },
    )
    return updater


class FactoredSwarmCoordinator:
    """Factored Damped Max-Sum + Priority-DAG + Eco-Glide + VoI V2X + DBN Routing coordinator."""

    def __init__(
        self,
        policy_name: str,
        topology: RoadTopology,
        dbn_updater: DBNUpdaterProtocol,
        *,
        belief_tracker: BeliefTrackerProtocol | None = None,
        control_config: ControlConfig | None = None,
        observation_config: ObservationConfig | None = None,
        prediction_horizon_s: float = 6.0,
    ) -> None:
        if policy_name not in DBN_COORDINATOR_POLICY_NAMES:
            raise ConfigValidationError(
                f"Unsupported policy_name {policy_name!r} for FactoredSwarmCoordinator; "
                f"expected one of {sorted(DBN_COORDINATOR_POLICY_NAMES)}.",
                context={
                    "policy_name": policy_name,
                    "allowed_policies": sorted(DBN_COORDINATOR_POLICY_NAMES),
                },
            )
        self._policy_name = policy_name
        self._topology = topology
        self._dbn_updater = dbn_updater
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
        self._route_broadcaster = DBNRouteCostBroadcaster(self._control_config, policy_name)
        self._v2x_selector = VoIV2XSelector(self._obs_config)

        self._last_belief_state: SwarmBeliefState | None = None
        self._last_dbn_snapshot: DBNStateSnapshot | None = None
        self._last_maxsum_result: MaxSumResult | None = None
        self._last_schedule: ScheduleResult | None = None

        self._prev_post_safety_accels: dict[str, float] = {}
        self._prev_v2x_polls: tuple[str, ...] = ()
        self._speed_controlled_vehs: set[str] = set()
        self._prev_sim_time: float = 0.0
        self._has_stepped: bool = False

    @property
    def policy_name(self) -> str:
        """Return canonical DBN policy or ablation identifier."""
        return self._policy_name

    @property
    def last_dbn_snapshot(self) -> DBNStateSnapshot | None:
        """Return the :class:`DBNStateSnapshot` inferred during the most recent ``act`` call."""
        return self._last_dbn_snapshot

    @property
    def last_belief_state(self) -> SwarmBeliefState | None:
        """Return the filtered :class:`SwarmBeliefState` from the most recent ``act`` call."""
        return self._last_belief_state

    @property
    def last_maxsum_result(self) -> MaxSumResult | None:
        """Return the :class:`MaxSumResult` from the most recent ``act`` call."""
        return self._last_maxsum_result

    @property
    def last_schedule(self) -> ScheduleResult | None:
        """Return the :class:`ScheduleResult` from the most recent ``act`` call."""
        return self._last_schedule

    def reset(self) -> None:
        """Reset belief filter, DBN temporal memory, TLS holds, route timer, and V2X cooldowns."""
        self._belief_tracker.reset()
        self._dbn_updater.reset()
        self._scheduler.reset()
        self._route_broadcaster.reset()
        self._v2x_selector.reset()

        self._last_belief_state = None
        self._last_dbn_snapshot = None
        self._last_maxsum_result = None
        self._last_schedule = None
        self._prev_post_safety_accels = {}
        self._prev_v2x_polls = ()
        self._speed_controlled_vehs.clear()
        self._prev_sim_time = 0.0
        self._has_stepped = False

    def act(self, obs_batch: ObservationBatch) -> SwarmControlAction:
        """Filter ``obs_batch``, update DBN structure & marginals, and decode coordinated actions."""
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
                    f"in FactoredSwarmCoordinator({self._policy_name}); call reset() between episodes.",
                    context={
                        "policy_name": self._policy_name,
                        "sim_time": now,
                        "prev_sim_time": self._prev_sim_time,
                    },
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

        prev_snapshot_for_dbn = None if self._policy_name == "no_temporal_prior" else self._last_dbn_snapshot
        dbn_snapshot = self._dbn_updater.update_and_infer(
            belief=belief,
            prev_snapshot=prev_snapshot_for_dbn,
            topology=self._topology,
        )
        self._last_dbn_snapshot = dbn_snapshot

        maxsum_res = run_damped_maxsum(
            belief=belief,
            dbn_snapshot=dbn_snapshot,
            topology=self._topology,
            control_config=self._control_config,
        )
        self._last_maxsum_result = maxsum_res

        schedule = self._scheduler.decode(
            belief=belief,
            topology=self._topology,
            maxsum_result=maxsum_res,
            dbn_snapshot=dbn_snapshot,
            enable_spillback_gate=True,
        )
        self._last_schedule = schedule

        glide = compute_ecoglide_speeds(
            belief=belief,
            topology=self._topology,
            schedule=schedule,
            control_config=self._control_config,
            base_modes=maxsum_res.modes,
        )
        safety = self._safety_filter.apply(
            belief=belief,
            topology=self._topology,
            desired_speeds=glide.desired_speeds,
            must_hold_stopline=schedule.must_hold_stopline,
        )
        edge_costs = self._route_broadcaster.maybe_broadcast(
            belief=belief,
            dbn_snapshot=dbn_snapshot,
            topology=self._topology,
        )
        v2x_polls = self._v2x_selector.select_polls(
            belief=belief,
            topology=self._topology,
            dbn_snapshot=dbn_snapshot,
        )

        v_dead = self._control_config.zero_speed_deadzone_mps
        modes: dict[str, LongitudinalMode] = {}
        for vid, mode in glide.effective_modes.items():
            veh = belief.vehicles[vid]
            if veh.in_junction_id is None and safety.target_speeds.get(vid, 0.0) <= v_dead:
                modes[vid] = LongitudinalMode.YIELD
            else:
                modes[vid] = mode

        actuated_speeds, self._speed_controlled_vehs = select_actuated_target_speeds(
            belief=belief,
            topology=self._topology,
            safety=safety,
            control_config=self._control_config,
            must_hold_stopline=schedule.must_hold_stopline,
            granted_vehicles=schedule.granted_vehicles,
            previously_controlled=self._speed_controlled_vehs,
        )

        self._prev_post_safety_accels = dict(safety.post_safety_accels)
        self._prev_v2x_polls = v2x_polls
        self._prev_sim_time = now
        self._has_stepped = True

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return SwarmControlAction(
            sim_time=now,
            target_speeds=actuated_speeds,
            post_safety_accels=safety.post_safety_accels,
            modes=modes,
            tls_desired_greens=schedule.tls_desired_greens,
            broadcast_edge_costs=edge_costs,
            reroute_hysteresis_s=self._control_config.reroute_hysteresis_s,
            v2x_polls=v2x_polls,
            safety_clamp_count=safety.safety_clamp_count,
            zero_speed_deadzone_count=safety.zero_speed_deadzone_count,
            control_latency_ms=latency_ms,
        )


def _resolve_controller_configs(
    config: ExperimentConfig | ScenarioConfig | ControlConfig | None,
    control_config: ControlConfig | None,
    observation_config: ObservationConfig | None,
    dbn_config: DBNConfig | None,
) -> tuple[ControlConfig, ObservationConfig, DBNConfig, ExperimentConfig | None]:
    """Extract validated ``(ControlConfig, ObservationConfig, DBNConfig, optional ExperimentConfig)``."""
    exp_cfg: ExperimentConfig | None = None
    if config is None:
        ctrl = control_config or ControlConfig()
        obs = observation_config or ObservationConfig()
        dbn = dbn_config or DBNConfig()
    elif isinstance(config, ExperimentConfig):
        exp_cfg = config
        ctrl = control_config or config.control
        obs = observation_config or config.scenario.observation
        dbn = dbn_config or config.dbn
    elif isinstance(config, ScenarioConfig):
        ctrl = control_config or ControlConfig()
        obs = observation_config or config.observation
        dbn = dbn_config or DBNConfig()
    elif isinstance(config, ControlConfig):
        ctrl = control_config or config
        obs = observation_config or ObservationConfig()
        dbn = dbn_config or DBNConfig()
    else:
        raise ConfigValidationError(
            f"Unsupported config type {type(config).__name__} passed to build_policy_controller; "
            "expected ExperimentConfig, ScenarioConfig, ControlConfig, or None.",
            context={"config_type": type(config).__name__},
        )
    return ctrl, obs, dbn, exp_cfg


def build_policy_controller(
    policy_name: str,
    topology: RoadTopology,
    config: ExperimentConfig | ScenarioConfig | ControlConfig | None = None,
    *,
    control_config: ControlConfig | None = None,
    observation_config: ObservationConfig | None = None,
    dbn_config: DBNConfig | None = None,
    checkpoint_path: Path | str | None = None,
    belief_tracker: BeliefTrackerProtocol | None = None,
    dbn_updater: DBNUpdaterProtocol | None = None,
    seed: int = 0,
) -> PolicyControllerProtocol:
    """Factory constructing any of the 10 pre-registered :data:`~swarm.contracts.POLICY_NAMES`.

    Parameters
    ----------
    policy_name:
        Must be one of ``POLICY_NAMES``.
    topology:
        Static :class:`~swarm.contracts.RoadTopology` for the active network.
    config:
        Optional :class:`ExperimentConfig`, :class:`ScenarioConfig`, or :class:`ControlConfig`.
    control_config, observation_config, dbn_config:
        Optional explicit sub-configs overriding ``config``.
    checkpoint_path:
        Required path to calibrated ``STGraphTransformerDBN`` checkpoint when ``policy_name``
        is in ``NEURAL_POLICY_NAMES`` and ``dbn_updater`` is ``None``.
    belief_tracker:
        Optional pre-constructed :class:`BeliefTrackerProtocol` (lazily instantiates
        ``swarm.env.CityBeliefTracker`` when ``None`` for belief-driven policies).
    dbn_updater:
        Optional pre-constructed :class:`DBNUpdaterProtocol` (lazily invokes
        ``swarm.dbn.build_dbn_updater`` when ``None`` for DBN policies).
    seed:
        Non-negative integer seed forwarded when lazily constructing ``dbn_updater``.
    """
    if policy_name not in POLICY_NAMES:
        raise ConfigValidationError(
            f"Unknown policy_name {policy_name!r}; valid policies are {list(POLICY_NAMES)}.",
            context={"policy_name": policy_name, "valid_policies": list(POLICY_NAMES)},
        )
    if not isinstance(topology, RoadTopology):
        raise ConfigValidationError(
            f"topology must be a RoadTopology instance, got {type(topology).__name__}.",
            context={"topology_type": type(topology).__name__},
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigValidationError(
            f"seed must be a non-negative strict int, got {seed!r}.",
            context={"seed": repr(seed)},
        )

    ctrl_cfg, obs_cfg, dbn_cfg, exp_cfg = _resolve_controller_configs(
        config=config,
        control_config=control_config,
        observation_config=observation_config,
        dbn_config=dbn_config,
    )

    if policy_name == "fixed_signal":
        return FixedSignalController(
            topology=topology,
            control_config=ctrl_cfg,
            belief_tracker=belief_tracker,
        )
    if policy_name == "actuated_signal":
        return ActuatedSignalController(
            topology=topology,
            control_config=ctrl_cfg,
            belief_tracker=belief_tracker,
        )
    if policy_name == "fcfs_reservation":
        return FCFSReservationController(
            topology=topology,
            control_config=ctrl_cfg,
            observation_config=obs_cfg,
            belief_tracker=belief_tracker,
            prediction_horizon_s=dbn_cfg.prediction_horizon_s,
        )

    resolved_updater = (
        dbn_updater
        if dbn_updater is not None
        else _build_default_dbn_updater(
            policy_name=policy_name,
            topology=topology,
            dbn_config=dbn_cfg,
            checkpoint_path=checkpoint_path,
            exp_config=exp_cfg,
            seed=seed,
        )
    )
    return FactoredSwarmCoordinator(
        policy_name=policy_name,
        topology=topology,
        dbn_updater=resolved_updater,
        belief_tracker=belief_tracker,
        control_config=ctrl_cfg,
        observation_config=obs_cfg,
        prediction_horizon_s=dbn_cfg.prediction_horizon_s,
    )


__all__ = [
    "DBN_COORDINATOR_POLICY_NAMES",
    "FactoredSwarmCoordinator",
    "build_policy_controller",
]

"""Real headless SUMO urban traffic adapter, yellow-clearance guard, rerouter, and cohort metrics.

Boundary Lock C note:
- ``libsumo`` / ``traci`` and ground-truth contracts (:class:`VehicleRawTelemetry`,
  :class:`GroundTruthStepRecord`, :class:`EpisodeMetrics`) are strictly confined to this module
  within ``swarm/env/*``.
- Backend selection is 100% explicit via :func:`validate_sumo_backend` (``"libsumo"`` or ``"traci"``)
  with zero silent fallback.
- Process-global ``libsumo`` singleton lifecycle violations, disconnected onboard routes, and
  conflicting simultaneous TLS link requests fail loudly via typed :class:`SwarmError` subclasses.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import heapq
import importlib
import math
from pathlib import Path
import tempfile
import time
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import sumolib

from ..config import (
    ControlConfig,
    ScenarioConfig,
    config_sha256,
    file_sha256,
    validate_sumo_backend,
)
from ..contracts import (
    DBNStateSnapshot,
    DepartedVehicleEvent,
    EpisodeMetrics,
    GroundTruthStepRecord,
    JunctionSpec,
    ObservationBatch,
    RoadTopology,
    SwarmControlAction,
    VehicleRawTelemetry,
)
from ..logging_utils import (
    ConfigValidationError,
    EpisodeExecutionError,
    NumericalInvariantError,
    RoutingIntegrityError,
    SimulationLifecycleError,
    TopologyIntegrityError,
    get_logger,
    require_finite,
    update_crash_context,
)
from .corridors import InternalEdgeMapping, build_internal_edge_lookup, load_or_build_topology
from .network_builder import generate_episode_routes, resolve_sumo_bin_dir
from .observations import ObservableSourceVehicle, ObservationChannel

_LOGGER = get_logger("env.sumo_adapter")

# Process-wide guard for SUMO's C++ in-process singleton (libsumo)
_ACTIVE_LIBSUMO_INSTANCE: SumoSimulationAdapter | None = None
_TRACI_LABEL_COUNTER: int = 0


@dataclass(frozen=True)
class _TripInfoRecord:
    """Parsed per-vehicle SUMO ``<tripinfo>`` record (completed, unfinished, or undeparted)."""

    veh_id: str
    depart: float
    depart_delay: float
    arrival: float
    duration: float
    time_loss: float
    waiting_time: float
    waiting_count: int
    reroute_no: int


def _parse_tripinfo_xml(tripinfo_path: Path) -> dict[str, _TripInfoRecord]:
    """Parse SUMO ``tripinfo.xml`` written upon simulation close."""
    if not tripinfo_path.is_file():
        raise EpisodeExecutionError(
            f"Expected SUMO tripinfo output file does not exist: {tripinfo_path}",
            context={"tripinfo_path": str(tripinfo_path)},
        )
    try:
        root = ET.parse(tripinfo_path).getroot()
    except ET.ParseError as exc:
        _LOGGER.error("Malformed SUMO tripinfo XML %s: %s", tripinfo_path, exc)
        raise EpisodeExecutionError(
            f"Malformed SUMO tripinfo XML at {tripinfo_path}: {exc}",
            context={"tripinfo_path": str(tripinfo_path)},
        ) from exc

    records: dict[str, _TripInfoRecord] = {}
    for elem in root.findall("tripinfo"):
        vid = elem.attrib.get("id", "")
        if not vid:
            continue
        records[vid] = _TripInfoRecord(
            veh_id=vid,
            depart=float(elem.attrib.get("depart", "-1.0")),
            depart_delay=float(elem.attrib.get("departDelay", "0.0")),
            arrival=float(elem.attrib.get("arrival", "-1.0")),
            duration=float(elem.attrib.get("duration", "0.0")),
            time_loss=float(elem.attrib.get("timeLoss", "0.0")),
            waiting_time=float(elem.attrib.get("waitingTime", "0.0")),
            waiting_count=int(float(elem.attrib.get("waitingCount", "0"))),
            reroute_no=int(float(elem.attrib.get("rerouteNo", "0"))),
        )
    return records


def _normalize_ryg_char(ch: str) -> str:
    if ch in ("G", "g"):
        return "G"
    if ch in ("y", "Y"):
        return "y"
    return "r"


class SumoSimulationAdapter:
    """Headless Eclipse SUMO microscopic simulator implementing :class:`SimulationAdapterProtocol`.

    Enforces:
    - Mandatory junction collision detection (``--collision.check-junctions true``), zero teleports
      (``--time-to-teleport -1``), and ``speedMode=31``.
    - ``3.0s`` yellow (``y``) clearance transition guard before conflicting TLS foe links open.
    - Mid-episode arterial lane closures / speed bottlenecks (:class:`IncidentSpec`) reflected in
      :attr:`ObservationBatch.open_lanes_by_edge`.
    - Public-cost onboard ``sumolib`` Dijkstra rerouter with hysteresis and connectivity checks.
    - Exact scheduled-cohort ``[warmup_duration_s, cohort_end_s)`` :class:`EpisodeMetrics`.
    """

    def __init__(
        self,
        scenario: ScenarioConfig,
        *,
        control_config: ControlConfig | None = None,
        policy_name: str = "actuated_signal",
        seed: int = 0,
        work_dir: Path | str | None = None,
        repo_root: Path | str | None = None,
        topology: RoadTopology | None = None,
    ) -> None:
        if not isinstance(scenario, ScenarioConfig):
            raise ConfigValidationError(
                f"SumoSimulationAdapter expected ScenarioConfig, got {type(scenario).__name__}."
            )
        self._backend_name = validate_sumo_backend(scenario.sumo_backend)
        self._backend_mod: Any = importlib.import_module(self._backend_name)
        self._tc: Any = self._backend_mod.constants

        self._scenario = scenario
        self._control_config = control_config if control_config is not None else ControlConfig()
        self._policy_name = str(policy_name)
        self._seed = int(seed)
        self._repo_root = Path(repo_root) if repo_root is not None else None
        self._explicit_work_dir = Path(work_dir) if work_dir is not None else None
        self._net_xml_path, self._poly_xml_path = scenario.resolve_paths(
            repo_root=self._repo_root, must_exist=True
        )
        self._sumo_bin_dir = resolve_sumo_bin_dir()

        self._topology = (
            topology
            if topology is not None
            else load_or_build_topology(scenario, repo_root=self._repo_root)
        )
        self._internal_edge_lookup: dict[str, InternalEdgeMapping] = build_internal_edge_lookup(
            self._topology
        )

        # Static sumolib geometry, passenger connectivity, lane counts, and vType minGaps
        self._lane_lengths: dict[str, float] = {}
        self._edge_num_lanes: dict[str, int] = {}
        self._base_open_lanes: dict[str, int] = {}
        self._base_edge_max_speed: dict[str, float] = {}
        self._base_lane_disallowed: dict[str, tuple[str, ...]] = {}
        self._edge_free_tt: dict[str, float] = {}
        self._edge_adj: dict[str, tuple[str, ...]] = {}
        self._reverse_edge_adj: dict[str, tuple[str, ...]] = {}
        self._preindex_static_network()

        self._min_gap_by_vtype: dict[str, float] = {
            vt.vtype_id: float(vt.min_gap_m) for vt in scenario.vehicle_types
        }
        self._validate_incident_specs()

        self._tls_to_junction: dict[str, JunctionSpec] = {}
        self._tls_foe_lists: dict[str, tuple[tuple[int, ...], ...]] = {}
        for junc in self._topology.junctions.values():
            if junc.tls_id is not None:
                self._tls_to_junction[junc.tls_id] = junc
                foes_per_link = tuple(
                    tuple(
                        int(k2)
                        for k2 in range(junc.num_tls_links)
                        if k2 != k1 and bool(junc.tls_link_foe_matrix[k1, k2])
                    )
                    for k1 in range(junc.num_tls_links)
                )
                self._tls_foe_lists[junc.tls_id] = foes_per_link

        self._obs_channel = ObservationChannel(
            config=scenario.observation,
            topology=self._topology,
            seed=self._seed,
        )

        self._sub_vars: tuple[int, ...] = (
            int(self._tc.VAR_POSITION),
            int(self._tc.VAR_ANGLE),
            int(self._tc.VAR_SPEED),
            int(self._tc.VAR_ACCELERATION),
            int(self._tc.VAR_SIGNALS),
            int(self._tc.VAR_LENGTH),
            int(self._tc.VAR_WIDTH),
            int(self._tc.VAR_TYPE),
            int(self._tc.VAR_ROAD_ID),
            int(self._tc.VAR_LANE_ID),
            int(self._tc.VAR_LANE_INDEX),
            int(self._tc.VAR_LANEPOSITION),
            int(self._tc.VAR_ROUTE_INDEX),
            int(self._tc.VAR_TIMELOSS),
            int(self._tc.VAR_WAITING_TIME),
            int(self._tc.VAR_CO2EMISSION),
        )
        self._leader_sub_dist: float = max(100.0, float(scenario.observation.radar_range_m))

        # Per-episode mutable runtime state
        self._conn: Any = None
        self._traci_label: str | None = None
        self._is_active: bool = False
        self._temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
        self._tripinfo_xml_path: Path | None = None
        self._parsed_tripinfos: dict[str, _TripInfoRecord] | None = None
        self._scheduled_departs: dict[str, float] = {}

        self._sim_time: float = 0.0
        self._step_idx: int = 0
        self._episode_start_wall: float = 0.0
        self._episode_elapsed_wall_s: float = 0.0

        self._subscribed_vehs: set[str] = set()
        self._route_cache: dict[str, tuple[str, ...]] = {}
        self._last_source_by_id: dict[str, ObservableSourceVehicle] = {}
        self._last_reroute_time: dict[str, float] = {}
        self._active_incident_indices: set[int] = set()
        self._open_lanes_by_edge: dict[str, int] = dict(self._base_open_lanes)

        self._tls_guard_state: dict[str, list[str]] = {}
        self._tls_guard_elapsed_s: dict[str, np.ndarray] = {}

        # Episode-level ground-truth telemetry accumulators
        self._total_collisions: int = 0
        self._total_teleports: int = 0
        self._total_emergency_stops: int = 0
        self._total_emergency_brakes: int = 0
        self._total_ttc_near_misses: int = 0
        self._episode_min_ttc_s: float = 999.0
        self._total_veh_meters: float = 0.0
        self._total_reroutes: int = 0
        self._spillback_corridor_seconds: float = 0.0
        self._co2_mg_by_veh: dict[str, float] = defaultdict(float)
        self._total_safety_clamps: int = 0
        self._total_control_decisions: int = 0
        self._total_zero_speed_deadzones: int = 0
        self._total_v2x_polls_requested: int = 0
        self._control_latencies_ms: list[float] = []

    @property
    def topology(self) -> RoadTopology:
        """Static road topology and junction conflict specification."""
        return self._topology

    @property
    def is_active(self) -> bool:
        """Whether an underlying SUMO simulation process/singleton is currently open."""
        return self._is_active

    @property
    def sim_time(self) -> float:
        """Current simulation timestamp in seconds."""
        return self._sim_time

    @property
    def scheduled_departs(self) -> Mapping[str, float]:
        """Privileged map ``veh_id -> scheduled_depart_s`` for offline cohort accounting only."""
        return self._scheduled_departs

    def __enter__(self) -> SumoSimulationAdapter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _preindex_static_network(self) -> None:
        net = sumolib.net.readNet(str(self._net_xml_path), withInternal=True)
        for edge in net.getEdges():
            for lane in edge.getLanes():
                self._lane_lengths[lane.getID()] = float(lane.getLength())

            if edge.isSpecial():
                continue
            eid = edge.getID()
            lanes = edge.getLanes()
            self._edge_num_lanes[eid] = len(lanes)
            open_cnt = sum(1 for ln in lanes if ln.allows("passenger"))
            if open_cnt == 0 and eid not in self._topology.edge_to_corridor:
                continue
            self._base_open_lanes[eid] = open_cnt
            self._base_edge_max_speed[eid] = float(edge.getSpeed())
            for ln in lanes:
                # Record non-passenger baseline permissions; normal urban passenger lanes allow all
                self._base_lane_disallowed[ln.getID()] = ()

            free_tt = float(edge.getLength()) / max(0.5, float(edge.getSpeed()))
            self._edge_free_tt[eid] = free_tt

        adj_sets: dict[str, set[str]] = {eid: set() for eid in self._edge_free_tt}
        rev_sets: dict[str, set[str]] = {eid: set() for eid in self._edge_free_tt}
        for edge in net.getEdges():
            if edge.isSpecial():
                continue
            eid = edge.getID()
            if eid not in adj_sets:
                continue
            for out_edge, conns in edge.getOutgoing().items():
                if out_edge.isSpecial():
                    continue
                oid = out_edge.getID()
                if oid not in adj_sets:
                    continue
                if any(
                    c.getFromLane().allows("passenger") and c.getToLane().allows("passenger")
                    for c in conns
                ):
                    adj_sets[eid].add(oid)
                    rev_sets[oid].add(eid)

        self._edge_adj = {eid: tuple(sorted(targets)) for eid, targets in adj_sets.items()}
        self._reverse_edge_adj = {eid: tuple(sorted(srcs)) for eid, srcs in rev_sets.items()}

    def _validate_incident_specs(self) -> None:
        for idx, inc in enumerate(self._scenario.incidents):
            for eid in inc.edge_ids:
                if eid not in self._edge_num_lanes:
                    raise TopologyIntegrityError(
                        f"IncidentSpec[{idx}] references unknown passenger edge {eid!r} in "
                        f"network {self._scenario.network_id!r}.",
                        context={"incident_idx": idx, "edge_id": eid, "network_id": self._scenario.network_id},
                    )
                n_lanes = self._edge_num_lanes[eid]
                for lidx in inc.closed_lane_indices:
                    if lidx < 0 or lidx >= n_lanes:
                        raise ConfigValidationError(
                            f"IncidentSpec[{idx}] closed_lane_index={lidx} out of bounds for "
                            f"edge {eid!r} with {n_lanes} lane(s).",
                            context={"incident_idx": idx, "edge_id": eid, "lane_index": lidx, "num_lanes": n_lanes},
                        )

    def _reset_accumulators(self) -> None:
        self._sim_time = 0.0
        self._step_idx = 0
        self._subscribed_vehs.clear()
        self._route_cache.clear()
        self._last_source_by_id.clear()
        self._last_reroute_time.clear()
        self._active_incident_indices.clear()
        self._open_lanes_by_edge = dict(self._base_open_lanes)
        self._tls_guard_state.clear()
        self._tls_guard_elapsed_s.clear()
        self._parsed_tripinfos = None

        self._total_collisions = 0
        self._total_teleports = 0
        self._total_emergency_stops = 0
        self._total_emergency_brakes = 0
        self._total_ttc_near_misses = 0
        self._episode_min_ttc_s = 999.0
        self._total_veh_meters = 0.0
        self._total_reroutes = 0
        self._spillback_corridor_seconds = 0.0
        self._co2_mg_by_veh.clear()
        self._total_safety_clamps = 0
        self._total_control_decisions = 0
        self._total_zero_speed_deadzones = 0
        self._total_v2x_polls_requested = 0
        self._control_latencies_ms.clear()

    def start(self, seed: int | None = None) -> tuple[ObservationBatch, GroundTruthStepRecord]:
        """Explicitly start SUMO, raising :class:`SimulationLifecycleError` if already active."""
        if self._is_active:
            raise SimulationLifecycleError(
                "SumoSimulationAdapter.start() called while simulation is already active; "
                "call close() first or use reset().",
                context={"scenario_id": self._scenario.scenario_id, "backend": self._backend_name},
            )
        return self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> tuple[ObservationBatch, GroundTruthStepRecord]:
        """Launch a fresh deterministic SUMO episode for ``seed`` and advance to first step."""
        global _ACTIVE_LIBSUMO_INSTANCE, _TRACI_LABEL_COUNTER

        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ConfigValidationError(
                    f"reset(seed=...) requires a non-negative int, got {seed!r}.",
                    context={"seed": repr(seed)},
                )
            self._seed = seed

        if self._backend_name == "libsumo":
            if _ACTIVE_LIBSUMO_INSTANCE is not None and _ACTIVE_LIBSUMO_INSTANCE is not self:
                raise SimulationLifecycleError(
                    "Cannot start a second libsumo SumoSimulationAdapter while another instance is active "
                    "in the same process (libsumo is a process-global C++ singleton).",
                    context={
                        "active_scenario": _ACTIVE_LIBSUMO_INSTANCE._scenario.scenario_id,
                        "requested_scenario": self._scenario.scenario_id,
                    },
                )

        if self._is_active:
            self.close()

        self._reset_accumulators()
        self._obs_channel.reset(self._seed)

        if self._explicit_work_dir is not None:
            run_dir = self._explicit_work_dir / f"{self._scenario.scenario_id}_seed{self._seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._temp_dir_obj = tempfile.TemporaryDirectory(
                prefix=f"sumo_{self._scenario.scenario_id}_s{self._seed}_"
            )
            run_dir = Path(self._temp_dir_obj.name)

        rou_path, scheduled_departs = generate_episode_routes(
            self._scenario,
            self._seed,
            run_dir,
            topology=self._topology,
            repo_root=self._repo_root,
        )
        self._scheduled_departs = scheduled_departs
        self._tripinfo_xml_path = run_dir / f"{self._scenario.scenario_id}_seed{self._seed}.tripinfo.xml"

        sumo_cmd = [
            str(self._sumo_bin_dir / "sumo"),
            "--net-file",
            str(self._net_xml_path),
            "--route-files",
            str(rou_path),
            "--step-length",
            f"{self._scenario.step_length_s:.4f}",
            "--seed",
            str(self._seed),
            "--collision.check-junctions",
            "true",
            "--collision.action",
            "warn",
            "--time-to-teleport",
            "-1",
            "--tripinfo-output",
            str(self._tripinfo_xml_path),
            "--tripinfo-output.write-unfinished",
            "true",
            "--tripinfo-output.write-undeparted",
            "true",
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
        ]

        self._episode_start_wall = time.perf_counter()
        if self._backend_name == "libsumo":
            self._backend_mod.start(sumo_cmd)
            self._conn = self._backend_mod
            _ACTIVE_LIBSUMO_INSTANCE = self
        else:
            _TRACI_LABEL_COUNTER += 1
            label = f"swarm_traci_{self._scenario.scenario_id}_{_TRACI_LABEL_COUNTER}"
            self._backend_mod.start(sumo_cmd, label=label)
            self._traci_label = label
            self._conn = self._backend_mod.getConnection(label)

        self._is_active = True

        try:
            # Explicitly select 'fixed' (static) vs '0' (actuated) TLS program
            # (SUMO activates whichever <tlLogic> appears last in .net.xml by default!)
            tls_ids = tuple(self._conn.trafficlight.getIDList())
            target_prog = "fixed" if self._policy_name == "fixed_signal" else "0"
            for tls_id in tls_ids:
                avail_progs = {
                    str(lg.programID)
                    for lg in self._conn.trafficlight.getAllProgramLogics(tls_id)
                }
                if target_prog in avail_progs:
                    self._conn.trafficlight.setProgram(tls_id, target_prog)

            # Initialize yellow-clearance guard state from live SUMO TLS link states
            min_hold = self._control_config.tls_min_green_hold_s
            for tls_id in tls_ids:
                ryg = str(self._conn.trafficlight.getRedYellowGreenState(tls_id))
                self._tls_guard_state[tls_id] = [_normalize_ryg_char(c) for c in ryg]
                self._tls_guard_elapsed_s[tls_id] = np.full((len(ryg),), min_hold, dtype=np.float64)

            self._sync_incidents(0.0)

            # Advance 1 simulation step so initial t = step_length_s departures are loaded
            self._conn.simulationStep()
            self._step_idx = 1
            self._sim_time = float(self._conn.simulation.getTime())
            self._sync_incidents(self._sim_time)

            obs_batch, gt_record = self._collect_step_outputs(reroutes_applied=0, v2x_sent=0)
        except Exception as exc:
            _LOGGER.error("SUMO reset failed after startup: %s", exc)
            self.close()
            raise

        update_crash_context(
            scenario_id=self._scenario.scenario_id,
            policy_name=self._policy_name,
            seed=self._seed,
            sim_time=self._sim_time,
            step_idx=self._step_idx,
            active_veh_count=len(obs_batch.active_veh_ids),
        )
        return obs_batch, gt_record

    def _sync_incidents(self, sim_time: float) -> None:
        """Apply or restore scheduled mid-episode lane closures and speed bottlenecks."""
        if not self._scenario.incidents:
            return

        desired_active = {
            idx
            for idx, inc in enumerate(self._scenario.incidents)
            if inc.start_time_s <= sim_time < inc.end_time_s
        }
        if desired_active == self._active_incident_indices:
            return

        # Recompute closed lanes & speed overrides cleanly across all currently active incidents
        closed_lanes_by_edge: dict[str, set[int]] = defaultdict(set)
        speed_override_by_edge: dict[str, float] = {}
        affected_edges: set[str] = set()
        for inc in self._scenario.incidents:
            affected_edges.update(inc.edge_ids)

        for idx in sorted(desired_active):
            inc = self._scenario.incidents[idx]
            for eid in inc.edge_ids:
                closed_lanes_by_edge[eid].update(inc.closed_lane_indices)
                if inc.reduced_speed_mps is not None:
                    prev_spd = speed_override_by_edge.get(eid, self._base_edge_max_speed[eid])
                    speed_override_by_edge[eid] = min(prev_spd, float(inc.reduced_speed_mps))

        for eid in sorted(affected_edges):
            n_lanes = self._edge_num_lanes[eid]
            closed_set = closed_lanes_by_edge.get(eid, set())
            open_count = 0
            for lidx in range(n_lanes):
                lane_id = f"{eid}_{lidx}"
                if lidx in closed_set:
                    self._conn.lane.setDisallowed(lane_id, ["passenger"])
                else:
                    self._conn.lane.setDisallowed(lane_id, list(self._base_lane_disallowed.get(lane_id, ())))
                    open_count += 1
            self._open_lanes_by_edge[eid] = open_count

            target_spd = speed_override_by_edge.get(eid, self._base_edge_max_speed[eid])
            self._conn.edge.setMaxSpeed(eid, float(target_spd))

        self._active_incident_indices = desired_active

    def _apply_tls_guard(self, desired_greens: Mapping[str, Any], dt: float) -> None:
        """Enforce ``3.0s`` yellow clearance transition before granting conflicting foe links."""
        min_green = self._control_config.tls_min_green_hold_s
        yellow_dur = self._control_config.tls_yellow_clearance_s

        for tls_id, req_links_raw in desired_greens.items():
            junc = self._tls_to_junction.get(tls_id)
            if junc is None or tls_id not in self._tls_guard_state:
                raise TopologyIntegrityError(
                    f"Controller requested TLS state for unknown tls_id={tls_id!r}.",
                    context={"tls_id": tls_id, "known_tls_ids": sorted(self._tls_to_junction)},
                )

            n_links = junc.num_tls_links
            req_links: set[int] = set()
            for l_idx in req_links_raw:
                if isinstance(l_idx, bool) or not isinstance(l_idx, int) or l_idx < 0 or l_idx >= n_links:
                    raise TopologyIntegrityError(
                        f"Invalid TLS link index {l_idx!r} for tls_id={tls_id!r} (num_tls_links={n_links}).",
                        context={"tls_id": tls_id, "link_index": repr(l_idx), "num_tls_links": n_links},
                    )
                req_links.add(l_idx)

            # Fail loudly if controller simultaneously requests mutually conflicting foe links
            ordered_req = sorted(req_links)
            foe_mat = junc.tls_link_foe_matrix
            for i_pos, l1 in enumerate(ordered_req):
                for l2 in ordered_req[i_pos + 1 :]:
                    if bool(foe_mat[l1, l2]):
                        raise TopologyIntegrityError(
                            f"Conflicting simultaneous green links ({l1}, {l2}) requested at tls_id={tls_id!r}.",
                            context={"tls_id": tls_id, "conflicting_pair": [l1, l2], "desired": ordered_req},
                        )

            cur_state = self._tls_guard_state[tls_id]
            elapsed = self._tls_guard_elapsed_s[tls_id]
            elapsed += dt
            foes_of = self._tls_foe_lists[tls_id]

            # Step 1: Finish yellow clearance -> red
            for l_idx in range(n_links):
                if cur_state[l_idx] == "y" and float(elapsed[l_idx]) >= yellow_dur - 1e-9:
                    cur_state[l_idx] = "r"
                    elapsed[l_idx] = 0.0

            # Step 2: Transition green -> yellow (NEVER green -> red directly!)
            for l_idx in range(n_links):
                if cur_state[l_idx] == "G" and l_idx not in req_links:
                    if float(elapsed[l_idx]) >= min_green - 1e-9:
                        cur_state[l_idx] = "y"
                        elapsed[l_idx] = 0.0

            # Step 3: Grant requested red -> green ONLY when every foe link is fully red ('r')
            for l_idx in ordered_req:
                if cur_state[l_idx] == "r":
                    if all(cur_state[foe_idx] == "r" for foe_idx in foes_of[l_idx]):
                        cur_state[l_idx] = "G"
                        elapsed[l_idx] = 0.0

            self._conn.trafficlight.setRedYellowGreenState(tls_id, "".join(cur_state))

    def validate_and_apply_route(self, veh_id: str, new_edges: Sequence[str]) -> None:
        """Strictly validate passenger connectivity of ``new_edges`` and apply ``vehicle.setRoute``."""
        if not self._is_active or self._conn is None:
            raise SimulationLifecycleError(
                "validate_and_apply_route called when SumoSimulationAdapter is not active.",
                context={"veh_id": veh_id},
            )
        if not new_edges:
            raise RoutingIntegrityError(
                f"Cannot assign empty route to vehicle {veh_id!r}.",
                context={"veh_id": veh_id},
            )

        active_ids = set(self._conn.vehicle.getIDList())
        if veh_id not in active_ids:
            raise RoutingIntegrityError(
                f"Cannot reroute non-active vehicle {veh_id!r}.",
                context={"veh_id": veh_id},
            )

        current_road = str(self._conn.vehicle.getRoadID(veh_id))
        if current_road.startswith(":"):
            raise RoutingIntegrityError(
                f"Cannot reroute vehicle {veh_id!r} while inside internal junction edge {current_road!r}.",
                context={"veh_id": veh_id, "current_road": current_road},
            )

        current_route = tuple(self._conn.vehicle.getRoute(veh_id))
        if not current_route:
            raise RoutingIntegrityError(
                f"Vehicle {veh_id!r} has an empty existing route in SUMO.",
                context={"veh_id": veh_id},
            )
        route_goal = current_route[-1]

        edge_chain = [str(e) for e in new_edges]
        if edge_chain[0] != current_road:
            raise RoutingIntegrityError(
                f"New route for {veh_id!r} must start at current road {current_road!r}, got {edge_chain[0]!r}.",
                context={"veh_id": veh_id, "current_road": current_road, "new_start": edge_chain[0]},
            )
        if edge_chain[-1] != route_goal:
            raise RoutingIntegrityError(
                f"New route for {veh_id!r} must preserve destination {route_goal!r}, got {edge_chain[-1]!r}.",
                context={"veh_id": veh_id, "expected_goal": route_goal, "new_goal": edge_chain[-1]},
            )

        for idx, eid in enumerate(edge_chain):
            if eid not in self._edge_adj:
                raise RoutingIntegrityError(
                    f"Route for {veh_id!r} contains unknown or non-passenger edge {eid!r} at index {idx}.",
                    context={"veh_id": veh_id, "edge_id": eid, "index": idx},
                )
            if idx + 1 < len(edge_chain):
                nxt = edge_chain[idx + 1]
                if nxt not in self._edge_adj[eid]:
                    raise RoutingIntegrityError(
                        f"Disconnected passenger edge transition {eid!r} -> {nxt!r} in route for {veh_id!r}.",
                        context={"veh_id": veh_id, "from_edge": eid, "to_edge": nxt, "route": edge_chain},
                    )

        try:
            self._conn.vehicle.setRoute(veh_id, edge_chain)
        except Exception as exc:
            _LOGGER.error("SUMO setRoute failed for %s on %s: %s", veh_id, edge_chain, exc)
            raise RoutingIntegrityError(
                f"SUMO rejected route assignment for {veh_id!r}: {exc}",
                context={"veh_id": veh_id, "route": edge_chain},
            ) from exc

        # Keep full route cache synchronized: prefix before current road + new edge_chain
        r_idx = max(0, int(self._conn.vehicle.getRouteIndex(veh_id)))
        prefix = current_route[:r_idx] if r_idx <= len(current_route) else ()
        self._route_cache[veh_id] = tuple(prefix) + tuple(edge_chain)

    def _compute_reverse_shortest_paths(
        self,
        goal_edge: str,
        edge_costs: Mapping[str, float],
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Run backward Dijkstra from ``goal_edge`` over passenger-open edges."""
        dist: dict[str, float] = {goal_edge: float(edge_costs[goal_edge])}
        next_hop: dict[str, str] = {}
        pq: list[tuple[float, str]] = [(dist[goal_edge], goal_edge)]

        while pq:
            d_cur, cur = heapq.heappop(pq)
            if d_cur > dist.get(cur, math.inf) + 1e-12:
                continue
            for pred in self._reverse_edge_adj.get(cur, ()):
                # Disallow routing through completely closed intermediate edges (open_lanes == 0),
                # while still allowing vehicles currently standing on `pred` to evaluate exiting it.
                step_cost = float(edge_costs[pred])
                if self._open_lanes_by_edge.get(pred, 1) <= 0:
                    step_cost += 1.0e6
                cand = d_cur + step_cost
                if cand + 1e-12 < dist.get(pred, math.inf):
                    dist[pred] = cand
                    next_hop[pred] = cur
                    heapq.heappush(pq, (cand, pred))
        return dist, next_hop

    def _apply_broadcast_rerouting(
        self,
        broadcast_edge_costs: Mapping[str, float],
        reroute_hysteresis_s: float,
    ) -> int:
        """Execute decentralized onboard shortest-path rerouting from public edge travel costs."""
        require_finite(reroute_hysteresis_s, "SwarmControlAction.reroute_hysteresis_s")
        effective_costs: dict[str, float] = dict(self._edge_free_tt)
        for eid, val in broadcast_edge_costs.items():
            require_finite(val, f"broadcast_edge_costs[{eid}]")
            fval = float(val)
            if fval <= 0.0:
                raise NumericalInvariantError(
                    f"broadcast_edge_costs[{eid!r}] must be strictly positive, got {fval}.",
                    context={"edge_id": eid, "cost": fval},
                )
            if eid in effective_costs:
                effective_costs[eid] = fval

        min_stopline_dist = self._control_config.reroute_min_stopline_dist_m
        cooldown = self._control_config.reroute_interval_s
        sp_cache: dict[str, tuple[dict[str, float], dict[str, str]]] = {}
        reroutes_applied = 0

        for vid in tuple(self._conn.vehicle.getIDList()):
            if self._sim_time - self._last_reroute_time.get(vid, -1e9) < cooldown - 1e-9:
                continue
            src = self._last_source_by_id.get(vid)
            if src is None or src.in_junction_id is not None:
                continue
            road_id = str(self._conn.vehicle.getRoadID(vid))
            if road_id.startswith(":") or road_id not in self._topology.edge_to_corridor:
                continue

            cid = self._topology.edge_to_corridor[road_id]
            spec = self._topology.corridors[cid]
            # Reroute only on corridor entry trunk (before turn-only pockets) and > 35m from stopline
            if road_id != spec.edges[0]:
                continue
            lane_pos = float(self._conn.vehicle.getLanePosition(vid))
            corridor_pos = self._topology.edge_offset_in_corridor[road_id] + lane_pos
            if spec.length_m - corridor_pos <= min_stopline_dist:
                continue

            cur_route = tuple(self._conn.vehicle.getRoute(vid))
            r_idx = int(self._conn.vehicle.getRouteIndex(vid))
            if not cur_route or r_idx < 0 or r_idx >= len(cur_route) - 1:
                continue
            rem_route = cur_route[r_idx:]
            if rem_route[0] != road_id:
                continue
            goal_edge = cur_route[-1]
            if road_id == goal_edge or goal_edge not in effective_costs:
                continue

            if goal_edge not in sp_cache:
                sp_cache[goal_edge] = self._compute_reverse_shortest_paths(goal_edge, effective_costs)
            dist_map, next_hop = sp_cache[goal_edge]
            if road_id not in dist_map:
                continue

            best_cost = dist_map[road_id]
            cur_cost = 0.0
            for idx_e, r_edge in enumerate(rem_route):
                c_val = effective_costs.get(r_edge, 30.0)
                if idx_e > 0 and self._open_lanes_by_edge.get(r_edge, 1) <= 0:
                    c_val += 1.0e6
                cur_cost += c_val

            if cur_cost - best_cost <= float(reroute_hysteresis_s):
                continue

            # Reconstruct shortest path from road_id -> goal_edge
            new_chain: list[str] = [road_id]
            cursor = road_id
            max_hops = len(effective_costs) + 2
            while cursor != goal_edge and len(new_chain) <= max_hops:
                nxt = next_hop.get(cursor)
                if nxt is None:
                    break
                new_chain.append(nxt)
                cursor = nxt

            if new_chain[-1] != goal_edge or tuple(new_chain) == rem_route:
                continue

            self.validate_and_apply_route(vid, new_chain)
            self._last_reroute_time[vid] = self._sim_time
            reroutes_applied += 1
            self._total_reroutes += 1

        return reroutes_applied

    def step(self, action: SwarmControlAction) -> tuple[ObservationBatch, GroundTruthStepRecord]:
        """Apply ``action``, advance SUMO by ``step_length_s``, and return observations + ground truth."""
        if not self._is_active or self._conn is None:
            raise SimulationLifecycleError(
                "SumoSimulationAdapter.step() called before reset() or after close().",
                context={"scenario_id": self._scenario.scenario_id},
            )

        dt = self._scenario.step_length_s
        require_finite(action.sim_time, "SwarmControlAction.sim_time")
        require_finite(action.control_latency_ms, "SwarmControlAction.control_latency_ms")

        # 1. Enqueue budgeted V2X queries against current pre-step physical vehicle states
        self._total_v2x_polls_requested += len(action.v2x_polls)
        v2x_sent = self._obs_channel.enqueue_v2x_polls(
            action.v2x_polls,
            sim_time=self._sim_time,
            step_idx=self._step_idx,
            dt=dt,
            source_by_id=self._last_source_by_id,
        )

        # 2. Apply per-vehicle target speeds (speedMode=31 preserves physical safety)
        active_now = set(self._conn.vehicle.getIDList())
        for vid, raw_v_cmd in action.target_speeds.items():
            require_finite(raw_v_cmd, f"target_speeds[{vid}]")
            if vid not in active_now:
                continue
            v_cmd = float(raw_v_cmd)
            if v_cmd < 0.0:
                self._conn.vehicle.setSpeed(vid, -1.0)
            else:
                self._conn.vehicle.setSpeed(vid, max(0.0, v_cmd))

        # 3. Apply TLS link grants through 3.0s yellow clearance transition guard
        if action.tls_desired_greens is not None:
            self._apply_tls_guard(action.tls_desired_greens, dt)

        # 4. Apply public edge-cost onboard rerouting
        reroutes_applied = 0
        if action.broadcast_edge_costs is not None:
            reroutes_applied = self._apply_broadcast_rerouting(
                action.broadcast_edge_costs,
                action.reroute_hysteresis_s,
            )

        # Record controller safety/latency diagnostics
        self._total_safety_clamps += max(0, int(action.safety_clamp_count))
        self._total_control_decisions += max(1, len(action.modes) if action.modes else len(active_now))
        self._total_zero_speed_deadzones += max(0, int(action.zero_speed_deadzone_count))
        self._control_latencies_ms.append(max(0.0, float(action.control_latency_ms)))

        # 5. Advance microscopic physics by 1 step
        self._conn.simulationStep()
        self._step_idx += 1
        self._sim_time = float(self._conn.simulation.getTime())
        self._sync_incidents(self._sim_time)

        obs_batch, gt_record = self._collect_step_outputs(
            reroutes_applied=reroutes_applied,
            v2x_sent=v2x_sent,
        )
        update_crash_context(
            sim_time=self._sim_time,
            step_idx=self._step_idx,
            active_veh_count=len(obs_batch.active_veh_ids),
        )
        return obs_batch, gt_record

    def _resolve_corridor_coordinates(
        self,
        veh_id: str,
        road_id: str,
        lane_pos: float,
    ) -> tuple[str, str, float, str | None]:
        """Map SUMO ``(road_id, lane_pos)`` onto ``(corridor_id, anchor_edge_id, corridor_pos_m, in_junc)``."""
        if road_id in self._topology.edge_to_corridor:
            cid = self._topology.edge_to_corridor[road_id]
            spec = self._topology.corridors[cid]
            offset = self._topology.edge_offset_in_corridor[road_id]
            c_pos = float(np.clip(offset + lane_pos, 0.0, spec.length_m))
            return cid, road_id, c_pos, None

        imap = self._internal_edge_lookup.get(road_id)
        if imap is not None:
            spec = self._topology.corridors[imap.corridor_id]
            c_pos = float(np.clip(imap.corridor_pos_m, 0.0, spec.length_m))
            return imap.corridor_id, imap.anchor_edge_id, c_pos, imap.in_junction_id

        raise TopologyIntegrityError(
            f"Active vehicle {veh_id!r} occupies unmapped SUMO road_id={road_id!r}.",
            context={"veh_id": veh_id, "road_id": road_id, "network_id": self._scenario.network_id},
        )

    def _infer_true_turn_index(
        self,
        corridor_id: str,
        route_edges: tuple[str, ...],
        route_index: int,
    ) -> int | None:
        spec = self._topology.corridors[corridor_id]
        if not spec.turn_options or not route_edges:
            return None
        start_i = max(0, min(route_index, len(route_edges) - 1))
        for next_edge in route_edges[start_i:]:
            next_cid = self._topology.edge_to_corridor.get(next_edge)
            if next_cid is not None and next_cid != corridor_id:
                for opt in spec.turn_options:
                    if opt.to_corridor == next_cid:
                        return opt.turn_idx
                return None
        return None

    def _collect_step_outputs(
        self,
        *,
        reroutes_applied: int,
        v2x_sent: int,
    ) -> tuple[ObservationBatch, GroundTruthStepRecord]:
        tc = self._tc
        dt = self._scenario.step_length_s

        departed_ids = tuple(str(v) for v in self._conn.simulation.getDepartedIDList())
        arrived_ids = tuple(str(v) for v in self._conn.simulation.getArrivedIDList())

        for arrived_id in arrived_ids:
            self._subscribed_vehs.discard(arrived_id)
            self._route_cache.pop(arrived_id, None)
            self._last_reroute_time.pop(arrived_id, None)

        for dep_id in departed_ids:
            self._conn.vehicle.setSpeedMode(dep_id, 31)
            self._conn.vehicle.subscribe(dep_id, self._sub_vars)
            self._conn.vehicle.subscribeLeader(dep_id, self._leader_sub_dist)
            self._route_cache[dep_id] = tuple(str(e) for e in self._conn.vehicle.getRoute(dep_id))
            self._subscribed_vehs.add(dep_id)

        active_ids = tuple(sorted(str(v) for v in self._conn.vehicle.getIDList()))
        for vid in active_ids:
            if vid not in self._subscribed_vehs:
                self._conn.vehicle.setSpeedMode(vid, 31)
                self._conn.vehicle.subscribe(vid, self._sub_vars)
                self._conn.vehicle.subscribeLeader(vid, self._leader_sub_dist)
                self._route_cache[vid] = tuple(str(e) for e in self._conn.vehicle.getRoute(vid))
                self._subscribed_vehs.add(vid)

        all_subs: Mapping[str, Mapping[int, Any]] = self._conn.vehicle.getAllSubscriptionResults()

        raw_list: list[VehicleRawTelemetry] = []
        raw_by_id: dict[str, VehicleRawTelemetry] = {}
        corridor_coords_by_id: dict[str, tuple[str, str, float, str | None]] = {}

        for vid in active_ids:
            subs = all_subs.get(vid)
            if subs is None or tc.VAR_POSITION not in subs:
                # Immediate fallback to direct getters only if subscription registered post-step
                fx, fy = self._conn.vehicle.getPosition(vid)
                ang_deg = float(self._conn.vehicle.getAngle(vid))
                speed = float(self._conn.vehicle.getSpeed(vid))
                accel = float(self._conn.vehicle.getAcceleration(vid))
                sig = int(self._conn.vehicle.getSignals(vid))
                length = float(self._conn.vehicle.getLength(vid))
                width = float(self._conn.vehicle.getWidth(vid))
                vtype = str(self._conn.vehicle.getTypeID(vid))
                road_id = str(self._conn.vehicle.getRoadID(vid))
                lane_id = str(self._conn.vehicle.getLaneID(vid))
                lane_index = max(0, int(self._conn.vehicle.getLaneIndex(vid)))
                lane_pos = float(self._conn.vehicle.getLanePosition(vid))
                route_index = max(0, int(self._conn.vehicle.getRouteIndex(vid)))
                route_edges = self._route_cache.get(vid) or tuple(
                    str(e) for e in self._conn.vehicle.getRoute(vid)
                )
                self._route_cache[vid] = route_edges
                time_loss = float(self._conn.vehicle.getTimeLoss(vid))
                wait_time = float(self._conn.vehicle.getWaitingTime(vid))
                co2_mg_s = float(self._conn.vehicle.getCO2Emission(vid))
                raw_leader = self._conn.vehicle.getLeader(vid, self._leader_sub_dist)
            else:
                fx, fy = subs[tc.VAR_POSITION]
                ang_deg = float(subs[tc.VAR_ANGLE])
                speed = float(subs[tc.VAR_SPEED])
                accel = float(subs[tc.VAR_ACCELERATION])
                sig = int(subs[tc.VAR_SIGNALS])
                length = float(subs[tc.VAR_LENGTH])
                width = float(subs[tc.VAR_WIDTH])
                vtype = str(subs[tc.VAR_TYPE])
                road_id = str(subs[tc.VAR_ROAD_ID])
                lane_id = str(subs[tc.VAR_LANE_ID])
                lane_index = max(0, int(subs[tc.VAR_LANE_INDEX]))
                lane_pos = float(subs[tc.VAR_LANEPOSITION])
                route_index = max(0, int(subs[tc.VAR_ROUTE_INDEX]))
                route_edges = self._route_cache.get(vid)
                if route_edges is None:
                    route_edges = tuple(str(e) for e in self._conn.vehicle.getRoute(vid))
                    self._route_cache[vid] = route_edges
                time_loss = float(subs[tc.VAR_TIMELOSS])
                wait_time = float(subs[tc.VAR_WAITING_TIME])
                co2_mg_s = float(subs[tc.VAR_CO2EMISSION])
                raw_leader = subs.get(tc.VAR_LEADER)

            ang_rad = math.radians(ang_deg)
            hx = math.sin(ang_rad)
            hy = math.cos(ang_rad)
            cx = float(fx) - 0.5 * length * hx
            cy = float(fy) - 0.5 * length * hy
            yaw_ccw = float(math.atan2(hy, hx))

            brake_light = bool(sig & 8)
            turn_signal = -1 if (sig & 1) else (1 if (sig & 2) else 0)

            leader_id: str | None = None
            leader_gap: float | None = None
            if raw_leader is not None and str(raw_leader[0]) != "":
                leader_id = str(raw_leader[0])
                min_gap = self._min_gap_by_vtype.get(vtype, 2.0)
                leader_gap = max(0.05, float(raw_leader[1]) + min_gap)

            lane_len = self._lane_lengths.get(lane_id, max(1.0, lane_pos))
            route_goal = route_edges[-1] if route_edges else road_id

            cid, anchor_edge, c_pos, in_junc = self._resolve_corridor_coordinates(vid, road_id, lane_pos)
            corridor_coords_by_id[vid] = (cid, anchor_edge, c_pos, in_junc)

            raw_veh = VehicleRawTelemetry(
                veh_id=vid,
                vtype=vtype,
                center_xy=(cx, cy),
                yaw_ccw_rad=yaw_ccw,
                speed=max(0.0, speed),
                accel=accel,
                road_id=road_id,
                lane_id=lane_id,
                lane_index=lane_index,
                lane_pos=max(0.0, lane_pos),
                lane_length=lane_len,
                route_edges=route_edges,
                route_index=route_index,
                route_goal=route_goal,
                length=length,
                width=width,
                brake_light=brake_light,
                turn_signal=turn_signal,
                leader_id=leader_id,
                leader_gap=leader_gap,
                time_loss=max(0.0, time_loss),
                waiting_time=max(0.0, wait_time),
                co2_mg_s=max(0.0, co2_mg_s),
                scheduled_depart=float(self._scheduled_departs.get(vid, self._sim_time)),
            )
            raw_list.append(raw_veh)
            raw_by_id[vid] = raw_veh
            self._co2_mg_by_veh[vid] += raw_veh.co2_mg_s * dt
            self._total_veh_meters += raw_veh.speed * dt

        # Build DepartedVehicleEvents strictly at current sim_time (zero departDelay leak!)
        departed_events: list[DepartedVehicleEvent] = []
        for dep_id in departed_ids:
            rv = raw_by_id.get(dep_id)
            if rv is None:
                continue
            _, anchor_edge, _, _ = corridor_coords_by_id[dep_id]
            departed_events.append(
                DepartedVehicleEvent(
                    veh_id=dep_id,
                    entry_sim_time=float(self._sim_time),
                    entry_edge_id=anchor_edge,
                    entry_lane_index=rv.lane_index,
                    length_m=rv.length,
                    width_m=rv.width,
                )
            )

        # Build ObservableSourceVehicles for range-limited radar & delayed V2X channel
        source_by_id: dict[str, ObservableSourceVehicle] = {}
        corridor_veh_counts: dict[str, int] = defaultdict(int)
        corridor_speed_sums: dict[str, float] = defaultdict(float)
        corridor_tail_blocked: set[str] = set()

        for rv in raw_list:
            vid = rv.veh_id
            cid, anchor_edge, c_pos, in_junc = corridor_coords_by_id[vid]
            corridor_veh_counts[cid] += 1
            corridor_speed_sums[cid] += rv.speed
            if not self._topology.corridors[cid].is_entry_fringe and c_pos <= 22.0 and rv.speed < 1.5:
                corridor_tail_blocked.add(cid)

            true_turn_idx = self._infer_true_turn_index(cid, rv.route_edges, rv.route_index)
            lead_spd: float | None = None
            if rv.leader_id is not None and rv.leader_id in raw_by_id:
                lead_spd = raw_by_id[rv.leader_id].speed

            source_by_id[vid] = ObservableSourceVehicle(
                veh_id=vid,
                edge_id=anchor_edge,
                corridor_id=cid,
                in_junction_id=in_junc,
                lane_index=rv.lane_index,
                corridor_pos_m=c_pos,
                speed_mps=rv.speed,
                turn_signal=rv.turn_signal,
                true_turn_idx=true_turn_idx,
                leader_id=rv.leader_id if lead_spd is not None else None,
                leader_gap_m=rv.leader_gap if lead_spd is not None else None,
                leader_speed_mps=lead_spd,
            )

        self._last_source_by_id = source_by_id

        tls_states = {
            str(tls_id): str(self._conn.trafficlight.getRedYellowGreenState(tls_id))
            for tls_id in self._conn.trafficlight.getIDList()
        }

        obs_batch, v2x_delivered = self._obs_channel.step_observations(
            sim_time=self._sim_time,
            step_idx=self._step_idx,
            dt=dt,
            active_veh_ids=active_ids,
            departed=departed_events,
            arrived_veh_ids=arrived_ids,
            source_by_id=source_by_id,
            open_lanes_by_edge=self._open_lanes_by_edge,
            tls_states=tls_states,
        )

        # Compute physical step safety & congestion counters
        collisions = len(self._conn.simulation.getCollisions())
        teleports = int(self._conn.simulation.getStartingTeleportNumber())
        emergency_stops = int(self._conn.simulation.getEmergencyStoppingVehiclesNumber())
        emergency_brakes = sum(1 for rv in raw_list if rv.accel < -4.5)

        near_misses_ttc = 0
        step_min_ttc = 999.0
        for rv in raw_list:
            if rv.leader_id is not None and rv.leader_gap is not None and rv.leader_id in raw_by_id:
                closing_v = rv.speed - raw_by_id[rv.leader_id].speed
                if closing_v > 0.1:
                    ttc = max(0.0, rv.leader_gap) / closing_v
                    if ttc < step_min_ttc:
                        step_min_ttc = ttc
                    if ttc < 1.5:
                        near_misses_ttc += 1

        spillback_corridors = set(corridor_tail_blocked)
        for cid, cnt in corridor_veh_counts.items():
            spec = self._topology.corridors[cid]
            if spec.is_entry_fringe or cnt == 0:
                continue
            cap = self._topology.effective_capacity(cid, self._open_lanes_by_edge)
            mean_v = corridor_speed_sums[cid] / float(cnt)
            if (cnt / float(cap)) >= 0.85 and mean_v < 2.0:
                spillback_corridors.add(cid)
        spillback_count = len(spillback_corridors)

        step_co2_mg = sum(rv.co2_mg_s * dt for rv in raw_list)

        self._total_collisions += collisions
        self._total_teleports += teleports
        self._total_emergency_stops += emergency_stops
        self._total_emergency_brakes += emergency_brakes
        self._total_ttc_near_misses += near_misses_ttc
        if step_min_ttc < self._episode_min_ttc_s:
            self._episode_min_ttc_s = step_min_ttc
        if self._sim_time >= self._scenario.warmup_duration_s - 1e-9:
            self._spillback_corridor_seconds += float(spillback_count) * dt

        gt_record = GroundTruthStepRecord(
            sim_time=float(self._sim_time),
            raw_vehicles=tuple(raw_list),
            collisions=collisions,
            teleports=teleports,
            emergency_stops=emergency_stops,
            emergency_brakes=emergency_brakes,
            near_misses_ttc=near_misses_ttc,
            min_ttc_s=float(step_min_ttc),
            reroutes_applied=reroutes_applied,
            v2x_sent=v2x_sent,
            v2x_delivered=v2x_delivered,
            step_co2_mg=float(step_co2_mg),
            spillback_corridor_count=spillback_count,
        )
        return obs_batch, gt_record

    def close(self) -> None:
        """Close the active SUMO session, flush & parse ``tripinfo.xml``, and release temp dirs."""
        global _ACTIVE_LIBSUMO_INSTANCE

        if self._is_active and self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._is_active = False
                self._conn = None
                self._traci_label = None
                if self._backend_name == "libsumo" and _ACTIVE_LIBSUMO_INSTANCE is self:
                    _ACTIVE_LIBSUMO_INSTANCE = None
                self._episode_elapsed_wall_s = max(1e-6, time.perf_counter() - self._episode_start_wall)

            if self._tripinfo_xml_path is not None and self._tripinfo_xml_path.is_file():
                self._parsed_tripinfos = _parse_tripinfo_xml(self._tripinfo_xml_path)

        if self._temp_dir_obj is not None:
            self._temp_dir_obj.cleanup()
            self._temp_dir_obj = None

    def compute_episode_metrics(
        self,
        *,
        policy_name: str | None = None,
        dbn_snapshots: Sequence[DBNStateSnapshot] | None = None,
        mean_dbn_update_latency_ms: float | None = None,
        mean_dbn_inference_latency_ms: float | None = None,
        mean_control_latency_ms: float | None = None,
        edge_churn_hamming_rate: float | None = None,
        wall_time_s: float | None = None,
        checkpoint_sha256: str | None = None,
        experiment_config: Any | None = None,
    ) -> EpisodeMetrics:
        """Flush SUMO ``tripinfo.xml`` (closing if still open) and compute exact cohort ``EpisodeMetrics``."""
        if self._step_idx <= 0:
            raise SimulationLifecycleError(
                "compute_episode_metrics() called before any episode was started via reset().",
                context={"scenario_id": self._scenario.scenario_id},
            )

        if self._is_active:
            self.close()

        if self._parsed_tripinfos is None:
            if self._tripinfo_xml_path is None or not self._tripinfo_xml_path.is_file():
                raise EpisodeExecutionError(
                    "Tripinfo records unavailable when computing EpisodeMetrics.",
                    context={"scenario_id": self._scenario.scenario_id, "seed": self._seed},
                )
            self._parsed_tripinfos = _parse_tripinfo_xml(self._tripinfo_xml_path)

        warmup_s = self._scenario.warmup_duration_s
        cohort_end_s = self._scenario.cohort_end_s
        cohort_ids = sorted(
            vid for vid, t_sched in self._scheduled_departs.items() if warmup_s <= t_sched < cohort_end_s
        )
        cohort_size = len(cohort_ids)
        if cohort_size == 0:
            raise EpisodeExecutionError(
                f"Scheduled cohort [{warmup_s}, {cohort_end_s}) contains zero vehicles.",
                context={"scenario_id": self._scenario.scenario_id, "seed": self._seed},
            )

        completed_count = 0
        unfinished_count = 0
        undeparted_count = 0
        total_delays: list[float] = []
        completed_delays: list[float] = []
        stops_per_departed: list[float] = []
        reroutes_per_departed: list[float] = []
        cohort_co2_g: list[float] = []

        for vid in cohort_ids:
            t_sched = self._scheduled_departs[vid]
            ti = self._parsed_tripinfos.get(vid)
            if ti is None or ti.depart < 0.0:
                undeparted_count += 1
                dep_delay = ti.depart_delay if ti is not None else 0.0
                total_delays.append(max(0.0, dep_delay, self._sim_time - t_sched))
            elif ti.arrival < 0.0:
                unfinished_count += 1
                total_delays.append(max(0.0, ti.depart_delay) + max(0.0, ti.time_loss))
                stops_per_departed.append(float(ti.waiting_count))
                reroutes_per_departed.append(float(ti.reroute_no))
                cohort_co2_g.append(self._co2_mg_by_veh.get(vid, 0.0) / 1000.0)
            else:
                completed_count += 1
                d_val = max(0.0, ti.depart_delay) + max(0.0, ti.time_loss)
                total_delays.append(d_val)
                completed_delays.append(d_val)
                stops_per_departed.append(float(ti.waiting_count))
                reroutes_per_departed.append(float(ti.reroute_no))
                cohort_co2_g.append(self._co2_mg_by_veh.get(vid, 0.0) / 1000.0)

        if completed_count + unfinished_count + undeparted_count != cohort_size:
            raise EpisodeExecutionError(
                "Cohort partition accounting mismatch.",
                context={
                    "cohort_size": cohort_size,
                    "completed": completed_count,
                    "unfinished": unfinished_count,
                    "undeparted": undeparted_count,
                },
            )

        completion_rate = float(completed_count) / float(cohort_size)
        mean_total_delay = float(np.mean(total_delays))
        p95_total_delay = float(np.percentile(total_delays, 95.0))
        mean_completed_delay = float(np.mean(completed_delays)) if completed_delays else mean_total_delay

        throughput_vph = float(completed_count) * (3600.0 / self._scenario.cohort_duration_s)
        mean_stops = float(np.mean(stops_per_departed)) if stops_per_departed else 0.0
        mean_co2_g = (
            float(np.mean(cohort_co2_g))
            if cohort_co2_g
            else (sum(self._co2_mg_by_veh.values()) / 1000.0) / max(1, len(self._co2_mg_by_veh))
        )
        reroutes_pv = (
            float(np.mean(reroutes_per_departed))
            if reroutes_per_departed
            else float(self._total_reroutes) / float(cohort_size)
        )

        veh_km = max(1e-3, self._total_veh_meters / 1000.0)
        ebrake_rate = float(self._total_emergency_brakes) / veh_km
        clamp_rate = float(self._total_safety_clamps) / float(max(1, self._total_control_decisions))
        v2x_eff = (
            float(self._obs_channel.total_v2x_delivered) / float(self._obs_channel.total_v2x_sent)
            if self._obs_channel.total_v2x_sent > 0
            else 1.0
        )

        # DBN latencies & edge churn from snapshots if provided
        if dbn_snapshots is not None and len(dbn_snapshots) > 0:
            if mean_dbn_update_latency_ms is None:
                mean_dbn_update_latency_ms = float(np.mean([s.update_latency_ms for s in dbn_snapshots]))
            if mean_dbn_inference_latency_ms is None:
                mean_dbn_inference_latency_ms = float(
                    np.mean([s.inference_latency_ms for s in dbn_snapshots])
                )
            if edge_churn_hamming_rate is None:
                churn_vals: list[float] = []
                for idx in range(1, len(dbn_snapshots)):
                    prev_s = dbn_snapshots[idx - 1]
                    cur_s = dbn_snapshots[idx]
                    common_vehs = set(prev_s.veh_ids) & set(cur_s.veh_ids)
                    if not common_vehs:
                        continue
                    prev_e = {(e.src_id, e.dst_id) for e in prev_s.edges if e.src_id in common_vehs and e.dst_id in common_vehs}
                    cur_e = {(e.src_id, e.dst_id) for e in cur_s.edges if e.src_id in common_vehs and e.dst_id in common_vehs}
                    union_n = len(prev_e | cur_e)
                    if union_n > 0:
                        churn_vals.append(float(len(prev_e ^ cur_e)) / float(union_n))
                    else:
                        churn_vals.append(0.0)
                edge_churn_hamming_rate = float(np.mean(churn_vals)) if churn_vals else 0.0

        upd_ms = float(mean_dbn_update_latency_ms) if mean_dbn_update_latency_ms is not None else 0.0
        inf_ms = float(mean_dbn_inference_latency_ms) if mean_dbn_inference_latency_ms is not None else 0.0
        ctrl_ms = (
            float(mean_control_latency_ms)
            if mean_control_latency_ms is not None
            else (float(np.mean(self._control_latencies_ms)) if self._control_latencies_ms else 0.0)
        )
        churn_rate = float(edge_churn_hamming_rate) if edge_churn_hamming_rate is not None else 0.0

        elapsed_wall = float(wall_time_s) if wall_time_s is not None else max(1e-6, self._episode_elapsed_wall_s)
        sps = float(self._step_idx) / max(1e-6, elapsed_wall)

        cfg_target = experiment_config if experiment_config is not None else self._scenario
        cfg_hash = config_sha256(cfg_target)
        net_hash = file_sha256(self._net_xml_path)

        metrics = EpisodeMetrics(
            scenario_id=self._scenario.scenario_id,
            policy_name=policy_name if policy_name is not None else self._policy_name,
            seed=self._seed,
            cohort_size=cohort_size,
            cohort_completed_count=completed_count,
            cohort_unfinished_count=unfinished_count,
            cohort_undeparted_count=undeparted_count,
            cohort_completion_rate=completion_rate,
            cohort_mean_total_delay_s=mean_total_delay,
            cohort_p95_total_delay_s=p95_total_delay,
            cohort_mean_completed_delay_s=mean_completed_delay,
            throughput_veh_per_hr=throughput_vph,
            mean_stops_per_veh=mean_stops,
            spillback_corridor_seconds=float(self._spillback_corridor_seconds),
            mean_co2_g_per_veh=mean_co2_g,
            reroutes_per_veh=reroutes_pv,
            collision_count=self._total_collisions,
            teleport_count=self._total_teleports,
            emergency_stop_count=self._total_emergency_stops,
            emergency_brake_count=self._total_emergency_brakes,
            emergency_brake_rate_per_veh_km=ebrake_rate,
            ttc_near_miss_count=self._total_ttc_near_misses,
            min_ttc_s=float(self._episode_min_ttc_s),
            safety_filter_clamp_count=self._total_safety_clamps,
            safety_filter_clamp_rate=clamp_rate,
            zero_speed_deadzone_count=self._total_zero_speed_deadzones,
            v2x_polls_requested=self._total_v2x_polls_requested,
            v2x_packets_delivered=self._obs_channel.total_v2x_delivered,
            v2x_delivery_efficiency=v2x_eff,
            mean_dbn_update_latency_ms=upd_ms,
            mean_dbn_inference_latency_ms=inf_ms,
            mean_control_latency_ms=ctrl_ms,
            edge_churn_hamming_rate=churn_rate,
            wall_time_s=elapsed_wall,
            sim_steps=self._step_idx,
            steps_per_sec=sps,
            checkpoint_sha256=checkpoint_sha256,
            config_sha256=cfg_hash,
            net_xml_sha256=net_hash,
        )
        return metrics


__all__ = [
    "SumoSimulationAdapter",
]

"""Canonical data contracts, feature schemas, firewall boundaries, and cross-slice protocols.

Architecture & Firewall Rules (enforced by static AST checks in ``tests/test_train_eval.py``):
1. Runtime perception/inference/control modules (``swarm/env/belief.py``, ``swarm/dbn/*``, and
   ``swarm/control/*``) MUST ONLY consume observable/filtered contracts (:class:`ObservationBatch`,
   :class:`SwarmBeliefState`, :class:`DBNStateSnapshot`, :class:`RoadTopology`, :class:`SwarmControlAction`).
2. Evaluator and 3D visualizer ground-truth contracts (:class:`VehicleRawTelemetry`,
   :class:`GroundTruthStepRecord`, :class:`EpisodeMetrics`) are strictly restricted to
   ``swarm/env/sumo_adapter.py``, ``swarm/pipelines/*``, and ``swarm/viz/*``.
"""

from __future__ import annotations

from collections.abc import Mapping, Set as AbstractSet
from dataclasses import asdict, dataclass
from enum import IntEnum
import math
from typing import Any, Final, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np

from .logging_utils import (
    NumericalInvariantError,
    TopologyIntegrityError,
    require_finite,
)

SCHEMA_VERSION: Final[int] = 2

NODE_FEATURE_DIM: Final[int] = 21
CORRIDOR_FEATURE_DIM: Final[int] = 12
EDGE_FEATURE_DIM: Final[int] = 18
CORRIDOR_EDGE_FEATURE_DIM: Final[int] = 6

LEFT_TURN_THRESHOLD_RAD: Final[float] = math.radians(20.0)

NODE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "rel_dist_to_stopline",
    "corridor_len_norm",
    "norm_corridor_progress",
    "v_norm",
    "v_abs_scaled",
    "accel_est_scaled",
    "pos_std_scaled",
    "vel_std_scaled",
    "obs_age_scaled",
    "pi_left",
    "pi_thru",
    "pi_right",
    "norm_intent_entropy",
    "prev_delay_post",
    "free_eta_scaled",
    "wait_time_scaled",
    "corridor_occ",
    "corridor_tail_block",
    "lane_norm",
    "exp_down_occ",
    "exp_down_tail_block",
)

CORRIDOR_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "length_norm",
    "speed_limit_norm",
    "effective_lanes_norm",
    "lane_open_ratio",
    "storage_cap_norm",
    "occupancy_ratio",
    "tail_blockage_prob",
    "expected_inflow_norm",
    "mean_corridor_speed_norm",
    "tls_green_fraction",
    "prev_spillback_prob",
    "is_fringe_indicator",
)

EDGE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "rel_same_corridor",
    "rel_same_junc_conflict",
    "rel_down_spillback",
    "rel_up_feeder",
    "signed_d_eta",
    "abs_d_eta",
    "signed_path_gap",
    "rel_speed",
    "p_conflict",
    "p_merge_outcorr",
    "eta_overlap_kernel",
    "same_lane",
    "joint_pos_std",
    "sum_obs_age",
    "sum_intent_entropy",
    "prev_edge_prob",
    "prior_edge_logit",
    "edge_persisted_flag",
)

CORRIDOR_EDGE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "turn_sin",
    "turn_cos",
    "is_thru",
    "is_backpressure_reverse",
    "tls_turn_green",
    "cap_ratio",
)

if len(NODE_FEATURE_NAMES) != NODE_FEATURE_DIM:
    raise AssertionError(f"NODE_FEATURE_NAMES length {len(NODE_FEATURE_NAMES)} != {NODE_FEATURE_DIM}")
if len(CORRIDOR_FEATURE_NAMES) != CORRIDOR_FEATURE_DIM:
    raise AssertionError(f"CORRIDOR_FEATURE_NAMES length {len(CORRIDOR_FEATURE_NAMES)} != {CORRIDOR_FEATURE_DIM}")
if len(EDGE_FEATURE_NAMES) != EDGE_FEATURE_DIM:
    raise AssertionError(f"EDGE_FEATURE_NAMES length {len(EDGE_FEATURE_NAMES)} != {EDGE_FEATURE_DIM}")
if len(CORRIDOR_EDGE_FEATURE_NAMES) != CORRIDOR_EDGE_FEATURE_DIM:
    raise AssertionError(
        f"CORRIDOR_EDGE_FEATURE_NAMES length {len(CORRIDOR_EDGE_FEATURE_NAMES)} != {CORRIDOR_EDGE_FEATURE_DIM}"
    )

MAIN_POLICY_NAMES: Final[tuple[str, ...]] = (
    "fixed_signal",
    "actuated_signal",
    "fcfs_reservation",
    "independent_belief",
    "heuristic_radius_dbn",
    "online_bic_dbn",
    "neural_dbn",
)

ABLATION_POLICY_NAMES: Final[tuple[str, ...]] = (
    "no_temporal_prior",
    "static_cpd",
    "fixed_route",
)

POLICY_NAMES: Final[tuple[str, ...]] = MAIN_POLICY_NAMES + ABLATION_POLICY_NAMES

NEURAL_POLICY_NAMES: Final[frozenset[str]] = frozenset(
    {
        "neural_dbn",
        "no_temporal_prior",
        "static_cpd",
        "fixed_route",
    }
)

SumoBackend: TypeAlias = Literal["libsumo", "traci"]
ComputeDevice: TypeAlias = Literal["cpu", "mps", "cuda"]
PrecisionMode: TypeAlias = Literal["fp32", "bf16"]
ModelPresetName: TypeAlias = Literal["ci_micro", "sota_local", "sota_cloud_xl"]


class RelationType(IntEnum):
    """Directed topological relationship class between candidate vehicle pairs ``(src -> dst)``."""

    SAME_CORRIDOR = 0
    SAME_JUNCTION_CONFLICT = 1
    DOWNSTREAM_SPILLBACK = 2
    UPSTREAM_FEEDER = 3


class LongitudinalMode(IntEnum):
    """Factored discrete longitudinal coordination action selected by Damped Max-Sum."""

    GO_WAVE = 0
    CRUISE = 1
    ECO_GLIDE = 2
    YIELD = 3


# =============================================================================
# 1. Road Network & Corridor Topology Contracts
# =============================================================================


@dataclass(frozen=True)
class TurnOption:
    """One legal outgoing turn movement from the terminal edge of a :class:`CorridorSpec`."""

    turn_idx: int
    from_edge: str
    to_corridor: str
    to_edge: str
    angle_rad: float
    tls_id: str | None
    tls_link_indices: tuple[int, ...]
    junction_req_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        require_finite(self.angle_rad, f"TurnOption[{self.from_edge}->{self.to_edge}].angle_rad")
        if self.turn_idx < 0:
            raise TopologyIntegrityError(
                f"Negative turn_idx={self.turn_idx} on {self.from_edge}->{self.to_edge}.",
                context={"turn_idx": self.turn_idx, "from_edge": self.from_edge, "to_edge": self.to_edge},
            )


@dataclass(frozen=True)
class CorridorSpec:
    """Unified street block between two junctions aggregating split trunk + turn-pocket edges.

    Boundary Lock A (``#19`` §1):
    - When ``is_exit_fringe == True``, ``len(turn_options) == 0`` is valid and downstream outgoing
      corridor context evaluates to zero by contract.
    - When ``is_exit_fringe == False``, ``len(turn_options) == 0`` raises :class:`TopologyIntegrityError`.
    """

    corridor_id: str
    edges: tuple[str, ...]
    edge_offsets_m: tuple[float, ...]
    from_junction: str
    to_junction: str
    length_m: float
    base_lanes: int
    speed_limit_mps: float
    turn_options: tuple[TurnOption, ...]
    is_entry_fringe: bool
    is_exit_fringe: bool

    def __post_init__(self) -> None:
        if not self.edges:
            raise TopologyIntegrityError(
                f"Corridor {self.corridor_id!r} has zero constituent edges.",
                context={"corridor_id": self.corridor_id},
            )
        if len(self.edges) != len(self.edge_offsets_m):
            raise TopologyIntegrityError(
                f"Corridor {self.corridor_id!r} edges/edge_offsets_m length mismatch.",
                context={
                    "corridor_id": self.corridor_id,
                    "num_edges": len(self.edges),
                    "num_offsets": len(self.edge_offsets_m),
                },
            )
        require_finite(self.length_m, f"CorridorSpec[{self.corridor_id}].length_m")
        require_finite(self.speed_limit_mps, f"CorridorSpec[{self.corridor_id}].speed_limit_mps")
        if self.length_m <= 0.0:
            raise TopologyIntegrityError(
                f"Corridor {self.corridor_id!r} must have positive length_m, got {self.length_m}.",
                context={"corridor_id": self.corridor_id, "length_m": self.length_m},
            )
        if self.base_lanes < 1:
            raise TopologyIntegrityError(
                f"Corridor {self.corridor_id!r} must have base_lanes >= 1, got {self.base_lanes}.",
                context={"corridor_id": self.corridor_id, "base_lanes": self.base_lanes},
            )
        if self.speed_limit_mps <= 0.0:
            raise TopologyIntegrityError(
                f"Corridor {self.corridor_id!r} must have positive speed_limit_mps, got {self.speed_limit_mps}.",
                context={"corridor_id": self.corridor_id, "speed_limit_mps": self.speed_limit_mps},
            )
        if not self.is_exit_fringe and len(self.turn_options) == 0:
            raise TopologyIntegrityError(
                f"Boundary Lock A violation: non-exit corridor {self.corridor_id!r} has zero turn_options.",
                context={
                    "corridor_id": self.corridor_id,
                    "to_junction": self.to_junction,
                    "is_exit_fringe": self.is_exit_fringe,
                },
            )


@dataclass(frozen=True)
class JunctionSpec:
    """Static geometry, signal link count, and exact connection foe matrices for one junction."""

    junction_id: str
    tls_id: str | None
    coord_xy: tuple[float, float]
    incoming_corridors: tuple[str, ...]
    outgoing_corridors: tuple[str, ...]
    num_tls_links: int
    tls_link_foe_matrix: np.ndarray
    req_foe_matrix: np.ndarray

    def __post_init__(self) -> None:
        require_finite(self.coord_xy[0], f"JunctionSpec[{self.junction_id}].x")
        require_finite(self.coord_xy[1], f"JunctionSpec[{self.junction_id}].y")
        if self.num_tls_links < 0:
            raise TopologyIntegrityError(
                f"Junction {self.junction_id!r} has negative num_tls_links={self.num_tls_links}.",
                context={"junction_id": self.junction_id, "num_tls_links": self.num_tls_links},
            )
        if self.tls_link_foe_matrix.ndim != 2 or self.tls_link_foe_matrix.shape != (
            self.num_tls_links,
            self.num_tls_links,
        ):
            raise TopologyIntegrityError(
                f"Junction {self.junction_id!r} tls_link_foe_matrix shape {self.tls_link_foe_matrix.shape} "
                f"!= ({self.num_tls_links}, {self.num_tls_links}).",
                context={
                    "junction_id": self.junction_id,
                    "actual_shape": list(self.tls_link_foe_matrix.shape),
                    "num_tls_links": self.num_tls_links,
                },
            )
        if self.req_foe_matrix.ndim != 2 or self.req_foe_matrix.shape[0] != self.req_foe_matrix.shape[1]:
            raise TopologyIntegrityError(
                f"Junction {self.junction_id!r} req_foe_matrix must be square 2D, got {self.req_foe_matrix.shape}.",
                context={"junction_id": self.junction_id, "actual_shape": list(self.req_foe_matrix.shape)},
            )


@dataclass(frozen=True)
class RoadTopology:
    """Complete topology-invariant representation of a procedural or real-world OSM SUMO network."""

    network_id: str
    net_xml_path: str
    poly_xml_path: str | None
    corridors: dict[str, CorridorSpec]
    edge_to_corridor: dict[str, str]
    edge_offset_in_corridor: dict[str, float]
    junctions: dict[str, JunctionSpec]
    successor_corridors: dict[str, tuple[str, ...]]
    predecessor_corridors: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        self.validate_integrity()

    def validate_integrity(self) -> None:
        """Fail loudly if any corridor, edge mapping, turn target, or Boundary Lock A check fails."""
        if not self.corridors:
            raise TopologyIntegrityError(
                f"RoadTopology {self.network_id!r} contains zero corridors.",
                context={"network_id": self.network_id},
            )
        if not self.junctions:
            raise TopologyIntegrityError(
                f"RoadTopology {self.network_id!r} contains zero junctions.",
                context={"network_id": self.network_id},
            )

        for cid, spec in self.corridors.items():
            if spec.corridor_id != cid:
                raise TopologyIntegrityError(
                    f"Corridor key {cid!r} != spec.corridor_id {spec.corridor_id!r}.",
                    context={"key": cid, "spec_id": spec.corridor_id},
                )
            if not spec.is_exit_fringe and len(spec.turn_options) == 0:
                raise TopologyIntegrityError(
                    f"Boundary Lock A violation on corridor {cid!r}: non-exit corridor has empty turn_options.",
                    context={"corridor_id": cid},
                )
            if spec.from_junction not in self.junctions or spec.to_junction not in self.junctions:
                raise TopologyIntegrityError(
                    f"Corridor {cid!r} references missing junction(s) "
                    f"({spec.from_junction!r}, {spec.to_junction!r}).",
                    context={
                        "corridor_id": cid,
                        "from_junction": spec.from_junction,
                        "to_junction": spec.to_junction,
                    },
                )
            for edge_id, offset in zip(spec.edges, spec.edge_offsets_m, strict=True):
                mapped = self.edge_to_corridor.get(edge_id)
                if mapped != cid:
                    raise TopologyIntegrityError(
                        f"Edge {edge_id!r} in corridor {cid!r} maps to {mapped!r} in edge_to_corridor.",
                        context={"edge_id": edge_id, "corridor_id": cid, "mapped": mapped},
                    )
                if edge_id not in self.edge_offset_in_corridor:
                    raise TopologyIntegrityError(
                        f"Edge {edge_id!r} missing from edge_offset_in_corridor.",
                        context={"edge_id": edge_id, "corridor_id": cid},
                    )
                require_finite(offset, f"CorridorSpec[{cid}].offset[{edge_id}]")

            for turn in spec.turn_options:
                if turn.to_corridor not in self.corridors:
                    raise TopologyIntegrityError(
                        f"Turn {turn.from_edge!r}->{turn.to_edge!r} targets unknown corridor {turn.to_corridor!r}.",
                        context={"corridor_id": cid, "to_corridor": turn.to_corridor},
                    )

    def turn_conflict(self, junction_id: str, opt_a: TurnOption, opt_b: TurnOption) -> bool:
        """Return ``True`` iff turn movements ``opt_a`` and ``opt_b`` conflict at ``junction_id``."""
        junc = self.junctions.get(junction_id)
        if junc is None:
            raise TopologyIntegrityError(
                f"Unknown junction_id {junction_id!r} in turn_conflict.",
                context={"junction_id": junction_id},
            )

        if opt_a.from_edge == opt_b.from_edge and opt_a.turn_idx == opt_b.turn_idx:
            return False

        checked = False
        if opt_a.junction_req_indices and opt_b.junction_req_indices and junc.req_foe_matrix.size > 0:
            checked = True
            n_req = junc.req_foe_matrix.shape[0]
            for r1 in opt_a.junction_req_indices:
                for r2 in opt_b.junction_req_indices:
                    if r1 < 0 or r1 >= n_req or r2 < 0 or r2 >= n_req:
                        raise TopologyIntegrityError(
                            f"Junction {junction_id!r} request index out of bounds ({r1}, {r2}) vs {n_req}.",
                            context={"junction_id": junction_id, "r1": r1, "r2": r2, "n_req": n_req},
                        )
                    if bool(junc.req_foe_matrix[r1, r2]) or bool(junc.req_foe_matrix[r2, r1]):
                        return True

        if opt_a.tls_link_indices and opt_b.tls_link_indices and junc.tls_link_foe_matrix.size > 0:
            checked = True
            n_tls = junc.num_tls_links
            for l1 in opt_a.tls_link_indices:
                for l2 in opt_b.tls_link_indices:
                    if l1 < 0 or l1 >= n_tls or l2 < 0 or l2 >= n_tls:
                        raise TopologyIntegrityError(
                            f"Junction {junction_id!r} TLS link index out of bounds ({l1}, {l2}) vs {n_tls}.",
                            context={"junction_id": junction_id, "l1": l1, "l2": l2, "n_tls": n_tls},
                        )
                    if bool(junc.tls_link_foe_matrix[l1, l2]) or bool(junc.tls_link_foe_matrix[l2, l1]):
                        return True

        if not checked and len(junc.incoming_corridors) > 1:
            raise TopologyIntegrityError(
                f"Multi-approach junction {junction_id!r} lacks request/TLS foe entries for "
                f"({opt_a.from_edge}->{opt_a.to_edge}) vs ({opt_b.from_edge}->{opt_b.to_edge}).",
                context={
                    "junction_id": junction_id,
                    "opt_a": f"{opt_a.from_edge}->{opt_a.to_edge}",
                    "opt_b": f"{opt_b.from_edge}->{opt_b.to_edge}",
                },
            )
        return False

    def effective_lanes(self, corridor_id: str, open_lanes_by_edge: Mapping[str, int]) -> int:
        """Return bottleneck open lane count across ``corridor_id`` given observable closures."""
        spec = self.corridors.get(corridor_id)
        if spec is None:
            raise TopologyIntegrityError(
                f"Unknown corridor_id {corridor_id!r} in effective_lanes.",
                context={"corridor_id": corridor_id},
            )
        counts = [max(0, int(open_lanes_by_edge.get(edge_id, spec.base_lanes))) for edge_id in spec.edges]
        return min(counts)

    def effective_capacity(
        self,
        corridor_id: str,
        open_lanes_by_edge: Mapping[str, int],
        jam_spacing_m: float = 7.0,
    ) -> int:
        """Return observable vehicle storage capacity ``C_e^eff(t) >= 1`` for ``corridor_id``."""
        if jam_spacing_m <= 0.0:
            raise NumericalInvariantError(
                f"jam_spacing_m must be positive, got {jam_spacing_m}.",
                context={"jam_spacing_m": jam_spacing_m},
            )
        spec = self.corridors.get(corridor_id)
        if spec is None:
            raise TopologyIntegrityError(
                f"Unknown corridor_id {corridor_id!r} in effective_capacity.",
                context={"corridor_id": corridor_id},
            )
        lane_meters = 0.0
        n_edges = len(spec.edges)
        for idx, edge_id in enumerate(spec.edges):
            next_offset = spec.edge_offsets_m[idx + 1] if idx + 1 < n_edges else spec.length_m
            seg_len = max(0.0, next_offset - spec.edge_offsets_m[idx])
            open_lanes = max(0, int(open_lanes_by_edge.get(edge_id, spec.base_lanes)))
            lane_meters += open_lanes * seg_len
        return max(1, int(math.floor(lane_meters / jam_spacing_m)))


# =============================================================================
# 2. Runtime Observation, Belief, DBN, & Control Contracts (Firewall-Safe)
# =============================================================================


@dataclass(frozen=True)
class DepartedVehicleEvent:
    """Observable perimeter/entry detector event when a vehicle physically enters the road.

    ``entry_sim_time`` MUST equal the current simulation clock ``sim_time`` and NEVER expose
    unobservable virtual-queue ``departDelay`` or scheduled departure timestamps (``#13`` §1).
    """

    veh_id: str
    entry_sim_time: float
    entry_edge_id: str
    entry_lane_index: int
    length_m: float
    width_m: float


@dataclass(frozen=True)
class NoisyVehicleObservation:
    """Range-limited, dropout-affected Gaussian radar + noisy turn-indicator measurement."""

    veh_id: str
    obs_time: float
    edge_id: str
    corridor_id: str
    in_junction_id: str | None
    lane_index: int
    noisy_corridor_pos_m: float
    noisy_speed_mps: float
    pos_noise_std: float
    speed_noise_std: float
    turn_likelihood: tuple[float, ...]
    noisy_leader_gap_m: float | None
    noisy_leader_speed_mps: float | None
    leader_id: str | None


@dataclass(frozen=True)
class V2XPacket:
    """Delayed, dropout-subject budgeted V2X status packet delivered at ``received_time``."""

    veh_id: str
    sent_time: float
    received_time: float
    edge_id: str
    corridor_id: str
    lane_index: int
    noisy_corridor_pos_m: float
    noisy_speed_mps: float
    pos_noise_std: float
    speed_noise_std: float
    turn_likelihood: tuple[float, ...]


@dataclass(frozen=True)
class ObservationBatch:
    """Complete runtime-observable sensor & V2X packet batch emitted by ``SumoSimulationAdapter``."""

    sim_time: float
    dt: float
    active_veh_ids: tuple[str, ...]
    departed: tuple[DepartedVehicleEvent, ...]
    arrived_veh_ids: tuple[str, ...]
    radar_obs: tuple[NoisyVehicleObservation, ...]
    v2x_packets: tuple[V2XPacket, ...]
    open_lanes_by_edge: dict[str, int]
    tls_states: dict[str, str]


@dataclass(frozen=True)
class VehicleBelief:
    """Filtered per-vehicle 2D kinematic state, covariance, and degree-agnostic turn belief."""

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
    obs_age_s: float
    wait_time_s: float
    turn_probs: tuple[float, ...]
    free_eta_s: float
    eta_std_s: float
    tail_blockage_score: float
    leader_id: str | None
    noisy_leader_gap_m: float | None
    noisy_leader_speed_mps: float | None
    last_v2x_poll_time: float


@dataclass(frozen=True)
class CorridorBeliefStats:
    """Observable macroscopic congestion statistics aggregated strictly from :class:`VehicleBelief`."""

    corridor_id: str
    active_count: int
    effective_lanes: int
    storage_capacity: int
    occupancy_ratio: float
    tail_blockage_prob: float
    expected_inflow: float
    mean_speed_mps: float


@dataclass(frozen=True)
class SwarmBeliefState:
    """Joint city-wide filtered belief state ``b_t`` passed to DBN updaters and controllers."""

    sim_time: float
    dt: float
    vehicles: dict[str, VehicleBelief]
    corridors: dict[str, CorridorBeliefStats]
    open_lanes_by_edge: dict[str, int]
    tls_states: dict[str, str]


@dataclass(frozen=True)
class DBNTemporalMemory:
    """Recurrent latent GRU hidden states carried across steps by ``STGraphTransformerDBN``.

    Omitted from JSON 3D trace serialization; cleared on ``DBNUpdaterProtocol.reset()``.
    """

    sim_time: float
    veh_hidden: dict[str, np.ndarray]
    corridor_hidden: dict[str, np.ndarray]
    edge_hidden: dict[tuple[str, str], np.ndarray]


@dataclass(frozen=True)
class DBNEdge:
    """Directed acyclic intra-slice dependency ``src_id -> dst_id`` with amortized CPD weights."""

    src_id: str
    dst_id: str
    relation: RelationType
    omega: float
    kappa: float
    gamma: float
    p_conflict: float
    p_merge: float


@dataclass(frozen=True)
class DBNStateSnapshot:
    """2-slice Dynamic Bayesian Network structure, marginals, cascade values, and latent memory."""

    sim_time: float
    veh_ids: tuple[str, ...]
    precedence_scores: dict[str, float]
    edges: tuple[DBNEdge, ...]
    edges_by_dst: dict[str, tuple[DBNEdge, ...]]
    edges_by_src: dict[str, tuple[DBNEdge, ...]]
    delay_pred: dict[str, float]
    delay_post: dict[str, float]
    expected_delay_s: dict[str, float]
    cascade_score: dict[str, float]
    corridor_spillback_pred: dict[str, float]
    raw_edge_probs: dict[tuple[str, str], float]
    update_latency_ms: float
    inference_latency_ms: float
    temporal_memory: DBNTemporalMemory | None = None

    def to_trace_dict(self, max_edges: int | None = None) -> dict[str, Any]:
        """Serialize DBN snapshot for 3D WebGL replay, explicitly excluding ``temporal_memory``."""
        selected_edges = self.edges
        if max_edges is not None and len(selected_edges) > max_edges:
            selected_edges = tuple(sorted(selected_edges, key=lambda e: e.omega, reverse=True)[:max_edges])
        return {
            "sim_time": self.sim_time,
            "precedence_scores": dict(self.precedence_scores),
            "delay_pred": dict(self.delay_pred),
            "delay_post": dict(self.delay_post),
            "expected_delay_s": dict(self.expected_delay_s),
            "cascade_score": dict(self.cascade_score),
            "corridor_spillback_pred": dict(self.corridor_spillback_pred),
            "update_latency_ms": self.update_latency_ms,
            "inference_latency_ms": self.inference_latency_ms,
            "edges": [
                {
                    "src_id": e.src_id,
                    "dst_id": e.dst_id,
                    "relation": int(e.relation),
                    "omega": e.omega,
                    "kappa": e.kappa,
                    "gamma": e.gamma,
                    "p_conflict": e.p_conflict,
                    "p_merge": e.p_merge,
                }
                for e in selected_edges
            ],
        }


@dataclass(frozen=True)
class SwarmControlAction:
    """Joint longitudinal speed, TLS link grant, public link-cost broadcast, and V2X poll command."""

    sim_time: float
    target_speeds: dict[str, float]
    post_safety_accels: dict[str, float]
    modes: dict[str, LongitudinalMode]
    tls_desired_greens: Mapping[str, AbstractSet[int]] | None
    broadcast_edge_costs: dict[str, float] | None
    reroute_hysteresis_s: float = 3.0
    v2x_polls: tuple[str, ...] = ()
    safety_clamp_count: int = 0
    zero_speed_deadzone_count: int = 0
    control_latency_ms: float = 0.0


# =============================================================================
# 3. Ground-Truth Telemetry & Evaluation Contracts (FORBIDDEN in Runtime Control)
# =============================================================================


@dataclass(frozen=True)
class VehicleRawTelemetry:
    """Privileged SUMO ground-truth vehicle state for offline labeling, metrics, and 3D replay."""

    veh_id: str
    vtype: str
    center_xy: tuple[float, float]
    yaw_ccw_rad: float
    speed: float
    accel: float
    road_id: str
    lane_id: str
    lane_index: int
    lane_pos: float
    lane_length: float
    route_edges: tuple[str, ...]
    route_index: int
    route_goal: str
    length: float
    width: float
    brake_light: bool
    turn_signal: int
    leader_id: str | None
    leader_gap: float | None
    time_loss: float
    waiting_time: float
    co2_mg_s: float
    scheduled_depart: float = 0.0


@dataclass(frozen=True)
class GroundTruthStepRecord:
    """Per-step privileged SUMO telemetry and physical safety counters emitted alongside observations."""

    sim_time: float
    raw_vehicles: tuple[VehicleRawTelemetry, ...]
    collisions: int
    teleports: int
    emergency_stops: int
    emergency_brakes: int
    near_misses_ttc: int
    min_ttc_s: float
    reroutes_applied: int
    v2x_sent: int
    v2x_delivered: int
    step_co2_mg: float
    spillback_corridor_count: int = 0


@dataclass(frozen=True)
class EpisodeMetrics:
    """Complete held-out cohort, safety, communication, latency, and provenance summary for 1 episode."""

    scenario_id: str
    policy_name: str
    seed: int
    cohort_size: int
    cohort_completed_count: int
    cohort_unfinished_count: int
    cohort_undeparted_count: int
    cohort_completion_rate: float
    cohort_mean_total_delay_s: float
    cohort_p95_total_delay_s: float
    cohort_mean_completed_delay_s: float
    throughput_veh_per_hr: float
    mean_stops_per_veh: float
    spillback_corridor_seconds: float
    mean_co2_g_per_veh: float
    reroutes_per_veh: float
    collision_count: int
    teleport_count: int
    emergency_stop_count: int
    emergency_brake_count: int
    emergency_brake_rate_per_veh_km: float
    ttc_near_miss_count: int
    min_ttc_s: float
    safety_filter_clamp_count: int
    safety_filter_clamp_rate: float
    zero_speed_deadzone_count: int
    v2x_polls_requested: int
    v2x_packets_delivered: int
    v2x_delivery_efficiency: float
    mean_dbn_update_latency_ms: float
    mean_dbn_inference_latency_ms: float
    mean_control_latency_ms: float
    edge_churn_hamming_rate: float
    wall_time_s: float
    sim_steps: int
    steps_per_sec: float
    checkpoint_sha256: str | None = None
    config_sha256: str | None = None
    net_xml_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a flat JSON/CSV-ready dictionary of all episode metrics."""
        return asdict(self)


# =============================================================================
# 4. Cross-Slice Runtime Protocols
# =============================================================================


@runtime_checkable
class SimulationAdapterProtocol(Protocol):
    """Protocol implemented by ``swarm.env.SumoSimulationAdapter``."""

    @property
    def topology(self) -> RoadTopology: ...

    def reset(self, seed: int | None = None) -> tuple[ObservationBatch, GroundTruthStepRecord]: ...

    def step(self, action: SwarmControlAction) -> tuple[ObservationBatch, GroundTruthStepRecord]: ...

    def close(self) -> None: ...


@runtime_checkable
class BeliefTrackerProtocol(Protocol):
    """Protocol implemented by ``swarm.env.CityBeliefTracker``."""

    def reset(self) -> None: ...

    def observe(self, obs_batch: ObservationBatch) -> SwarmBeliefState: ...

    def predict(
        self,
        dt: float,
        post_safety_accels: Mapping[str, float],
        v2x_polled_ids: tuple[str, ...] = (),
        sim_time: float | None = None,
    ) -> None: ...


@runtime_checkable
class DBNUpdaterProtocol(Protocol):
    """Protocol implemented by ``NeuralDBNUpdater`` and baseline DBN updaters in ``swarm.dbn``."""

    def reset(self) -> None: ...

    def update_and_infer(
        self,
        belief: SwarmBeliefState,
        prev_snapshot: DBNStateSnapshot | None,
        topology: RoadTopology,
    ) -> DBNStateSnapshot: ...


@runtime_checkable
class PolicyControllerProtocol(Protocol):
    """Protocol implemented by all 10 controllers/ablations in ``swarm.control``."""

    @property
    def policy_name(self) -> str: ...

    @property
    def last_dbn_snapshot(self) -> DBNStateSnapshot | None: ...

    def reset(self) -> None: ...

    def act(self, obs_batch: ObservationBatch) -> SwarmControlAction: ...


@runtime_checkable
class TraceRecorderProtocol(Protocol):
    """Protocol implemented by ``swarm.viz.EpisodeTraceRecorder``."""

    def record_step(
        self,
        gt_record: GroundTruthStepRecord,
        belief: SwarmBeliefState | None,
        dbn_snapshot: DBNStateSnapshot | None,
        action: SwarmControlAction,
        tls_states: Mapping[str, str],
    ) -> None: ...

    def finalize(self, metrics: EpisodeMetrics | None = None) -> dict[str, Any]: ...


__all__ = [
    "ABLATION_POLICY_NAMES",
    "CORRIDOR_EDGE_FEATURE_DIM",
    "CORRIDOR_EDGE_FEATURE_NAMES",
    "CORRIDOR_FEATURE_DIM",
    "CORRIDOR_FEATURE_NAMES",
    "ComputeDevice",
    "CorridorBeliefStats",
    "CorridorSpec",
    "DBNEdge",
    "DBNStateSnapshot",
    "DBNTemporalMemory",
    "DBNUpdaterProtocol",
    "DepartedVehicleEvent",
    "EDGE_FEATURE_DIM",
    "EDGE_FEATURE_NAMES",
    "EpisodeMetrics",
    "GroundTruthStepRecord",
    "JunctionSpec",
    "LEFT_TURN_THRESHOLD_RAD",
    "LongitudinalMode",
    "MAIN_POLICY_NAMES",
    "ModelPresetName",
    "NEURAL_POLICY_NAMES",
    "NODE_FEATURE_DIM",
    "NODE_FEATURE_NAMES",
    "NoisyVehicleObservation",
    "ObservationBatch",
    "POLICY_NAMES",
    "PolicyControllerProtocol",
    "PrecisionMode",
    "RelationType",
    "RoadTopology",
    "SCHEMA_VERSION",
    "SimulationAdapterProtocol",
    "SumoBackend",
    "SwarmBeliefState",
    "SwarmControlAction",
    "TraceRecorderProtocol",
    "TurnOption",
    "V2XPacket",
    "VehicleBelief",
    "VehicleRawTelemetry",
    "BeliefTrackerProtocol",
]

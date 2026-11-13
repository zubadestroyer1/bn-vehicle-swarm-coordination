"""Reproducible SUMO urban network builders, static/actuated dual TLS programs, and seeded OD routing.

Builds:
1. Procedural multi-intersection city grids (``grid3x3.net.xml``, ``grid4x4.net.xml``) with
   2-lane arterials, 45m dedicated left-turn pockets, ``--tls.discard-simple true``,
   ``--tls.ungroup-signals true``, and dual embedded TLS programs (``programID="0"`` actuated
   and ``programID="fixed"`` static cycle) for fair signal baseline comparison.
2. Real-world OpenStreetMap Downtown Palo Alto (``palo_alto_clean.net.xml`` + 3D building
   footprints ``palo_alto_buildings.poly.xml``) via ``netconvert`` and ``polyconvert``.
3. Deterministic per-``(scenario_id, seed)`` exogenous multi-OD trip & ``duarouter`` route files
   so every controller faces identical vehicle arrivals, vehicle classes, and OD destinations.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import sumolib

from ..config import (
    DEFAULT_VEHICLE_TYPES,
    IncidentSpec,
    ObservationConfig,
    ScenarioConfig,
    load_scenario_config,
    save_config_json,
)
from ..contracts import RoadTopology
from ..logging_utils import (
    BackendUnavailableError,
    ConfigValidationError,
    RoutingIntegrityError,
    TopologyIntegrityError,
    get_logger,
)

PALO_ALTO_BBOX: tuple[float, float, float, float] = (-122.1660, 37.4415, -122.1550, 37.4490)
PALO_ALTO_OSM_API_URL: str = (
    f"https://api.openstreetmap.org/api/0.6/map?"
    f"bbox={PALO_ALTO_BBOX[0]},{PALO_ALTO_BBOX[1]},{PALO_ALTO_BBOX[2]},{PALO_ALTO_BBOX[3]}"
)


def resolve_sumo_bin_dir() -> Path:
    """Locate the native SUMO binary directory from ``eclipse-sumo`` or ``SUMO_HOME`` (fail-loud)."""
    if importlib.util.find_spec("sumo") is not None:
        import sumo  # type: ignore[import-untyped]

        sumo_home = Path(sumo.SUMO_HOME)
        os.environ.setdefault("SUMO_HOME", str(sumo_home))
        bin_dir = sumo_home / "bin"
        if (bin_dir / "sumo").is_file():
            return bin_dir

    env_home = os.environ.get("SUMO_HOME")
    if env_home:
        bin_dir = Path(env_home) / "bin"
        if (bin_dir / "sumo").is_file():
            return bin_dir

    raise BackendUnavailableError(
        "Unable to locate SUMO binaries ('sumo', 'netgenerate', 'netconvert', 'duarouter'); "
        "install eclipse-sumo==1.28.0 or set SUMO_HOME.",
        context={"SUMO_HOME": env_home},
    )


def _run_sumo_cli(cmd: list[str], tool_name: str) -> None:
    """Execute a SUMO CLI tool and raise :class:`TopologyIntegrityError` or :class:`RoutingIntegrityError`."""
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err_cls = RoutingIntegrityError if tool_name == "duarouter" else TopologyIntegrityError
        raise err_cls(
            f"SUMO {tool_name} failed with exit code {proc.returncode}: {proc.stderr.strip()}",
            context={
                "tool": tool_name,
                "returncode": proc.returncode,
                "cmd": cmd,
                "stderr": proc.stderr[-1500:],
                "stdout": proc.stdout[-500:],
            },
        )


def inject_static_fixed_tls_programs(net_xml_path: Path) -> None:
    """Ensure every actuated ``<tlLogic>`` in ``net_xml_path`` has a companion ``programID='fixed'`` static plan.

    In SUMO, ``programID="0"`` remains the default gap-based ``actuated`` controller used by
    ``actuated_signal``, while ``programID="fixed"`` provides an exact fixed-cycle ``static``
    controller on identical signal phases for ``fixed_signal``.
    """
    if not net_xml_path.is_file():
        raise TopologyIntegrityError(
            f"Cannot inject static TLS program into missing network XML: {net_xml_path}",
            context={"net_xml_path": str(net_xml_path)},
        )

    tree = ET.parse(net_xml_path)
    root = tree.getroot()

    existing_programs: set[tuple[str, str]] = set()
    tl_elements = root.findall("tlLogic")
    for tl in tl_elements:
        existing_programs.add((tl.attrib.get("id", ""), tl.attrib.get("programID", "0")))

    insert_positions: list[tuple[int, ET.Element]] = []
    children = list(root)
    for idx, elem in enumerate(children):
        if elem.tag != "tlLogic":
            continue
        tls_id = elem.attrib.get("id", "")
        prog_id = elem.attrib.get("programID", "0")
        if prog_id != "0" or (tls_id, "fixed") in existing_programs:
            continue

        fixed_elem = copy.deepcopy(elem)
        fixed_elem.attrib["type"] = "static"
        fixed_elem.attrib["programID"] = "fixed"
        for param in list(fixed_elem.findall("param")):
            fixed_elem.remove(param)
        for phase in fixed_elem.findall("phase"):
            phase.attrib.pop("minDur", None)
            phase.attrib.pop("maxDur", None)
            phase.attrib.pop("next", None)
        insert_positions.append((idx + 1, fixed_elem))
        existing_programs.add((tls_id, "fixed"))

    for offset, (pos, fixed_elem) in enumerate(insert_positions):
        root.insert(pos + offset, fixed_elem)

    if insert_positions:
        tmp_path = net_xml_path.parent / f".{net_xml_path.name}.pid{os.getpid()}.tmp"
        tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
        os.replace(tmp_path, net_xml_path)


def build_procedural_grid_network(
    output_net_xml: Path | str,
    *,
    grid_number: int,
    block_length_m: float = 150.0,
    attach_length_m: float = 120.0,
    lanes: int = 2,
    turn_lanes: int = 1,
    turn_lane_length_m: float = 45.0,
    speed_limit_mps: float = 13.89,
) -> Path:
    """Build a multi-intersection urban grid via ``netgenerate`` with actuated + fixed TLS programs."""
    if grid_number < 2:
        raise ConfigValidationError(
            f"grid_number must be >= 2, got {grid_number}.",
            context={"grid_number": grid_number},
        )
    bin_dir = resolve_sumo_bin_dir()
    out_path = Path(output_net_xml)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(bin_dir / "netgenerate"),
        "--grid",
        "--grid.number",
        str(grid_number),
        "--grid.length",
        f"{block_length_m:.2f}",
        "--grid.attach-length",
        f"{attach_length_m:.2f}",
        "--default.lanenumber",
        str(lanes),
        "--turn-lanes",
        str(turn_lanes),
        "--turn-lanes.length",
        f"{turn_lane_length_m:.2f}",
        "--default.speed",
        f"{speed_limit_mps:.2f}",
        "--default.junctions.keep-clear",
        "true",
        "-j",
        "traffic_light",
        "--tls.discard-simple",
        "true",
        "--tls.ungroup-signals",
        "true",
        "--tls.default-type",
        "actuated",
        "--tls.min-dur",
        "5",
        "--tls.max-dur",
        "35",
        "--tls.yellow.time",
        "3",
        "--tls.left-green.time",
        "8",
        "--no-turnarounds",
        "true",
        "--output-file",
        str(out_path),
        "--no-warnings",
        "true",
    ]
    _run_sumo_cli(cmd, "netgenerate")
    inject_static_fixed_tls_programs(out_path)
    return out_path


def fetch_or_reuse_palo_alto_osm(
    osm_cache_path: Path | str,
    *,
    allow_download: bool = True,
) -> Path:
    """Return a valid raw OpenStreetMap XML for Downtown Palo Alto, downloading only if needed."""
    target = Path(osm_cache_path)
    if target.is_file() and target.stat().st_size > 10_000:
        return target

    probe_cache = Path("/tmp/sumo_scout_probe/osm_palo_alto/palo_alto_downtown.osm.xml")
    if probe_cache.is_file() and probe_cache.stat().st_size > 10_000:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(probe_cache.read_bytes())
        return target

    if not allow_download:
        raise ConfigValidationError(
            f"Cached OSM XML not found at {target} and allow_download=False.",
            context={"osm_cache_path": str(target)},
        )

    get_logger("env.network_builder").info(
        "Downloading Downtown Palo Alto OpenStreetMap extract from %s", PALO_ALTO_OSM_API_URL
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        PALO_ALTO_OSM_API_URL,
        headers={"User-Agent": "Stanford-CS238-BN-Vehicle-Swarm/2.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    if len(data) < 10_000:
        raise TopologyIntegrityError(
            f"Downloaded OSM payload from {PALO_ALTO_OSM_API_URL} is unexpectedly small ({len(data)} bytes).",
            context={"byte_length": len(data)},
        )
    tmp_path = target.parent / f".{target.name}.pid{os.getpid()}.tmp"
    tmp_path.write_bytes(data)
    os.replace(tmp_path, target)
    return target


def build_palo_alto_osm_network(
    output_net_xml: Path | str,
    output_poly_xml: Path | str,
    *,
    osm_source_xml: Path | str | None = None,
    allow_download: bool = True,
) -> tuple[Path, Path]:
    """Convert Downtown Palo Alto OSM into ``palo_alto_clean.net.xml`` + 3D building ``.poly.xml``."""
    bin_dir = resolve_sumo_bin_dir()
    net_out = Path(output_net_xml)
    poly_out = Path(output_poly_xml)
    net_out.parent.mkdir(parents=True, exist_ok=True)
    poly_out.parent.mkdir(parents=True, exist_ok=True)

    cache_path = (
        Path(osm_source_xml)
        if osm_source_xml is not None
        else Path("/tmp/sumo_scout_probe/osm_palo_alto/palo_alto_downtown.osm.xml")
    )
    osm_file = fetch_or_reuse_palo_alto_osm(cache_path, allow_download=allow_download)

    netconvert_cmd = [
        str(bin_dir / "netconvert"),
        "--osm-files",
        str(osm_file),
        "--output-file",
        str(net_out),
        "--output.street-names",
        "true",
        "--keep-edges.by-type",
        "highway.primary,highway.primary_link,highway.secondary,highway.secondary_link,"
        "highway.tertiary,highway.tertiary_link,highway.residential",
        "--keep-edges.by-vclass",
        "passenger",
        "--keep-edges.components",
        "1",
        "--geometry.remove",
        "true",
        "--roundabouts.guess",
        "true",
        "--ramps.guess",
        "true",
        "--junctions.join",
        "true",
        "--junctions.join-dist",
        "15",
        "--tls.guess-signals",
        "true",
        "--tls.discard-simple",
        "true",
        "--tls.ungroup-signals",
        "true",
        "--tls.join",
        "true",
        "--tls.default-type",
        "actuated",
        "--tls.min-dur",
        "5",
        "--tls.max-dur",
        "35",
        "--tls.yellow.time",
        "3",
        "--default.junctions.keep-clear",
        "true",
        "--no-turnarounds",
        "true",
        "--no-warnings",
        "true",
    ]
    _run_sumo_cli(netconvert_cmd, "netconvert")
    inject_static_fixed_tls_programs(net_out)

    polyconvert_cmd = [
        str(bin_dir / "polyconvert"),
        "--net-file",
        str(net_out),
        "--osm-files",
        str(osm_file),
        "--output-file",
        str(poly_out),
        "--no-warnings",
        "true",
    ]
    _run_sumo_cli(polyconvert_cmd, "polyconvert")
    return net_out, poly_out


def _deterministic_scenario_salt(scenario_id: str, network_id: str) -> int:
    digest = hashlib.sha256(f"{scenario_id}:{network_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def generate_episode_routes(
    scenario: ScenarioConfig,
    seed: int,
    output_dir: Path | str,
    *,
    topology: RoadTopology | None = None,
    repo_root: Path | str | None = None,
) -> tuple[Path, dict[str, float]]:
    """Generate deterministic multi-OD ``.rou.xml`` and ``scheduled_departs[veh_id] -> t_sched``.

    Vehicles are injected across ``[0.0, scenario.cohort_end_s)``, followed by ``drain_duration_s``
    with zero new injections. RNG streams depend strictly on ``(scenario, seed)`` so all policies
    evaluated on the same ``seed`` receive bit-identical vehicle arrivals, types, and OD routes.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigValidationError(
            f"Episode seed must be a non-negative integer, got {seed!r}.",
            context={"seed": repr(seed)},
        )

    net_path, _ = scenario.resolve_paths(repo_root=repo_root, must_exist=True)
    bin_dir = resolve_sumo_bin_dir()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    net = sumolib.net.readNet(str(net_path))
    normal_edges = [e for e in net.getEdges() if not e.isSpecial() and e.allows("passenger")]
    if not normal_edges:
        raise TopologyIntegrityError(
            f"Network {net_path} has zero passenger-drivable edges.",
            context={"net_xml_path": str(net_path)},
        )

    if topology is not None:
        entry_corridors = [c for c in topology.corridors.values() if c.is_entry_fringe]
        exit_corridors = [c for c in topology.corridors.values() if c.is_exit_fringe]
        edge_map = {e.getID(): e for e in normal_edges}
        entry_edges = [edge_map[c.edges[0]] for c in entry_corridors if c.edges[0] in edge_map]
        exit_edges = [edge_map[c.edges[-1]] for c in exit_corridors if c.edges[-1] in edge_map]
    else:
        entry_edges = [
            e
            for e in normal_edges
            if len([ie for ie in e.getFromNode().getIncoming() if not ie.isSpecial()]) == 0
        ]
        exit_edges = [
            e
            for e in normal_edges
            if len([oe for oe in e.getToNode().getOutgoing() if not oe.isSpecial()]) == 0
        ]

    # On real-world urban networks (e.g. Palo Alto OSM), mix boundary feeders with interior arterials >= 35m.
    long_edges = [e for e in normal_edges if e.getLength() >= 35.0]
    if len(entry_edges) < 8:
        entry_edges = long_edges if len(long_edges) >= 4 else normal_edges
    if len(exit_edges) < 8:
        exit_edges = long_edges if len(long_edges) >= 4 else normal_edges

    # Compute network bounding box for SW -> NE asymmetric directional commuter wave
    xs = [n.getCoord()[0] for n in net.getNodes()]
    ys = [n.getCoord()[1] for n in net.getNodes()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)

    sw_origins = [
        e
        for e in entry_edges
        if (e.getFromNode().getCoord()[0] - min_x) / span_x <= 0.42
        or (e.getFromNode().getCoord()[1] - min_y) / span_y <= 0.42
    ]
    ne_destinations = [
        e
        for e in exit_edges
        if (e.getToNode().getCoord()[0] - min_x) / span_x >= 0.58
        or (e.getToNode().getCoord()[1] - min_y) / span_y >= 0.58
    ]
    if not sw_origins:
        sw_origins = entry_edges
    if not ne_destinations:
        ne_destinations = exit_edges

    salt = _deterministic_scenario_salt(scenario.scenario_id, scenario.network_id)
    rng = np.random.default_rng(np.random.SeedSequence([seed, salt, 701]))

    vtypes = scenario.vehicle_types
    shares = np.array([vt.share for vt in vtypes], dtype=np.float64)
    shares /= shares.sum()

    routes_root = ET.Element("routes")
    for vt in vtypes:
        ET.SubElement(
            routes_root,
            "vType",
            id=vt.vtype_id,
            carFollowModel=vt.car_follow_model,
            laneChangeModel=vt.lane_change_model,
            length=f"{vt.length_m:.2f}",
            width=f"{vt.width_m:.2f}",
            minGap=f"{vt.min_gap_m:.2f}",
            tau=f"{vt.tau_s:.2f}",
            accel=f"{vt.accel_mps2:.2f}",
            decel=f"{vt.decel_mps2:.2f}",
            emergencyDecel=f"{vt.emergency_decel_mps2:.2f}",
            maxSpeed=f"{vt.max_speed_mps:.2f}",
            speedDev=f"{vt.speed_dev:.3f}",
        )

    horizon_s = scenario.cohort_end_s
    rate = scenario.demand_rate_veh_per_s
    min_dist = scenario.min_od_euclidean_dist_m

    t_now = 0.0
    trip_counter = 0
    tentative_departs: dict[str, float] = {}

    while t_now < horizon_s:
        t_now += float(rng.exponential(1.0 / rate))
        if t_now >= horizon_s:
            break

        use_wave = scenario.asymmetric_rush_wave and (float(rng.random()) < scenario.asymmetric_wave_fraction)
        origin_pool = sw_origins if use_wave else entry_edges
        dest_pool = ne_destinations if use_wave else exit_edges

        src_edge = origin_pool[int(rng.integers(len(origin_pool)))]
        sx, sy = src_edge.getFromNode().getCoord()

        valid_dsts = [
            e
            for e in dest_pool
            if e.getID() != src_edge.getID()
            and e.getToNode().getID() != src_edge.getFromNode().getID()
            and math.hypot(e.getToNode().getCoord()[0] - sx, e.getToNode().getCoord()[1] - sy) >= min_dist
        ]
        if not valid_dsts:
            valid_dsts = [
                e
                for e in exit_edges
                if e.getID() != src_edge.getID() and e.getToNode().getID() != src_edge.getFromNode().getID()
            ]
        if not valid_dsts:
            raise RoutingIntegrityError(
                f"No valid destination edge reachable from origin {src_edge.getID()!r} in {scenario.scenario_id!r}.",
                context={"origin_edge": src_edge.getID(), "scenario_id": scenario.scenario_id},
            )

        dst_edge = valid_dsts[int(rng.integers(len(valid_dsts)))]
        vt_spec = vtypes[int(rng.choice(len(vtypes), p=shares))]
        veh_id = f"veh_{trip_counter:04d}"
        depart_t = round(t_now, 2)
        tentative_departs[veh_id] = depart_t

        trip_el = ET.SubElement(
            routes_root,
            "trip",
            id=veh_id,
            type=vt_spec.vtype_id,
            depart=f"{depart_t:.2f}",
            departLane="best",
            departSpeed="max",
        )
        trip_el.set("from", src_edge.getID())
        trip_el.set("to", dst_edge.getID())
        trip_counter += 1

    trips_path = out_dir / f"{scenario.scenario_id}_seed{seed}.trips.xml"
    rou_path = out_dir / f"{scenario.scenario_id}_seed{seed}.rou.xml"
    ET.ElementTree(routes_root).write(trips_path, encoding="utf-8", xml_declaration=True)

    duarouter_seed = int((seed + (zlib.crc32(scenario.scenario_id.encode("utf-8")) & 0xFFFF)) & 0x7FFFFFFF)
    duarouter_cmd = [
        str(bin_dir / "duarouter"),
        "--net-file",
        str(net_path),
        "--route-files",
        str(trips_path),
        "--output-file",
        str(rou_path),
        "--remove-loops",
        "true",
        "--ignore-errors",
        "true",
        "--seed",
        str(duarouter_seed),
        "--no-warnings",
        "true",
    ]
    _run_sumo_cli(duarouter_cmd, "duarouter")

    if not rou_path.is_file():
        raise RoutingIntegrityError(
            f"duarouter did not produce output route file {rou_path}.",
            context={"rou_path": str(rou_path)},
        )

    routed_tree = ET.parse(rou_path).getroot()
    scheduled_departs: dict[str, float] = {}
    for veh_elem in routed_tree.findall("vehicle"):
        vid = veh_elem.attrib["id"]
        scheduled_departs[vid] = float(veh_elem.attrib["depart"])

    if len(scheduled_departs) == 0:
        raise RoutingIntegrityError(
            f"Zero valid connected vehicle routes generated for scenario={scenario.scenario_id!r}, seed={seed}.",
            context={"scenario_id": scenario.scenario_id, "seed": seed, "attempted_trips": trip_counter},
        )

    return rou_path, scheduled_departs


def default_scenario_configs() -> dict[str, ScenarioConfig]:
    """Return the 5 canonical HCM/NGSIM-calibrated ID & Zero-Shot OOD :class:`ScenarioConfig` specs."""
    return {
        "grid3x3_nominal": ScenarioConfig(
            scenario_id="grid3x3_nominal",
            network_id="grid3x3",
            net_xml_path="configs/networks/grid3x3.net.xml",
            poly_xml_path=None,
            sumo_backend="libsumo",
            step_length_s=0.5,
            warmup_duration_s=60.0,
            cohort_duration_s=240.0,
            drain_duration_s=180.0,
            demand_rate_veh_per_s=1.25,
            asymmetric_rush_wave=False,
            asymmetric_wave_fraction=0.55,
            min_od_euclidean_dist_m=220.0,
            is_ood_regime=False,
            incidents=(),
            vehicle_types=DEFAULT_VEHICLE_TYPES,
            observation=ObservationConfig(),
        ),
        "grid3x3_rush_asymmetric": ScenarioConfig(
            scenario_id="grid3x3_rush_asymmetric",
            network_id="grid3x3",
            net_xml_path="configs/networks/grid3x3.net.xml",
            poly_xml_path=None,
            sumo_backend="libsumo",
            step_length_s=0.5,
            warmup_duration_s=60.0,
            cohort_duration_s=240.0,
            drain_duration_s=210.0,
            demand_rate_veh_per_s=2.10,
            asymmetric_rush_wave=True,
            asymmetric_wave_fraction=0.60,
            min_od_euclidean_dist_m=220.0,
            is_ood_regime=False,
            incidents=(),
            vehicle_types=DEFAULT_VEHICLE_TYPES,
            observation=ObservationConfig(),
        ),
        "grid3x3_noisy_dropout": ScenarioConfig(
            scenario_id="grid3x3_noisy_dropout",
            network_id="grid3x3",
            net_xml_path="configs/networks/grid3x3.net.xml",
            poly_xml_path=None,
            sumo_backend="libsumo",
            step_length_s=0.5,
            warmup_duration_s=60.0,
            cohort_duration_s=240.0,
            drain_duration_s=210.0,
            demand_rate_veh_per_s=1.95,
            asymmetric_rush_wave=True,
            asymmetric_wave_fraction=0.55,
            min_od_euclidean_dist_m=220.0,
            is_ood_regime=False,
            incidents=(),
            vehicle_types=DEFAULT_VEHICLE_TYPES,
            observation=ObservationConfig(
                radar_range_m=85.0,
                pos_noise_std_m=3.0,
                speed_noise_std_mps=1.125,
                sensor_dropout_prob=0.35,
                turn_sensor_accuracy=0.68,
                v2x_pos_noise_std_m=0.50,
                v2x_speed_noise_std_mps=0.20,
                v2x_turn_accuracy=0.94,
                v2x_dropout_prob=0.40,
                v2x_max_delay_steps=3,
                v2x_budget_per_step=6,
                v2x_poll_cooldown_s=1.0,
                kalman_process_pos_var=0.80,
                kalman_process_vel_var=1.40,
                intent_forgetting_rate=0.05,
            ),
        ),
        "grid4x4_incident": ScenarioConfig(
            scenario_id="grid4x4_incident",
            network_id="grid4x4",
            net_xml_path="configs/networks/grid4x4.net.xml",
            poly_xml_path=None,
            sumo_backend="libsumo",
            step_length_s=0.5,
            warmup_duration_s=60.0,
            cohort_duration_s=240.0,
            drain_duration_s=240.0,
            demand_rate_veh_per_s=2.85,
            asymmetric_rush_wave=True,
            asymmetric_wave_fraction=0.60,
            min_od_euclidean_dist_m=280.0,
            is_ood_regime=True,
            incidents=(
                IncidentSpec(
                    edge_ids=("B1C1", "B2C2"),
                    start_time_s=120.0,
                    end_time_s=260.0,
                    closed_lane_indices=(0,),
                    reduced_speed_mps=4.5,
                ),
            ),
            vehicle_types=DEFAULT_VEHICLE_TYPES,
            observation=ObservationConfig(),
        ),
        "osm_palo_alto_rush": ScenarioConfig(
            scenario_id="osm_palo_alto_rush",
            network_id="palo_alto_clean",
            net_xml_path="configs/networks/palo_alto_clean.net.xml",
            poly_xml_path="configs/networks/palo_alto_buildings.poly.xml",
            sumo_backend="libsumo",
            step_length_s=0.5,
            warmup_duration_s=60.0,
            cohort_duration_s=240.0,
            drain_duration_s=240.0,
            demand_rate_veh_per_s=1.55,
            asymmetric_rush_wave=True,
            asymmetric_wave_fraction=0.55,
            min_od_euclidean_dist_m=280.0,
            is_ood_regime=True,
            incidents=(),
            vehicle_types=DEFAULT_VEHICLE_TYPES,
            observation=ObservationConfig(),
        ),
    }


def build_all_networks_and_scenarios(
    repo_root: Path | str | None = None,
    *,
    allow_osm_download: bool = True,
    write_scenarios: bool = True,
) -> dict[str, Path]:
    """Build ``grid3x3``, ``grid4x4``, & Downtown Palo Alto OSM networks + 5 scenario JSONs."""
    base = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    nets_dir = base / "configs" / "networks"
    scen_dir = base / "configs" / "scenarios"
    nets_dir.mkdir(parents=True, exist_ok=True)

    g3_path = build_procedural_grid_network(nets_dir / "grid3x3.net.xml", grid_number=3)
    g4_path = build_procedural_grid_network(nets_dir / "grid4x4.net.xml", grid_number=4)
    pa_net, pa_poly = build_palo_alto_osm_network(
        nets_dir / "palo_alto_clean.net.xml",
        nets_dir / "palo_alto_buildings.poly.xml",
        allow_download=allow_osm_download,
    )

    outputs: dict[str, Path] = {
        "grid3x3": g3_path,
        "grid4x4": g4_path,
        "palo_alto_clean": pa_net,
        "palo_alto_buildings": pa_poly,
    }

    if write_scenarios:
        scen_dir.mkdir(parents=True, exist_ok=True)
        for scen_id, cfg in default_scenario_configs().items():
            json_path = scen_dir / f"{scen_id}.json"
            save_config_json(cfg, json_path)
            loaded = load_scenario_config(json_path, must_exist_network_files=True, repo_root=base)
            # Validate incident edge IDs exist in target network
            if loaded.incidents:
                net = sumolib.net.readNet(str(base / loaded.net_xml_path))
                known_edges = {e.getID(): e for e in net.getEdges() if not e.isSpecial()}
                for inc in loaded.incidents:
                    for eid in inc.edge_ids:
                        if eid not in known_edges:
                            raise TopologyIntegrityError(
                                f"Incident edge {eid!r} missing from {loaded.net_xml_path}.",
                                context={"scenario_id": scen_id, "edge_id": eid},
                            )
                        n_lanes = len(known_edges[eid].getLanes())
                        for lidx in inc.closed_lane_indices:
                            if lidx >= n_lanes:
                                raise TopologyIntegrityError(
                                    f"Incident lane index {lidx} out of range on edge {eid!r} (lanes={n_lanes}).",
                                    context={"edge_id": eid, "lane_index": lidx, "num_lanes": n_lanes},
                                )
            outputs[f"scenario:{scen_id}"] = json_path

    return outputs


__all__ = [
    "PALO_ALTO_BBOX",
    "PALO_ALTO_OSM_API_URL",
    "build_all_networks_and_scenarios",
    "build_palo_alto_osm_network",
    "build_procedural_grid_network",
    "default_scenario_configs",
    "fetch_or_reuse_palo_alto_osm",
    "generate_episode_routes",
    "inject_static_fixed_tls_programs",
    "resolve_sumo_bin_dir",
]

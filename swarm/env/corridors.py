"""Unified corridor aggregation, turn-angle sorting, and junction foe-matrix extraction.

Implements:
- Merging of consecutive split trunk + turn-pocket SUMO edges between junctions into a single
  :class:`~swarm.contracts.CorridorSpec` (``#13`` §2).
- Degree-agnostic :class:`~swarm.contracts.TurnOption` extraction ordered by signed CCW turn angle
  ``angle_rad`` in ``[-pi, pi]`` (``> +20 deg`` left, ``[-20, +20]`` thru, ``< -20 deg`` right).
- Exact ``<junction><request index=k foes=.../>`` bitmask parsing and TLS link conflict matrix
  construction on :class:`~swarm.contracts.JunctionSpec`.
- Strict Boundary Lock A enforcement (:class:`~swarm.logging_utils.TopologyIntegrityError` on any
  broken interior corridor, missing foe table, or unmapped edge — zero synthetic fallbacks).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import sumolib

from ..config import ScenarioConfig
from ..contracts import CorridorSpec, JunctionSpec, RoadTopology, TurnOption
from ..logging_utils import ConfigValidationError, TopologyIntegrityError


@dataclass(frozen=True)
class InternalEdgeMapping:
    """O(1) resolution of SUMO ``':'`` internal edges to unified corridor & junction coordinates."""

    corridor_id: str
    anchor_edge_id: str
    corridor_pos_m: float
    in_junction_id: str | None


_TOPOLOGY_CACHE: dict[tuple[str, int, str | None], RoadTopology] = {}
_INTERNAL_MAP_CACHE: dict[tuple[str, int], dict[str, InternalEdgeMapping]] = {}


def _edge_unit_tangent(edge: sumolib.net.edge.Edge, *, at_end: bool) -> tuple[float, float]:
    shape = edge.getShape()
    if len(shape) < 2:
        raise TopologyIntegrityError(
            f"Edge {edge.getID()!r} has degenerate shape with fewer than 2 points.",
            context={"edge_id": edge.getID(), "num_points": len(shape)},
        )
    if at_end:
        x0, y0 = shape[-2]
        x1, y1 = shape[-1]
    else:
        x0, y0 = shape[0]
        x1, y1 = shape[1]
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        # Use full start-to-end vector if terminal segment has zero length
        dx = float(shape[-1][0] - shape[0][0])
        dy = float(shape[-1][1] - shape[0][1])
        norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        raise TopologyIntegrityError(
            f"Edge {edge.getID()!r} has zero geometric length in shape coordinates.",
            context={"edge_id": edge.getID()},
        )
    return dx / norm, dy / norm


def compute_turn_angle_rad(
    from_edge: sumolib.net.edge.Edge,
    to_edge: sumolib.net.edge.Edge,
) -> float:
    """Compute signed CCW turn angle in ``[-pi, pi]`` (positive = left, negative = right)."""
    ux_in, uy_in = _edge_unit_tangent(from_edge, at_end=True)
    ux_out, uy_out = _edge_unit_tangent(to_edge, at_end=False)
    cross = ux_in * uy_out - uy_in * ux_out
    dot = ux_in * ux_out + uy_in * uy_out
    return float(math.atan2(cross, dot))


def _passenger_lane_count(edge: sumolib.net.edge.Edge) -> int:
    count = sum(1 for lane in edge.getLanes() if lane.allows("passenger"))
    return count


def _non_turnaround_passenger_connections(
    edge: sumolib.net.edge.Edge,
    valid_edge_ids: set[str],
) -> list[sumolib.net.connection.Connection]:
    out: list[sumolib.net.connection.Connection] = []
    for target_edge, conn_list in edge.getOutgoing().items():
        if target_edge.isSpecial() or target_edge.getID() not in valid_edge_ids:
            continue
        for conn in conn_list:
            if conn.getDirection() in ("t", "T"):
                continue
            from_lane = conn.getFromLane()
            to_lane = conn.getToLane()
            if from_lane.allows("passenger") and to_lane.allows("passenger"):
                out.append(conn)
    return out


def _parse_junction_foe_tables(
    xml_root: ET.Element,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Parse ``<junction><request index=k foes=.../>`` and ``<tlLogic>`` link counts from XML."""
    req_matrices: dict[str, np.ndarray] = {}
    for j_elem in xml_root.findall("junction"):
        jid = j_elem.attrib.get("id", "")
        jtype = j_elem.attrib.get("type", "")
        if jtype == "internal":
            continue
        reqs = j_elem.findall("request")
        if not reqs:
            req_matrices[jid] = np.zeros((0, 0), dtype=bool)
            continue
        n_req = len(reqs)
        mat = np.zeros((n_req, n_req), dtype=bool)
        seen_indices: set[int] = set()
        for r_elem in reqs:
            idx = int(r_elem.attrib["index"])
            foes_bits_str = r_elem.attrib.get("foes", "")
            if idx < 0 or idx >= n_req or len(foes_bits_str) != n_req:
                raise TopologyIntegrityError(
                    f"Malformed <request> entry at junction {jid!r}: index={idx}, len(foes)={len(foes_bits_str)}, n_req={n_req}.",
                    context={"junction_id": jid, "index": idx, "foes": foes_bits_str, "n_req": n_req},
                )
            seen_indices.add(idx)
            mask = int(foes_bits_str, 2)
            for k2 in range(n_req):
                if k2 != idx and ((mask >> k2) & 1):
                    mat[idx, k2] = True
                    mat[k2, idx] = True
        if len(seen_indices) != n_req:
            raise TopologyIntegrityError(
                f"Incomplete <request> indices at junction {jid!r}: got {sorted(seen_indices)}.",
                context={"junction_id": jid},
            )
        req_matrices[jid] = mat

    tls_link_counts: dict[str, int] = {}
    for tl_elem in xml_root.findall("tlLogic"):
        tls_id = tl_elem.attrib.get("id", "")
        prog_id = tl_elem.attrib.get("programID", "0")
        phases = tl_elem.findall("phase")
        if not phases:
            raise TopologyIntegrityError(
                f"<tlLogic id={tls_id!r}> has zero <phase> definitions.",
                context={"tls_id": tls_id, "program_id": prog_id},
            )
        state_len = len(phases[0].attrib.get("state", ""))
        if state_len == 0:
            raise TopologyIntegrityError(
                f"<tlLogic id={tls_id!r}> has empty phase state string.",
                context={"tls_id": tls_id},
            )
        for ph in phases:
            if len(ph.attrib.get("state", "")) != state_len:
                raise TopologyIntegrityError(
                    f"Inconsistent phase state lengths inside <tlLogic id={tls_id!r}>.",
                    context={"tls_id": tls_id},
                )
        if tls_id in tls_link_counts and tls_link_counts[tls_id] != state_len:
            raise TopologyIntegrityError(
                f"Conflicting link counts across programs of TLS {tls_id!r}.",
                context={"tls_id": tls_id, "a": tls_link_counts[tls_id], "b": state_len},
            )
        tls_link_counts[tls_id] = state_len

    return req_matrices, tls_link_counts


def load_or_build_topology(
    source: ScenarioConfig | Path | str,
    *,
    network_id: str | None = None,
    poly_xml_path: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> RoadTopology:
    """Parse and validate a SUMO ``.net.xml`` into a unified :class:`~swarm.contracts.RoadTopology`."""
    resolved_poly_str: str | None = None
    if isinstance(source, ScenarioConfig):
        net_path, poly_p = source.resolve_paths(repo_root=repo_root, must_exist=True)
        net_id = network_id or source.network_id
        resolved_poly_str = str(poly_p) if poly_p is not None else None
    elif isinstance(source, (Path, str)):
        net_path = Path(source)
        if not net_path.is_absolute() and repo_root is not None:
            net_path = (Path(repo_root) / net_path).resolve()
        else:
            net_path = net_path.resolve()
        if not net_path.is_file():
            raise ConfigValidationError(
                f"Network XML file does not exist: {net_path}",
                context={"net_xml_path": str(net_path)},
            )
        net_id = network_id or net_path.stem.replace(".net", "")
        if poly_xml_path is not None:
            pp = Path(poly_xml_path)
            if not pp.is_absolute() and repo_root is not None:
                pp = (Path(repo_root) / pp).resolve()
            else:
                pp = pp.resolve()
            if not pp.is_file():
                raise ConfigValidationError(
                    f"Polygon XML file does not exist: {pp}",
                    context={"poly_xml_path": str(pp)},
                )
            resolved_poly_str = str(pp)
    else:
        raise ConfigValidationError(
            f"Unsupported source type {type(source).__name__} passed to load_or_build_topology."
        )

    mtime_ns = net_path.stat().st_mtime_ns
    cache_key = (str(net_path), mtime_ns, resolved_poly_str)
    if cache_key in _TOPOLOGY_CACHE:
        cached = _TOPOLOGY_CACHE[cache_key]
        if cached.network_id == net_id:
            return cached

    xml_root = ET.parse(net_path).getroot()
    req_matrices, tls_link_counts = _parse_junction_foe_tables(xml_root)
    net = sumolib.net.readNet(str(net_path), withInternal=True)

    normal_edges = [e for e in net.getEdges() if not e.isSpecial() and _passenger_lane_count(e) > 0]
    if not normal_edges:
        raise TopologyIntegrityError(
            f"Network {net_path} contains zero passenger-drivable normal edges.",
            context={"net_xml_path": str(net_path)},
        )

    edge_by_id: dict[str, sumolib.net.edge.Edge] = {e.getID(): e for e in normal_edges}
    valid_edge_ids = set(edge_by_id.keys())

    # Identify TLS-controlled junction node IDs
    tls_by_node_id: dict[str, str] = {}
    for tls in net.getTrafficLights():
        tls_id = tls.getID()
        for in_lane, _, _ in tls.getConnections():
            nid = in_lane.getEdge().getToNode().getID()
            if nid in tls_by_node_id and tls_by_node_id[nid] != tls_id:
                raise TopologyIntegrityError(
                    f"Junction {nid!r} is controlled by multiple distinct TLS IDs "
                    f"({tls_by_node_id[nid]!r}, {tls_id!r}).",
                    context={"junction_id": nid},
                )
            tls_by_node_id[nid] = tls_id

    # Map outgoing/incoming target edges per edge (excluding turnarounds)
    out_targets_of_edge: dict[str, set[str]] = {}
    in_sources_of_edge: dict[str, set[str]] = {eid: set() for eid in valid_edge_ids}
    conns_by_edge: dict[str, list[sumolib.net.connection.Connection]] = {}

    for eid, edge in edge_by_id.items():
        conns = _non_turnaround_passenger_connections(edge, valid_edge_ids)
        conns_by_edge[eid] = conns
        targets = {c.getTo().getID() for c in conns}
        out_targets_of_edge[eid] = targets
        for tgt in targets:
            in_sources_of_edge[tgt].add(eid)

    # Identify pure internal split/continuation nodes (e.g. A0A1 -> A0A1.105 turn-pocket split)
    continuation_next: dict[str, str] = {}
    continuation_prev: dict[str, str] = {}
    split_node_ids: set[str] = set()

    for node in net.getNodes():
        nid = node.getID()
        ntype = node.getType()
        if nid in tls_by_node_id or "traffic_light" in ntype:
            continue
        in_e = [e.getID() for e in node.getIncoming() if e.getID() in valid_edge_ids]
        out_e = [e.getID() for e in node.getOutgoing() if e.getID() in valid_edge_ids]
        if not in_e or len(in_e) != len(out_e) or len(in_e) > 2:
            continue

        req_mat = req_matrices.get(nid, np.zeros((0, 0), dtype=bool))
        if req_mat.size > 0 and bool(req_mat.any()):
            continue

        # Verify strict 1-to-1 bijection without U-turns at nid
        bijective_pairs: list[tuple[str, str]] = []
        valid_split = True
        for ie_id in in_e:
            tgts = out_targets_of_edge[ie_id]
            if len(tgts) != 1:
                valid_split = False
                break
            oe_id = next(iter(tgts))
            if oe_id not in out_e or len(in_sources_of_edge[oe_id]) != 1:
                valid_split = False
                break
            if edge_by_id[oe_id].getToNode().getID() == edge_by_id[ie_id].getFromNode().getID():
                valid_split = False
                break
            bijective_pairs.append((ie_id, oe_id))

        if valid_split and len(bijective_pairs) == len(in_e):
            split_node_ids.add(nid)
            for ie_id, oe_id in bijective_pairs:
                continuation_next[ie_id] = oe_id
                continuation_prev[oe_id] = ie_id

    # Trace maximal corridor edge chains between true junctions
    visited_edges: set[str] = set()
    raw_chains: list[list[str]] = []

    # First trace chains starting from edges with no continuation_prev
    for eid in sorted(valid_edge_ids):
        if eid in visited_edges or eid in continuation_prev:
            continue
        chain: list[str] = []
        cur: str | None = eid
        while cur is not None and cur not in visited_edges:
            visited_edges.add(cur)
            chain.append(cur)
            nxt = continuation_next.get(cur)
            # Break before closing a pure cycle back to chain[0]'s start node
            if nxt is not None and edge_by_id[nxt].getToNode().getID() == edge_by_id[chain[0]].getFromNode().getID():
                break
            cur = nxt
        raw_chains.append(chain)

    # Catch any remaining closed continuation loops deterministically
    for eid in sorted(valid_edge_ids):
        if eid in visited_edges:
            continue
        chain = []
        cur = eid
        while cur is not None and cur not in visited_edges:
            visited_edges.add(cur)
            chain.append(cur)
            cur = continuation_next.get(cur)
        raw_chains.append(chain)

    # Assign unique corridor IDs (first edge ID if unique, which is always unique because edges partition!)
    edge_to_corridor: dict[str, str] = {}
    edge_offset_in_corridor: dict[str, float] = {}
    chain_by_cid: dict[str, list[str]] = {}

    for chain in raw_chains:
        cid = chain[0]
        chain_by_cid[cid] = chain
        cum_offset = 0.0
        for seg_eid in chain:
            edge_to_corridor[seg_eid] = cid
            edge_offset_in_corridor[seg_eid] = float(cum_offset)
            cum_offset += float(edge_by_id[seg_eid].getLength())

    # Collect true junctions referenced as endpoints of corridors
    junction_node_ids: set[str] = set()
    incoming_corrs_by_junc: dict[str, list[str]] = {}
    outgoing_corrs_by_junc: dict[str, list[str]] = {}

    for cid, chain in sorted(chain_by_cid.items()):
        from_jid = edge_by_id[chain[0]].getFromNode().getID()
        to_jid = edge_by_id[chain[-1]].getToNode().getID()
        junction_node_ids.add(from_jid)
        junction_node_ids.add(to_jid)
        outgoing_corrs_by_junc.setdefault(from_jid, []).append(cid)
        incoming_corrs_by_junc.setdefault(to_jid, []).append(cid)

    # Build TurnOptions and CorridorSpecs
    corridors: dict[str, CorridorSpec] = {}
    successor_corridors: dict[str, tuple[str, ...]] = {}
    predecessor_set: dict[str, set[str]] = {cid: set() for cid in chain_by_cid}

    for cid, chain in sorted(chain_by_cid.items()):
        first_edge = edge_by_id[chain[0]]
        last_edge = edge_by_id[chain[-1]]
        from_jid = first_edge.getFromNode().getID()
        to_jid = last_edge.getToNode().getID()

        offsets = tuple(edge_offset_in_corridor[eid] for eid in chain)
        total_len = float(sum(edge_by_id[eid].getLength() for eid in chain))
        base_lanes = min(_passenger_lane_count(edge_by_id[eid]) for eid in chain)
        # Length-weighted mean speed limit across segments
        weighted_speed = sum(
            float(edge_by_id[eid].getSpeed()) * float(edge_by_id[eid].getLength()) for eid in chain
        ) / max(total_len, 1e-6)

        # Group outgoing non-turnaround connections from last_edge by target corridor
        grouped_conns: dict[str, list[sumolib.net.connection.Connection]] = {}
        for conn in conns_by_edge[last_edge.getID()]:
            to_eid = conn.getTo().getID()
            to_cid = edge_to_corridor.get(to_eid)
            if to_cid is None:
                raise TopologyIntegrityError(
                    f"Outgoing edge {to_eid!r} from {last_edge.getID()!r} is missing from edge_to_corridor.",
                    context={"from_edge": last_edge.getID(), "to_edge": to_eid},
                )
            grouped_conns.setdefault(to_cid, []).append(conn)

        unsorted_turns: list[tuple[float, str, str, str | None, tuple[int, ...], tuple[int, ...]]] = []
        for to_cid, c_list in grouped_conns.items():
            target_first_eid = chain_by_cid[to_cid][0]
            to_edge_obj = edge_by_id[target_first_eid]
            angle = compute_turn_angle_rad(last_edge, to_edge_obj)
            tls_ids_present = {c.getTLSID() for c in c_list if c.getTLSID()}
            if len(tls_ids_present) > 1:
                raise TopologyIntegrityError(
                    f"Turn {last_edge.getID()!r}->{target_first_eid!r} spans multiple TLS IDs {tls_ids_present}.",
                    context={"from_edge": last_edge.getID(), "to_edge": target_first_eid},
                )
            tls_id_val = next(iter(tls_ids_present)) if tls_ids_present else tls_by_node_id.get(to_jid)
            tls_links = tuple(sorted({int(c.getTLLinkIndex()) for c in c_list if int(c.getTLLinkIndex()) >= 0}))
            req_indices = tuple(
                sorted({int(c.getJunctionIndex()) for c in c_list if int(c.getJunctionIndex()) >= 0})
            )
            unsorted_turns.append((angle, to_cid, target_first_eid, tls_id_val, tls_links, req_indices))

        unsorted_turns.sort(key=lambda item: (item[0], item[1], item[2]))
        turn_options = tuple(
            TurnOption(
                turn_idx=idx,
                from_edge=last_edge.getID(),
                to_corridor=to_cid,
                to_edge=to_eid,
                angle_rad=angle,
                tls_id=tls_id_val,
                tls_link_indices=tls_links,
                junction_req_indices=req_indices,
            )
            for idx, (angle, to_cid, to_eid, tls_id_val, tls_links, req_indices) in enumerate(unsorted_turns)
        )

        is_entry = len(in_sources_of_edge[chain[0]]) == 0

        # Boundary Lock A: A corridor has is_exit_fringe == True iff it terminates at a genuine
        # boundary/cul-de-sac sink (zero outgoing connections AND to_jid has no non-returning outgoing corridors).
        if len(turn_options) == 0:
            non_returning_out = [
                ocid
                for ocid in outgoing_corrs_by_junc.get(to_jid, [])
                if edge_by_id[chain_by_cid[ocid][-1]].getToNode().getID() != from_jid
            ]
            is_exit = len(non_returning_out) == 0
        else:
            is_exit = False

        corridors[cid] = CorridorSpec(
            corridor_id=cid,
            edges=tuple(chain),
            edge_offsets_m=offsets,
            from_junction=from_jid,
            to_junction=to_jid,
            length_m=total_len,
            base_lanes=base_lanes,
            speed_limit_mps=float(weighted_speed),
            turn_options=turn_options,
            is_entry_fringe=is_entry,
            is_exit_fringe=is_exit,
        )

        succs = tuple(t.to_corridor for t in turn_options)
        successor_corridors[cid] = succs
        for scid in succs:
            predecessor_set[scid].add(cid)

    predecessor_corridors: dict[str, tuple[str, ...]] = {
        cid: tuple(sorted(preds)) for cid, preds in predecessor_set.items()
    }

    # Build JunctionSpecs with exact req_foe_matrix & tls_link_foe_matrix
    junctions: dict[str, JunctionSpec] = {}
    for jid in sorted(junction_node_ids):
        node = net.getNode(jid)
        cx, cy = node.getCoord()
        inc_corrs = tuple(sorted(incoming_corrs_by_junc.get(jid, [])))
        out_corrs = tuple(sorted(outgoing_corrs_by_junc.get(jid, [])))
        tls_id = tls_by_node_id.get(jid)

        req_mat = req_matrices.get(jid, np.zeros((0, 0), dtype=bool))
        # Fail loudly if a multi-approach junction with active turns has empty <request> foe matrix
        total_turns_across_junc = sum(len(corridors[ic].turn_options) for ic in inc_corrs)
        if len(inc_corrs) > 1 and total_turns_across_junc > 1 and req_mat.size == 0:
            raise TopologyIntegrityError(
                f"Multi-approach junction {jid!r} is missing <request index=k foes=.../> entries in {net_path}.",
                context={"junction_id": jid, "incoming_corridors": list(inc_corrs)},
            )

        if tls_id is not None:
            if tls_id not in tls_link_counts:
                raise TopologyIntegrityError(
                    f"Junction {jid!r} references TLS {tls_id!r} missing from <tlLogic> definitions.",
                    context={"junction_id": jid, "tls_id": tls_id},
                )
            n_tls = tls_link_counts[tls_id]
            tls_foe = np.zeros((n_tls, n_tls), dtype=bool)
            # Map each TLS link index -> set of junction request indices at jid
            reqs_for_tls_link: dict[int, set[int]] = {l_idx: set() for l_idx in range(n_tls)}
            for conn in node.getConnections():
                tl_idx = int(conn.getTLLinkIndex())
                rq_idx = int(conn.getJunctionIndex())
                if 0 <= tl_idx < n_tls and 0 <= rq_idx < req_mat.shape[0]:
                    reqs_for_tls_link[tl_idx].add(rq_idx)

            for l1 in range(n_tls):
                for l2 in range(l1 + 1, n_tls):
                    conflict = any(
                        bool(req_mat[r1, r2])
                        for r1 in reqs_for_tls_link[l1]
                        for r2 in reqs_for_tls_link[l2]
                    )
                    if conflict:
                        tls_foe[l1, l2] = True
                        tls_foe[l2, l1] = True
        else:
            n_tls = 0
            tls_foe = np.zeros((0, 0), dtype=bool)

        junctions[jid] = JunctionSpec(
            junction_id=jid,
            tls_id=tls_id,
            coord_xy=(float(cx), float(cy)),
            incoming_corridors=inc_corrs,
            outgoing_corridors=out_corrs,
            num_tls_links=n_tls,
            tls_link_foe_matrix=tls_foe,
            req_foe_matrix=req_mat,
        )

    topology = RoadTopology(
        network_id=net_id,
        net_xml_path=str(net_path),
        poly_xml_path=resolved_poly_str,
        corridors=corridors,
        edge_to_corridor=edge_to_corridor,
        edge_offset_in_corridor=edge_offset_in_corridor,
        junctions=junctions,
        successor_corridors=successor_corridors,
        predecessor_corridors=predecessor_corridors,
    )
    _TOPOLOGY_CACHE[cache_key] = topology
    return topology


def build_internal_edge_lookup(topology: RoadTopology) -> dict[str, InternalEdgeMapping]:
    """Precompute O(1) mapping for all SUMO ``':'`` internal edges (junctions & split pockets)."""
    net_path = Path(topology.net_xml_path)
    mtime_ns = net_path.stat().st_mtime_ns
    cache_key = (str(net_path), mtime_ns)
    if cache_key in _INTERNAL_MAP_CACHE:
        return _INTERNAL_MAP_CACHE[cache_key]

    net = sumolib.net.readNet(str(net_path), withInternal=True)
    mapping: dict[str, InternalEdgeMapping] = {}

    for edge in net.getEdges():
        if not edge.isSpecial():
            continue
        eid = edge.getID()
        # BFS forward through chained internal edges (e.g. left-turn waiting internal lane) to normal edge
        q: deque[sumolib.net.edge.Edge] = deque([edge])
        seen: set[str] = {eid}
        target_normal: str | None = None
        while q:
            cur = q.popleft()
            for out_e in cur.getOutgoing():
                oid = out_e.getID()
                if not out_e.isSpecial() and oid in topology.edge_to_corridor:
                    target_normal = oid
                    break
                if out_e.isSpecial() and oid not in seen:
                    seen.add(oid)
                    q.append(out_e)
            if target_normal is not None:
                break

        # Also check incoming normal edge in case an internal edge terminates at a dead-end
        if target_normal is None:
            q_in: deque[sumolib.net.edge.Edge] = deque([edge])
            seen_in: set[str] = {eid}
            while q_in:
                cur = q_in.popleft()
                for in_e in cur.getIncoming():
                    iid = in_e.getID()
                    if not in_e.isSpecial() and iid in topology.edge_to_corridor:
                        target_normal = iid
                        break
                    if in_e.isSpecial() and iid not in seen_in:
                        seen_in.add(iid)
                        q_in.append(in_e)
                if target_normal is not None:
                    break

        if target_normal is None:
            # Non-passenger internal edge (e.g. pedestrian/rail crossing if any); skip safely
            continue

        cid = topology.edge_to_corridor[target_normal]
        offset = topology.edge_offset_in_corridor[target_normal]

        # Extract node ID from ':NODE_ID_idx' by checking known topology junctions (longest prefix match)
        node_id = edge.getFromNode().getID()
        if node_id not in topology.junctions:
            # Internal edge may have internal waiting node; strip leading ':' and match known junction IDs
            raw_body = eid[1:] if eid.startswith(":") else eid
            matched_jid: str | None = None
            for cand_jid in topology.junctions:
                if raw_body.startswith(f"{cand_jid}_"):
                    if matched_jid is None or len(cand_jid) > len(matched_jid):
                        matched_jid = cand_jid
            in_junc = matched_jid
        else:
            in_junc = node_id

        mapping[eid] = InternalEdgeMapping(
            corridor_id=cid,
            anchor_edge_id=target_normal,
            corridor_pos_m=float(offset),
            in_junction_id=in_junc,
        )

    _INTERNAL_MAP_CACHE[cache_key] = mapping
    return mapping


__all__ = [
    "InternalEdgeMapping",
    "build_internal_edge_lookup",
    "compute_turn_angle_rad",
    "load_or_build_topology",
]

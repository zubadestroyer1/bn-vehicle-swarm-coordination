"""3D urban scene geometry extractor from SUMO ``.net.xml`` and OSM ``.poly.xml`` artifacts.

Extracts multi-lane asphalt ribbons, lane divider markings, stop bars, directional turn arrows,
triangulated multi-vertex junction polygons, zebra crosswalks, 3D traffic signal poles/heads mapped
1-to-1 to controlled TLS link indices, and either real OpenStreetMap 3D extruded building footprints
(``poly_xml_path``) or setback-verified procedural urban city blocks when ``poly_xml_path is None``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import sumolib

from ..contracts import LEFT_TURN_THRESHOLD_RAD, RoadTopology
from ..logging_utils import (
    ConfigValidationError,
    NumericalInvariantError,
    TopologyIntegrityError,
    require_finite,
)

SCENE_SCHEMA_VERSION = 2


def _stable_unit_hash(seed_key: str) -> float:
    """Return deterministic float in ``[0, 1)`` from ``seed_key``."""
    digest = hashlib.blake2b(seed_key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _signed_polygon_area(pts: Sequence[tuple[float, float]]) -> float:
    """Return signed 2D shoelace area (positive iff counter-clockwise)."""
    n = len(pts)
    if n < 3:
        return 0.0
    acc = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        acc += x0 * y1 - x1 * y0
    return 0.5 * acc


def _clean_polygon_ring(
    raw_pts: Sequence[Sequence[float]],
    *,
    min_edge_m: float = 0.08,
) -> list[tuple[float, float]]:
    """Remove duplicate closing/colinear vertices and normalize polygon ring to CCW winding."""
    pts: list[tuple[float, float]] = []
    for p in raw_pts:
        x = float(p[0])
        y = float(p[1])
        require_finite(x, "polygon.x")
        require_finite(y, "polygon.y")
        if not pts or math.hypot(x - pts[-1][0], y - pts[-1][1]) >= min_edge_m:
            pts.append((x, y))

    if len(pts) >= 2 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < min_edge_m:
        pts.pop()

    if len(pts) < 3:
        return []

    area = _signed_polygon_area(pts)
    if abs(area) < 1e-3:
        return []
    if area < 0.0:
        pts.reverse()
    return pts


def _point_in_triangle_2d(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> bool:
    """Return ``True`` if ``(px, py)`` lies inside or on CCW triangle ``(a, b, c)``."""
    eps = -1e-7
    c0 = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    c1 = (cx - bx) * (py - by) - (cy - by) * (px - bx)
    c2 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx)
    return c0 >= eps and c1 >= eps and c2 >= eps


def triangulate_polygon_2d(ring_ccw: Sequence[tuple[float, float]]) -> list[int]:
    """Ear-clipping triangulator supporting concave OSM buildings and 24+ vertex junctions.

    Returns flat 0-based triangle index list ``[i0, i1, i2, ...]`` into ``ring_ccw``.
    """
    n = len(ring_ccw)
    if n < 3:
        return []
    if n == 3:
        return [0, 1, 2]

    remaining = list(range(n))
    triangles: list[int] = []
    guard = 0
    max_iters = n * n + 16

    while len(remaining) > 3 and guard < max_iters:
        guard += 1
        ear_found = False
        m = len(remaining)
        for idx in range(m):
            i_prev = remaining[(idx - 1) % m]
            i_curr = remaining[idx]
            i_next = remaining[(idx + 1) % m]
            ax, ay = ring_ccw[i_prev]
            bx, by = ring_ccw[i_curr]
            cx, cy = ring_ccw[i_next]

            cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if cross <= 1e-8:
                continue

            contains_other = False
            for other in remaining:
                if other in (i_prev, i_curr, i_next):
                    continue
                px, py = ring_ccw[other]
                if _point_in_triangle_2d(px, py, ax, ay, bx, by, cx, cy):
                    contains_other = True
                    break

            if not contains_other:
                triangles.extend((i_prev, i_curr, i_next))
                remaining.pop(idx)
                ear_found = True
                break

        if not ear_found:
            # Degenerate self-touching OSM vertex fallback: clip sharpest convex/near-colinear corner
            best_idx = 0
            best_cross = -float("inf")
            for idx in range(m):
                ax, ay = ring_ccw[remaining[(idx - 1) % m]]
                bx, by = ring_ccw[remaining[idx]]
                cx, cy = ring_ccw[remaining[(idx + 1) % m]]
                cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                if cross > best_cross:
                    best_cross = cross
                    best_idx = idx
            i_prev = remaining[(best_idx - 1) % m]
            i_curr = remaining[best_idx]
            i_next = remaining[(best_idx + 1) % m]
            triangles.extend((i_prev, i_curr, i_next))
            remaining.pop(best_idx)

    if len(remaining) == 3:
        triangles.extend((remaining[0], remaining[1], remaining[2]))
    return triangles


def _polyline_normals(pts: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Compute unit left-hand normal ``(-dy, dx)`` at each vertex of a 2D polyline."""
    n = len(pts)
    if n < 2:
        return [(0.0, 1.0)] * n
    seg_normals: list[tuple[float, float]] = []
    for i in range(n - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            seg_normals.append(seg_normals[-1] if seg_normals else (0.0, 1.0))
        else:
            seg_normals.append((-dy / norm, dx / norm))

    vertex_normals: list[tuple[float, float]] = [seg_normals[0]]
    for i in range(1, n - 1):
        nx = seg_normals[i - 1][0] + seg_normals[i][0]
        ny = seg_normals[i - 1][1] + seg_normals[i][1]
        norm = math.hypot(nx, ny)
        if norm < 1e-6:
            vertex_normals.append(seg_normals[i])
        else:
            # Miter scale clamped to [1.0, 1.6] for clean sharp bends
            dot = max(-0.9, min(1.0, seg_normals[i - 1][0] * (nx / norm) + seg_normals[i - 1][1] * (ny / norm)))
            scale = min(1.6, 1.0 / max(0.5, dot))
            vertex_normals.append(((nx / norm) * scale, (ny / norm) * scale))
    vertex_normals.append(seg_normals[-1])
    return vertex_normals


def _offset_polyline(pts: Sequence[tuple[float, float]], lateral_offset_m: float) -> list[list[float]]:
    """Shift a 2D polyline laterally (+left, -right) in meters."""
    normals = _polyline_normals(pts)
    return [
        [round(x + nx * lateral_offset_m, 3), round(y + ny * lateral_offset_m, 3)]
        for (x, y), (nx, ny) in zip(pts, normals, strict=True)
    ]


def _polyline_heading_end(pts: Sequence[tuple[float, float]]) -> float:
    """Return terminal tangent angle ``atan2(dy, dx)`` in CCW radians."""
    for i in range(len(pts) - 1, 0, -1):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        if math.hypot(dx, dy) > 1e-4:
            return math.atan2(dy, dx)
    return 0.0


def _polyline_heading_start(pts: Sequence[tuple[float, float]]) -> float:
    """Return initial tangent angle ``atan2(dy, dx)`` in CCW radians."""
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        if math.hypot(dx, dy) > 1e-4:
            return math.atan2(dy, dx)
    return 0.0


def _wrap_angle_rad(angle: float) -> float:
    """Wrap angle into ``[-pi, pi]``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _parse_shape_attr(shape_str: str) -> list[tuple[float, float]]:
    """Parse SUMO space-separated ``x,y`` shape attribute."""
    out: list[tuple[float, float]] = []
    for token in shape_str.strip().split():
        parts = token.split(",")
        if len(parts) < 2:
            continue
        x = float(parts[0])
        y = float(parts[1])
        require_finite(x, "poly.shape.x")
        require_finite(y, "poly.shape.y")
        out.append((x, y))
    return out


def _extract_osm_polygons(
    poly_path: Path,
    bbox: tuple[float, float, float, float],
    margin_m: float = 145.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse real-world OSM 3D buildings and greenbelt/park polygons from SUMO ``polyconvert`` XML."""
    min_x, min_y, max_x, max_y = bbox
    lo_x, lo_y = min_x - margin_m, min_y - margin_m
    hi_x, hi_y = max_x + margin_m, max_y + margin_m

    try:
        root = ET.parse(poly_path).getroot()
    except ET.ParseError as exc:
        raise ConfigValidationError(
            f"Malformed polygon XML file {poly_path}: {exc}",
            context={"poly_xml_path": str(poly_path)},
        ) from exc

    building_types = {"building", "shop", "commercial", "office", "apartments", "retail", "university", "civic"}
    park_types = {"park", "forest", "grass", "meadow", "recreation_ground", "garden", "pitch"}

    seen_building_bases: set[str] = set()
    buildings: list[dict[str, Any]] = []
    parks: list[dict[str, Any]] = []

    palettes = (
        [0.84, 0.81, 0.76],  # Warm Stanford limestone
        [0.68, 0.75, 0.82],  # Architectural glass-steel
        [0.79, 0.67, 0.60],  # Terracotta / brick
        [0.76, 0.78, 0.80],  # Cool modern concrete
        [0.87, 0.84, 0.79],  # Cream stucco
    )

    for poly_el in root.findall("poly"):
        raw_id = poly_el.attrib.get("id", "")
        ptype = poly_el.attrib.get("type", "").lower()
        shape_str = poly_el.attrib.get("shape", "")
        if not shape_str:
            continue

        is_bldg = any(bt in ptype for bt in building_types)
        is_park = (not is_bldg) and any(pt in ptype for pt in park_types)
        if not is_bldg and not is_park:
            continue

        raw_pts = _parse_shape_attr(shape_str)
        if len(raw_pts) < 3:
            continue

        xs = [p[0] for p in raw_pts]
        ys = [p[1] for p in raw_pts]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        if cx < lo_x or cx > hi_x or cy < lo_y or cy > hi_y:
            continue

        ring = _clean_polygon_ring(raw_pts)
        if len(ring) < 3:
            continue
        area = _signed_polygon_area(ring)
        if area < 12.0:
            continue

        if is_bldg:
            base_id = raw_id.split("#")[0]
            if base_id in seen_building_bases:
                continue
            seen_building_bases.add(base_id)
            if area > 42000.0:
                continue

            tri_indices = triangulate_polygon_2d(ring)
            if not tri_indices:
                continue

            h_seed = _stable_unit_hash(f"height:{base_id}")
            c_seed = int(_stable_unit_hash(f"color:{base_id}") * len(palettes)) % len(palettes)
            # Realistic urban height scaling by footprint area + deterministic architectural variance
            base_floors = 2.0 + min(6.0, math.sqrt(area) / 11.5) * (0.55 + 0.75 * h_seed)
            if "commercial" in ptype or "office" in ptype or area > 950.0:
                base_floors += 1.5 * h_seed
            height_m = round(max(6.8, min(38.0, base_floors * 3.5)), 2)

            buildings.append(
                {
                    "id": base_id,
                    "footprint": [[round(x, 2), round(y, 2)] for x, y in ring],
                    "triangles": tri_indices,
                    "height_m": height_m,
                    "color_rgb": palettes[c_seed],
                }
            )
        elif is_park and area <= 180000.0:
            tri_indices = triangulate_polygon_2d(ring)
            if tri_indices:
                parks.append(
                    {
                        "id": raw_id,
                        "footprint": [[round(x, 2), round(y, 2)] for x, y in ring],
                        "triangles": tri_indices,
                        "color_rgb": [0.16, 0.27, 0.19],
                    }
                )

    return buildings, parks


def _point_to_segment_dist_sq(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Return squared Euclidean distance from point ``(px, py)`` to segment ``(a, b)``."""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq <= 1e-9:
        return wx * wx + wy * wy
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len_sq))
    dx = px - (ax + t * vx)
    dy = py - (ay + t * vy)
    return dx * dx + dy * dy


def _generate_procedural_city_blocks(
    junction_coords: Sequence[tuple[float, float]],
    road_segments: Sequence[tuple[float, float, float, float, float]],
    bbox: tuple[float, float, float, float],
    *,
    axis_coords: Sequence[tuple[float, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Synthesize setback-verified architectural 3D city blocks & green plazas for grid networks."""
    min_x, min_y, max_x, max_y = bbox
    span_x = max_x - min_x
    span_y = max_y - min_y
    if span_x < 40.0 or span_y < 40.0:
        return [], []

    palettes = (
        [0.72, 0.77, 0.84],
        [0.82, 0.79, 0.74],
        [0.64, 0.71, 0.79],
        [0.78, 0.72, 0.68],
        [0.75, 0.80, 0.82],
    )

    major_juncs = list(axis_coords) if axis_coords is not None and len(axis_coords) >= 4 else list(junction_coords)

    def _box_clear_of_roads(cx: float, cy: float, hw: float, hd: float, clearance_m: float) -> bool:
        test_pts = (
            (cx, cy),
            (cx - hw, cy - hd),
            (cx + hw, cy - hd),
            (cx + hw, cy + hd),
            (cx - hw, cy + hd),
            (cx - hw, cy),
            (cx + hw, cy),
            (cx, cy - hd),
            (cx, cy + hd),
            (cx - hw, cy - 0.5 * hd),
            (cx - hw, cy + 0.5 * hd),
            (cx + hw, cy - 0.5 * hd),
            (cx + hw, cy + 0.5 * hd),
            (cx - 0.5 * hw, cy - hd),
            (cx + 0.5 * hw, cy - hd),
            (cx - 0.5 * hw, cy + hd),
            (cx + 0.5 * hw, cy + hd),
        )
        for px, py in test_pts:
            for ax, ay, bx, by, half_w in road_segments:
                thresh = half_w + clearance_m
                if _point_to_segment_dist_sq(px, py, ax, ay, bx, by) < thresh * thresh:
                    return False
            for jx, jy in major_juncs:
                j_thresh = 16.0 + clearance_m * 0.5
                if (px - jx) ** 2 + (py - jy) ** 2 < j_thresh * j_thresh:
                    return False
        return True

    # Snap unique grid axes from major/signalized intersections + outer network boundary
    # (prevents mid-corridor turn-pocket split nodes from slicing 150m blocks into narrow strips).
    if axis_coords is not None and len(axis_coords) >= 4:
        raw_xs = {round(c[0], 0) for c in axis_coords} | {round(min_x, 0), round(max_x, 0)}
        raw_ys = {round(c[1], 0) for c in axis_coords} | {round(min_y, 0), round(max_y, 0)}
    else:
        raw_xs = {round(c[0], 0) for c in junction_coords}
        raw_ys = {round(c[1], 0) for c in junction_coords}

    xs_sorted = sorted(raw_xs)
    ys_sorted = sorted(raw_ys)

    def _merge_close(vals: list[float], tol: float = 28.0) -> list[float]:
        merged: list[float] = []
        for v in vals:
            if not merged or abs(v - merged[-1]) > tol:
                merged.append(v)
        return merged

    x_axes = _merge_close(xs_sorted)
    y_axes = _merge_close(ys_sorted)

    buildings: list[dict[str, Any]] = []
    parks: list[dict[str, Any]] = []
    b_idx = 0

    if len(x_axes) >= 2 and len(y_axes) >= 2:
        for ix in range(len(x_axes) - 1):
            for iy in range(len(y_axes) - 1):
                x0, x1 = x_axes[ix], x_axes[ix + 1]
                y0, y1 = y_axes[iy], y_axes[iy + 1]
                cell_w = x1 - x0
                cell_h = y1 - y0
                if cell_w < 55.0 or cell_h < 55.0:
                    continue

                setback = 21.5 if min(cell_w, cell_h) >= 110.0 else 18.5
                bx0, bx1 = x0 + setback, x1 - setback
                by0, by1 = y0 + setback, y1 - setback
                if bx1 - bx0 < 18.0 or by1 - by0 < 18.0:
                    continue

                # Central courtyard green plaza inside the urban block
                pcx, pcy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
                pw, ph = 0.46 * (bx1 - bx0), 0.46 * (by1 - by0)
                if _box_clear_of_roads(pcx, pcy, pw * 0.5, ph * 0.5, 4.0):
                    p_ring = [
                        [round(bx0, 2), round(by0, 2)],
                        [round(bx1, 2), round(by0, 2)],
                        [round(bx1, 2), round(by1, 2)],
                        [round(bx0, 2), round(by1, 2)],
                    ]
                    parks.append(
                        {
                            "id": f"plaza_{ix}_{iy}",
                            "footprint": p_ring,
                            "triangles": [0, 1, 2, 0, 2, 3],
                            "color_rgb": [0.14, 0.24, 0.18],
                        }
                    )

                # Four architectural corner parcels per city block
                gap = 5.5
                sub_boxes = (
                    (bx0, pcx - gap * 0.5, by0, pcy - gap * 0.5),
                    (pcx + gap * 0.5, bx1, by0, pcy - gap * 0.5),
                    (pcx + gap * 0.5, bx1, pcy + gap * 0.5, by1),
                    (bx0, pcx - gap * 0.5, pcy + gap * 0.5, by1),
                )
                for q_idx, (sx0, sx1, sy0, sy1) in enumerate(sub_boxes):
                    sw, sh = sx1 - sx0, sy1 - sy0
                    if sw < 10.0 or sh < 10.0:
                        continue
                    scx, scy = 0.5 * (sx0 + sx1), 0.5 * (sy0 + sy1)
                    if not _box_clear_of_roads(scx, scy, sw * 0.5, sh * 0.5, 6.5):
                        continue
                    key = f"blk_{ix}_{iy}_{q_idx}"
                    u_h = _stable_unit_hash(f"h:{key}")
                    # Taller downtown core near center of grid
                    dist_center = math.hypot(
                        scx - 0.5 * (min_x + max_x),
                        scy - 0.5 * (min_y + max_y),
                    )
                    core_boost = max(0.0, 1.0 - dist_center / max(180.0, 0.55 * max(span_x, span_y)))
                    height_m = round(11.0 + 18.0 * u_h + 15.0 * core_boost, 2)
                    c_idx = int(_stable_unit_hash(f"c:{key}") * len(palettes)) % len(palettes)
                    ring = [
                        [round(sx0, 2), round(sy0, 2)],
                        [round(sx1, 2), round(sy0, 2)],
                        [round(sx1, 2), round(sy1, 2)],
                        [round(sx0, 2), round(sy1, 2)],
                    ]
                    buildings.append(
                        {
                            "id": f"bldg_{b_idx}",
                            "footprint": ring,
                            "triangles": [0, 1, 2, 0, 2, 3],
                            "height_m": height_m,
                            "color_rgb": palettes[c_idx],
                        }
                    )
                    b_idx += 1

    # Single-junction or corridor fallback when grid cells < 2x2: place corner quadrant buildings
    if not buildings:
        cx_net = 0.5 * (min_x + max_x)
        cy_net = 0.5 * (min_y + max_y)
        for qx in (-1.0, 1.0):
            for qy in (-1.0, 1.0):
                for step_r in (34.0, 68.0):
                    bcx = cx_net + qx * step_r
                    bcy = cy_net + qy * step_r
                    hw, hd = 11.0, 11.0
                    if _box_clear_of_roads(bcx, bcy, hw, hd, 6.5):
                        key = f"quad_{qx}_{qy}_{step_r}"
                        height_m = round(12.0 + 16.0 * _stable_unit_hash(key), 2)
                        c_idx = b_idx % len(palettes)
                        buildings.append(
                            {
                                "id": f"bldg_{b_idx}",
                                "footprint": [
                                    [round(bcx - hw, 2), round(bcy - hd, 2)],
                                    [round(bcx + hw, 2), round(bcy - hd, 2)],
                                    [round(bcx + hw, 2), round(bcy + hd, 2)],
                                    [round(bcx - hw, 2), round(bcy + hd, 2)],
                                ],
                                "triangles": [0, 1, 2, 0, 2, 3],
                                "height_m": height_m,
                                "color_rgb": palettes[c_idx],
                            }
                        )
                        b_idx += 1

    return buildings, parks


def synthesize_procedural_blocks_for_scene(
    scene_data: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Regenerate procedural 3D city blocks & plazas from an exported scene dictionary."""
    bounds = scene_data["bounds"]
    bbox = (
        float(bounds["min_x"]),
        float(bounds["min_y"]),
        float(bounds["max_x"]),
        float(bounds["max_y"]),
    )
    junctions = scene_data.get("junctions") or ()
    signals = scene_data.get("signals") or ()
    lanes = scene_data.get("lanes") or ()

    junction_coords: list[tuple[float, float]] = [
        (float(j["coord_xy"][0]), float(j["coord_xy"][1]))
        for j in junctions
        if j.get("coord_xy") and len(j["coord_xy"]) >= 2
    ]

    tls_coords: list[tuple[float, float]] = [
        (float(j["coord_xy"][0]), float(j["coord_xy"][1]))
        for j in junctions
        if "traffic_light" in str(j.get("node_type", "")).lower()
        and j.get("coord_xy")
        and len(j["coord_xy"]) >= 2
    ]
    if len(tls_coords) < 4:
        for sig in signals:
            links = sig.get("links") or ()
            if links:
                lx = sum(float(lk["head_xyz"][0]) for lk in links) / len(links)
                ly = sum(float(lk["head_xyz"][1]) for lk in links) / len(links)
                tls_coords.append((lx, ly))

    road_segments: list[tuple[float, float, float, float, float]] = []
    for ln in lanes:
        pts = ln.get("shape") or ()
        width = float(ln.get("width_m", 3.2))
        half_w = max(3.5, width * 0.7)
        for s_i in range(len(pts) - 1):
            road_segments.append(
                (
                    float(pts[s_i][0]),
                    float(pts[s_i][1]),
                    float(pts[s_i + 1][0]),
                    float(pts[s_i + 1][1]),
                    half_w,
                )
            )

    return _generate_procedural_city_blocks(
        junction_coords=junction_coords,
        road_segments=road_segments,
        bbox=bbox,
        axis_coords=tls_coords if len(tls_coords) >= 4 else None,
    )


def compute_scene_focus_bounds(
    bounds: Mapping[str, Any],
    junctions: Sequence[Mapping[str, Any]],
    lanes: Sequence[Mapping[str, Any]] = (),
    signals: Sequence[Mapping[str, Any]] = (),
) -> dict[str, float]:
    """Compute tight urban core framing bounds from signalized intersections or core road geometry.

    Prevents distant highway approach ramps (such as perimeter arterial approaches in OpenStreetMap
    Downtown Palo Alto) from skewing the initial 3D isometric camera away from the signalized grid.
    """
    min_x = float(bounds["min_x"])
    min_y = float(bounds["min_y"])
    max_x = float(bounds["max_x"])
    max_y = float(bounds["max_y"])
    require_finite(min_x, "bounds.min_x")
    require_finite(min_y, "bounds.min_y")
    require_finite(max_x, "bounds.max_x")
    require_finite(max_y, "bounds.max_y")

    tls_pts: list[tuple[float, float]] = []
    for sig in signals:
        links = sig.get("links") or ()
        if links:
            lx = sum(float(lk["head_xyz"][0]) for lk in links) / len(links)
            ly = sum(float(lk["head_xyz"][1]) for lk in links) / len(links)
            tls_pts.append((lx, ly))

    if len(tls_pts) < 4:
        for junc in junctions:
            if "traffic_light" in str(junc.get("node_type", "")).lower():
                coord = junc.get("coord_xy")
                if coord and len(coord) >= 2:
                    tls_pts.append((float(coord[0]), float(coord[1])))

    if len(tls_pts) >= 18:
        # Large OSM urban network with perimeter arterial signals: frame p10..p90 core tightly
        xs = np.array([p[0] for p in tls_pts], dtype=np.float64)
        ys = np.array([p[1] for p in tls_pts], dtype=np.float64)
        fx0, fx1 = float(np.percentile(xs, 10.0)), float(np.percentile(xs, 90.0))
        fy0, fy1 = float(np.percentile(ys, 10.0)), float(np.percentile(ys, 90.0))
        pad_x = max(40.0, 0.08 * (fx1 - fx0))
        pad_y = max(40.0, 0.08 * (fy1 - fy0))
    elif len(tls_pts) >= 4:
        xs = np.array([p[0] for p in tls_pts], dtype=np.float64)
        ys = np.array([p[1] for p in tls_pts], dtype=np.float64)
        fx0, fx1 = float(np.percentile(xs, 5.0)), float(np.percentile(xs, 95.0))
        fy0, fy1 = float(np.percentile(ys, 5.0)), float(np.percentile(ys, 95.0))
        pad_x = max(45.0, 0.12 * (fx1 - fx0))
        pad_y = max(45.0, 0.12 * (fy1 - fy0))
    else:
        fallback_pts: list[tuple[float, float]] = [
            (float(j["coord_xy"][0]), float(j["coord_xy"][1]))
            for j in junctions
            if j.get("coord_xy") and len(j["coord_xy"]) >= 2
        ]
        if not fallback_pts:
            for ln in lanes:
                for pt in ln.get("shape") or ():
                    fallback_pts.append((float(pt[0]), float(pt[1])))
        if len(fallback_pts) >= 4:
            xs = np.array([p[0] for p in fallback_pts], dtype=np.float64)
            ys = np.array([p[1] for p in fallback_pts], dtype=np.float64)
            fx0, fx1 = float(np.percentile(xs, 5.0)), float(np.percentile(xs, 95.0))
            fy0, fy1 = float(np.percentile(ys, 5.0)), float(np.percentile(ys, 95.0))
            pad_x = max(30.0, 0.10 * (fx1 - fx0))
            pad_y = max(30.0, 0.10 * (fy1 - fy0))
        else:
            fx0, fx1, fy0, fy1 = min_x, max_x, min_y, max_y
            pad_x = 0.0
            pad_y = 0.0

    fx0 = max(min_x, fx0 - pad_x)
    fx1 = min(max_x, fx1 + pad_x)
    fy0 = max(min_y, fy0 - pad_y)
    fy1 = min(max_y, fy1 + pad_y)
    if fx1 - fx0 < 40.0:
        cx = 0.5 * (fx0 + fx1)
        fx0, fx1 = max(min_x, cx - 20.0), min(max_x, cx + 20.0)
    if fy1 - fy0 < 40.0:
        cy = 0.5 * (fy0 + fy1)
        fy0, fy1 = max(min_y, cy - 20.0), min(max_y, cy + 20.0)

    return {
        "min_x": round(fx0, 3),
        "min_y": round(fy0, 3),
        "max_x": round(fx1, 3),
        "max_y": round(fy1, 3),
        "center_x": round(0.5 * (fx0 + fx1), 3),
        "center_y": round(0.5 * (fy0 + fy1), 3),
        "span_x": round(max(10.0, fx1 - fx0), 3),
        "span_y": round(max(10.0, fy1 - fy0), 3),
    }


def export_scene_3d_from_files(
    net_xml_path: Path | str,
    poly_xml_path: Path | str | None = None,
    *,
    network_id: str = "urban_network",
    edge_to_corridor: Mapping[str, str] | None = None,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Extract a complete WebGL2-ready 3D urban scene dictionary from SUMO network/polygon XMLs."""
    net_p = Path(net_xml_path)
    if not net_p.is_file():
        raise ConfigValidationError(
            f"SUMO network XML does not exist: {net_p}",
            context={"net_xml_path": str(net_p)},
        )

    poly_p: Path | None = None
    if poly_xml_path is not None:
        poly_p = Path(poly_xml_path)
        if not poly_p.is_file():
            raise ConfigValidationError(
                f"SUMO polygon XML does not exist: {poly_p}",
                context={"poly_xml_path": str(poly_p)},
            )

    net = sumolib.net.readNet(str(net_p), withInternal=True)
    corridor_lookup = dict(edge_to_corridor) if edge_to_corridor is not None else {}

    lanes_out: list[dict[str, Any]] = []
    lane_dividers_out: list[dict[str, Any]] = []
    stop_bars_out: list[dict[str, Any]] = []
    turn_arrows_out: list[dict[str, Any]] = []
    crosswalks_out: list[dict[str, Any]] = []
    corridor_ribbons: dict[str, list[list[list[float]]]] = {}

    road_segments: list[tuple[float, float, float, float, float]] = []
    all_xs: list[float] = []
    all_ys: list[float] = []

    normal_edges = [e for e in net.getEdges() if not e.isSpecial() and e.allows("passenger")]
    if not normal_edges:
        normal_edges = [e for e in net.getEdges() if not e.isSpecial()]
    if not normal_edges:
        raise TopologyIntegrityError(
            f"SUMO network {net_p} contains zero drivable non-internal edges.",
            context={"net_xml_path": str(net_p)},
        )

    # Track two-way opposing node pairs for yellow center-divider rendering
    directed_node_pairs = {(e.getFromNode().getID(), e.getToNode().getID()) for e in normal_edges}

    for edge in normal_edges:
        edge_id = edge.getID()
        corridor_id = corridor_lookup.get(edge_id, edge_id)
        street_name = edge.getName() or ""
        lanes = edge.getLanes()
        num_lanes = len(lanes)
        from_id = edge.getFromNode().getID()
        to_id = edge.getToNode().getID()
        to_node_type = edge.getToNode().getType() or ""
        has_opposing = (to_id, from_id) in directed_node_pairs

        edge_lane_shapes: list[list[tuple[float, float]]] = []
        edge_lane_widths: list[float] = []

        for l_idx, lane in enumerate(lanes):
            raw_shape = lane.getShape()
            pts: list[tuple[float, float]] = []
            for p in raw_shape:
                x, y = float(p[0]), float(p[1])
                require_finite(x, f"lane[{lane.getID()}].x")
                require_finite(y, f"lane[{lane.getID()}].y")
                pts.append((x, y))
                all_xs.append(x)
                all_ys.append(y)

            if len(pts) < 2:
                continue

            width = float(lane.getWidth())
            require_finite(width, f"lane[{lane.getID()}].width")
            edge_lane_shapes.append(pts)
            edge_lane_widths.append(width)

            for s_i in range(len(pts) - 1):
                road_segments.append(
                    (pts[s_i][0], pts[s_i][1], pts[s_i + 1][0], pts[s_i + 1][1], max(3.5, width * 0.7))
                )

            shape_rounded = [[round(x, 3), round(y, 3)] for x, y in pts]
            lanes_out.append(
                {
                    "lane_id": lane.getID(),
                    "edge_id": edge_id,
                    "corridor_id": corridor_id,
                    "lane_index": int(l_idx),
                    "width_m": round(width, 3),
                    "length_m": round(float(lane.getLength()), 3),
                    "speed_limit_mps": round(float(lane.getSpeed()), 3),
                    "street_name": street_name,
                    "shape": shape_rounded,
                }
            )
            corridor_ribbons.setdefault(corridor_id, []).append(shape_rounded)

            # Stop-bar segment & directional turn arrows at controlled/multi-arm intersections
            end_heading = _polyline_heading_end(pts)
            ex, ey = pts[-1]
            nx, ny = -math.sin(end_heading), math.cos(end_heading)
            outgoing_conns = lane.getOutgoing()
            if outgoing_conns and ("traffic_light" in to_node_type or len(edge.getToNode().getIncoming()) >= 2):
                hw = 0.46 * width
                stop_bars_out.append(
                    {
                        "lane_id": lane.getID(),
                        "corridor_id": corridor_id,
                        "p0": [round(ex - nx * hw, 3), round(ey - ny * hw, 3)],
                        "p1": [round(ex + nx * hw, 3), round(ey + ny * hw, 3)],
                    }
                )

            if outgoing_conns and float(lane.getLength()) >= 14.0:
                dirs: set[str] = set()
                for conn in outgoing_conns:
                    to_lane = conn.getToLane()
                    if to_lane is None or to_lane.getEdge().isSpecial():
                        continue
                    out_pts = [(float(p[0]), float(p[1])) for p in to_lane.getShape()]
                    if len(out_pts) >= 2:
                        out_heading = _polyline_heading_start(out_pts)
                        delta = _wrap_angle_rad(out_heading - end_heading)
                        if delta > LEFT_TURN_THRESHOLD_RAD:
                            dirs.add("L")
                        elif delta < -LEFT_TURN_THRESHOLD_RAD:
                            dirs.add("R")
                        else:
                            dirs.add("S")
                if dirs:
                    back_d = min(6.5, 0.35 * float(lane.getLength()))
                    ax_pos = ex - math.cos(end_heading) * back_d
                    ay_pos = ey - math.sin(end_heading) * back_d
                    turn_arrows_out.append(
                        {
                            "lane_id": lane.getID(),
                            "xy": [round(ax_pos, 3), round(ay_pos, 3)],
                            "yaw_ccw_rad": round(end_heading, 4),
                            "dirs": sorted(dirs),
                        }
                    )

        # Dashed white lane dividers between adjacent same-direction lanes
        for l_idx in range(len(edge_lane_shapes) - 1):
            pts = edge_lane_shapes[l_idx]
            half_w = 0.5 * edge_lane_widths[l_idx]
            lane_dividers_out.append(
                {
                    "kind": "white_dashed",
                    "points": _offset_polyline(pts, half_w),
                }
            )

        # Solid double-yellow center line on innermost lane of two-way streets
        if has_opposing and edge_lane_shapes:
            inner_pts = edge_lane_shapes[-1]
            half_w = 0.5 * edge_lane_widths[-1]
            lane_dividers_out.append(
                {
                    "kind": "yellow_center",
                    "points": _offset_polyline(inner_pts, half_w),
                }
            )

        # Zebra crosswalk across multi-arm signalized intersection approaches
        if (
            edge_lane_shapes
            and float(edge.getLength()) >= 22.0
            and ("traffic_light" in to_node_type or len(edge.getToNode().getIncoming()) >= 3)
        ):
            total_w = sum(edge_lane_widths)
            mid_pts = edge_lane_shapes[len(edge_lane_shapes) // 2]
            end_h = _polyline_heading_end(mid_pts)
            # Shift lateral center if even number of lanes
            lat_shift = 0.5 * edge_lane_widths[0] if (num_lanes % 2 == 0) else 0.0
            mx = mid_pts[-1][0] - math.cos(end_h) * 2.1 - math.sin(end_h) * lat_shift
            my = mid_pts[-1][1] - math.sin(end_h) * 2.1 + math.cos(end_h) * lat_shift
            crosswalks_out.append(
                {
                    "edge_id": edge_id,
                    "xy": [round(mx, 3), round(my, 3)],
                    "yaw_ccw_rad": round(end_h, 4),
                    "width_m": round(total_w * 0.92, 3),
                    "depth_m": 2.4,
                }
            )

    # Junction polygons
    junctions_out: list[dict[str, Any]] = []
    junction_coords: list[tuple[float, float]] = []
    for node in net.getNodes():
        nid = node.getID()
        if nid.startswith(":"):
            continue
        nx_c, ny_c = float(node.getCoord()[0]), float(node.getCoord()[1])
        require_finite(nx_c, f"junction[{nid}].x")
        require_finite(ny_c, f"junction[{nid}].y")
        junction_coords.append((nx_c, ny_c))
        all_xs.append(nx_c)
        all_ys.append(ny_c)

        raw_j_shape = node.getShape()
        ring = _clean_polygon_ring(raw_j_shape, min_edge_m=0.04) if raw_j_shape else []
        tri_indices = triangulate_polygon_2d(ring) if len(ring) >= 3 else []
        junctions_out.append(
            {
                "junction_id": nid,
                "node_type": node.getType() or "priority",
                "coord_xy": [round(nx_c, 3), round(ny_c, 3)],
                "polygon": [[round(x, 3), round(y, 3)] for x, y in ring],
                "triangles": tri_indices,
            }
        )

    # 3D Traffic Signal Poles & Controlled Link Bulbs (1-to-1 with tls_states[tls_id][link_index])
    signals_out: list[dict[str, Any]] = []
    for tls in net.getTrafficLights():
        tls_id = tls.getID()
        # Group connections by linkIndex so every index k in 0..max_idx has a 3D signal head
        by_link_idx: dict[int, list[Any]] = {}
        for conn in tls.getConnections():
            in_lane, out_lane, link_idx = conn[0], conn[1], int(conn[2])
            by_link_idx.setdefault(link_idx, []).append((in_lane, out_lane))

        if not by_link_idx:
            continue

        links_meta: list[dict[str, Any]] = []
        pole_anchors: dict[str, dict[str, Any]] = {}

        for link_idx in sorted(by_link_idx):
            in_lane, out_lane = by_link_idx[link_idx][0]
            in_pts = [(float(p[0]), float(p[1])) for p in in_lane.getShape()]
            out_pts = [(float(p[0]), float(p[1])) for p in out_lane.getShape()]
            if len(in_pts) < 2:
                continue
            heading = _polyline_heading_end(in_pts)
            sx, sy = in_pts[-1]
            edge_obj = in_lane.getEdge()
            edge_id = edge_obj.getID()

            # Place shared curbside gantry pole on the right curb of the incoming edge
            if edge_id not in pole_anchors:
                right_lane_pts = [(float(p[0]), float(p[1])) for p in edge_obj.getLanes()[0].getShape()]
                rh = _polyline_heading_end(right_lane_pts)
                rx, ry = right_lane_pts[-1]
                r_norm_x, r_norm_y = math.sin(rh), -math.cos(rh)
                pole_x = rx + r_norm_x * (0.5 * float(edge_obj.getLanes()[0].getWidth()) + 1.6)
                pole_y = ry + r_norm_y * (0.5 * float(edge_obj.getLanes()[0].getWidth()) + 1.6)
                pole_anchors[edge_id] = {
                    "edge_id": edge_id,
                    "base_xy": [round(pole_x, 3), round(pole_y, 3)],
                    "yaw_ccw_rad": round(rh, 4),
                    "height_m": 5.6,
                }

            links_meta.append(
                {
                    "link_index": link_idx,
                    "in_lane_id": in_lane.getID(),
                    "out_lane_id": out_lane.getID(),
                    "edge_id": edge_id,
                    "head_xyz": [round(sx, 3), round(sy, 3), 5.35],
                    "yaw_ccw_rad": round(heading + math.pi, 4),
                    "target_xy": [round(out_pts[0][0], 3), round(out_pts[0][1], 3)] if out_pts else [round(sx, 3), round(sy, 3)],
                }
            )

        signals_out.append(
            {
                "tls_id": tls_id,
                "num_links": max(by_link_idx) + 1,
                "poles": list(pole_anchors.values()),
                "links": links_meta,
            }
        )

    min_x, max_x = float(min(all_xs)), float(max(all_xs))
    min_y, max_y = float(min(all_ys)), float(max(all_ys))
    bbox = (min_x, min_y, max_x, max_y)

    if poly_p is not None:
        buildings_out, parks_out = _extract_osm_polygons(poly_p, bbox=bbox)
    else:
        tls_junction_coords = [
            (float(j["coord_xy"][0]), float(j["coord_xy"][1]))
            for j in junctions_out
            if "traffic_light" in str(j.get("node_type", "")).lower()
        ]
        buildings_out, parks_out = _generate_procedural_city_blocks(
            junction_coords=junction_coords,
            road_segments=road_segments,
            bbox=bbox,
            axis_coords=tls_junction_coords if len(tls_junction_coords) >= 4 else None,
        )

    full_bounds: dict[str, Any] = {
        "min_x": round(min_x, 3),
        "min_y": round(min_y, 3),
        "max_x": round(max_x, 3),
        "max_y": round(max_y, 3),
        "center_x": round(0.5 * (min_x + max_x), 3),
        "center_y": round(0.5 * (min_y + max_y), 3),
        "span_x": round(max(10.0, max_x - min_x), 3),
        "span_y": round(max(10.0, max_y - min_y), 3),
    }
    focus_bounds = compute_scene_focus_bounds(
        bounds=full_bounds,
        junctions=junctions_out,
        lanes=lanes_out,
        signals=signals_out,
    )
    full_bounds["focus_bounds"] = dict(focus_bounds)

    scene_dict: dict[str, Any] = {
        "schema_version": SCENE_SCHEMA_VERSION,
        "network_id": network_id,
        "source_mode": "osm_poly" if poly_p is not None else "procedural_blocks",
        "bounds": full_bounds,
        "focus_bounds": focus_bounds,
        "lanes": lanes_out,
        "lane_dividers": lane_dividers_out,
        "stop_bars": stop_bars_out,
        "turn_arrows": turn_arrows_out,
        "crosswalks": crosswalks_out,
        "junctions": junctions_out,
        "signals": signals_out,
        "buildings": buildings_out,
        "parks": parks_out,
        "corridor_ribbons": corridor_ribbons,
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = out_p.parent / f".{out_p.name}.pid{os.getpid()}.tmp"
        tmp_p.write_text(json.dumps(scene_dict, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_p, out_p)

    return scene_dict


def export_scene_3d(
    topology: RoadTopology,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Export WebGL2 3D urban geometry for a validated :class:`RoadTopology`."""
    if not isinstance(topology, RoadTopology):
        raise ConfigValidationError(
            f"export_scene_3d expects RoadTopology, got {type(topology).__name__}.",
            context={"type": type(topology).__name__},
        )
    topology.validate_integrity()
    return export_scene_3d_from_files(
        net_xml_path=topology.net_xml_path,
        poly_xml_path=topology.poly_xml_path,
        network_id=topology.network_id,
        edge_to_corridor=topology.edge_to_corridor,
        output_path=output_path,
    )


__all__ = [
    "SCENE_SCHEMA_VERSION",
    "compute_scene_focus_bounds",
    "export_scene_3d",
    "export_scene_3d_from_files",
    "synthesize_procedural_blocks_for_scene",
    "triangulate_polygon_2d",
]

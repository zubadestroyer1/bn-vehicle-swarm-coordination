"""Self-contained offline WebGL2 3D City & Neural DBN HTML replay compiler and SoA Gzip codec.

Compiles extracted 3D urban scene geometry, primary (and optional synchronized split-screen
baseline) episode trace data, and the local native WebGL2 shader engine into a single
zero-dependency HTML file making zero network requests.

Uses Structure-of-Arrays (SoA) string/vehicle interning + deterministic Level-6 Gzip + Base64
embedding (`swarm_3d_soa_v2`) so 1080-frame dual-policy urban replays shrink ~25x-30x (from
150-175 MB down to ~4-7 MB) and decompress natively in <100 ms via browser `DecompressionStream`.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
import gzip
import html
import json
import os
from pathlib import Path
import re
from typing import Any

from ..logging_utils import ConfigValidationError
from .scene_builder import compute_scene_focus_bounds, synthesize_procedural_blocks_for_scene

_VIZ_DIR = Path(__file__).resolve().parent
_TEMPLATE_PATH = _VIZ_DIR / "templates" / "city_dbn_3d.html"
_ENGINE_JS_PATH = _VIZ_DIR / "assets" / "city_engine_webgl2.js"
_EXTERNAL_URL_PATTERN = re.compile(r"https?://", flags=re.IGNORECASE)

PAYLOAD_FORMAT_VERSION = "swarm_3d_soa_v2"
_EXACT_PRESERVE_MAX_FRAMES = 20
_LARGE_TRACE_MIN_OMEGA = 0.12
_LARGE_TRACE_MAX_EDGES = 64
_LARGE_TRACE_MIN_HEAT = 0.08


def _validate_scene_dict(scene_data: Mapping[str, Any]) -> None:
    if not isinstance(scene_data, Mapping):
        raise ConfigValidationError(
            f"scene_data must be a Mapping, got {type(scene_data).__name__}.",
            context={"type": type(scene_data).__name__},
        )
    for req_key in ("network_id", "bounds", "lanes", "junctions", "buildings"):
        if req_key not in scene_data:
            raise ConfigValidationError(
                f"scene_data missing required key {req_key!r}.",
                context={"missing_key": req_key},
            )


def _validate_trace_dict(trace_data: Mapping[str, Any], label: str) -> None:
    if not isinstance(trace_data, Mapping):
        raise ConfigValidationError(
            f"{label} must be a Mapping, got {type(trace_data).__name__}.",
            context={"label": label, "type": type(trace_data).__name__},
        )
    frames = trace_data.get("frames")
    packed_frames = trace_data.get("packed_frames")
    has_frames = isinstance(frames, list) and len(frames) > 0
    has_packed = isinstance(packed_frames, list) and len(packed_frames) > 0
    if not has_frames and not has_packed:
        raise ConfigValidationError(
            f"{label} must contain a non-empty 'frames' list.",
            context={"label": label},
        )


def _prepare_scene_for_viewer(scene_data: Mapping[str, Any]) -> dict[str, Any]:
    """Copy scene dictionary, backfill `focus_bounds`, and repair legacy procedural city blocks."""
    scene_copy = dict(scene_data)
    bounds_copy = dict(scene_copy["bounds"])

    signals = scene_copy.get("signals") or ()
    buildings = scene_copy.get("buildings") or ()
    if (
        scene_copy.get("source_mode") == "procedural_blocks"
        and len(signals) >= 4
        and len(buildings) <= 24
    ):
        rebuilt_bldgs, rebuilt_parks = synthesize_procedural_blocks_for_scene(scene_copy)
        if len(rebuilt_bldgs) > len(buildings):
            scene_copy["buildings"] = rebuilt_bldgs
            scene_copy["parks"] = rebuilt_parks

    focus = scene_copy.get("focus_bounds") or bounds_copy.get("focus_bounds")
    if not isinstance(focus, Mapping):
        focus = compute_scene_focus_bounds(
            bounds=bounds_copy,
            junctions=scene_copy.get("junctions") or (),
            lanes=scene_copy.get("lanes") or (),
            signals=signals,
        )
    else:
        focus = dict(focus)
    bounds_copy["focus_bounds"] = dict(focus)
    scene_copy["bounds"] = bounds_copy
    scene_copy["focus_bounds"] = dict(focus)
    return scene_copy


def _select_spatially_diverse_edges(
    raw_edges: list[Mapping[str, Any]],
    veh_corridor_by_id: Mapping[str, str],
    *,
    min_omega: float,
    max_edges: int,
) -> list[Mapping[str, Any]]:
    """Select up to `max_edges` high-confidence DBN edges distributed across active corridors."""
    candidates = [e for e in raw_edges if float(e.get("omega", 0.0)) >= min_omega]
    if len(candidates) <= max_edges:
        candidates.sort(key=lambda e: float(e.get("omega", 0.0)), reverse=True)
        return candidates

    candidates.sort(key=lambda e: float(e.get("omega", 0.0)), reverse=True)

    # Bucket by source corridor (falling back to src vehicle ID) in order of strongest edge per corridor
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    bucket_order: list[str] = []
    for edge in candidates:
        src_id = str(edge.get("src", ""))
        corr_key = veh_corridor_by_id.get(src_id) or src_id
        if corr_key not in buckets:
            buckets[corr_key] = []
            bucket_order.append(corr_key)
        buckets[corr_key].append(edge)

    selected: list[Mapping[str, Any]] = []
    selected_idx_set: set[int] = set()
    src_counts: dict[str, int] = {}

    # Pass 1: Round-robin across corridors with at most 2 edges per source vehicle
    made_progress = True
    cursors = {ck: 0 for ck in bucket_order}
    while len(selected) < max_edges and made_progress:
        made_progress = False
        for ck in bucket_order:
            b_list = buckets[ck]
            cur = cursors[ck]
            while cur < len(b_list):
                cand = b_list[cur]
                cur += 1
                src_id = str(cand.get("src", ""))
                if src_counts.get(src_id, 0) < 2:
                    selected.append(cand)
                    selected_idx_set.add(id(cand))
                    src_counts[src_id] = src_counts.get(src_id, 0) + 1
                    made_progress = True
                    break
            cursors[ck] = cur
            if len(selected) >= max_edges:
                break

    # Pass 2: Fill any remaining capacity with strongest unselected edges
    if len(selected) < max_edges:
        for cand in candidates:
            if id(cand) not in selected_idx_set:
                selected.append(cand)
                if len(selected) >= max_edges:
                    break

    selected.sort(key=lambda e: float(e.get("omega", 0.0)), reverse=True)
    return selected


def _zero_elide(arr: list[int]) -> list[int]:
    """Return an empty list when all integer entries are zero to compress baseline columns."""
    for val in arr:
        if val != 0:
            return arr
    return []


def _pack_trace_soa(
    trace_data: Mapping[str, Any],
    *,
    min_omega: float = _LARGE_TRACE_MIN_OMEGA,
    max_edges_per_frame: int = _LARGE_TRACE_MAX_EDGES,
) -> dict[str, Any]:
    """Pack an episode trace into compact integer-quantized Structure-of-Arrays (`soa_v2`)."""
    if trace_data.get("encoding") == "soa_v2" and isinstance(trace_data.get("packed_frames"), list):
        return dict(trace_data)

    frames = list(trace_data.get("frames") or ())
    preserve_exact = len(frames) <= _EXACT_PRESERVE_MAX_FRAMES

    strings: list[str] = []
    str_map: dict[str, int] = {}

    def _intern_str(val: Any) -> int:
        s = str(val or "")
        idx = str_map.get(s)
        if idx is None:
            idx = len(strings)
            str_map[s] = idx
            strings.append(s)
        return idx

    # Static per-vehicle metadata table: [id_str, vtype_sidx, L, W, goal_sidx]
    veh_meta: list[list[Any]] = []
    veh_map: dict[str, int] = {}

    def _intern_veh_from_dict(v: Mapping[str, Any]) -> int:
        vid = str(v["id"])
        idx = veh_map.get(vid)
        if idx is None:
            idx = len(veh_meta)
            veh_map[vid] = idx
            veh_meta.append(
                [
                    vid,
                    _intern_str(v.get("vtype", "")),
                    round(float(v.get("L", 4.5)), 2),
                    round(float(v.get("W", 1.85)), 2),
                    _intern_str(v.get("goal", "")),
                ]
            )
        return idx

    def _intern_veh_id_only(vid_raw: Any) -> int:
        vid = str(vid_raw)
        idx = veh_map.get(vid)
        if idx is None:
            idx = len(veh_meta)
            veh_map[vid] = idx
            veh_meta.append([vid, _intern_str(""), 4.5, 1.85, _intern_str("")])
        return idx

    tls_keys = sorted({str(k) for f in frames for k in (f.get("tls") or {}).keys()})
    packed_frames: list[list[Any]] = []

    for f in frames:
        vehs = f.get("vehicles") or ()
        veh_corr_by_id: dict[str, str] = {}
        if vehs:
            v_idx: list[int] = []
            x_cm: list[int] = []
            y_cm: list[int] = []
            yaw_mrad: list[int] = []
            v_cms: list[int] = []
            a_cms: list[int] = []
            corr_sidx: list[int] = []
            state_bits: list[int] = []
            d_post_pm: list[int] = []
            psi_pm: list[int] = []
            std_cm: list[int] = []

            for v in vehs:
                vid_s = str(v["id"])
                corr_s = str(v.get("corridor", ""))
                veh_corr_by_id[vid_s] = corr_s
                v_idx.append(_intern_veh_from_dict(v))
                x_cm.append(int(round(float(v["x"]) * 100.0)))
                y_cm.append(int(round(float(v["y"]) * 100.0)))
                yaw_mrad.append(int(round(float(v["yaw"]) * 1000.0)))
                v_cms.append(int(round(float(v["v"]) * 100.0)))
                a_cms.append(int(round(float(v.get("a", 0.0)) * 100.0)))
                corr_sidx.append(_intern_str(corr_s))

                turn_clamped = max(-1, min(6, int(v.get("turn", 0))))
                mode_clamped = max(-1, min(14, int(v.get("mode", -1))))
                sb = (
                    (1 if v.get("brake") else 0)
                    | (((turn_clamped + 1) & 7) << 1)
                    | (((mode_clamped + 1) & 15) << 4)
                )
                state_bits.append(sb)
                d_post_pm.append(int(round(float(v.get("d_post", 0.0)) * 1000.0)))
                psi_pm.append(int(round(float(v.get("psi", 0.0)) * 1000.0)))
                std_cm.append(int(round(float(v.get("pos_std", 0.0)) * 100.0)))

            veh_cols: list[list[int]] = [
                v_idx,
                x_cm,
                y_cm,
                yaw_mrad,
                v_cms,
                _zero_elide(a_cms),
                corr_sidx,
                state_bits,
                _zero_elide(d_post_pm),
                _zero_elide(psi_pm),
                _zero_elide(std_cm),
            ]
        else:
            veh_cols = []

        raw_edges = list(f.get("dbn_edges") or ())
        if preserve_exact:
            kept_edges = raw_edges
        else:
            kept_edges = _select_spatially_diverse_edges(
                raw_edges,
                veh_corr_by_id,
                min_omega=min_omega,
                max_edges=max_edges_per_frame,
            )

        if kept_edges:
            e_src: list[int] = []
            e_dst: list[int] = []
            e_rel: list[int] = []
            e_om: list[int] = []
            e_ka: list[int] = []
            e_ga: list[int] = []
            e_pc: list[int] = []
            for edge in kept_edges:
                e_src.append(_intern_veh_id_only(edge["src"]))
                e_dst.append(_intern_veh_id_only(edge["dst"]))
                e_rel.append(int(edge.get("rel", 0)))
                e_om.append(int(round(float(edge["omega"]) * 1000.0)))
                e_ka.append(int(round(float(edge.get("kappa", 0.0)) * 1000.0)))
                e_ga.append(int(round(float(edge.get("gamma", 0.0)) * 1000.0)))
                e_pc.append(int(round(float(edge.get("p_conf", 0.0)) * 1000.0)))
            edge_cols: list[list[int]] = [
                e_src,
                e_dst,
                e_rel,
                e_om,
                _zero_elide(e_ka),
                _zero_elide(e_ga),
                _zero_elide(e_pc),
            ]
        else:
            edge_cols = []

        raw_heat = f.get("corridor_heat") or {}
        h_cid: list[int] = []
        h_occ: list[int] = []
        h_tail: list[int] = []
        h_spill: list[int] = []
        for cid, hv in raw_heat.items():
            occ = float(hv.get("occ", 0.0))
            tail = float(hv.get("tail", 0.0))
            spill = float(hv.get("spill", 0.0))
            if preserve_exact or max(occ, tail, spill) >= _LARGE_TRACE_MIN_HEAT:
                h_cid.append(_intern_str(cid))
                h_occ.append(int(round(occ * 1000.0)))
                h_tail.append(int(round(tail * 1000.0)))
                h_spill.append(int(round(spill * 1000.0)))
        heat_cols: list[list[int]] = [h_cid, h_occ, h_tail, h_spill] if h_cid else []

        tls_map = f.get("tls") or {}
        tls_vals = [str(tls_map.get(tk, "")) for tk in tls_keys] if tls_keys else []
        poll_vidxs = [_intern_veh_id_only(pid) for pid in (f.get("v2x_polls") or ())]

        kpi = f.get("kpi") or {}
        kpi_arr = [
            int(kpi.get("active_veh", len(vehs))),
            int(kpi.get("halted_veh", 0)),
            round(float(kpi.get("mean_speed_mps", 0.0)), 3),
            int(kpi.get("active_edges", len(raw_edges))),
            round(float(kpi.get("mean_psi", 0.0)), 4),
            int(kpi.get("spillback_corridors", 0)),
            int(kpi.get("cum_collisions", 0)),
            int(kpi.get("cum_emerg_brakes", 0)),
            int(kpi.get("cum_reroutes", 0)),
            round(float(kpi.get("dbn_ms", 0.0)), 3),
        ]

        packed_frames.append(
            [
                round(float(f["t"]), 3),
                veh_cols,
                edge_cols,
                heat_cols,
                tls_vals,
                poll_vidxs,
                kpi_arr,
            ]
        )

    packed: dict[str, Any] = {
        "schema_version": trace_data.get("schema_version", 1),
        "scenario_id": trace_data.get("scenario_id", "unknown"),
        "policy_name": trace_data.get("policy_name", "unknown"),
        "seed": int(trace_data.get("seed", 0)),
        "dt": float(trace_data.get("dt", 0.5)),
        "frame_count": len(frames),
        "timeline": trace_data.get("timeline"),
        "summary_metrics": trace_data.get("summary_metrics"),
        "encoding": "soa_v2",
        "strings": strings,
        "veh_meta": veh_meta,
        "tls_keys": tls_keys,
        "packed_frames": packed_frames,
    }
    if preserve_exact:
        packed["frames"] = frames
    return packed


def _unpack_trace_soa(trace_data: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Restore standard `{..., 'frames': [...]}` dictionary from a packed `soa_v2` trace."""
    if trace_data is None:
        return None
    if isinstance(trace_data.get("frames"), list) and len(trace_data["frames"]) > 0:
        out = {
            k: v
            for k, v in trace_data.items()
            if k not in ("encoding", "strings", "veh_meta", "tls_keys", "packed_frames")
        }
        return out
    if trace_data.get("encoding") != "soa_v2" or not isinstance(trace_data.get("packed_frames"), list):
        return dict(trace_data)

    strings: list[str] = list(trace_data.get("strings") or ())
    veh_meta: list[list[Any]] = list(trace_data.get("veh_meta") or ())
    tls_keys: list[str] = list(trace_data.get("tls_keys") or ())
    packed_frames: list[list[Any]] = list(trace_data["packed_frames"])

    frames_out: list[dict[str, Any]] = []
    for pf in packed_frames:
        sim_t = float(pf[0])
        veh_cols = pf[1]
        edge_cols = pf[2]
        heat_cols = pf[3]
        tls_vals = pf[4]
        poll_vidxs = pf[5]
        kpi_arr = pf[6]

        vehicles_frame: list[dict[str, Any]] = []
        if veh_cols:
            v_idx, x_cm, y_cm, yaw_mrad, v_cms, a_cms, corr_sidx, state_bits, d_post_pm, psi_pm, std_cm = veh_cols
            n_v = len(v_idx)
            for i in range(n_v):
                vm = veh_meta[v_idx[i]]
                sb = state_bits[i]
                d_post_val = round(d_post_pm[i] * 0.001, 4) if d_post_pm else 0.0
                vehicles_frame.append(
                    {
                        "id": vm[0],
                        "vtype": strings[vm[1]],
                        "x": round(x_cm[i] * 0.01, 3),
                        "y": round(y_cm[i] * 0.01, 3),
                        "yaw": round(yaw_mrad[i] * 0.001, 4),
                        "v": round(v_cms[i] * 0.01, 3),
                        "a": round(a_cms[i] * 0.01, 3) if a_cms else 0.0,
                        "L": float(vm[2]),
                        "W": float(vm[3]),
                        "brake": bool(sb & 1),
                        "turn": ((sb >> 1) & 7) - 1,
                        "goal": strings[vm[4]],
                        "corridor": strings[corr_sidx[i]],
                        "mode": ((sb >> 4) & 15) - 1,
                        "pos_std": round(std_cm[i] * 0.01, 3) if std_cm else 0.0,
                        "entropy": 0.0,
                        "obs_age": 0.0,
                        "d_pred": d_post_val,
                        "d_post": d_post_val,
                        "psi": round(psi_pm[i] * 0.001, 4) if psi_pm else 0.0,
                        "r_score": 0.0,
                    }
                )

        edges_frame: list[dict[str, Any]] = []
        if edge_cols:
            e_src, e_dst, e_rel, e_om, e_ka, e_ga, e_pc = edge_cols
            n_e = len(e_src)
            for i in range(n_e):
                edges_frame.append(
                    {
                        "src": veh_meta[e_src[i]][0],
                        "dst": veh_meta[e_dst[i]][0],
                        "rel": int(e_rel[i]),
                        "omega": round(e_om[i] * 0.001, 4),
                        "kappa": round(e_ka[i] * 0.001, 4) if e_ka else 0.0,
                        "gamma": round(e_ga[i] * 0.001, 4) if e_ga else 0.0,
                        "p_conf": round(e_pc[i] * 0.001, 4) if e_pc else 0.0,
                    }
                )

        corridor_heat: dict[str, dict[str, float]] = {}
        if heat_cols:
            h_cid, h_occ, h_tail, h_spill = heat_cols
            for i in range(len(h_cid)):
                corridor_heat[strings[h_cid[i]]] = {
                    "occ": round(h_occ[i] * 0.001, 4),
                    "tail": round(h_tail[i] * 0.001, 4),
                    "spill": round(h_spill[i] * 0.001, 4),
                }

        tls_dict = {tls_keys[i]: tls_vals[i] for i in range(min(len(tls_keys), len(tls_vals)))}
        polls_list = [veh_meta[vi][0] for vi in poll_vidxs]
        kpi_dict = {
            "active_veh": int(kpi_arr[0]),
            "halted_veh": int(kpi_arr[1]),
            "mean_speed_mps": float(kpi_arr[2]),
            "active_edges": int(kpi_arr[3]),
            "mean_psi": float(kpi_arr[4]),
            "spillback_corridors": int(kpi_arr[5]),
            "cum_collisions": int(kpi_arr[6]),
            "cum_emerg_brakes": int(kpi_arr[7]),
            "cum_reroutes": int(kpi_arr[8]),
            "dbn_ms": float(kpi_arr[9]),
        }

        frames_out.append(
            {
                "t": sim_t,
                "vehicles": vehicles_frame,
                "dbn_edges": edges_frame,
                "corridor_heat": corridor_heat,
                "tls": tls_dict,
                "v2x_polls": polls_list,
                "kpi": kpi_dict,
            }
        )

    return {
        "schema_version": trace_data.get("schema_version", 1),
        "scenario_id": trace_data.get("scenario_id", "unknown"),
        "policy_name": trace_data.get("policy_name", "unknown"),
        "seed": int(trace_data.get("seed", 0)),
        "dt": float(trace_data.get("dt", 0.5)),
        "frame_count": int(trace_data.get("frame_count", len(frames_out))),
        "frames": frames_out,
        "timeline": trace_data.get("timeline"),
        "summary_metrics": trace_data.get("summary_metrics"),
    }


def pack_viewer_payload(
    scene_data: Mapping[str, Any],
    primary_trace: Mapping[str, Any],
    comparison_trace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build compact `swarm_3d_soa_v2` viewer payload dictionary ready for Gzip compression."""
    _validate_scene_dict(scene_data)
    _validate_trace_dict(primary_trace, "primary_trace")
    if comparison_trace is not None:
        _validate_trace_dict(comparison_trace, "comparison_trace")

    prepared_scene = _prepare_scene_for_viewer(scene_data)
    return {
        "format": PAYLOAD_FORMAT_VERSION,
        "scene": prepared_scene,
        "primary_trace": _pack_trace_soa(primary_trace),
        "comparison_trace": _pack_trace_soa(comparison_trace) if comparison_trace is not None else None,
    }


def unpack_viewer_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Unpack a `swarm_3d_soa_v2` (or raw v1) payload dictionary into `{scene, primary_trace, comparison_trace}`."""
    if not isinstance(payload, Mapping) or "scene" not in payload or "primary_trace" not in payload:
        raise ConfigValidationError("Invalid viewer payload: expected Mapping with 'scene' and 'primary_trace'.")
    return {
        "scene": dict(payload["scene"]),
        "primary_trace": _unpack_trace_soa(payload["primary_trace"]),
        "comparison_trace": _unpack_trace_soa(payload.get("comparison_trace")),
    }


def decode_viewer_html_payload(html_text: str) -> dict[str, Any]:
    """Extract, Base64+Gzip decompress, and unpack the embedded 3D viewer payload from compiled HTML."""
    m_gz = re.search(
        r'<script\s+id="swarm-payload-gz"[^>]*>\s*(.*?)\s*</script>',
        html_text,
        flags=re.DOTALL,
    )
    if m_gz is not None:
        b64_str = m_gz.group(1).strip()
        try:
            gz_bytes = base64.b64decode(b64_str, validate=False)
            raw_json = gzip.decompress(gz_bytes).decode("utf-8")
            packed = json.loads(raw_json)
        except (ValueError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConfigValidationError(f"Failed to decode swarm-payload-gz: {exc}") from exc
        return unpack_viewer_payload(packed)

    m_raw = re.search(
        r'<script\s+id="swarm-payload"\s+type="application/json">\s*(.*?)\s*</script>',
        html_text,
        flags=re.DOTALL,
    )
    if m_raw is not None:
        try:
            parsed = json.loads(m_raw.group(1).replace("<\\/", "</"))
        except json.JSONDecodeError as exc:
            raise ConfigValidationError(f"Failed to parse legacy swarm-payload JSON: {exc}") from exc
        return unpack_viewer_payload(parsed)

    raise ConfigValidationError("Compiled HTML contains neither 'swarm-payload-gz' nor 'swarm-payload'.")


def write_city_3d_viewer(
    scene_data: Mapping[str, Any],
    primary_trace: Mapping[str, Any],
    output_html: Path | str,
    comparison_trace: Mapping[str, Any] | None = None,
    *,
    title: str | None = None,
) -> Path:
    """Compile a self-contained offline 3D WebGL2 City & Neural DBN interactive HTML viewer."""
    _validate_scene_dict(scene_data)
    _validate_trace_dict(primary_trace, "primary_trace")
    if comparison_trace is not None:
        _validate_trace_dict(comparison_trace, "comparison_trace")

    if not _TEMPLATE_PATH.is_file():
        raise ConfigValidationError(
            f"Missing 3D HTML template at {_TEMPLATE_PATH}.",
            context={"template_path": str(_TEMPLATE_PATH)},
        )
    if not _ENGINE_JS_PATH.is_file():
        raise ConfigValidationError(
            f"Missing WebGL2 engine asset at {_ENGINE_JS_PATH}.",
            context={"engine_js_path": str(_ENGINE_JS_PATH)},
        )

    template_text = _TEMPLATE_PATH.read_text(encoding="utf-8")
    engine_js_text = _ENGINE_JS_PATH.read_text(encoding="utf-8")

    # Verify asset purity (zero external URLs in template or engine JS)
    if _EXTERNAL_URL_PATTERN.search(template_text) or _EXTERNAL_URL_PATTERN.search(engine_js_text):
        raise ConfigValidationError(
            "Offline integrity violation: external URL detected in 3D viewer template or JS."
        )

    packed_payload = pack_viewer_payload(
        scene_data=scene_data,
        primary_trace=primary_trace,
        comparison_trace=comparison_trace,
    )
    prepared_scene = packed_payload["scene"]
    packed_primary = packed_payload["primary_trace"]
    packed_compare = packed_payload["comparison_trace"]

    net_id = str(prepared_scene.get("network_id", "city"))
    p_name = str(packed_primary.get("policy_name", "neural_dbn"))
    if title is None:
        if packed_compare is not None:
            c_name = str(packed_compare.get("policy_name", "baseline"))
            resolved_title = f"3D City Swarm DBN — {p_name} vs {c_name} ({net_id})"
        else:
            resolved_title = f"3D City Swarm DBN — {p_name} ({net_id})"
    else:
        resolved_title = str(title)

    packed_json_bytes = json.dumps(packed_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    gz_bytes = gzip.compress(packed_json_bytes, compresslevel=6, mtime=0)
    payload_gz_b64 = base64.b64encode(gz_bytes).decode("ascii")

    meta_obj: dict[str, Any] = {
        "format": PAYLOAD_FORMAT_VERSION,
        "title": resolved_title,
        "network_id": net_id,
        "scenario_id": str(packed_primary.get("scenario_id", net_id)),
        "seed": int(packed_primary.get("seed", 0)),
        "dt": float(packed_primary.get("dt", 0.5)),
        "frame_count": int(packed_primary.get("frame_count", 0)),
        "primary_policy": p_name,
        "comparison_policy": str(packed_compare.get("policy_name")) if packed_compare is not None else None,
        "has_comparison": packed_compare is not None,
        "primary_summary": packed_primary.get("summary_metrics"),
        "comparison_summary": packed_compare.get("summary_metrics") if packed_compare is not None else None,
        "bounds": prepared_scene["bounds"],
        "focus_bounds": prepared_scene["focus_bounds"],
        "packed_json_bytes": len(packed_json_bytes),
        "gzip_bytes": len(gz_bytes),
    }
    payload_meta_json = json.dumps(meta_obj, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")

    rendered_html = (
        template_text.replace("{{PAGE_TITLE}}", html.escape(resolved_title))
        .replace("{{PAYLOAD_META_JSON}}", payload_meta_json)
        .replace("{{PAYLOAD_GZ_B64}}", payload_gz_b64)
    )
    if "{{PAYLOAD_JSON}}" in rendered_html:
        legacy_obj = unpack_viewer_payload(packed_payload)
        legacy_json = json.dumps(legacy_obj, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")
        rendered_html = rendered_html.replace("{{PAYLOAD_JSON}}", legacy_json)
    rendered_html = rendered_html.replace("{{ENGINE_JS}}", engine_js_text)

    out_p = Path(output_html)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = out_p.parent / f".{out_p.name}.pid{os.getpid()}.tmp"
    tmp_p.write_text(rendered_html, encoding="utf-8")
    os.replace(tmp_p, out_p)
    return out_p


__all__ = [
    "PAYLOAD_FORMAT_VERSION",
    "decode_viewer_html_payload",
    "pack_viewer_payload",
    "unpack_viewer_payload",
    "write_city_3d_viewer",
]

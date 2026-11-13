"""Self-contained offline WebGL2 3D City & Neural DBN HTML replay builder.

Embeds extracted 3D urban scene geometry, primary (and optional synchronized split-screen
baseline) episode trace data, and the local native WebGL2 shader engine into a single
zero-dependency HTML file making zero network requests.
"""

from __future__ import annotations

from collections.abc import Mapping
import html
import json
import os
from pathlib import Path
import re
from typing import Any

from ..logging_utils import ConfigValidationError

_VIZ_DIR = Path(__file__).resolve().parent
_TEMPLATE_PATH = _VIZ_DIR / "templates" / "city_dbn_3d.html"
_ENGINE_JS_PATH = _VIZ_DIR / "assets" / "city_engine_webgl2.js"
_EXTERNAL_URL_PATTERN = re.compile(r"https?://", flags=re.IGNORECASE)


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
    if not isinstance(frames, list) or len(frames) == 0:
        raise ConfigValidationError(
            f"{label} must contain a non-empty 'frames' list.",
            context={"label": label},
        )


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
            "Offline integrity violation: external http/https URL detected in 3D viewer template or JS."
        )

    net_id = str(scene_data.get("network_id", "city"))
    p_name = str(primary_trace.get("policy_name", "neural_dbn"))
    if title is None:
        if comparison_trace is not None:
            c_name = str(comparison_trace.get("policy_name", "baseline"))
            resolved_title = f"3D City Swarm DBN — {p_name} vs {c_name} ({net_id})"
        else:
            resolved_title = f"3D City Swarm DBN — {p_name} ({net_id})"
    else:
        resolved_title = str(title)

    payload_obj: dict[str, Any] = {
        "scene": dict(scene_data),
        "primary_trace": dict(primary_trace),
        "comparison_trace": dict(comparison_trace) if comparison_trace is not None else None,
    }
    # Escape `</` so embedded JSON can never prematurely close `<script>`
    payload_json = json.dumps(payload_obj, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")

    rendered_html = (
        template_text.replace("{{PAGE_TITLE}}", html.escape(resolved_title))
        .replace("{{PAYLOAD_JSON}}", payload_json)
        .replace("{{ENGINE_JS}}", engine_js_text)
    )

    out_p = Path(output_html)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = out_p.parent / f".{out_p.name}.pid{os.getpid()}.tmp"
    tmp_p.write_text(rendered_html, encoding="utf-8")
    os.replace(tmp_p, out_p)
    return out_p


__all__ = [
    "write_city_3d_viewer",
]

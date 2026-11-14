"""Flagship offline WebGL2 3D City & Neural DBN interactive visualizer and 300-DPI 2D plots."""

from __future__ import annotations

from .plots_2d import (
    POLICY_COLORS,
    POLICY_LABELS,
    SCENARIO_LABELS,
    plot_ablation_breakdown,
    plot_calibration_and_roc_pr,
    plot_latency_scaling,
    plot_paired_forest,
    plot_scenario_comparison,
    render_publication_plots,
)
from .scene_builder import (
    SCENE_SCHEMA_VERSION,
    compute_scene_focus_bounds,
    export_scene_3d,
    export_scene_3d_from_files,
    synthesize_procedural_blocks_for_scene,
    triangulate_polygon_2d,
)
from .trace_recorder import EpisodeTraceRecorder, load_trace_json
from .viewer_3d import (
    PAYLOAD_FORMAT_VERSION,
    decode_viewer_html_payload,
    pack_viewer_payload,
    unpack_viewer_payload,
    write_city_3d_viewer,
)

__all__ = [
    "EpisodeTraceRecorder",
    "PAYLOAD_FORMAT_VERSION",
    "POLICY_COLORS",
    "POLICY_LABELS",
    "SCENARIO_LABELS",
    "SCENE_SCHEMA_VERSION",
    "compute_scene_focus_bounds",
    "decode_viewer_html_payload",
    "export_scene_3d",
    "export_scene_3d_from_files",
    "load_trace_json",
    "pack_viewer_payload",
    "plot_ablation_breakdown",
    "plot_calibration_and_roc_pr",
    "plot_latency_scaling",
    "plot_paired_forest",
    "plot_scenario_comparison",
    "render_publication_plots",
    "synthesize_procedural_blocks_for_scene",
    "triangulate_polygon_2d",
    "unpack_viewer_payload",
    "write_city_3d_viewer",
]

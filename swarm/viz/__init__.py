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
    export_scene_3d,
    export_scene_3d_from_files,
    triangulate_polygon_2d,
)
from .trace_recorder import EpisodeTraceRecorder, load_trace_json
from .viewer_3d import write_city_3d_viewer

__all__ = [
    "EpisodeTraceRecorder",
    "POLICY_COLORS",
    "POLICY_LABELS",
    "SCENARIO_LABELS",
    "SCENE_SCHEMA_VERSION",
    "export_scene_3d",
    "export_scene_3d_from_files",
    "load_trace_json",
    "plot_ablation_breakdown",
    "plot_calibration_and_roc_pr",
    "plot_latency_scaling",
    "plot_paired_forest",
    "plot_scenario_comparison",
    "render_publication_plots",
    "triangulate_polygon_2d",
    "write_city_3d_viewer",
]

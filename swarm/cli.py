"""Fail-loud command-line entrypoint for ``bn-vehicle-swarm-coordination``.

Exposes 9 publication-grade pipeline commands:
1. ``build-nets``      — Generate procedural 3x3/4x4 urban grids & Downtown Palo Alto OSM networks + scenario JSONs.
2. ``collect``         — Collect leak-free mixed-policy SUMO rollouts + offline causal future supervision shards.
3. ``train``           — Train ``STGraphTransformerDBN`` via unrolled BPTT + Boundary Lock B L-BFGS calibration.
4. ``eval-predictive`` — Evaluate held-out ID & Zero-Shot OOD predictive structure/delay ROC-AUC, PR-AUC, & ECE.
5. ``simulate``        — Execute one seeded SUMO episode with optional trace recording & offline WebGL2 3D HTML.
6. ``evaluate``        — Run multi-process paired statistical evaluation (10,000-resample CIs + exact Wilcoxon).
7. ``benchmark``       — Run warmed CPU DBN structure-update & LBP latency scaling benchmark across N=16..256.
8. ``render-3d``       — Build self-contained offline WebGL2 3D City & DBN HTML viewer (single or split-screen).
9. ``demo``            — End-to-end single/split-screen SUMO simulation + interactive 3D HTML replay generator.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from typing import Any

from . import __version__
from .config import (
    ControlConfig,
    DBNConfig,
    ExperimentConfig,
    MODEL_PRESETS,
    ScenarioConfig,
    TEST_SEEDS,
    TRAIN_SEEDS,
    TrainConfig,
    VAL_SEEDS,
    load_scenario_config,
)
from .contracts import MAIN_POLICY_NAMES, NEURAL_POLICY_NAMES, POLICY_NAMES
from .logging_utils import (
    ConfigValidationError,
    configure_logging,
    fail_loud_entrypoint,
    get_logger,
    update_crash_context,
)
from .pipelines.benchmark import DEFAULT_BENCHMARK_METHODS, DEFAULT_SWARM_SIZES, run_scaling_benchmark
from .pipelines.evaluator import evaluate_experiment_matrix, run_single_episode
from .pipelines.rollout_collector import collect_rollout_dataset
from .pipelines.trainer import evaluate_heldout_predictive, train_neural_dbn

_LOG = get_logger("cli")


def _resolve_error_report_dir(argv: Sequence[str] | None = None) -> Path:
    """Resolve crash report directory from ``--error-report-dir`` or ``SWARM_ERROR_REPORT_DIR``."""
    tokens = list(argv) if argv is not None else []
    for idx, tok in enumerate(tokens):
        if tok == "--error-report-dir" and idx + 1 < len(tokens):
            return Path(tokens[idx + 1]).expanduser().resolve()
        if tok.startswith("--error-report-dir="):
            return Path(tok.split("=", 1)[1]).expanduser().resolve()
    env_dir = os.environ.get("SWARM_ERROR_REPORT_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path.cwd().resolve()


def _resolve_scenario_arg(
    scenario_arg: str,
    *,
    backend_override: str | None = None,
    must_exist_network_files: bool = True,
) -> ScenarioConfig:
    """Resolve a ``ScenarioConfig`` from a JSON path or canonical scenario identifier."""
    cand_path = Path(scenario_arg)
    if cand_path.is_file():
        cfg = load_scenario_config(cand_path, must_exist_network_files=must_exist_network_files)
    else:
        default_json = Path("configs") / "scenarios" / f"{scenario_arg}.json"
        if default_json.is_file():
            cfg = load_scenario_config(default_json, must_exist_network_files=must_exist_network_files)
        else:
            from .env.network_builder import default_scenario_configs

            canon = default_scenario_configs()
            if scenario_arg not in canon:
                raise ConfigValidationError(
                    f"Scenario {scenario_arg!r} is neither an existing JSON path nor a canonical scenario ID "
                    f"({sorted(canon)}).",
                    context={"scenario_arg": scenario_arg, "canonical_ids": sorted(canon)},
                )
            cfg = canon[scenario_arg]
            if must_exist_network_files:
                cfg.resolve_paths(must_exist=True)

    if backend_override is not None:
        cfg = replace(cfg, sumo_backend=backend_override)  # type: ignore[arg-type]
    return cfg


def _atomic_write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = path.with_name(f".{path.name}.pid{os.getpid()}.tmp")
    tmp_p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_p, path)
    return path


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``swarm`` CLI argument parser and all 9 subcommands."""
    parser = argparse.ArgumentParser(
        prog="swarm",
        description="City-scale autonomous vehicle swarm coordination via neurally updated DBNs in SUMO.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Structured logging verbosity level.",
    )
    parser.add_argument(
        "--error-report-dir",
        type=Path,
        default=None,
        help="Directory where fail-loud error_report_<UTC>_pid<PID>.json diagnostics are written.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # 1. build-nets
    p_nets = sub.add_parser(
        "build-nets",
        help="Build procedural 3x3/4x4 SUMO urban grids & Downtown Palo Alto OSM networks + scenarios.",
    )
    p_nets.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root directory.")
    p_nets.add_argument(
        "--no-osm-download",
        action="store_true",
        help="Require cached Palo Alto OSM XML rather than querying OpenStreetMap API.",
    )
    p_nets.add_argument(
        "--no-write-scenarios",
        action="store_true",
        help="Skip writing canonical ScenarioConfig JSON files under configs/scenarios/.",
    )

    # 2. collect
    p_col = sub.add_parser(
        "collect",
        help="Collect zero-leakage SUMO rollout shards with offline future causal supervision labels.",
    )
    p_col.add_argument(
        "--scenario",
        required=True,
        nargs="+",
        help="One or more scenario JSON paths or canonical scenario IDs.",
    )
    p_col.add_argument(
        "--split",
        required=True,
        choices=("train", "val", "eval_id", "eval_ood", "test"),
        help="Dataset split (strictly enforces seed & ID/OOD topology separation).",
    )
    p_col.add_argument("--output-dir", type=Path, required=True, help="Output shard directory.")
    p_col.add_argument("--seeds", type=int, nargs="*", default=None, help="Explicit seed list override.")
    p_col.add_argument("--workers", type=int, default=4, help="Parallel rollout worker processes.")
    p_col.add_argument("--backend", choices=("libsumo", "traci"), default=None, help="SUMO backend override.")
    p_col.add_argument("--behavior-policy", default=None, help="Optional single behavior policy override.")
    p_col.add_argument(
        "--max-windows-per-episode",
        type=int,
        default=None,
        help="Optional cap of evenly spaced contiguous trajectory windows recorded per episode.",
    )
    p_col.add_argument(
        "--window-len-steps",
        type=int,
        default=16,
        help="Contiguous step count per recorded trajectory window when --max-windows-per-episode is set.",
    )

    # 3. train
    p_trn = sub.add_parser(
        "train",
        help="Train STGraphTransformerDBN via unrolled BPTT & fit Boundary Lock B temperature scaling.",
    )
    p_trn.add_argument("--train-dir", type=Path, required=True, help="Training rollout shard directory.")
    p_trn.add_argument("--val-dir", type=Path, required=True, help="Validation rollout shard directory.")
    p_trn.add_argument(
        "--output-checkpoint",
        type=Path,
        required=True,
        help="Output .pt checkpoint path (or directory).",
    )
    p_trn.add_argument("--report-path", type=Path, default=None, help="Optional JSON training report path.")
    p_trn.add_argument(
        "--preset",
        choices=tuple(sorted(MODEL_PRESETS)),
        default="sota_local",
        help="Neural DBN architecture preset.",
    )
    p_trn.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu", help="Compute device.")
    p_trn.add_argument("--precision", choices=("fp32", "bf16"), default="fp32", help="Floating-point precision.")
    p_trn.add_argument("--compile", action="store_true", help="Enable torch.compile.")
    p_trn.add_argument("--distributed", action="store_true", help="Enable DistributedDataParallel (torchrun).")
    p_trn.add_argument("--epochs", type=int, default=40, help="Maximum training epochs.")
    p_trn.add_argument("--batch-size", type=int, default=16, help="Trajectory windows per optimizer step.")
    p_trn.add_argument("--lr", type=float, default=3e-4, help="Peak AdamW learning rate.")
    p_trn.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay.")
    p_trn.add_argument("--patience", type=int, default=8, help="Early stopping validation patience.")
    p_trn.add_argument("--seed", type=int, default=2026, help="Deterministic RNG seed.")

    # 4. eval-predictive
    p_pred = sub.add_parser(
        "eval-predictive",
        help="Evaluate held-out ID & Zero-Shot OOD predictive ROC-AUC, PR-AUC, Brier, & 10-bin ECE.",
    )
    p_pred.add_argument("--checkpoint", type=Path, required=True, help="Calibrated neural DBN checkpoint.")
    p_pred.add_argument("--eval-id-dir", type=Path, default=None, help="Held-out ID rollout shard directory.")
    p_pred.add_argument("--eval-ood-dir", type=Path, default=None, help="Zero-shot OOD rollout shard directory.")
    p_pred.add_argument("--dataset-dir", type=Path, default=None, help="Single held-out rollout directory.")
    p_pred.add_argument(
        "--named-split",
        nargs="*",
        default=None,
        help="Named held-out splits as NAME=DIR pairs (e.g. id_grid3x3=... ood_grid4x4=... ood_palo_alto=...).",
    )
    p_pred.add_argument("--output", type=Path, required=True, help="Output predictive report JSON path.")
    p_pred.add_argument("--plot-output", type=Path, default=None, help="Optional ROC/PR/ECE figure PNG path.")
    p_pred.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu", help="Inference device.")
    p_pred.add_argument("--allow-ci-micro", action="store_true", help="Allow ci_micro preset (unit tests only).")

    # 5. simulate
    p_sim = sub.add_parser(
        "simulate",
        help="Run a single seeded SUMO episode with optional trace export & 3D HTML replay.",
    )
    p_sim.add_argument("--scenario", required=True, help="Scenario JSON path or canonical scenario ID.")
    p_sim.add_argument(
        "--policy",
        default="neural_dbn",
        choices=tuple(sorted(POLICY_NAMES)),
        help="Swarm coordination policy.",
    )
    p_sim.add_argument("--seed", type=int, required=True, help="Deterministic episode seed.")
    p_sim.add_argument("--checkpoint", type=Path, default=None, help="Neural DBN checkpoint path.")
    p_sim.add_argument("--backend", choices=("libsumo", "traci"), default=None, help="SUMO backend override.")
    p_sim.add_argument("--output-metrics", type=Path, default=None, help="Optional output EpisodeMetrics JSON.")
    p_sim.add_argument("--trace-out", type=Path, default=None, help="Optional EpisodeTraceRecorder JSON path.")
    p_sim.add_argument("--html-out", type=Path, default=None, help="Optional offline WebGL2 3D HTML output.")
    p_sim.add_argument("--allow-ci-micro", action="store_true", help="Allow ci_micro preset (unit tests only).")

    # 6. evaluate
    p_eval = sub.add_parser(
        "evaluate",
        help="Run multi-process paired statistical evaluation across scenarios, policies, & TEST_SEEDS.",
    )
    p_eval.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenario JSON paths or canonical IDs (defaults to all 5 canonical scenarios).",
    )
    p_eval.add_argument(
        "--policies",
        nargs="+",
        default=list(MAIN_POLICY_NAMES),
        choices=tuple(sorted(POLICY_NAMES)),
        help="Policies to evaluate.",
    )
    p_eval.add_argument("--seeds", type=int, nargs="*", default=None, help="Seeds (defaults to TEST_SEEDS).")
    p_eval.add_argument("--checkpoint", type=Path, default=None, help="Calibrated neural DBN checkpoint.")
    p_eval.add_argument("--output-dir", type=Path, required=True, help="Evaluation artifacts output directory.")
    p_eval.add_argument("--workers", type=int, default=4, help="Parallel episode worker processes.")
    p_eval.add_argument("--backend", choices=("libsumo", "traci"), default=None, help="SUMO backend override.")
    p_eval.add_argument("--bootstrap-resamples", type=int, default=10_000, help="Paired bootstrap resamples.")
    p_eval.add_argument("--plots-dir", type=Path, default=None, help="Optional 300-DPI plots output directory.")
    p_eval.add_argument("--allow-ci-micro", action="store_true", help="Allow ci_micro preset (unit tests only).")
    p_eval.add_argument(
        "--allow-non-test-seeds",
        action="store_true",
        help="Bypass TEST_SEEDS lock (for local smoke checks only; never for reported claims).",
    )

    # 7. benchmark
    p_bench = sub.add_parser(
        "benchmark",
        help="Run warmed CPU DBN structure-update & LBP latency scaling benchmark across N=16..256.",
    )
    p_bench.add_argument("--output", type=Path, required=True, help="Output scaling.json path.")
    p_bench.add_argument("--plot-output", type=Path, default=None, help="Optional latency scaling PNG path.")
    p_bench.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SWARM_SIZES),
        help="Swarm vehicle counts N.",
    )
    p_bench.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_BENCHMARK_METHODS),
        help="DBN updaters to benchmark.",
    )
    p_bench.add_argument("--checkpoint", type=Path, default=None, help="Optional trained neural DBN checkpoint.")
    p_bench.add_argument(
        "--preset",
        choices=tuple(sorted(MODEL_PRESETS)),
        default="sota_local",
        help="Model preset when benchmarking neural_dbn.",
    )
    p_bench.add_argument("--warmup", type=int, default=5, help="Untimed warmup steps per N.")
    p_bench.add_argument("--repeats", type=int, default=20, help="Timed steady-state repeats per N.")
    p_bench.add_argument("--allow-ci-micro", action="store_true", help="Allow ci_micro preset (unit tests only).")
    p_bench.add_argument("--seed", type=int, default=2026, help="Benchmark RNG seed.")

    # 8. render-3d
    p_r3d = sub.add_parser(
        "render-3d",
        help="Render a self-contained offline WebGL2 3D City & Neural DBN HTML replay from trace JSON.",
    )
    p_r3d.add_argument("--trace", type=Path, required=True, help="Primary episode trace JSON path.")
    p_r3d.add_argument("--output-html", type=Path, required=True, help="Output self-contained HTML path.")
    p_r3d.add_argument("--compare-trace", type=Path, default=None, help="Optional comparison trace JSON path.")
    p_r3d.add_argument("--scene-json", type=Path, default=None, help="Optional pre-exported 3D scene JSON.")
    p_r3d.add_argument("--scenario", default=None, help="Scenario JSON path or ID to resolve network geometry.")
    p_r3d.add_argument("--net-xml", type=Path, default=None, help="Explicit SUMO .net.xml path.")
    p_r3d.add_argument("--poly-xml", type=Path, default=None, help="Optional SUMO/OSM .poly.xml buildings path.")
    p_r3d.add_argument("--title", default=None, help="Custom HTML viewer header title.")

    # 9. demo
    p_demo = sub.add_parser(
        "demo",
        help="Run an end-to-end single or split-screen SUMO episode and generate interactive 3D HTML.",
    )
    p_demo.add_argument(
        "--scenario",
        default="grid3x3_nominal",
        help="Scenario JSON path or canonical scenario ID.",
    )
    p_demo.add_argument(
        "--policy",
        default=None,
        choices=tuple(sorted(POLICY_NAMES)),
        help="Primary policy (defaults to neural_dbn if --checkpoint given, else online_bic_dbn).",
    )
    p_demo.add_argument(
        "--compare-policy",
        default=None,
        choices=tuple(sorted(POLICY_NAMES)),
        help="Optional baseline policy to run on the same seed for synchronized split-screen comparison.",
    )
    p_demo.add_argument("--seed", type=int, default=3000, help="Deterministic episode seed.")
    p_demo.add_argument("--checkpoint", type=Path, default=None, help="Optional neural DBN checkpoint.")
    p_demo.add_argument("--backend", choices=("libsumo", "traci"), default=None, help="SUMO backend override.")
    p_demo.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Optional cohort duration override in seconds for rapid interactive demos.",
    )
    p_demo.add_argument(
        "--output-html",
        type=Path,
        default=Path("outputs/demo/city_dbn_3d_demo.html"),
        help="Output self-contained 3D HTML path.",
    )
    p_demo.add_argument("--allow-ci-micro", action="store_true", help="Allow ci_micro preset (unit tests only).")

    return parser


def _cmd_build_nets(args: argparse.Namespace) -> int:
    from .env.network_builder import build_all_networks_and_scenarios

    built = build_all_networks_and_scenarios(
        repo_root=args.repo_root,
        allow_osm_download=not bool(args.no_osm_download),
        write_scenarios=not bool(args.no_write_scenarios),
    )
    _LOG.info("Built %d network and scenario artifacts under %s", len(built), args.repo_root)
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    raw_scen = args.scenario if isinstance(args.scenario, (list, tuple)) else [args.scenario]
    scenarios = [_resolve_scenario_arg(s, backend_override=args.backend) for s in raw_scen]
    if args.seeds:
        seeds: Sequence[int] = tuple(args.seeds)
    elif args.split == "train":
        seeds = TRAIN_SEEDS
    elif args.split == "val":
        seeds = VAL_SEEDS
    else:
        seeds = TEST_SEEDS

    manifest = collect_rollout_dataset(
        scenarios=scenarios,
        seeds=seeds,
        split=args.split,
        output_dir=args.output_dir,
        dbn_cfg=DBNConfig(),
        max_workers=args.workers,
        behavior_policy_override=args.behavior_policy,
        max_windows_per_episode=args.max_windows_per_episode,
        window_len_steps=args.window_len_steps,
    )
    _LOG.info(
        "Collected %d %s rollout shards into %s",
        len(manifest.get("shards", [])),
        args.split,
        args.output_dir,
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    train_cfg = TrainConfig(
        model_preset=args.preset,
        device=args.device,
        precision=args.precision,
        compile_model=bool(args.compile),
        distributed=bool(args.distributed),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        seed=args.seed,
    )
    dbn_cfg = DBNConfig(model_preset=args.preset)
    report = train_neural_dbn(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_checkpoint_path=args.output_checkpoint,
        train_cfg=train_cfg,
        dbn_cfg=dbn_cfg,
        report_path=args.report_path,
    )
    _LOG.info(
        "Trained %s checkpoint at %s (best_val_loss=%.4f, sha256=%s)",
        report["model_preset"],
        report["checkpoint_path"],
        report["best_val_loss"],
        report["checkpoint_sha256"][:12],
    )
    return 0


def _cmd_eval_predictive(args: argparse.Namespace) -> int:
    dataset_dirs: dict[str, Path] = {}
    if args.eval_id_dir is not None:
        dataset_dirs["eval_id"] = args.eval_id_dir
    if args.eval_ood_dir is not None:
        dataset_dirs["eval_ood"] = args.eval_ood_dir
    if args.dataset_dir is not None:
        dataset_dirs["held_out"] = args.dataset_dir
    if args.named_split:
        for pair in args.named_split:
            if "=" not in pair:
                raise ConfigValidationError(
                    f"Invalid --named-split entry {pair!r}; expected NAME=DIR."
                )
            label, raw_dir = pair.split("=", 1)
            label = label.strip()
            if not label or not raw_dir.strip():
                raise ConfigValidationError(
                    f"Invalid --named-split entry {pair!r}; expected non-empty NAME=DIR."
                )
            dataset_dirs[label] = Path(raw_dir.strip())
    if not dataset_dirs:
        raise ConfigValidationError(
            "eval-predictive requires at least one of --eval-id-dir, --eval-ood-dir, --dataset-dir, or --named-split."
        )

    dbn_cfg = DBNConfig(inference_device=args.device)
    report = evaluate_heldout_predictive(
        checkpoint_path=args.checkpoint,
        dataset_dirs=dataset_dirs,
        output_path=args.output,
        dbn_cfg=dbn_cfg,
        allow_ci_micro=bool(args.allow_ci_micro),
    )
    if args.plot_output is not None:
        from .viz import plot_calibration_and_roc_pr

        plot_calibration_and_roc_pr(report, args.plot_output)
    _LOG.info("Saved held-out predictive report to %s", args.output)
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    scenario = _resolve_scenario_arg(args.scenario, backend_override=args.backend)
    preset = "ci_micro" if args.allow_ci_micro else "sota_local"
    if args.checkpoint is not None:
        from .dbn.updater import load_dbn_checkpoint

        ckpt_file = (
            args.checkpoint / "neural_dbn.pt"
            if args.checkpoint.is_dir() and (args.checkpoint / "neural_dbn.pt").is_file()
            else args.checkpoint
        )
        bundle = load_dbn_checkpoint(ckpt_file, require_calibrated=True, device="cpu")
        preset = bundle.model.preset_name

    exp_cfg = ExperimentConfig(
        scenario=scenario,
        dbn=DBNConfig(model_preset=preset),  # type: ignore[arg-type]
        control=ControlConfig(),
    )
    need_trace = args.trace_out is not None or args.html_out is not None
    metrics, trace_data = run_single_episode(
        exp_cfg=exp_cfg,
        policy_name=args.policy,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        allow_ci_micro=bool(args.allow_ci_micro),
        record_trace=need_trace,
        trace_output_path=args.trace_out,
    )
    if args.output_metrics is not None:
        _atomic_write_json(args.output_metrics, asdict(metrics))

    if args.html_out is not None and trace_data is not None:
        from .env import load_or_build_topology
        from .viz import export_scene_3d, write_city_3d_viewer

        topology = load_or_build_topology(scenario)
        scene = export_scene_3d(topology)
        write_city_3d_viewer(scene, trace_data, args.html_out)

    _LOG.info(
        "Simulated %s | policy=%s seed=%d -> mean_total_delay=%.2fs throughput=%.1f veh/h",
        scenario.scenario_id,
        args.policy,
        args.seed,
        metrics.cohort_mean_total_delay_s,
        metrics.throughput_veh_per_hr,
    )
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    if args.scenarios:
        scenarios = [_resolve_scenario_arg(s, backend_override=args.backend) for s in args.scenarios]
    else:
        from .env.network_builder import default_scenario_configs

        scenarios = [
            _resolve_scenario_arg(sid, backend_override=args.backend)
            for sid in default_scenario_configs()
        ]

    seeds = tuple(args.seeds) if args.seeds else TEST_SEEDS
    preset = "sota_local"
    if args.checkpoint is not None:
        from .dbn.updater import load_dbn_checkpoint

        ckpt_file = (
            args.checkpoint / "neural_dbn.pt"
            if args.checkpoint.is_dir() and (args.checkpoint / "neural_dbn.pt").is_file()
            else args.checkpoint
        )
        bundle = load_dbn_checkpoint(ckpt_file, require_calibrated=True, device="cpu")
        preset = bundle.model.preset_name

    summary = evaluate_experiment_matrix(
        scenarios=scenarios,
        policies=args.policies,
        seeds=seeds,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        dbn_cfg=DBNConfig(model_preset=preset),  # type: ignore[arg-type]
        control_cfg=ControlConfig(),
        max_workers=args.workers,
        bootstrap_resamples=args.bootstrap_resamples,
        require_test_seeds=not bool(args.allow_non_test_seeds),
        allow_ci_micro=bool(args.allow_ci_micro),
    )
    if args.plots_dir is not None:
        from .viz import render_publication_plots

        render_publication_plots(summary, output_dir=args.plots_dir)
    _LOG.info("Completed evaluation matrix (%d episodes) -> %s", summary["num_episodes"], args.output_dir)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    report = run_scaling_benchmark(
        output_path=args.output,
        swarm_sizes=args.sizes,
        methods=args.methods,
        checkpoint_path=args.checkpoint,
        model_preset=args.preset,
        warmup_repeats=args.warmup,
        repeats=args.repeats,
        allow_ci_micro=bool(args.allow_ci_micro),
        seed=args.seed,
    )
    if args.plot_output is not None:
        from .viz import plot_latency_scaling

        plot_latency_scaling(report, args.plot_output)
    _LOG.info("Saved CPU scaling benchmark across sizes %s to %s", report["swarm_sizes"], args.output)
    return 0


def _cmd_render_3d(args: argparse.Namespace) -> int:
    from .env import load_or_build_topology
    from .viz import (
        export_scene_3d,
        export_scene_3d_from_files,
        load_trace_json,
        write_city_3d_viewer,
    )

    primary_trace = load_trace_json(args.trace)
    comparison_trace = load_trace_json(args.compare_trace) if args.compare_trace is not None else None

    if args.scene_json is not None:
        scene_data = json.loads(args.scene_json.read_text(encoding="utf-8"))
    elif args.scenario is not None:
        scenario = _resolve_scenario_arg(args.scenario)
        topology = load_or_build_topology(scenario)
        scene_data = export_scene_3d(topology)
    elif args.net_xml is not None:
        scene_data = export_scene_3d_from_files(
            net_xml_path=args.net_xml,
            poly_xml_path=args.poly_xml,
            network_id=str(primary_trace.get("network_id", args.net_xml.stem)),
        )
    else:
        scen_id = primary_trace.get("scenario_id")
        if not isinstance(scen_id, str) or not scen_id:
            raise ConfigValidationError(
                "render-3d requires one of --scene-json, --scenario, or --net-xml when trace lacks scenario_id."
            )
        scenario = _resolve_scenario_arg(scen_id)
        topology = load_or_build_topology(scenario)
        scene_data = export_scene_3d(topology)

    out_html = write_city_3d_viewer(
        scene_data=scene_data,
        primary_trace=primary_trace,
        output_html=args.output_html,
        comparison_trace=comparison_trace,
        title=args.title,
    )
    _LOG.info("Wrote self-contained 3D HTML viewer to %s", out_html)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .env import load_or_build_topology
    from .viz import export_scene_3d, write_city_3d_viewer

    scenario = _resolve_scenario_arg(args.scenario, backend_override=args.backend)
    if args.duration_s is not None:
        scenario = replace(
            scenario,
            warmup_duration_s=min(scenario.warmup_duration_s, 20.0),
            cohort_duration_s=float(args.duration_s),
            drain_duration_s=min(scenario.drain_duration_s, 45.0),
        )

    policy = args.policy or ("neural_dbn" if args.checkpoint is not None else "online_bic_dbn")
    preset = "sota_local"
    if args.checkpoint is not None:
        from .dbn.updater import load_dbn_checkpoint

        ckpt_file = (
            args.checkpoint / "neural_dbn.pt"
            if args.checkpoint.is_dir() and (args.checkpoint / "neural_dbn.pt").is_file()
            else args.checkpoint
        )
        bundle = load_dbn_checkpoint(ckpt_file, require_calibrated=True, device="cpu")
        preset = bundle.model.preset_name

    exp_cfg = ExperimentConfig(
        scenario=scenario,
        dbn=DBNConfig(model_preset=preset),  # type: ignore[arg-type]
        control=ControlConfig(),
    )

    _, primary_trace = run_single_episode(
        exp_cfg=exp_cfg,
        policy_name=policy,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        allow_ci_micro=bool(args.allow_ci_micro),
        record_trace=True,
    )
    if primary_trace is None:
        raise ConfigValidationError("Demo failed to capture primary trajectory trace.")

    comparison_trace = None
    if args.compare_policy is not None:
        _, comparison_trace = run_single_episode(
            exp_cfg=exp_cfg,
            policy_name=args.compare_policy,
            seed=args.seed,
            checkpoint_path=args.checkpoint if args.compare_policy in NEURAL_POLICY_NAMES else None,
            allow_ci_micro=bool(args.allow_ci_micro),
            record_trace=True,
        )

    topology = load_or_build_topology(scenario)
    scene_data = export_scene_3d(topology)
    out_html = write_city_3d_viewer(
        scene_data=scene_data,
        primary_trace=primary_trace,
        output_html=args.output_html,
        comparison_trace=comparison_trace,
    )
    _LOG.info("Generated interactive 3D demo viewer at %s", out_html)
    return 0


@fail_loud_entrypoint(_resolve_error_report_dir)
def main(argv: Sequence[str] | None = None) -> int:
    """Execute the ``swarm`` CLI with fail-loud crash diagnostics."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(level=args.log_level)
    update_crash_context(command=str(args.command))

    dispatch = {
        "build-nets": _cmd_build_nets,
        "collect": _cmd_collect,
        "train": _cmd_train,
        "eval-predictive": _cmd_eval_predictive,
        "simulate": _cmd_simulate,
        "evaluate": _cmd_evaluate,
        "benchmark": _cmd_benchmark,
        "render-3d": _cmd_render_3d,
        "demo": _cmd_demo,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        raise ConfigValidationError(f"Unknown CLI subcommand: {args.command!r}")
    return int(handler(args))


__all__ = ["build_parser", "main"]

"""Run `python -m swarm --help` for the complete workflow."""

import argparse
import json
from pathlib import Path
from .config import Config
from .experiment import run_episode, evaluate, write_json, benchmark_structure
from .neural import EdgeMLP
from .policies import POLICIES


def parser():
    root = argparse.ArgumentParser(description="Neurally updated Bayesian networks for vehicle coordination")
    commands = root.add_subparsers(dest="command", required=True)
    for name, help_text in (("train", "Train the edge MLP on synthetic traffic scenes"),
                            ("simulate", "Run one policy and generate a visual replay"),
                            ("evaluate", "Compare policies on identical seeds"),
                            ("benchmark", "Measure structure-update scaling"),
                            ("demo", "Train, compare, benchmark, and create a replay")):
        p = commands.add_parser(name, help=help_text)
        p.add_argument("--config", type=Path)
        p.add_argument("--model", type=Path, default=Path("models/edge_mlp.npz"))
        p.add_argument("--output", type=Path, default=Path("outputs")/name)
        if name in ("train", "demo"):
            p.add_argument("--scenes", type=int, default=700)
            p.add_argument("--epochs", type=int, default=60)
        if name in ("train", "simulate", "benchmark", "demo"):
            p.add_argument("--seed", type=int, default=7)
        if name in ("simulate", "evaluate", "demo"):
            p.add_argument("--duration", type=float)
            p.add_argument("--arrival-rate", type=float)
        if name == "simulate":
            p.add_argument("--policy", choices=POLICIES, default="neural")
        if name in ("evaluate", "demo"):
            p.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
            p.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
            p.add_argument("--scenarios", nargs="+", choices=("nominal", "busy", "noisy", "no-communication"), default=["nominal"])
        if name == "benchmark":
            p.add_argument("--sizes", nargs="+", type=int, default=[8, 16, 32, 64])
            p.add_argument("--repeats", type=int, default=8)
    return root


def load_model(path):
    if not path.exists():
        raise ValueError(f"Model not found: {path}. Run `python -m swarm train` first, or use `python -m swarm demo`.")
    return EdgeMLP.load(path)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        cfg = Config.load(args.config, duration=getattr(args, "duration", None),
                          arrival_rate=getattr(args, "arrival_rate", None))
        if getattr(args, "seed", 0) < 0:
            raise ValueError("seed must be nonnegative")
        args.output.mkdir(parents=True, exist_ok=True)
        if args.command in ("train", "demo"):
            from .training import train_model
            from .render import training_plot
            training_output = args.output/"training" if args.command == "demo" else args.output
            report = train_model(cfg, args.model, training_output, args.scenes, args.epochs, args.seed)
            training_plot(report, training_output/"test_predictions.npz", training_output/"training.png")
            print("Held-out edge prediction:", json.dumps(report["test"], indent=2), flush=True)
            if args.command == "train":
                print(f"Saved model: {args.model.resolve()}")
                return
        if args.command in ("simulate", "demo"):
            policy = args.policy if args.command == "simulate" else "neural"
            model = load_model(args.model) if policy == "neural" else None
            metrics, trace = run_episode(cfg, policy, args.seed, model, record=True)
            from .render import write_replay, trajectory_preview
            write_json(args.output/"metrics.json", metrics)
            write_json(args.output/"trace.json", trace)
            write_replay(trace, args.output/"replay.html")
            trajectory_preview(trace, args.output/"preview.png")
            print(json.dumps(metrics, indent=2), flush=True)
            print(f"Replay: {(args.output/'replay.html').resolve()}", flush=True)
        if args.command in ("evaluate", "demo"):
            model = load_model(args.model) if "neural" in args.policies else None
            output = args.output/"evaluation" if args.command == "demo" else args.output
            evaluate(cfg, args.policies, args.seeds, model, output, args.scenarios,
                     progress=lambda message: print(message, flush=True))
            print(f"Comparison: {(output/'comparison.png').resolve()}", flush=True)
        if args.command in ("benchmark", "demo"):
            model = load_model(args.model)
            report = benchmark_structure(cfg, model, getattr(args, "sizes", (8, 16, 32, 64)),
                                         getattr(args, "repeats", 8), args.seed)
            from .render import scaling_plot
            output = args.output/"benchmark" if args.command == "demo" else args.output
            output.mkdir(parents=True, exist_ok=True)
            write_json(output/"scaling.json", report)
            scaling_plot(report, output/"scaling.png")
            print(json.dumps(report, indent=2), flush=True)
    except (ValueError, FileNotFoundError) as error:
        parser().error(str(error))

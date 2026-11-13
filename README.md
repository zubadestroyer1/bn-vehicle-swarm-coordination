# Neurally Updated Bayesian Networks for Vehicle Swarm Coordination

A runnable first version of the CS 238 proposal by Kieran, Aadi, and Arthur. Vehicles approach a four-way intersection, infer uncertain interactions, choose crossing priorities and speeds, and request a limited number of messages. A trained neural network updates the dependency graph at every simulation step.

The project includes a simulator, probabilistic belief tracking, a NumPy neural model, a two-slice Bayesian network, five controllers, reproducible experiments, and an interactive visual replay. It needs no SUMO server, GPU, online service, or external dataset.

## Run it

Use Python 3.10 or newer; the checked development environment uses Python 3.12. The local `.venv` is already installed. For a fresh checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Train, run a replay, compare all five policies, and benchmark graph-update scaling:

```bash
python -m swarm demo --duration 120
```

Open `outputs/demo/replay.html` in a browser. It is a self-contained file with play/pause, time scrubbing, playback speeds, vehicle colors, crossing reservations, and dependency overlays. It makes no network requests. On macOS:

```bash
open outputs/demo/replay.html
```

The already generated local example is in `outputs/demo/`; the checked multi-scenario experiment is in `outputs/evaluation/`. See [the recorded results](docs/RESULTS.md) for the current numbers and their limitations.

## Individual commands

```bash
# Train with independent train, validation, and test scenes.
python -m swarm train --scenes 700 --epochs 60 --seed 7

# One episode, plus JSON trace, metrics, static preview, and interactive replay.
python -m swarm simulate --policy neural --duration 120 --seed 11

# Rule-based policies work without a trained model.
python -m swarm simulate --policy fcfs --duration 120

# Paired comparisons: 5 policies x 3 seeds x 4 conditions = 60 episodes.
python -m swarm evaluate --duration 120 --seeds 11 12 13 \
  --scenarios nominal busy noisy no-communication

# Increase the number of seeds for research conclusions.
python -m swarm evaluate --duration 300 --seeds 20 21 22 23 24 25 26 27 28 29

# Isolate structure-update cost as vehicle count grows.
python -m swarm benchmark --sizes 8 16 32 64 --repeats 8

# Change the simulated environment.
python -m swarm simulate --config configs/stress.json --policy neural

# Behavioral and mathematical regression checks.
python -m unittest discover -s tests -v
```

Every command accepts `--output`; model-using commands accept `--model`. `swarm` is also installed as a command-line entry point. Use `python -m swarm --help` or a subcommand's `--help` for all options. Missing checkpoints fail with a training instruction; an untrained model is never silently substituted by the CLI.

## What is implemented

| Proposal component | First implementation |
| --- | --- |
| Dynamic driving society | Poisson arrivals, four incoming lanes, left/straight/right trajectories, bounded speed and acceleration, upstream queues, departures |
| Partial observability | Gaussian position/speed measurements, sensor range limits, missed detections, noisy maneuver labels, delayed and dropped packets |
| Belief state | Per-vehicle Gaussian kinematic filter and categorical maneuver belief |
| Neural dependency updates | Shared pairwise MLP, supervised by synthetic latent route/occupancy interactions, evaluated on independent scenes |
| Changing Bayesian network | Ordered DAG per step; Bernoulli edge uncertainty, noisy-OR CPDs, temporal self-dependencies, particle-based approximate inference |
| Sequential decisions | Receding-horizon enumeration of up to 24 crossing-priority orders, speed targets, adaptive cruise control, budgeted message selection |
| Rewards and constraints | Throughput reward; delay, stop, collision and communication penalties; probabilistic route-conflict gate and a shared physical safety shield |
| Evaluation | Paired demand/noise seeds, delay including unfinished demand, throughput, stops, collisions, safety interventions, communication, graph-update latency |

The observation-based policies do not receive `Vehicle` objects, hidden maneuvers, realized driver noise, or exact positions. Entry and exit detectors provide vehicle identity and approach. Ground truth is available only to simulation dynamics, evaluation, and the shared safety shield.

## Controllers

| CLI name | Behavior |
| --- | --- |
| `signal` | Four fixed approach phases with clearance intervals; one incoming approach gets green at a time |
| `fcfs` | Reservation scheduling by requested arrival time, with compatible routes allowed concurrently |
| `independent` | Same belief-based scheduling machinery as neural, with all cross-vehicle edges removed |
| `greedy` | Relearns an ordered, bounded-parent network each step using binary BIC on a sliding observation window |
| `neural` | Trained MLP supplies probabilistic edges; graph inference informs occupancy durations and message priorities |

`independent` is the primary graph ablation. `greedy` compares an online structure-learning approach. `signal` is intentionally a simple conservative reference, not an optimized traffic-engineering baseline. Policies share the same sensor model, physical dynamics, communication budget, and safety layer, but choose different messages and therefore receive different realized packet sets.

## Reading the output

- `training.json`, `training.png`: learning curve, held-out accuracy, AUC, log loss, Brier score and calibration. `test_predictions.npz` contains labels and probabilities.
- `metrics.json`, `trace.json`, `replay.html`, `preview.png`: one simulated episode and visualizations.
- `episodes.csv`, `episodes.json`: every episode, including collision and intervention counts.
- `summary.json`, `comparison.png`: per-scenario means, variation across seeds, and paired neural-minus-baseline differences.
- `scaling.json`, `scaling.png`: warmed structure-update timings for neural and greedy methods.

Models and generated outputs are ignored by Git. The small [results snapshot](examples/validation/) is included as a reviewable record. Simulation outcomes are seeded; timings depend on hardware and process load.

## Scope and assumptions

This is a research prototype of the proposed loop. It is a centralized belief-based controller for a POMDP-like simulator, not a solved POMDP or a fully decentralized swarm. The neural model learns synthetic interaction labels, not causal structure from real driving data. Network CPDs and the delay-to-control mapping are designed heuristics. Calibration is measured against synthetic labels, not real collision probabilities.

The safety shield uses true state, conservative whole-route reservations, and position/velocity projection. It can apply abrupt corrections beyond physical braking limits. Zero recorded collisions therefore do **not** establish learned-controller safety. Inspect intervention counts and unfinished demand alongside throughput and delay. The route-conflict threshold is a heuristic risk gate, not a formal chance-constrained safety guarantee.

Neural inference scores every candidate pair, so graph construction remains quadratic in the number of vehicles. The current controller compares at most four lane heads. Timing gains over this bounded greedy learner do not prove scalability relative to every structure-learning method.

See [METHODS.md](docs/METHODS.md) for the model equations, data construction, metric definitions, and next research steps.

## Code map

```text
swarm/
  geometry.py       Fixed routes, arc-length positions, conflict geometry
  simulator.py      Demand, dynamics, sensors, packets, shield, metrics
  belief.py         Gaussian/categorical belief tracking
  features.py       Shared observable pair feature extraction
  neural.py         NumPy MLP, Adam training, metrics, checkpoint I/O
  training.py       Independent synthetic scenes and interaction labels
  bayes.py          Neural/BIC graph updates and temporal inference
  policies.py       Five controllers and communication selection
  experiment.py     Episodes, paired comparisons, scaling benchmark
  render.py         Scientific plots and self-contained replay export
  replay.html       Offline canvas replay template
  cli.py            train / simulate / evaluate / benchmark / demo
configs/            Nominal and stress settings
tests/              Regression checks
docs/               Method and measured results
```

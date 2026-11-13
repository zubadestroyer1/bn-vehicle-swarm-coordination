"""Simulation-supervised edge learning, explicitly not causal discovery.

Labels identify pairs whose latent free-flight conflict-zone occupancy intervals
overlap (or whose same-lane headway is short). Whole independent scenes are split
before pair expansion, so a vehicle cannot leak across train/validation/test.
"""

from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .belief import Track
from .features import FEATURE_NAMES, pair_features
from .geometry import MANEUVER_PRIOR, arrival_time, conflict_tensor, route
from .neural import EdgeMLP, binary_metrics


def sample_scene(rng, config, n=None):
    n = int(rng.integers(4, 13)) if n is None else n
    lane = rng.integers(0, 4, n)
    maneuver = rng.choice(3, n, p=MANEUVER_PRIOR)
    progress = rng.uniform(0, config.approach_length+12, n)
    speed = rng.uniform(0, config.speed_limit, n)
    # Include queued, flowing, old-message, and ambiguous-intent examples.
    speed[rng.random(n) < .18] *= .1
    age = rng.uniform(0, 2.0, n)
    sigma = config.position_noise + .8*age
    observed_s = progress + rng.normal(size=n)*sigma
    observed_v = np.clip(speed+rng.normal(size=n)*(config.velocity_noise+.4*age), 0, config.speed_limit)
    tracks = []
    for i in range(n):
        correct = rng.random() < .88
        observed_intent = int(maneuver[i]) if correct else int(rng.choice([m for m in range(3) if m != maneuver[i]]))
        confidence = rng.uniform(.4, .995)
        intent = np.full(3, (1-confidence)/2)
        intent[observed_intent] = confidence
        tracks.append(Track(i, int(lane[i]), 0, np.array([observed_s[i], observed_v[i]]),
                            np.diag([sigma[i]**2, (config.velocity_noise+.4*age[i])**2]),
                            intent=intent, age=float(age[i])))
    tracks.sort(key=lambda t: (t.eta(config), t.vehicle_id))
    truth = {i: (lane[i], maneuver[i], progress[i], speed[i]) for i in range(n)}
    return tracks, truth


def make_dataset(config, scenes, seed):
    if scenes < 1:
        raise ValueError("scenes must be positive")
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    tensor = conflict_tensor(config.vehicle_radius)
    for _ in range(scenes):
        tracks, truth = sample_scene(rng, config)
        x, parents, children = pair_features(tracks, config)
        labels = []
        for a, b in zip(parents, children):
            la, ma, sa, va = truth[tracks[a].vehicle_id]
            lb, mb, sb, vb = truth[tracks[b].vehicle_id]
            ta = float(arrival_time(config.approach_length-sa, va, config.acceleration, config.speed_limit))
            tb = float(arrival_time(config.approach_length-sb, vb, config.acceleration, config.speed_limit))
            da = max(0, config.approach_length+route(int(la), int(ma)).length-sa)/max(va, 4.0)
            db = max(0, config.approach_length+route(int(lb), int(mb)).length-sb)/max(vb, 4.0)
            # Outside the intersection, use duration from its entry, not approach distance.
            da = min(da, route(int(la), int(ma)).length/max(va, 4.0)) + .6
            db = min(db, route(int(lb), int(mb)).length/max(vb, 4.0)) + .6
            overlap = ta < tb+db and tb < ta+da
            following = la == lb and abs(sa-sb) < config.following_gap+1.5*max(va, vb)
            labels.append(float((tensor[la, ma, lb, mb] and overlap) or following))
        xs.append(x)
        ys.append(labels)
    return np.vstack(xs), np.concatenate(ys)


def train_model(config, model_path, output_dir, scenes=700, epochs=60, seed=7):
    training = make_dataset(config, scenes, seed)
    validation = make_dataset(config, max(50, scenes//5), seed+10000)
    test = make_dataset(config, max(80, scenes//4), seed+20000)
    model = EdgeMLP(seed=seed)
    history = model.fit(*training, validation=validation, epochs=epochs, seed=seed)
    report = {
        "seed": seed, "config": asdict(config), "features": list(FEATURE_NAMES),
        "label": "Latent simulated route conflict and overlapping free-flight occupancy, or short same-lane headway",
        "split": "Independent scenes; test set unused for training or early stopping",
        "train_examples": len(training[1]), "epochs_run": len(history),
        "validation": binary_metrics(validation[1], model.predict(validation[0])),
        "test": binary_metrics(test[1], model.predict(test[0])),
        "constant_baseline_test": binary_metrics(test[1], np.full(len(test[1]), training[1].mean())),
        "history": history,
    }
    model.metadata = {"seed": seed, "training_scenes": scenes, "training_config": asdict(config),
                      "label": report["label"], "test": report["test"]}
    model.save(model_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output/"training.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(output/"test_predictions.npz", target=test[1], probability=model.predict(test[0]))
    return report

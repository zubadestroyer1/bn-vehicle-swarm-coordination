"""A small NumPy MLP: observable pair features -> Bernoulli edge probability."""

import json
from pathlib import Path
import numpy as np
from .features import FEATURE_NAMES, FEATURE_VERSION


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def binary_metrics(target, probability):
    y = np.asarray(target, dtype=float)
    p = np.clip(probability, 1e-7, 1-1e-7)
    if not len(y):
        raise ValueError("Metrics require at least one example")
    # Mann-Whitney AUC, averaging tied ranks.
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p), dtype=float)
    _, starts, counts = np.unique(p[order], return_index=True, return_counts=True)
    ranks[order] = np.repeat(starts + (counts + 1)/2, counts)
    positives = y.sum()
    negatives = len(y)-positives
    auc = ((ranks[y == 1].sum()-positives*(positives+1)/2)/(positives*negatives)
           if positives and negatives else None)
    ece = 0.0
    bins = np.minimum((p*10).astype(int), 9)
    for bin_id in range(10):
        mask = bins == bin_id
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean()-p[mask].mean())
    return {"examples": len(y), "positive_fraction": float(y.mean()),
            "accuracy": float(np.mean((p >= .5) == y)),
            "brier": float(np.mean((p-y)**2)),
            "log_loss": float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
            "roc_auc": None if auc is None else float(auc), "ece": float(ece)}


class EdgeMLP:
    def __init__(self, seed=0, hidden=32):
        rng = np.random.default_rng(seed)
        d = len(FEATURE_NAMES)
        self.parameters = {
            "w1": rng.normal(0, np.sqrt(2/d), (d, hidden)), "b1": np.zeros(hidden),
            "w2": rng.normal(0, np.sqrt(2/hidden), (hidden, 1)), "b2": np.zeros(1),
        }
        self.mean = np.zeros(d)
        self.scale = np.ones(d)
        self.metadata = {}

    def predict(self, x):
        if not len(x):
            return np.empty(0)
        z = (x-self.mean)/self.scale
        h = np.tanh(z @ self.parameters["w1"] + self.parameters["b1"])
        return sigmoid(h @ self.parameters["w2"] + self.parameters["b2"]).ravel()

    def fit(self, x, y, validation, epochs=50, batch_size=256, learning_rate=.003, seed=0):
        if epochs < 1 or len(x) < 1:
            raise ValueError("Training requires positive epochs and nonempty data")
        self.mean, self.scale = x.mean(axis=0), np.maximum(x.std(axis=0), 0.05)
        normalized = (x-self.mean)/self.scale
        rng = np.random.default_rng(seed)
        first = {k: np.zeros_like(v) for k, v in self.parameters.items()}
        second = {k: np.zeros_like(v) for k, v in self.parameters.items()}
        history, best_loss, best, step, stale = [], np.inf, None, 0, 0
        for epoch in range(epochs):
            for indices in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x)/batch_size)))):
                z, target = normalized[indices], y[indices, None]
                p = self.parameters
                h = np.tanh(z @ p["w1"]+p["b1"])
                output = sigmoid(h @ p["w2"]+p["b2"])
                error = (output-target)/len(indices)
                dh = (error @ p["w2"].T)*(1-h*h)
                gradients = {"w2": h.T @ error + 1e-5*p["w2"], "b2": error.sum(axis=0),
                             "w1": z.T @ dh + 1e-5*p["w1"], "b1": dh.sum(axis=0)}
                step += 1
                for key, gradient in gradients.items():
                    first[key] = .9*first[key]+.1*gradient
                    second[key] = .999*second[key]+.001*gradient*gradient
                    p[key] -= learning_rate*(first[key]/(1-.9**step))/(np.sqrt(second[key]/(1-.999**step))+1e-8)
            metrics = binary_metrics(validation[1], self.predict(validation[0]))
            history.append({"epoch": epoch+1, **metrics})
            if metrics["log_loss"] < best_loss-1e-5:
                best_loss, best, stale = metrics["log_loss"], {k: v.copy() for k, v in self.parameters.items()}, 0
            else:
                stale += 1
            if stale >= 10:
                break
        self.parameters = best
        return history

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            np.savez_compressed(f, **self.parameters, mean=self.mean, scale=self.scale,
                                feature_version=np.array(FEATURE_VERSION),
                                metadata=np.array(json.dumps(self.metadata)))

    @classmethod
    def load(cls, path):
        model = cls()
        with np.load(path, allow_pickle=False) as data:
            if int(data["feature_version"]) != FEATURE_VERSION:
                raise ValueError("Incompatible model feature version; retrain the model")
            model.parameters = {k: data[k].copy() for k in model.parameters}
            model.mean, model.scale = data["mean"].copy(), data["scale"].copy()
            model.metadata = json.loads(str(data["metadata"]))
        return model

"""A changing, ordered two-slice Bayesian network over binary delay variables.

Edges are uncertain Bernoulli variables. Marginalizing an edge with probability
w in a noisy-OR CPD gives (1-w*strength*parent_state). Temporal self-edges carry
delay information between slices. Ancestral samples preserve within-slice
correlations; local soft observation evidence is incorporated by importance
weighting. Previous-slice marginals are factorized (an explicit approximation).
"""

from collections import deque
from dataclasses import dataclass
import numpy as np
from .features import pair_features


@dataclass
class NetworkState:
    vehicle_ids: list
    adjacency: np.ndarray  # parent row -> child column, strictly upper triangular
    delay_probability: np.ndarray
    posterior: np.ndarray
    effective_sample_size: float


def noisy_or_probability(leak, previous, parents, weights, persistence=.65, strength=.75):
    return 1-(1-leak)*(1-persistence*previous)*np.prod(1-strength*parents*weights, axis=-1)


def infer_delay(tracks, adjacency, previous, rng, particles=128):
    n = len(tracks)
    if adjacency.shape != (n, n) or np.any(np.tril(adjacency) != 0):
        raise ValueError("The contemporaneous graph must be a strictly upper-triangular DAG")
    if np.any((adjacency < 0) | (adjacency > 1)):
        raise ValueError("Edge probabilities must lie in [0, 1]")
    if not n:
        return np.empty(0), np.empty(0), float(particles)
    old = np.array([previous.get(t.vehicle_id, .1) for t in tracks])
    old_samples = rng.random((particles, n)) < old
    samples = np.zeros((particles, n), dtype=float)
    # First ancestral pass produces a predictive marginal, before current evidence.
    leak = np.array([min(.2, .025+.015*t.age) for t in tracks])
    for j in range(n):
        p = noisy_or_probability(leak[j], old_samples[:, j], samples[:, :j], adjacency[:j, j])
        samples[:, j] = rng.random(particles) < p
    predictive = samples.mean(axis=0)
    # A stopped observation supports delay but is deliberately not deterministic.
    slow = np.array([1/(1+np.exp(np.clip((t.speed-2.0)*1.5, -30, 30))) for t in tracks])
    confidence = np.array([.8*np.exp(-t.age/3) for t in tracks])
    likelihood_delayed = .5+confidence*(slow-.5)
    log_weights = (samples*np.log(likelihood_delayed+1e-9)
                   +(1-samples)*np.log(1-likelihood_delayed+1e-9)).sum(axis=1)
    weights = np.exp(log_weights-log_weights.max())
    weights /= weights.sum()
    posterior = weights @ samples
    ess = float(1/np.sum(weights**2))
    return predictive, posterior, ess


class NeuralStructure:
    def __init__(self, model, config):
        self.model, self.config = model, config

    def update(self, tracks):
        x, a, b = pair_features(tracks, self.config)
        adjacency = np.zeros((len(tracks), len(tracks)))
        adjacency[a, b] = self.model.predict(x)
        # Keep only the most probable incoming edges, bounding inference cost.
        for child in range(len(tracks)):
            incoming = np.flatnonzero(adjacency[:, child] >= .15)
            keep = incoming[np.argsort(adjacency[incoming, child])[-self.config.max_parents:]] if self.config.max_parents else []
            mask = np.ones(len(tracks), dtype=bool)
            mask[keep] = False
            adjacency[mask, child] = 0
        return adjacency


def bic_score(data, child, parents):
    """Binary local maximum-likelihood BIC, counting all 2^k free parameters."""
    relevant = [child, *parents]
    rows = data[np.isfinite(data[:, relevant]).all(axis=1)]
    if len(rows) < 2:
        return -np.inf
    code = np.zeros(len(rows), dtype=int)
    for bit, parent in enumerate(parents):
        code += rows[:, parent].astype(int) << bit
    totals = np.bincount(code, minlength=2**len(parents))
    positives = np.bincount(code, weights=rows[:, child], minlength=len(totals))
    p = np.divide(positives, totals, out=np.full_like(positives, .5), where=totals > 0)
    p = np.clip(p, 1e-9, 1-1e-9)
    ll = np.sum(positives*np.log(p)+(totals-positives)*np.log(1-p))
    return float(ll - .5*(2**len(parents))*np.log(len(rows)))


class GreedyStructure:
    """Relearn a bounded-parent ordered graph from a sliding observation history.

    For each child all candidate additions are scored on one common complete-case
    row set. This avoids comparing likelihoods with different sample counts when
    vehicles enter/leave. New nodes without history start with an empty parent set.
    """
    def __init__(self, config):
        self.config = config
        self.history = deque(maxlen=config.history_window)

    def update(self, tracks):
        self.history.append({t.vehicle_id: float(t.speed < 2) for t in tracks if t.age < 1.5})
        n = len(tracks)
        data = np.array([[row.get(t.vehicle_id, np.nan) for t in tracks] for row in self.history])
        adjacency = np.zeros((n, n))
        for child in range(n):
            # Restrict to candidates sharing sufficient observations with this node.
            candidates = [p for p in range(child)
                          if np.isfinite(data[:, [p, child]]).all(axis=1).sum() >= 8]
            if not candidates:
                continue
            common = data[np.isfinite(data[:, [child, *candidates]]).all(axis=1)]
            if len(common) < 8:
                continue
            parents = []
            score = bic_score(common, child, parents)
            while candidates and len(parents) < self.config.max_parents:
                scores = [(bic_score(common, child, parents+[p]), p) for p in candidates]
                best_score, parent = max(scores)
                if best_score <= score+1e-8:
                    break
                # Noisy-OR only encodes positive delay influence. Reject negative
                # association, while retaining BIC as the structure selection rule.
                group0 = common[common[:, parent] == 0, child]
                group1 = common[common[:, parent] == 1, child]
                if len(group0) and len(group1) and group1.mean() > group0.mean():
                    adjacency[parent, child] = np.clip(group1.mean()-group0.mean(), .15, .95)
                    parents.append(parent)
                    score = best_score
                candidates.remove(parent)
        return adjacency


class IndependentStructure:
    def update(self, tracks):
        return np.zeros((len(tracks), len(tracks)))


class DynamicBayesNet:
    def __init__(self, structure, config, seed):
        self.structure, self.config = structure, config
        self.previous = {}
        self.rng = np.random.default_rng(seed)

    def update(self, tracks):
        adjacency = self.structure.update(tracks)
        predictive, posterior, ess = infer_delay(tracks, adjacency, self.previous, self.rng,
                                                self.config.inference_particles)
        self.previous = {t.vehicle_id: float(p) for t, p in zip(tracks, posterior)}
        return NetworkState([t.vehicle_id for t in tracks], adjacency, predictive, posterior, ess)

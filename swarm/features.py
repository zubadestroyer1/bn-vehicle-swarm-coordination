"""The same observable pair features are used during training and execution."""

import numpy as np
from .geometry import conflict_tensor, arrival_time

FEATURE_NAMES = (
    "parent_distance", "child_distance", "parent_speed", "child_speed",
    "eta_difference", "absolute_eta_difference", "same_approach", "route_conflict",
    "parent_position_std", "child_position_std", "observation_age",
    "intent_entropy", "absolute_progress_gap",
)
FEATURE_VERSION = 1


def pair_features(tracks, config):
    n = len(tracks)
    parents, children = np.triu_indices(n, 1)
    if not len(parents):
        return np.empty((0, len(FEATURE_NAMES))), parents, children
    s = np.array([t.progress for t in tracks])
    v = np.array([t.speed for t in tracks])
    lane = np.array([t.lane for t in tracks])
    intent = np.array([t.intent for t in tracks])
    sigma = np.sqrt([max(0.0, t.covariance[0, 0]) for t in tracks])
    age = np.array([t.age for t in tracks])
    entropy = -(intent * np.log(np.maximum(intent, 1e-9))).sum(axis=1)
    eta = arrival_time(config.approach_length - s, v, config.acceleration, config.speed_limit)
    a, b = parents, children
    conflicts = conflict_tensor(config.vehicle_radius)[lane[a], :, lane[b], :]
    p_conflict = np.einsum('bi,bij,bj->b', intent[a], conflicts, intent[b])
    x = np.column_stack([
        (config.approach_length-s[a])/70, (config.approach_length-s[b])/70,
        v[a]/10, v[b]/10, (eta[b]-eta[a])/6, np.abs(eta[b]-eta[a])/6,
        lane[a] == lane[b], p_conflict, sigma[a]/5, sigma[b]/5,
        (age[a]+age[b])/5, (entropy[a]+entropy[b])/2.2, np.abs(s[a]-s[b])/70,
    ])
    return x, parents, children

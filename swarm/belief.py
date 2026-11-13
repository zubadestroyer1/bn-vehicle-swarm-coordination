"""Gaussian kinematic filtering and categorical intent updates.

Policies receive these tracks, never simulation Vehicle objects. Delayed packets
are extrapolated and their covariance inflated; this is an approximation to an
out-of-sequence Kalman smoother. Older-than-last-accepted packets are discarded.
"""

from dataclasses import dataclass, field
import numpy as np
from .geometry import MANEUVER_PRIOR, arrival_time


@dataclass
class Observation:
    vehicle_id: int
    measured_at: float
    progress: float
    speed: float
    position_variance: float
    velocity_variance: float
    maneuver: int
    intent_accuracy: float
    source: str = "sensor"


@dataclass
class Track:
    vehicle_id: int
    lane: int
    entered_at: float
    mean: np.ndarray
    covariance: np.ndarray
    intent: np.ndarray = field(default_factory=lambda: MANEUVER_PRIOR.copy())
    last_observed_at: float = -1.0
    age: float = 0.0
    waiting: float = 0.0

    @property
    def progress(self):
        return float(self.mean[0])

    @property
    def speed(self):
        return float(self.mean[1])

    def eta(self, config):
        return float(arrival_time(config.approach_length - self.progress, self.speed,
                                  config.acceleration, config.speed_limit))


class BeliefTracker:
    def __init__(self, config):
        self.config = config
        self.tracks = {}

    def register(self, vehicle_id, lane, requested_at):
        # Entry detector supplies identity and approach only, not true speed/intent.
        self.tracks[vehicle_id] = Track(vehicle_id, lane, requested_at,
                                       np.array([0.0, self.config.speed_limit * 0.85]),
                                       np.diag([2.0, 4.0]))

    def remove(self, vehicle_id):
        self.tracks.pop(vehicle_id, None)

    def predict(self, accelerations):
        dt = self.config.dt
        f = np.array([[1.0, dt], [0.0, 1.0]])
        b = np.array([0.5 * dt**2, dt])
        q = np.outer(b, b) * (0.8 + self.config.driver_noise)**2 + np.diag([0.02, 0.02])
        for track in self.tracks.values():
            track.mean = f @ track.mean + b * accelerations.get(track.vehicle_id, 0.0)
            track.mean[1] = np.clip(track.mean[1], 0, self.config.speed_limit)
            track.covariance = f @ track.covariance @ f.T + q
            track.age += dt
            track.waiting += dt * float(track.speed < 1.0)

    def update(self, observations, now):
        for obs in sorted(observations, key=lambda o: o.measured_at):
            track = self.tracks.get(obs.vehicle_id)
            if track is None or obs.measured_at < track.last_observed_at - 1e-9:
                continue
            lag = max(0.0, now - obs.measured_at)
            z = np.array([obs.progress + obs.speed * lag, obs.speed])
            f = np.array([[1.0, lag], [0.0, 1.0]])
            r = f @ np.diag([obs.position_variance, obs.velocity_variance]) @ f.T
            r += np.diag([lag**4 * 0.25, lag**2])
            gain = np.linalg.solve((track.covariance + r).T, track.covariance.T).T
            track.mean += gain @ (z - track.mean)
            residual = np.eye(2) - gain
            track.covariance = residual @ track.covariance @ residual.T + gain @ r @ gain.T
            track.mean[1] = np.clip(track.mean[1], 0, self.config.speed_limit)
            # Small forgetting avoids permanently locking onto an early wrong label.
            prior = 0.98 * track.intent + 0.02 * MANEUVER_PRIOR
            likelihood = np.full(3, (1 - obs.intent_accuracy) / 2)
            likelihood[obs.maneuver] = obs.intent_accuracy
            track.intent = prior * likelihood
            track.intent /= track.intent.sum()
            track.last_observed_at = obs.measured_at
            track.age = lag

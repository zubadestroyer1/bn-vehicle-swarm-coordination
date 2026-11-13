"""Twelve fixed routes through a right-hand-traffic, four-way intersection.

Each route has a scalar arc-length coordinate. The square conflict region is
[-9, 9]^2. Geometry is deterministic; the maneuver taken is latent to the planner.
"""

from dataclasses import dataclass
from functools import lru_cache
import numpy as np

LANES = ("north", "east", "south", "west")
MANEUVERS = ("left", "straight", "right")
MANEUVER_PRIOR = np.array([0.20, 0.60, 0.20])


def rotate(points, lane):
    theta = -lane * np.pi / 2
    matrix = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
    return np.asarray(points) @ matrix.T


@dataclass(frozen=True)
class Route:
    lane: int
    maneuver: int
    points: np.ndarray
    distances: np.ndarray
    exit_lane: int

    @property
    def length(self):
        return float(self.distances[-1])

    def xy(self, progress, approach_length=70.0):
        s = float(progress) - approach_length
        if s < 0:
            return self.points[0] + rotate([0.0, -s], self.lane)
        if s > self.length:
            direction = self.points[-1] - self.points[-2]
            return self.points[-1] + direction / np.linalg.norm(direction) * (s - self.length)
        return np.array([np.interp(s, self.distances, self.points[:, j]) for j in (0, 1)])


@lru_cache(maxsize=12)
def route(lane, maneuver):
    if lane not in range(4) or maneuver not in range(3):
        raise ValueError("lane must be 0..3 and maneuver 0..2")
    start = np.array([-2.5, 9.0])
    if maneuver == 1:
        end = np.array([-2.5, -9.0])
        points = np.linspace(start, end, 80)
        destination = 2
    else:
        if maneuver == 0:
            end = np.array([9.0, -2.5])
            c1, c2, destination = start + [0, -8], end + [-8, 0], 1
        else:
            end = np.array([-9.0, 2.5])
            c1, c2, destination = start + [0, -4], end + [4, 0], 3
        t = np.linspace(0, 1, 80)[:, None]
        points = (1-t)**3*start + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*end
    points = rotate(points, lane)
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return Route(lane, maneuver, points, distances, (destination + lane) % 4)


@lru_cache(maxsize=16)
def conflict_tensor(radius=1.7):
    """Conservative path conflicts, including shared approaches and exit merges."""
    conflicts = np.zeros((4, 3, 4, 3), dtype=bool)
    for a in range(4):
        for ma in range(3):
            ra = route(a, ma)
            for b in range(4):
                for mb in range(3):
                    rb = route(b, mb)
                    distance = np.linalg.norm(ra.points[:, None] - rb.points[None, :], axis=2)
                    conflicts[a, ma, b, mb] = (a == b or ra.exit_lane == rb.exit_lane
                                               or distance.min() < 2 * radius + 0.5)
    conflicts.setflags(write=False)
    return conflicts


def conflict_probability(lane_a, probabilities_a, lane_b, probabilities_b, radius=1.7):
    return float(probabilities_a @ conflict_tensor(radius)[lane_a, :, lane_b, :] @ probabilities_b)


def arrival_time(distance, speed, acceleration=2.5, speed_limit=10.0):
    """Earliest arrival under bounded acceleration (works for stopped vehicles)."""
    d = np.maximum(0.0, distance)
    v = np.clip(speed, 0.0, speed_limit)
    t_acc = (speed_limit - v) / acceleration
    d_acc = v * t_acc + 0.5 * acceleration * t_acc**2
    return np.where(d <= d_acc,
                    (np.sqrt(v*v + 2*acceleration*d) - v) / acceleration,
                    t_acc + (d - d_acc) / speed_limit)

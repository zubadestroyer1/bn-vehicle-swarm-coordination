"""Units throughout are meters, seconds, and meters/second."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    duration: float = 180.0
    dt: float = 0.25
    arrival_rate: float = 0.45  # Total demand over all four approaches, vehicles/s.
    approach_length: float = 70.0
    exit_length: float = 28.0
    speed_limit: float = 10.0
    acceleration: float = 2.5
    braking: float = 5.0
    driver_noise: float = 0.25
    vehicle_radius: float = 1.7
    following_gap: float = 7.0
    sensor_range: float = 55.0
    position_noise: float = 1.5
    velocity_noise: float = 0.6
    sensor_dropout: float = 0.10
    message_dropout: float = 0.15
    max_message_delay: int = 3
    communication_budget: int = 4
    risk_threshold: float = 0.10
    signal_green: float = 13.0
    signal_clearance: float = 3.0
    history_window: int = 48
    max_parents: int = 3
    inference_particles: int = 128
    planning_horizon: float = 6.0
    throughput_reward: float = 20.0
    delay_penalty: float = 1.0
    stop_penalty: float = 2.0
    collision_penalty: float = 1000.0
    communication_penalty: float = 0.02

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        for name in ("duration", "dt", "approach_length", "exit_length", "speed_limit",
                     "acceleration", "braking", "vehicle_radius", "following_gap",
                     "position_noise", "velocity_noise", "sensor_range", "signal_green",
                     "signal_clearance", "planning_horizon"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("arrival_rate", "driver_noise", "throughput_reward", "delay_penalty",
                     "stop_penalty", "collision_penalty", "communication_penalty"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        for name in ("sensor_dropout", "message_dropout", "risk_threshold"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("max_message_delay", "communication_budget", "history_window",
                     "max_parents", "inference_particles"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.inference_particles < 1 or self.history_window < 2 or self.max_parents > 8:
            raise ValueError("Require particles >= 1, history >= 2, and max_parents <= 8")
        if self.dt > 0.5:
            raise ValueError("dt must be <= 0.5 for the discrete vehicle dynamics")
        if self.following_gap < 2 * self.vehicle_radius + 0.5:
            raise ValueError("following_gap must exceed the vehicle diameter by 0.5 m")

    @classmethod
    def load(cls, path=None, **overrides):
        values = json.loads(Path(path).read_text()) if path else {}
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(values)-set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def to_dict(self):
        return asdict(self)

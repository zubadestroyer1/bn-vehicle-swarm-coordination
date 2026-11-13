"""Route motion, exogenous demand, imperfect observations, and shielding.

The shield knows true route occupancy and projects motions onto stop-line/headway
constraints. This assumption is shared by every policy, not learned safety.
"""

from collections import deque
from dataclasses import dataclass
import math
import numpy as np
from .belief import Observation
from .geometry import MANEUVER_PRIOR, route, conflict_tensor


@dataclass(frozen=True)
class Demand:
    vehicle_id: int
    requested_at: float
    lane: int
    maneuver: int
    cruise_speed: float


@dataclass
class Vehicle:
    demand: Demand
    progress: float = 0.0
    speed: float = 0.0
    stops: int = 0
    moving: bool = False

    @property
    def vehicle_id(self):
        return self.demand.vehicle_id

    @property
    def path(self):
        return route(self.demand.lane, self.demand.maneuver)


def generate_demand(config, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    demand, now = [], 0.0
    if config.arrival_rate == 0:
        return demand
    while True:
        now += float(rng.exponential(1/config.arrival_rate))
        if now >= config.duration:
            break
        demand.append(Demand(len(demand), now, int(rng.integers(4)),
                             int(rng.choice(3, p=MANEUVER_PRIOR)),
                             float(rng.uniform(.8, 1.0)*config.speed_limit)))
    return demand


class Intersection:
    def __init__(self, config, seed=0):
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        self.config, self.seed, self.tick = config, seed, 0
        self.demand = generate_demand(config, seed)
        self.next_demand = 0
        self.pending = [deque() for _ in range(4)]
        self.vehicles, self.completed = {}, {}
        self.reservations = set()
        self.packets = []
        self.messages_requested = self.messages_delivered = self.messages_dropped = 0
        self.shield_interventions = self.following_interventions = 0
        self.collision_pairs = set()
        self.total_stops = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.minimum_separation = math.inf
        self.conflicts = conflict_tensor(config.vehicle_radius)

    @property
    def time(self):
        return self.tick*self.config.dt

    def random(self, stream, vehicle_id):
        # Keyed streams keep noise and arrivals paired across policies.
        return np.random.default_rng(np.random.SeedSequence([self.seed, stream, self.tick, vehicle_id]))

    def prepare(self):
        """Admit due demand and return entry-detector registrations."""
        while self.next_demand < len(self.demand) and self.demand[self.next_demand].requested_at <= self.time:
            d = self.demand[self.next_demand]
            self.pending[d.lane].append(d)
            self.next_demand += 1
        registrations = []
        for lane, queue in enumerate(self.pending):
            approach = [v.progress for v in self.vehicles.values()
                        if v.demand.lane == lane and v.progress < self.config.approach_length]
            if queue and (not approach or min(approach) >= self.config.following_gap+1):
                d = queue.popleft()
                self.vehicles[d.vehicle_id] = Vehicle(d, speed=d.cruise_speed, moving=True)
                registrations.append((d.vehicle_id, lane, d.requested_at))
        return registrations

    def measurement(self, vehicle, communication=False):
        cfg = self.config
        rng = self.random(303 if communication else 202, vehicle.vehicle_id)
        dropout = cfg.message_dropout if communication else cfg.sensor_dropout
        if rng.random() < dropout:
            return None, rng
        p_sigma = .35 if communication else cfg.position_noise
        v_sigma = .15 if communication else cfg.velocity_noise
        accuracy = .97 if communication else .78
        maneuver = vehicle.demand.maneuver
        if rng.random() >= accuracy:
            maneuver = int(rng.choice([m for m in range(3) if m != maneuver]))
        return Observation(vehicle.vehicle_id, self.time,
                           vehicle.progress+float(rng.normal(0, p_sigma)),
                           vehicle.speed+float(rng.normal(0, v_sigma)),
                           p_sigma**2, v_sigma**2, maneuver, accuracy,
                           "message" if communication else "sensor"), rng

    def observe(self):
        observations, remaining = [], []
        for due_tick, observation in self.packets:
            if due_tick <= self.tick:
                self.messages_delivered += 1
                if observation.vehicle_id in self.vehicles:
                    observations.append(observation)
            else:
                remaining.append((due_tick, observation))
        self.packets = remaining
        for vehicle in self.vehicles.values():
            if abs(self.config.approach_length-vehicle.progress) <= self.config.sensor_range:
                obs, _ = self.measurement(vehicle)
                if obs is not None:
                    observations.append(obs)
        return observations

    def request_messages(self, requests):
        requests = list(dict.fromkeys(requests))[:self.config.communication_budget]
        count = 0
        for vehicle_id in requests:
            if vehicle_id not in self.vehicles:
                continue
            count += 1
            self.messages_requested += 1
            observation, rng = self.measurement(self.vehicles[vehicle_id], communication=True)
            if observation is None:
                self.messages_dropped += 1
            else:
                delay = 1+int(rng.integers(self.config.max_message_delay+1))
                self.packets.append((self.tick+delay, observation))
        return count

    def routes_conflict(self, first, second):
        a, b = first.demand, second.demand
        return bool(self.conflicts[a.lane, a.maneuver, b.lane, b.maneuver])

    def step(self, action):
        cfg, dt = self.config, self.config.dt
        gate = cfg.approach_length-cfg.vehicle_radius-.5
        message_count = self.request_messages(action.communicate)
        self.reservations = {i for i in self.reservations if i in self.vehicles
                             and self.vehicles[i].progress <= cfg.approach_length+self.vehicles[i].path.length+cfg.following_gap}
        before = {i: v.path.xy(v.progress, cfg.approach_length) for i, v in self.vehicles.items()}
        proposed, new_speed, accelerations = {}, {}, {}
        for i, v in self.vehicles.items():
            target = min(v.demand.cruise_speed, max(0.0, action.target_speeds.get(i, 0.0)))
            a = float(np.clip((target-v.speed)/dt, -cfg.braking, cfg.acceleration))
            a = float(np.clip(a+self.random(404, i).normal(0, cfg.driver_noise), -cfg.braking, cfg.acceleration))
            speed = float(np.clip(v.speed+a*dt, 0, cfg.speed_limit))
            proposed[i] = v.progress+(v.speed+speed)*.5*dt
            new_speed[i], accelerations[i] = speed, a
        priorities = {i: rank for rank, i in enumerate(action.priority)}
        entering = [i for i, v in self.vehicles.items() if v.progress <= gate and proposed[i] > gate]
        for i in sorted(entering, key=lambda j: (priorities.get(j, 10**9), j)):
            vehicle = self.vehicles[i]
            blocked = any(self.routes_conflict(vehicle, self.vehicles[j]) for j in self.reservations if j != i)
            blocked |= any(v.path.exit_lane == vehicle.path.exit_lane
                           and cfg.approach_length+v.path.length <= v.progress
                           < cfg.approach_length+v.path.length+cfg.following_gap
                           for j, v in self.vehicles.items() if j != i)
            if i in action.grants and not blocked:
                self.reservations.add(i)
            else:
                proposed[i], new_speed[i] = gate, 0.0
                self.shield_interventions += 1
        for lane in range(4):
            ids = sorted((i for i, v in self.vehicles.items() if v.demand.lane == lane
                          and v.progress < cfg.approach_length+cfg.following_gap), key=lambda i: self.vehicles[i].progress, reverse=True)
            for leader, follower in zip(ids, ids[1:]):
                limit = max(self.vehicles[follower].progress, proposed[leader]-cfg.following_gap)
                if proposed[follower] > limit:
                    proposed[follower] = limit
                    new_speed[follower] = min(new_speed[follower], new_speed[leader])
                    self.following_interventions += 1
        for lane in range(4):
            ids = sorted((i for i, v in self.vehicles.items() if v.path.exit_lane == lane
                          and v.progress >= cfg.approach_length+v.path.length),
                         key=lambda i: self.vehicles[i].progress-self.vehicles[i].path.length, reverse=True)
            for leader, follower in zip(ids, ids[1:]):
                limit = proposed[leader]-self.vehicles[leader].path.length+self.vehicles[follower].path.length-cfg.following_gap
                if proposed[follower] > limit:
                    proposed[follower] = max(self.vehicles[follower].progress, limit)
                    new_speed[follower] = min(new_speed[follower], new_speed[leader])
                    self.following_interventions += 1
        # A following constraint can retract a just-issued entry reservation.
        self.reservations = {i for i in self.reservations if proposed[i] > gate}
        stops = 0
        for i, vehicle in self.vehicles.items():
            # Hysteresis: a stop needs a fresh moving interval above 2 m/s.
            # Tiny noisy creep around zero must not count as repeated stops.
            if vehicle.moving and new_speed[i] < .3:
                vehicle.stops += 1
                stops += 1
                vehicle.moving = False
            vehicle.moving |= new_speed[i] >= 2.0
            vehicle.progress, vehicle.speed = proposed[i], new_speed[i]
        self.total_stops += stops
        ids = list(self.vehicles)
        collisions_before = len(self.collision_pairs)
        if len(ids) > 1:
            starts = np.array([before[i] for i in ids])
            ends = np.array([self.vehicles[i].path.xy(self.vehicles[i].progress, cfg.approach_length) for i in ids])
            a, b = np.triu_indices(len(ids), 1)
            offset = starts[a]-starts[b]
            delta = (ends[a]-starts[a])-(ends[b]-starts[b])
            fraction = np.clip(-np.sum(offset*delta, axis=1)/np.maximum(np.sum(delta*delta, axis=1), 1e-12), 0, 1)
            separation = np.linalg.norm(offset+fraction[:, None]*delta, axis=1)
            self.minimum_separation = min(self.minimum_separation, float(separation.min()))
            for j in np.flatnonzero(separation < 2*cfg.vehicle_radius-1e-7):
                self.collision_pairs.add(tuple(sorted((ids[a[j]], ids[b[j]]))))
        departed = []
        delay_cost = sum(len(q) for q in self.pending)*dt
        for i, vehicle in list(self.vehicles.items()):
            delay_cost += dt*max(0, 1-vehicle.speed/vehicle.demand.cruise_speed)
            distance = cfg.approach_length+vehicle.path.length+cfg.exit_length
            if vehicle.progress >= distance:
                actual = self.time+dt-vehicle.demand.requested_at
                self.completed[i] = {"travel_time": actual,
                                     "delay": max(0, actual-distance/vehicle.demand.cruise_speed),
                                     "stops": vehicle.stops}
                departed.append(i)
                del self.vehicles[i]
                self.reservations.discard(i)
        self.last_reward = (cfg.throughput_reward*len(departed)-cfg.delay_penalty*delay_cost
                            -cfg.stop_penalty*stops-cfg.communication_penalty*message_count
                            -cfg.collision_penalty*(len(self.collision_pairs)-collisions_before))
        self.total_reward += self.last_reward
        self.tick += 1
        return departed, accelerations

    def metrics(self):
        arrived = [d for d in self.demand if d.requested_at <= self.time]
        cumulative_delay = sum(v["delay"] for v in self.completed.values())
        for d in arrived:
            if d.vehicle_id in self.completed:
                continue
            vehicle = self.vehicles.get(d.vehicle_id)
            progress = vehicle.progress if vehicle else 0.0
            cumulative_delay += max(0, self.time-d.requested_at-progress/d.cruise_speed)
        count = len(self.completed)
        return {
            "duration_s": self.time, "arrived": len(arrived), "completed": count,
            "unfinished": len(arrived)-count,
            "completion_fraction": count/len(arrived) if arrived else 0.0,
            "throughput_veh_per_hour": count/max(self.time, 1e-9)*3600,
            "mean_completed_delay_s": float(np.mean([v["delay"] for v in self.completed.values()])) if count else None,
            "mean_accrued_delay_s": cumulative_delay/len(arrived) if arrived else 0.0,
            "total_stops": self.total_stops, "stops_per_arrival": self.total_stops/max(1, len(arrived)),
            "collision_pairs": len(self.collision_pairs), "shield_interventions": self.shield_interventions,
            "following_interventions": self.following_interventions,
            "minimum_separation_m": self.minimum_separation if math.isfinite(self.minimum_separation) else None,
            "messages_requested": self.messages_requested, "messages_delivered": self.messages_delivered,
            "messages_dropped": self.messages_dropped, "messages_in_flight": len(self.packets),
            "messages_per_arrival": self.messages_requested/max(1, len(arrived)), "reward": self.total_reward,
        }

    def frame(self, action, network):
        return {"time": self.time, "completed": len(self.completed),
                "queued": sum(len(q) for q in self.pending), "reward": self.total_reward,
                "vehicles": [{"id": i, "lane": v.demand.lane, "maneuver": v.demand.maneuver,
                              "x": float(v.path.xy(v.progress, self.config.approach_length)[0]),
                              "y": float(v.path.xy(v.progress, self.config.approach_length)[1]),
                              "speed": v.speed, "reserved": i in self.reservations}
                             for i, v in self.vehicles.items()],
                "edges": [] if network is None else
                [{"source": network.vehicle_ids[a], "target": network.vehicle_ids[b],
                  "probability": float(network.adjacency[a, b])}
                 for a, b in zip(*np.where(network.adjacency >= .3))],
                "grants": list(action.grants)}

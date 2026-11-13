"""Belief-only control: baseline rules or short-horizon priority enumeration."""

from dataclasses import dataclass, field
from itertools import permutations
from time import perf_counter
import numpy as np
from .bayes import DynamicBayesNet, NeuralStructure, GreedyStructure, IndependentStructure, infer_delay, NetworkState
from .geometry import conflict_probability, route

POLICIES = ("signal", "fcfs", "independent", "greedy", "neural")


@dataclass
class Action:
    target_speeds: dict = field(default_factory=dict)
    grants: set = field(default_factory=set)
    communicate: list = field(default_factory=list)
    priority: list = field(default_factory=list)


class Controller:
    def __init__(self, name, config, model=None, seed=0):
        if name not in POLICIES:
            raise ValueError(f"Unknown policy {name}; choose from {POLICIES}")
        if name == "neural" and model is None:
            raise ValueError("The neural policy requires a trained model")
        self.name, self.config, self.network = name, config, None
        structure = (NeuralStructure(model, config) if name == "neural" else
                     GreedyStructure(config) if name == "greedy" else IndependentStructure())
        self.dbn = DynamicBayesNet(structure, config, seed+5000)
        self.update_times, self.structure_times, self.control_times = [], [], []
        self.last_requests = {}
        self.ess_values = []

    def act(self, tracker, now):
        started = perf_counter()
        cfg = self.config
        tracks = sorted(tracker.tracks.values(), key=lambda t: (t.eta(cfg), t.vehicle_id))
        active = [t for t in tracks if t.progress <= cfg.approach_length+26]
        if self.name in ("neural", "greedy", "independent"):
            structure_started = perf_counter()
            adjacency = self.dbn.structure.update(active)
            self.structure_times.append((perf_counter()-structure_started)*1000)
            predictive, posterior, ess = infer_delay(active, adjacency, self.dbn.previous,
                                                     self.dbn.rng, cfg.inference_particles)
            self.dbn.previous = {t.vehicle_id: float(p) for t, p in zip(active, posterior)}
            self.network = NetworkState([t.vehicle_id for t in active], adjacency, predictive, posterior, ess)
            self.ess_values.append(ess)
        self.update_times.append((perf_counter()-started)*1000)
        planner_started = perf_counter()
        heads = []
        gate = cfg.approach_length-cfg.vehicle_radius-.5
        for lane in range(4):
            candidates = [t for t in tracks if t.lane == lane and t.progress < gate+1.5]
            if candidates:
                heads.append(max(candidates, key=lambda t: t.progress))
        heads = [t for t in heads if t.eta(cfg) <= cfg.planning_horizon]
        crossing = [t for t in tracks if gate+1.5 <= t.progress < cfg.approach_length+24]
        action = Action()
        for t in tracks:
            distance = max(0, gate-t.progress-np.sqrt(t.covariance[0, 0]))
            action.target_speeds[t.vehicle_id] = (cfg.speed_limit if t.progress > gate+1.5 else
                                                 min(cfg.speed_limit, np.sqrt(2*cfg.braking*distance)))
        if self.name == "signal":
            phase_length = cfg.signal_green+cfg.signal_clearance
            phase = int(now/phase_length) % 4
            ordered = [t for t in heads if t.lane == phase and now % phase_length < cfg.signal_green]
            slots = {t.vehicle_id: t.eta(cfg) for t in ordered}
        elif self.name == "fcfs":
            ordered = sorted(heads, key=lambda t: (t.entered_at, t.vehicle_id))
            slots = self.schedule(ordered, crossing, {})
        else:
            delays = dict(zip(self.network.vehicle_ids, self.network.delay_probability))
            best_cost, ordered, slots = np.inf, [], {}
            for ordering in permutations(heads):
                candidate_slots = self.schedule(ordering, crossing, delays)
                cost = 0.0
                for t in ordering:
                    wait = max(0, candidate_slots[t.vehicle_id]-t.eta(cfg))
                    pressure = sum(u.lane == t.lane and u.progress < t.progress for u in tracks)
                    fairness = 1+.04*max(0, now-t.entered_at)+.12*pressure
                    cost += cfg.delay_penalty*fairness*wait + cfg.stop_penalty*(wait > 1.0)
                    cost -= cfg.throughput_reward/(1+candidate_slots[t.vehicle_id]+3)
                if cost < best_cost:
                    best_cost, ordered, slots = cost, list(ordering), candidate_slots
        selected = []
        for t in ordered:
            start = slots.get(t.vehicle_id, np.inf)
            if start <= t.eta(cfg)+.35:
                if any(self.conflict(t, other) > cfg.risk_threshold for other in selected+crossing):
                    continue
                selected.append(t)
                action.grants.add(t.vehicle_id)
                action.target_speeds[t.vehicle_id] = cfg.speed_limit
            else:
                distance = max(0, gate-t.progress-2.0)
                action.target_speeds[t.vehicle_id] = min(action.target_speeds[t.vehicle_id], distance/max(.5, start))
        action.priority = [t.vehicle_id for t in ordered]
        # Belief-based adaptive cruise control for each incoming queue. The
        # simulator's headway shield remains a backstop for state-estimation error.
        for lane in range(4):
            queue = sorted((t for t in tracks if t.lane == lane and t.progress < cfg.approach_length+cfg.following_gap),
                           key=lambda t: t.progress, reverse=True)
            for leader, follower in zip(queue, queue[1:]):
                uncertainty = np.sqrt(leader.covariance[0, 0]+follower.covariance[0, 0])
                clearance = max(0, leader.progress-follower.progress-cfg.following_gap-uncertainty)
                limit = np.sqrt(leader.speed**2+2*cfg.braking*clearance)
                action.target_speeds[follower.vehicle_id] = min(action.target_speeds[follower.vehicle_id], limit)
        if self.name in ("neural", "greedy", "independent"):
            importance = {i: float(v) for i, v in zip(self.network.vehicle_ids, self.network.adjacency.sum(axis=0)+self.network.adjacency.sum(axis=1))}
            def value(t):
                entropy = -float(np.sum(t.intent*np.log(t.intent+1e-9)))
                return (t.age+np.sqrt(t.covariance[0, 0])*.2+entropy)*(1+importance.get(t.vehicle_id, 0))/(1+t.eta(cfg))
            eligible = [t for t in active if now-self.last_requests.get(t.vehicle_id, -100) >= 1.0]
            selected_messages = sorted(eligible, key=value, reverse=True)[:cfg.communication_budget]
            action.communicate = [t.vehicle_id for t in selected_messages]
        else:
            eligible = [t for t in heads if now-self.last_requests.get(t.vehicle_id, -100) >= 1.0]
            action.communicate = [t.vehicle_id for t in sorted(eligible, key=lambda t: -t.age)[:cfg.communication_budget]]
        self.last_requests.update({i: now for i in action.communicate})
        live = {t.vehicle_id for t in tracks}
        self.last_requests = {i: time for i, time in self.last_requests.items() if i in live}
        self.control_times.append((perf_counter()-planner_started)*1000)
        return action

    def conflict(self, a, b):
        return conflict_probability(a.lane, a.intent, b.lane, b.intent, self.config.vehicle_radius)

    def schedule(self, ordering, crossing, delays):
        cfg = self.config
        starts, finishes, earlier = {}, {}, []
        for t in crossing:
            length = sum(t.intent[m]*route(t.lane, m).length for m in range(3))
            remaining = max(0, cfg.approach_length+length+cfg.following_gap-t.progress)
            finishes[t.vehicle_id] = remaining/max(3.0, t.speed)
            earlier.append(t)
        for t in ordering:
            eta = t.eta(cfg)
            start = eta
            for previous in earlier:
                if self.conflict(t, previous) > cfg.risk_threshold:
                    start = max(start, finishes[previous.vehicle_id])
            length = sum(t.intent[m]*route(t.lane, m).length for m in range(3))
            uncertainty = .15*np.sqrt(t.covariance[0, 0])+min(1.0, t.age*.1)
            duration = (length+cfg.following_gap)/max(4.0, t.speed)+.4+uncertainty+1.5*delays.get(t.vehicle_id, 0)
            starts[t.vehicle_id], finishes[t.vehicle_id] = start, start+duration
            earlier.append(t)
        return starts

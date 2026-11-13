"""Behavioral and mathematical regression checks; run with unittest discovery."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from swarm.bayes import GreedyStructure, NeuralStructure, bic_score, infer_delay, noisy_or_probability
from swarm.belief import BeliefTracker, Observation, Track
from swarm.config import Config
from swarm.experiment import run_episode, paired_differences, summarize
from swarm.features import pair_features
from swarm.geometry import arrival_time, conflict_tensor, route
from swarm.neural import EdgeMLP, binary_metrics
from swarm.policies import Action, Controller
from swarm.simulator import Demand, Intersection, Vehicle, generate_demand
from swarm.training import make_dataset, sample_scene


class GeometryTests(unittest.TestCase):
    def test_routes_are_continuous_and_valid(self):
        for lane in range(4):
            for maneuver in range(3):
                r = route(lane, maneuver)
                self.assertTrue(np.all(np.diff(r.distances) > 0))
                for boundary in (70.0, 70.0+r.length):
                    self.assertLess(np.linalg.norm(r.xy(boundary+1e-5)-r.xy(boundary-1e-5)), 3e-5)

    def test_conflict_symmetry_and_opposing_straight_routes(self):
        c = conflict_tensor()
        np.testing.assert_array_equal(c, c.transpose(2, 3, 0, 1))
        self.assertFalse(c[0, 1, 2, 1])
        self.assertTrue(c[0, 1, 1, 1])
        for lane in range(4):
            self.assertTrue(c[lane, :, lane, :].all())

    def test_arrival_from_rest_and_at_speed(self):
        self.assertAlmostEqual(float(arrival_time(5, 0, 2.5, 10)), 2.0)
        self.assertAlmostEqual(float(arrival_time(20, 10)), 2.0)
        self.assertEqual(float(arrival_time(-1, 0)), 0)


class BeliefTests(unittest.TestCase):
    def test_updates_reduce_uncertainty_and_reject_old_packets(self):
        tracker = BeliefTracker(Config())
        tracker.register(0, 0, 0)
        original = np.trace(tracker.tracks[0].covariance)
        obs = Observation(0, 1, 12, 8, .1, .1, 2, .95)
        tracker.update([obs], 1)
        t = tracker.tracks[0]
        self.assertLess(np.trace(t.covariance), original)
        self.assertGreater(t.intent[2], .5)
        before = t.mean.copy()
        tracker.update([Observation(0, 0, -100, -100, .01, .01, 0, .95)], 1)
        np.testing.assert_array_equal(before, t.mean)
        self.assertTrue(np.all(np.linalg.eigvalsh(t.covariance) > 0))

    def test_missing_observations_increase_uncertainty(self):
        tracker = BeliefTracker(Config())
        tracker.register(0, 0, 0)
        before = tracker.tracks[0].covariance[0, 0]
        for _ in range(10):
            tracker.predict({0: 0})
        self.assertGreater(tracker.tracks[0].covariance[0, 0], before)
        self.assertAlmostEqual(tracker.tracks[0].age, 2.5)


class BayesianTests(unittest.TestCase):
    def test_noisy_or_matches_enumeration(self):
        p = noisy_or_probability(.1, 1, np.array([1, 0]), np.array([.8, .4]))
        self.assertAlmostEqual(p, 1-.9*.35*(1-.75*.8))

    def test_temporal_persistence_and_graph_validity(self):
        tracks, _ = sample_scene(np.random.default_rng(0), Config(), 4)
        a = np.zeros((4, 4))
        low, _, _ = infer_delay(tracks, a, {}, np.random.default_rng(0), 10000)
        high, post, ess = infer_delay(tracks, a, {t.vehicle_id: 1 for t in tracks}, np.random.default_rng(0), 10000)
        self.assertGreater(np.mean(high), np.mean(low)+.4)
        self.assertTrue(np.all((post >= 0) & (post <= 1)))
        self.assertGreater(ess, 0)
        with self.assertRaises(ValueError):
            infer_delay(tracks, np.eye(4), {}, np.random.default_rng(0))

    def test_neural_structure_is_acyclic_bounded_and_handles_churn(self):
        cfg = Config(max_parents=2)
        structure = NeuralStructure(EdgeMLP(), cfg)
        for n in (0, 1, 8, 3):
            tracks, _ = sample_scene(np.random.default_rng(n), cfg, n)
            a = structure.update(tracks)
            self.assertEqual(a.shape, (n, n))
            self.assertFalse(np.tril(a).any())
            self.assertTrue(np.all(np.count_nonzero(a, axis=0) <= 2))

    def test_bic_finds_a_predictive_parent(self):
        rng = np.random.default_rng(5)
        x = rng.integers(2, size=400)
        y = np.logical_xor(x, rng.random(400) < .05)
        independent = rng.integers(2, size=400)
        data = np.column_stack([x, independent, y])
        self.assertGreater(bic_score(data, 2, [0]), bic_score(data, 2, []))
        self.assertLess(bic_score(data, 2, [0, 1]), bic_score(data, 2, [0]))
        cfg = Config(history_window=400)
        learner = GreedyStructure(cfg)
        tracks = [Track(i, i, 0, np.array([20., 0.]), np.eye(2)) for i in range(3)]
        learner.history.extend([{i: float(row[i]) for i in range(3)} for row in data[:-1]])
        for i in range(3):
            tracks[i].mean[1] = 0 if data[-1, i] else 8
        a = learner.update(tracks)
        self.assertGreater(a[0, 2], .5)


class LearningTests(unittest.TestCase):
    def test_model_learns_and_round_trips(self):
        cfg = Config()
        x, y = make_dataset(cfg, 70, 8)
        validation = make_dataset(cfg, 30, 9)
        model = EdgeMLP(seed=4)
        model.fit(x, y, validation, epochs=18, seed=5)
        test = make_dataset(cfg, 30, 10)
        actual = binary_metrics(test[1], model.predict(test[0]))
        baseline = binary_metrics(test[1], np.full(len(test[1]), y.mean()))
        self.assertLess(actual["brier"], baseline["brier"]*.85)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.npz"
            model.save(path)
            loaded = EdgeMLP.load(path)
            np.testing.assert_array_equal(model.predict(test[0]), loaded.predict(test[0]))

    def test_auc_ties_and_separable_predictions(self):
        self.assertEqual(binary_metrics([0, 1], [.5, .5])["roc_auc"], .5)
        self.assertEqual(binary_metrics([0, 1], [.1, .9])["roc_auc"], 1)
        self.assertAlmostEqual(binary_metrics([0], [.5])["ece"], .5)


class SimulatorTests(unittest.TestCase):
    def test_shield_rejects_conflicting_crossings(self):
        cfg = Config(driver_noise=0)
        env = Intersection(cfg)
        gate = cfg.approach_length-cfg.vehicle_radius-.5
        for i in range(2):
            env.vehicles[i] = Vehicle(Demand(i, 0, i, 1, 10), gate-.05, 8, moving=True)
        action = Action({0: 10, 1: 10}, {0, 1}, [], [0, 1])
        env.step(action)
        self.assertGreater(env.vehicles[0].progress, gate)
        self.assertEqual(env.vehicles[1].progress, gate)
        self.assertEqual(env.shield_interventions, 1)
        self.assertEqual(env.metrics()["collision_pairs"], 0)

    def test_same_lane_spacing(self):
        env = Intersection(Config(driver_noise=0))
        env.vehicles[0] = Vehicle(Demand(0, 0, 0, 1, 10), 40, 0)
        env.vehicles[1] = Vehicle(Demand(1, 0, 0, 1, 10), 33, 10)
        env.step(Action({0: 0, 1: 10}))
        self.assertGreaterEqual(env.vehicles[0].progress-env.vehicles[1].progress, 7)
        self.assertEqual(env.metrics()["collision_pairs"], 0)

    def test_messages_respect_budget_delay_and_accounting(self):
        env = Intersection(Config(communication_budget=2, message_dropout=0, sensor_dropout=1))
        for i in range(3):
            env.vehicles[i] = Vehicle(Demand(i, 0, i, 1, 10), 20, 0)
        env.request_messages([0, 0, 1, 2])
        self.assertEqual(env.messages_requested, 2)
        self.assertEqual(env.observe(), [])
        for _ in range(5):
            env.tick += 1
            env.observe()
        self.assertEqual(env.messages_delivered, 2)
        self.assertEqual(len(env.packets), 0)

    def test_demand_determinism_and_seed_variation(self):
        cfg = Config(duration=40)
        self.assertEqual(generate_demand(cfg, 42), generate_demand(cfg, 42))
        self.assertNotEqual(generate_demand(cfg, 42), generate_demand(cfg, 43))

    def test_empty_episode_has_finite_serializable_metrics(self):
        metrics, trace = run_episode(Config(duration=2, arrival_rate=0), "independent", 1, record=True)
        self.assertEqual(metrics["arrived"], 0)
        self.assertEqual(metrics["completed"], 0)
        json.dumps(trace, allow_nan=False)

    def test_policies_move_traffic_and_respect_safety(self):
        for name in ("signal", "fcfs", "independent", "greedy", "neural"):
            with self.subTest(policy=name):
                metrics, trace = run_episode(Config(duration=65, arrival_rate=.3), name, 11,
                                             EdgeMLP(seed=7), record=True)
                self.assertGreater(metrics["completed"], 2)
                self.assertEqual(metrics["collision_pairs"], 0)
                self.assertEqual(metrics["completed"]+metrics["unfinished"], metrics["arrived"])
                self.assertEqual(metrics["messages_requested"], metrics["messages_delivered"]+
                                 metrics["messages_dropped"]+metrics["messages_in_flight"])
                self.assertTrue(trace["frames"])

    def test_repeat_episode_is_reproducible_except_timings(self):
        cfg = Config(duration=15)
        first, _ = run_episode(cfg, "independent", 10)
        second, _ = run_episode(cfg, "independent", 10)
        for key in first:
            if not key.endswith("_ms") and key != "wall_time_s":
                self.assertEqual(first[key], second[key], key)

    def test_bad_config_fails_early(self):
        for kwargs in ({"duration": 0}, {"arrival_rate": -1}, {"dt": 2},
                       {"message_dropout": 1.1}, {"communication_budget": 1.5},
                       {"risk_threshold": float("nan")}):
            with self.assertRaises(ValueError):
                Config(**kwargs)

    def test_stop_hysteresis_excludes_creep(self):
        env = Intersection(Config(driver_noise=0))
        env.vehicles[0] = Vehicle(Demand(0, 0, 0, 1, 10), 30, 0, moving=False)
        for target in [.4, 0, .6, 0]:
            env.step(Action({0: target}))
        self.assertEqual(env.total_stops, 0)
        env.vehicles[0].speed = 2.5
        env.vehicles[0].moving = True
        for _ in range(3):
            env.step(Action({0: 0}))
        self.assertEqual(env.total_stops, 1)

    def test_zero_communication_budget(self):
        metrics, _ = run_episode(Config(duration=30, communication_budget=0), "neural", 11, EdgeMLP())
        self.assertEqual(metrics["messages_requested"], 0)

    def test_noisy_episode_does_not_collide(self):
        cfg = Config(duration=60, arrival_rate=.8, position_noise=4, sensor_dropout=.4,
                     message_dropout=.5, max_message_delay=6)
        metrics, _ = run_episode(cfg, "neural", 12, EdgeMLP())
        self.assertEqual(metrics["collision_pairs"], 0)
        self.assertGreater(metrics["completed"], 0)


if __name__ == "__main__":
    unittest.main()

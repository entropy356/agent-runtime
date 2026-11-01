import copy
import math
import unittest

from memory_dynamics.model import DynamicsConfig, activate, decay, rank


class DynamicsModelTests(unittest.TestCase):
    def test_weight_decays_each_round(self):
        config = DynamicsConfig(decay_rate=0.2)
        memory = {"weight": 1.0, "stability": 0.0}
        result = decay(memory, config)
        self.assertAlmostEqual(memory["weight"], math.exp(-0.2))
        self.assertEqual(result["effective_decay_rate"], 0.2)

    def test_stability_reduces_future_decay_rate(self):
        config = DynamicsConfig(decay_rate=0.2, stability_protection=0.8)
        fragile = {"weight": 1.0, "stability": 0.0}
        stable = {"weight": 1.0, "stability": 0.75}
        fragile_result = decay(fragile, config)
        stable_result = decay(stable, config)
        self.assertLess(stable_result["effective_decay_rate"], fragile_result["effective_decay_rate"])
        self.assertGreater(stable["weight"], fragile["weight"])

    def test_activation_restores_weight_and_increases_stability(self):
        config = DynamicsConfig(recovery_rate=0.6, stability_gain=0.2)
        memory = {"weight": 0.4, "stability": 0.3}
        result = activate(memory, config)
        self.assertGreater(memory["weight"], 0.4)
        self.assertGreater(memory["stability"], 0.3)
        self.assertAlmostEqual(memory["weight"], 0.4 + 0.6 * 0.6)
        self.assertAlmostEqual(memory["stability"], 0.3 + 0.7 * 0.2 * 0.6)
        self.assertAlmostEqual(result["stability_gain"], memory["stability"] - 0.3)

    def test_harder_successful_retrieval_builds_more_stability(self):
        config = DynamicsConfig(stability_gain=0.2)
        hard = {"weight": 0.2, "stability": 0.0}
        easy = {"weight": 0.9, "stability": 0.0}
        activate(hard, config)
        activate(easy, config)
        self.assertGreater(hard["stability"], easy["stability"])

    def test_weight_only_ablation_has_no_stability_state(self):
        config = DynamicsConfig(mode="weight-only", stability_gain=0.9)
        memory = {"weight": 0.4, "stability": 0.8}
        activate(memory, config)
        self.assertEqual(memory["stability"], 0.0)
        decay(memory, config)
        self.assertEqual(memory["stability"], 0.0)

    def test_rank_combines_relevance_and_weight_without_mutating_state(self):
        memories = [
            {"id": 1, "content": "same", "vector": [1.0, 0.0], "weight": 0.2, "stability": 0.0},
            {"id": 2, "content": "other", "vector": [0.0, 1.0], "weight": 0.8, "stability": 0.0},
        ]
        before = copy.deepcopy(memories)
        rows = rank(memories, [1.0, 0.0], top_k=2)
        self.assertEqual([row["id"] for row in rows], [2, 1])
        self.assertAlmostEqual(rows[0]["score"], 0.4)
        self.assertAlmostEqual(rows[1]["score"], 0.2)
        self.assertEqual(memories, before)

    def test_internal_tick_ranks_by_weight_only(self):
        memories = [
            {"id": 1, "vector": [1.0, 0.0], "weight": 0.2, "stability": 0.0},
            {"id": 2, "vector": [0.0, 1.0], "weight": 0.8, "stability": 0.0},
        ]
        self.assertEqual([row["id"] for row in rank(memories, None)], [2, 1])

    def test_invalid_configuration_is_rejected(self):
        for kwargs in (
            {"decay_rate": -0.1},
            {"stability_protection": 1.0},
            {"recovery_rate": 1.1},
            {"stability_gain": -0.1},
            {"top_k": 0},
            {"mode": "tiers"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DynamicsConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()

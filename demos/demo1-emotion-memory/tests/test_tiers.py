import copy
import unittest
from unittest.mock import patch

from agent_runtime.providers import OfflineProvider
from agent_runtime.runtime import Runtime, rank, tier, usage_frequency
from tests import test_runtime as fixtures


class TierSelectionTests(unittest.TestCase):
    def memories(self):
        return [{"id": i, "vector": [1., 0.], "affect": [1., 0.],
                 "uses": 999, "recent_uses": recent, "write_strength": strength}
                for i, recent, strength in (
                    (1, list(range(9, 17)), .9), (2, list(range(9, 17)), .8),
                    (3, [15, 16], .3), (4, [], .1))]

    def test_layers_change_context_without_changing_scores(self):
        memories = self.memories()
        before = copy.deepcopy(memories)
        layered = rank(memories, [1., 0.], None, step=16)
        plain = rank(memories, [1., 0.], None, step=16, tiered=False)
        self.assertEqual([r["id"] for r in layered if r["selected"]], [1, 3, 4])
        self.assertEqual([r["id"] for r in plain if r["selected"]], [1, 2, 3])
        self.assertEqual([r["score"] for r in layered], [r["score"] for r in plain])
        self.assertEqual(memories, before)
        self.assertTrue(all(r["selected_by"] == "tier_quota" for r in layered if r["selected"]))

    def test_missing_layers_release_places_and_never_duplicate(self):
        memories = self.memories()
        for memory in memories:
            memory["recent_uses"] = []
        rows = rank(memories, None, None, step=16)
        self.assertEqual([r["id"] for r in rows if r["selected"]], [1, 2, 3])
        self.assertEqual([r["selected_by"] for r in rows[:3]],
                         ["tier_quota", "score_fill", "score_fill"])
        for budget in (0, 1, 2, 3, 8):
            with self.subTest(budget=budget):
                rows = rank(self.memories(), None, None, step=16, top_k=budget)
                ids = [r["id"] for r in rows if r["selected"]]
                self.assertEqual(len(ids), min(budget, 4))
                self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(rank([], None, None), [])

    def test_window_boundaries_duplicates_and_unknown_history(self):
        memory = {"uses": 999, "recent_uses": [0, 1, 1, 16, 17]}
        self.assertEqual(usage_frequency(memory, 16), 2 / 16)
        self.assertEqual(tier(usage_frequency(memory, 16)), "Warm")
        self.assertEqual(usage_frequency({"uses": 999}, 16), 0)
        self.assertEqual(usage_frequency(memory, 33), 0)

    def test_superseded_hot_memory_cannot_take_a_quota(self):
        memories = self.memories()
        memories[0]["superseded_by"] = 4
        rows = rank(memories, None, None, step=16)
        self.assertEqual([r["id"] for r in rows if r["selected"]], [2, 3, 4])


class TierLifecycleTests(unittest.TestCase):
    setUp = fixtures.RuntimeTests.setUp
    database_dump = fixtures.RuntimeTests.database_dump

    def run_rounds(self, count, ids):
        result = {"answer": "收到", "appraisal": "继续认知", "used_memory_ids": ids,
                  "memories": []}
        with patch.object(self.provider, "cognize", return_value=result):
            for _ in range(count):
                self.runtime.step("合作")

    def test_cold_promotes_then_unused_hot_cools_and_history_is_bounded(self):
        self.runtime.seed()
        target = self.runtime.retrieve("合作")[0]["id"]
        self.run_rounds(2, [target])
        find = lambda: next(m for m in self.runtime.inspect()["state"]["memories"] if m["id"] == target)
        self.assertEqual(find()["tier"], "Warm")
        self.run_rounds(6, [target])
        self.assertEqual(find()["tier"], "Hot")
        self.assertEqual(find()["usage_frequency"], .5)
        self.run_rounds(16, [])
        self.assertEqual(find()["tier"], "Cold")
        self.assertEqual(find()["recent_uses"], [])
        self.assertEqual(find()["uses"], 8)
        self.run_rounds(20, [target])
        self.assertEqual(len(find()["recent_uses"]), 16)

    def test_failed_round_and_read_only_operations_do_not_age_usage(self):
        self.runtime.seed()
        target = self.runtime.retrieve("合作")[0]["id"]
        self.run_rounds(8, [target])
        before = self.database_dump()
        self.runtime.retrieve("合作")
        self.runtime.inspect()
        self.runtime.compare("合作", "baseline", "A", "B")
        with patch.object(self.provider, "embed", side_effect=ValueError("unavailable")):
            with self.assertRaises(ValueError):
                self.runtime.step("合作")
        self.assertEqual(before, self.database_dump())

    def test_usage_and_selection_survive_restore_and_restart(self):
        self.runtime.seed()
        target = self.runtime.retrieve("合作")[0]["id"]
        self.run_rounds(8, [target])
        self.runtime.save("hot")
        expected = self.runtime.retrieve("合作")
        self.run_rounds(16, [])
        self.runtime.restore("hot")
        self.assertEqual(expected, self.runtime.retrieve("合作"))
        self.assertEqual(expected, Runtime(self.path, OfflineProvider()).retrieve("合作"))

    def test_compare_uses_common_library_usage_not_emotion_snapshot_usage(self):
        self.runtime.seed()
        target = self.runtime.retrieve("合作")[0]["id"]
        self.run_rounds(8, [target])
        self.runtime.save("common-hot")
        comparison = self.runtime.compare("合作", "common-hot", "A", "B")
        for trial in comparison["trials"]:
            row = next(r for r in trial["retrieval"] if r["id"] == target)
            self.assertEqual(row["tier"], "Hot")
            self.assertEqual(row["usage_frequency"], .5)
        off_a, off_b = comparison["trials"][2:]
        project = lambda t: [(r["id"], r["score"], r["tier"], r["selected"])
                             for r in t["retrieval"]]
        self.assertEqual(project(off_a), project(off_b))

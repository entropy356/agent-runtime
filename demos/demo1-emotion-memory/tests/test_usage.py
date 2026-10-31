import unittest
from unittest.mock import patch

from agent_runtime.runtime import USAGE_WINDOW, rank, usage_frequency
from tests import test_runtime as fixtures


class UsageObservationTests(unittest.TestCase):
    def test_window_boundaries_duplicates_and_unknown_history(self):
        memory = {"uses": 999, "recent_uses": [0, 1, 1, 16, 17]}
        self.assertEqual(usage_frequency(memory, 16), 2 / 16)
        self.assertEqual(usage_frequency({"uses": 999}, 16), 0)
        self.assertEqual(usage_frequency(memory, 33), 0)

    def test_frequency_never_changes_order_or_selection(self):
        # Same vectors and strengths, opposite citation history. Observation only means
        # the selected set and every score must be identical.
        memories = [{"id": i, "vector": [1., 0.], "affect": [1., 0.], "uses": 0,
                     "recent_uses": recent, "write_strength": strength}
                    for i, recent, strength in ((1, list(range(16)), .2), (2, [], .9), (3, [], .8))]
        quiet = [dict(memory, recent_uses=[]) for memory in memories]
        project = lambda rows: [(r["id"], r["score"], r["selected"]) for r in rows]
        for budget in (0, 1, 2, 3):
            with self.subTest(budget=budget):
                self.assertEqual(project(rank(memories, None, None, step=16, top_k=budget)),
                                 project(rank(quiet, None, None, step=16, top_k=budget)))
        self.assertEqual([r["id"] for r in rank(memories, None, None, step=16) if r["selected"]],
                         [2, 3, 1])

    def test_rows_carry_the_frequency_for_display(self):
        memories = [{"id": 1, "vector": [1., 0.], "affect": [1., 0.], "uses": 0,
                     "recent_uses": list(range(1, 17)), "write_strength": .5}]
        self.assertEqual(rank(memories, None, None, step=16)[0]["usage_frequency"], 1.0)


class UsageLifecycleTests(unittest.TestCase):
    setUp = fixtures.RuntimeTests.setUp
    database_dump = fixtures.RuntimeTests.database_dump

    def run_rounds(self, count, ids):
        result = {"answer": "收到", "appraisal": "继续认知", "used_memory_ids": ids,
                  "memories": []}
        with patch.object(self.provider, "cognize", return_value=result):
            for _ in range(count):
                self.runtime.step("合作")

    def test_citation_history_is_windowed_but_the_memory_never_expires(self):
        self.runtime.seed()
        target = self.runtime.retrieve("合作")[0]["id"]
        find = lambda: next(m for m in self.runtime.inspect()["state"]["memories"]
                            if m["id"] == target)
        self.run_rounds(8, [target])
        self.assertEqual(find()["usage_frequency"], .5)
        self.run_rounds(16, [])
        self.assertEqual(find()["recent_uses"], [])
        self.assertEqual(find()["usage_frequency"], 0)
        self.assertEqual(find()["uses"], 8)
        self.run_rounds(20, [target])
        self.assertEqual(len(find()["recent_uses"]), 16)
        self.assertEqual(find()["usage_frequency"], 1)

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
        self.runtime.save("cited")
        expected = self.runtime.retrieve("合作")
        self.run_rounds(16, [])
        self.runtime.restore("cited")
        self.assertEqual(expected, self.runtime.retrieve("合作"))
        self.assertEqual(expected, self.runtime.__class__(self.path, self.provider).retrieve("合作"))

    def test_compare_reads_usage_from_the_common_snapshot(self):
        self.runtime.seed()
        target = self.runtime.retrieve("合作")[0]["id"]
        self.run_rounds(8, [target])
        self.runtime.save("cited")
        comparison = self.runtime.compare("合作", "cited", "A", "B")
        for trial in comparison["trials"]:
            row = next(r for r in trial["retrieval"] if r["id"] == target)
            self.assertEqual(row["usage_frequency"], .5)
        off_a, off_b = comparison["trials"][2:]
        project = lambda t: [(r["id"], r["score"], r["selected"]) for r in t["retrieval"]]
        self.assertEqual(project(off_a), project(off_b))

    def test_step_trace_records_the_usage_window(self):
        self.runtime.seed()
        trace = self.runtime.step("合作")
        self.assertEqual(trace["usage_window"], USAGE_WINDOW)
        self.assertNotIn("tiered", trace)
        selected = [r for r in trace["retrieval"] if r["selected"]]
        self.assertEqual([r["id"] for r in selected], [r["id"] for r in trace["retrieval"][:3]])

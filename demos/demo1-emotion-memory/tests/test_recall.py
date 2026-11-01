import unittest
from unittest.mock import patch

from agent_runtime.runtime import RECALL_K, Runtime
from tests import test_runtime as fixtures


class RecallTests(unittest.TestCase):
    setUp = fixtures.RuntimeTests.setUp

    @staticmethod
    def answer(**overrides):
        base = {"answer": "收到", "appraisal": "平静", "used_memory_ids": [], "memories": []}
        base.update(overrides)
        return base

    def hidden_id(self, text="合作"):
        """Seeded library has four memories; the one outside the three-slot context."""
        top = [row["id"] for row in self.runtime.retrieve(text) if row["selected"]]
        return ({1, 2, 3, 4} - set(top)).pop()

    def test_recall_widens_the_context_and_its_memories_can_be_cited(self):
        self.runtime.seed()
        hidden = self.hidden_id()
        with patch.object(self.provider, "cognize", side_effect=[
                {"recall": "那次我们一起尝试合作，后来到底发生了什么"},
                self.answer(used_memory_ids=[hidden])]):
            trace = self.runtime.step("合作")
        self.assertEqual(len(trace["recalls"]), 1)
        self.assertEqual([row["id"] for row in trace["recalls"][0]["rows"]], [hidden])
        # Without the recall it was never offered, so citing it would have been rejected.
        self.assertNotIn(hidden, [row["id"] for row in trace["retrieval"] if row["selected"]])
        self.assertEqual(trace["cognition"]["used_memory_ids"], [hidden])
        stored = next(m for m in self.runtime.load()[1]["memories"] if m["id"] == hidden)
        self.assertEqual(stored["uses"], 1)

    def test_a_recalled_memory_can_be_replaced(self):
        self.runtime.seed()
        hidden = self.hidden_id()
        with patch.object(self.provider, "cognize", side_effect=[
                {"recall": "那次合作"},
                self.answer(memories=[{"content": "修正后的事实", "appraisal": "清楚了",
                                       "replaces": [hidden]}])]):
            trace = self.runtime.step("合作")
        self.assertEqual(trace["replaced_ids"], [hidden])
        self.assertNotIn(hidden, [m["id"] for m in self.runtime.load()[1]["memories"]])

    def test_recall_budget_bounds_the_number_of_requests(self):
        self.runtime.seed()
        # Ten memories: three in context and enough left over for two recalls.
        with patch.object(self.provider, "cognize", return_value=self.answer(
                memories=[{"content": f"填充 {i}", "appraisal": "平静"} for i in range(3)])):
            for _ in range(2):
                self.runtime.step("写几条")
        calls = []

        def fake(text, memories, recall_left=0, library_size=None):
            calls.append(recall_left)
            wanted = f"我想再想想第{len(calls)}次合作里对方说过什么"
            return {"recall": wanted} if len(calls) <= 2 else self.answer()

        with patch.object(self.provider, "cognize", side_effect=fake):
            trace = self.runtime.step("合作")
        self.assertEqual(calls, [2, 1, 0])
        self.assertEqual(len(trace["recalls"]), 2)
        self.assertEqual(trace["recall_limit"], 2)
        for recall in trace["recalls"]:
            self.assertTrue(recall["rows"])

    def test_asking_past_the_budget_fails_without_changing_state(self):
        self.runtime.seed()
        runtime = Runtime(self.path, self.provider, recall_limit=1)
        before = runtime.load()
        with patch.object(self.provider, "cognize", return_value={"recall": "再想想"}):
            with self.assertRaises(ValueError):
                runtime.step("合作")
        self.assertEqual(before, runtime.load())

    def test_zero_budget_rejects_the_first_request(self):
        self.runtime.seed()
        runtime = Runtime(self.path, self.provider, recall_limit=0)
        with patch.object(self.provider, "cognize", return_value={"recall": "想不起来"}):
            with self.assertRaises(ValueError):
                runtime.step("合作")

    def test_recall_request_cannot_carry_an_answer(self):
        self.runtime.seed()
        with patch.object(self.provider, "cognize",
                          return_value={"recall": "想", "answer": "顺便说一句"}):
            with self.assertRaises(ValueError):
                self.runtime.step("合作")

    def test_empty_library_recall_spends_the_budget_and_forces_an_answer(self):
        # Nothing to find: the runtime sets the budget to zero rather than asking again.
        calls = []

        def fake(text, memories, recall_left=0, library_size=None):
            calls.append((len(memories), recall_left))
            if len(calls) == 1:
                return {"recall": "任何往事"}
            return self.answer(answer="还没有可引用的往事")

        with patch.object(self.provider, "cognize", side_effect=fake):
            trace = self.runtime.step("你好")
        self.assertEqual(calls, [(0, 2), (0, 0)])
        self.assertEqual(trace["recalls"][0]["rows"], [])

    def test_trace_marks_which_route_admitted_each_memory(self):
        self.runtime.seed()
        hidden = self.hidden_id()
        with patch.object(self.provider, "cognize", side_effect=[
                {"recall": "那次合作"}, self.answer(used_memory_ids=[hidden])]):
            trace = self.runtime.step("合作")
        self.assertTrue(all(row["selected_by"] == "score"
                            for row in trace["retrieval"] if row["selected"]))
        self.assertTrue(all(row["selected_by"] == "recall"
                            for row in trace["recalls"][0]["rows"]))

    def test_recall_returns_at_most_k_new_memories(self):
        vector = self.provider.embed("记忆")
        memories = [{"id": i, "vector": vector, "affect": vector, "write_strength": .5,
                     "uses": 0, "recent_uses": []} for i in range(1, 13)]
        rows = self.runtime._recall(memories, "想找的事", None, True, {1, 2, 3}, None)
        self.assertEqual(len(rows), RECALL_K)
        self.assertFalse({row["id"] for row in rows} & {1, 2, 3})

    def test_invalid_budget_is_rejected(self):
        for bad in (-1, 1.0, True, "2"):
            with self.subTest(budget=bad):
                with self.assertRaises(ValueError):
                    Runtime(self.path, self.provider, recall_limit=bad)

    def test_offline_fixture_never_asks_to_recall(self):
        result = self.provider.cognize("合作", [], 2)
        self.assertNotIn("recall", result)

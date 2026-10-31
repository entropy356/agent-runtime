import copy
import unittest
from unittest.mock import patch

from tests import test_runtime as fixtures
from agent_runtime.providers import OfflineProvider
from agent_runtime.runtime import Runtime


class MemoryRelationTests(unittest.TestCase):
    setUp = fixtures.RuntimeTests.setUp
    database_dump = fixtures.RuntimeTests.database_dump

    def proposal(self, target):
        return {"answer": "已记录更新", "appraisal": "了解了新的情况",
                "used_memory_ids": [target], "memories": [{
                    "content": "更新后的事实", "appraisal": "收到明确更新",
                    "source_memory_ids": [target], "replaces": [target]}]}

    def target(self):
        self.runtime.seed()
        return self.runtime.retrieve("合作")[0]["id"]

    def test_replacement_removes_the_old_memory_and_keeps_only_a_debug_copy(self):
        self.runtime.seed()
        _, before = self.runtime.load()
        target = self.runtime.retrieve("合作")[0]["id"]
        content = next(m["content"] for m in before["memories"] if m["id"] == target)
        with patch.object(self.provider, "cognize", return_value=self.proposal(target)):
            trace = self.runtime.step("合作")
        _, state = self.runtime.load()
        replacement = state["memories"][-1]
        self.assertEqual([m["id"] for m in state["memories"]],
                         [m["id"] for m in before["memories"] if m["id"] != target]
                         + [replacement["id"]])
        self.assertFalse(any("superseded_by" in m for m in state["memories"]))
        self.assertEqual(trace["replaced_ids"], [target])
        self.assertEqual(replacement["replaces"], [target])
        self.assertEqual(replacement["source_memory_ids"], [target])
        self.assertEqual(replacement["created_step"], trace["step"])
        log = self.runtime.replaced_log()
        self.assertEqual([(r["id"], r["replaced_by"], r["content"]) for r in log],
                         [(target, replacement["id"], content)])

    def test_relations_require_visible_ids_and_integer_lists(self):
        target = self.target()
        before = self.database_dump()
        for field in ("source_memory_ids", "replaces"):
            for invalid in ([999], [True], "1", None):
                result = self.proposal(target)
                result["memories"][0][field] = invalid
                with self.subTest(field=field, invalid=invalid):
                    with patch.object(self.provider, "cognize", return_value=result):
                        with self.assertRaises(ValueError):
                            self.runtime.step("合作")
                    self.assertEqual(before, self.database_dump())
        visible = {m["id"] for m in self.runtime.retrieve("合作") if m["selected"]}
        hidden = ({1, 2, 3, 4} - visible).pop()
        result = self.proposal(target)
        result["memories"][0]["replaces"] = [hidden]
        with patch.object(self.provider, "cognize", return_value=result):
            with self.assertRaises(ValueError):
                self.runtime.step("合作")
        self.assertEqual(before, self.database_dump())

    def test_conflicting_replacements_roll_back(self):
        target = self.target()
        before = self.database_dump()
        result = self.proposal(target)
        result["memories"].append(copy.deepcopy(result["memories"][0]))
        with patch.object(self.provider, "cognize", return_value=result):
            with self.assertRaises(ValueError):
                self.runtime.step("合作")
        self.assertEqual(before, self.database_dump())
        self.assertEqual(self.runtime.replaced_log(), [])

    def test_failed_embedding_does_not_remove_the_old_memory(self):
        target = self.target()
        before = self.database_dump()
        original = self.provider.embed

        def fail_content(text, task=None):
            if text == "更新后的事实":
                raise ValueError("Embedding unavailable")
            return original(text, task)

        with patch.object(self.provider, "cognize", return_value=self.proposal(target)):
            with patch.object(self.provider, "embed", side_effect=fail_content):
                with self.assertRaises(ValueError):
                    self.runtime.step("合作")
        self.assertEqual(before, self.database_dump())
        self.assertEqual(self.runtime.replaced_log(), [])

    def test_old_state_and_snapshots_need_no_migration(self):
        self.runtime.seed()
        revision, state = self.runtime.load()
        fields = ("created_step", "created_revision", "origin", "source_memory_ids", "replaces")
        for memory in state["memories"]:
            for field in fields:
                memory.pop(field)
        self.runtime._commit(revision, state, snapshots={"legacy": state})
        self.runtime.restore("legacy")
        result = {"answer": "收到", "appraisal": "有所感受", "used_memory_ids": [],
                  "memories": [{"content": "旧格式的新记忆", "appraisal": "感受"}]}
        with patch.object(self.provider, "cognize", return_value=result):
            self.runtime.step("新输入")
        written = self.runtime.load()[1]["memories"][-1]
        self.assertEqual(written["replaces"], [])
        self.assertEqual(written["source_memory_ids"], [])
        self.assertEqual(written["origin"], "external_input")

    def test_restore_brings_back_a_replaced_memory(self):
        target = self.target()
        self.runtime.save("before-correction")
        with patch.object(self.provider, "cognize", return_value=self.proposal(target)):
            self.runtime.step("合作")
        self.runtime.save("after-correction")
        corrected = self.runtime.load()[1]
        self.assertNotIn(target, [m["id"] for m in corrected["memories"]])
        self.assertEqual(Runtime(self.path, OfflineProvider()).load()[1], corrected)
        self.runtime.restore("before-correction")
        self.assertIn(target, [m["id"] for m in self.runtime.load()[1]["memories"]])
        self.runtime.restore("after-correction")
        self.assertEqual(self.runtime.load()[1], corrected)

    def test_no_input_round_records_internal_origin_and_reported_sources(self):
        self.runtime.seed()
        trace = self.runtime.step()
        written = self.runtime.load()[1]["memories"][-1]
        self.assertEqual(written["origin"], "internal_cognition")
        self.assertEqual(written["source_memory_ids"], trace["cognition"]["used_memory_ids"])

    def test_one_proposal_may_replace_several_memories(self):
        self.runtime.seed()
        selected = [r["id"] for r in self.runtime.retrieve("合作") if r["selected"]]
        merged = {"answer": "合并", "appraisal": "理清了", "used_memory_ids": selected[:2],
                  "memories": [{"content": "合并后的事实", "appraisal": "清楚了",
                                "replaces": selected[:2], "source_memory_ids": selected[:2]}]}
        with patch.object(self.provider, "cognize", return_value=merged):
            trace = self.runtime.step("合作")
        state = self.runtime.load()[1]
        new_id = trace["written_ids"][0]
        self.assertEqual(state["memories"][-1]["replaces"], selected[:2])
        self.assertFalse({m["id"] for m in state["memories"]} & set(selected[:2]))
        self.assertEqual(sorted(r["id"] for r in self.runtime.replaced_log()), sorted(selected[:2]))
        self.assertTrue(all(r["replaced_by"] == new_id for r in self.runtime.replaced_log()))

    def test_two_proposals_may_replace_different_memories_in_one_round(self):
        self.runtime.seed()
        selected = [r["id"] for r in self.runtime.retrieve("合作") if r["selected"]]
        split = {"answer": "分别更新", "appraisal": "整理", "used_memory_ids": selected[:2],
                 "memories": [{"content": "A 的更新", "appraisal": "甲", "replaces": [selected[0]]},
                              {"content": "B 的更新", "appraisal": "乙", "replaces": [selected[1]]}]}
        with patch.object(self.provider, "cognize", return_value=split):
            trace = self.runtime.step("合作")
        ids = {m["id"] for m in self.runtime.load()[1]["memories"]}
        self.assertFalse(ids & set(selected[:2]))
        pairs = {r["id"]: r["replaced_by"] for r in self.runtime.replaced_log()}
        self.assertEqual(pairs, {selected[0]: trace["written_ids"][0],
                                 selected[1]: trace["written_ids"][1]})

    def test_a_replaced_memory_never_returns_in_later_rounds(self):
        self.runtime.seed()
        target = self.runtime.retrieve("合作")[0]["id"]
        with patch.object(self.provider, "cognize", return_value=self.proposal(target)):
            self.runtime.step("合作")
        quiet = {"answer": "好", "appraisal": "平静", "used_memory_ids": [], "memories": []}
        with patch.object(self.provider, "cognize", return_value=quiet):
            for _ in range(20):
                trace = self.runtime.step("合作")
        self.assertNotIn(target, [r["id"] for r in trace["retrieval"]])
        self.assertNotIn(target, [r["id"] for r in self.runtime.retrieve("合作")])
        self.assertEqual(len(self.runtime.replaced_log()), 1)

    def test_opening_a_database_with_tombstones_moves_them_out(self):
        self.runtime.seed()
        revision, state = self.runtime.load()
        state["memories"][0]["superseded_by"] = state["memories"][1]["id"]
        self.runtime._commit(revision, state)
        reopened = Runtime(self.path, OfflineProvider())
        self.assertNotIn(state["memories"][0]["id"],
                         [m["id"] for m in reopened.load()[1]["memories"]])
        self.assertEqual([(r["id"], r["replaced_by"]) for r in reopened.replaced_log()],
                         [(state["memories"][0]["id"], state["memories"][1]["id"])])

    def test_a_legacy_snapshot_cannot_resurrect_a_removed_memory(self):
        self.runtime.seed()
        revision, state = self.runtime.load()
        legacy = copy.deepcopy(state)
        legacy["memories"][0]["superseded_by"] = legacy["memories"][1]["id"]
        self.runtime._commit(revision, legacy, snapshots={"legacy": legacy})
        comparison = self.runtime.compare("合作", "legacy", "A", "B")
        self.assertNotIn(legacy["memories"][0]["id"],
                         [r["id"] for t in comparison["trials"] for r in t["retrieval"]])
        self.runtime.restore("legacy")
        self.assertNotIn(legacy["memories"][0]["id"],
                         [m["id"] for m in self.runtime.load()[1]["memories"]])
        self.assertEqual(self.runtime.replaced_log()[0]["id"], legacy["memories"][0]["id"])

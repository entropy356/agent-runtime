import copy
import unittest
from unittest.mock import patch

from tests import test_runtime as fixtures
from agent_runtime.providers import OfflineProvider
from agent_runtime.runtime import Runtime, rank


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

    def test_replacement_keeps_old_record_but_excludes_it_from_ranking_and_mean(self):
        target = self.target()
        with patch.object(self.provider, "cognize", return_value=self.proposal(target)):
            trace = self.runtime.step("合作")
        _, state = self.runtime.load()
        replacement = state["memories"][-1]
        old = next(m for m in state["memories"] if m["id"] == target)
        self.assertEqual(old["superseded_by"], replacement["id"])
        self.assertEqual(replacement["replaces"], [target])
        self.assertEqual(replacement["source_memory_ids"], [target])
        self.assertEqual(old["uses"], 1)
        self.assertEqual(replacement["created_step"], trace["step"])
        self.assertEqual(replacement["created_revision"], self.runtime.load()[0])
        query = self.provider.embed("合作")
        actual = rank(state["memories"], query, state["emotion"])
        active = [m for m in state["memories"] if m["id"] != target]
        self.assertEqual(actual, rank(active, query, state["emotion"]))
        self.assertNotIn(target, [m["id"] for m in actual])

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

    def test_failed_embedding_does_not_supersede_old_memory(self):
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

    def test_old_state_and_snapshots_need_no_migration(self):
        self.runtime.seed()
        revision, state = self.runtime.load()
        fields = ("created_step", "created_revision", "origin", "source_memory_ids",
                  "replaces", "superseded_by")
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

    def test_restore_and_restart_preserve_relation_state(self):
        target = self.target()
        self.runtime.save("before-correction")
        with patch.object(self.provider, "cognize", return_value=self.proposal(target)):
            self.runtime.step("合作")
        self.runtime.save("after-correction")
        corrected = self.runtime.load()[1]
        self.assertEqual(Runtime(self.path, OfflineProvider()).load()[1], corrected)
        self.runtime.restore("before-correction")
        self.assertIn(target, [m["id"] for m in self.runtime.retrieve("合作")])
        self.runtime.restore("after-correction")
        self.assertEqual(self.runtime.load()[1], corrected)

    def test_no_input_round_records_internal_origin_and_reported_sources(self):
        self.runtime.seed()
        trace = self.runtime.step()
        written = self.runtime.load()[1]["memories"][-1]
        self.assertEqual(written["origin"], "internal_cognition")
        self.assertEqual(written["source_memory_ids"], trace["cognition"]["used_memory_ids"])


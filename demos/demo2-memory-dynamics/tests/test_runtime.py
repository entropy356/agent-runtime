import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from memory_dynamics.model import DynamicsConfig
from memory_dynamics.providers import OfflineProvider
from memory_dynamics.runtime import Runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "state.sqlite3"
        self.provider = OfflineProvider()
        self.config = DynamicsConfig(
            decay_rate=0.1,
            stability_protection=0.8,
            recovery_rate=0.5,
            stability_gain=0.2,
            top_k=2,
        )
        self.runtime = Runtime(self.path, self.provider, self.config)
        self.runtime.initialize([
            "Mira prefers short written updates.",
            "The spare key is in the blue tin.",
            "Lin expects the draft on Friday.",
        ])

    def database_dump(self):
        with sqlite3.connect(self.path) as db:
            return list(db.iterdump())

    def result(self, used=None, memories=None):
        return {
            "answer": "test answer",
            "used_memory_ids": used or [],
            "memories": memories or [],
        }

    def test_only_memories_actually_used_are_reinforced(self):
        with patch.object(self.provider, "cognize", return_value=self.result([1])):
            trace = self.runtime.step("question")
        state = self.runtime.inspect()["state"]
        first, second = state["memories"][:2]
        self.assertEqual(trace["activated"][0]["id"], 1)
        self.assertGreater(first["weight"], second["weight"])
        self.assertGreater(first["stability"], 0.0)
        self.assertEqual(second["stability"], 0.0)
        self.assertEqual(second["activated_count"], 0)

    def test_successful_recall_changes_future_decay_rate(self):
        with patch.object(self.provider, "cognize", return_value=self.result([1])):
            self.runtime.step("question")
        with patch.object(self.provider, "cognize", return_value=self.result([])):
            trace = self.runtime.step("unrelated")
        rates = {item["id"]: item["effective_decay_rate"] for item in trace["decayed"]}
        self.assertLess(rates[1], rates[2])

    def test_new_memory_starts_fresh_and_is_not_aged_in_creation_round(self):
        proposal = [{"content": "A newly learned fact."}]
        with patch.object(self.provider, "cognize", return_value=self.result([], proposal)):
            trace = self.runtime.step("new fact")
        memory = self.runtime.inspect()["state"]["memories"][-1]
        self.assertEqual(trace["written_ids"], [4])
        self.assertEqual(memory["weight"], 1.0)
        self.assertEqual(memory["stability"], 0.0)
        self.assertEqual(memory["created_step"], 1)

    def test_tick_without_input_still_advances_and_decays(self):
        before = self.runtime.inspect()["state"]["memories"][0]["weight"]
        trace = self.runtime.step()
        after = self.runtime.inspect()["state"]["memories"][0]["weight"]
        self.assertIsNone(trace["input"])
        self.assertLess(after, before)

    def test_read_only_retrieval_does_not_age_memories(self):
        before = self.database_dump()
        self.runtime.retrieve("Mira update")
        self.assertEqual(before, self.database_dump())

    def test_invalid_citation_does_not_commit_any_state(self):
        before = self.database_dump()
        bad = self.result([999])
        with patch.object(self.provider, "cognize", return_value=bad):
            with self.assertRaises(ValueError):
                self.runtime.step("question")
        self.assertEqual(before, self.database_dump())

    def test_embedding_failure_after_cognition_does_not_commit_any_state(self):
        before = self.database_dump()
        original = self.provider.embed
        calls = 0

        def fail_on_new_memory(text, task=None):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("embedding service failed")
            return original(text, task)

        proposal = [{"content": "A new fact."}]
        with patch.object(self.provider, "cognize", return_value=self.result([], proposal)):
            with patch.object(self.provider, "embed", side_effect=fail_on_new_memory):
                with self.assertRaises(ValueError):
                    self.runtime.step("question")
        self.assertEqual(before, self.database_dump())

    def test_restart_preserves_memory_state_and_trace(self):
        with patch.object(self.provider, "cognize", return_value=self.result([1])):
            self.runtime.step("question")
        expected = self.runtime.inspect()
        reopened = Runtime(self.path, self.provider, self.config).inspect()
        self.assertEqual(expected, reopened)

    def test_database_rejects_changed_embedding_profile_or_config(self):
        other = OfflineProvider()
        other.profile = "different-embedding-space"
        with self.assertRaises(ValueError):
            Runtime(self.path, other, self.config)
        changed = DynamicsConfig(decay_rate=0.3)
        with self.assertRaises(ValueError):
            Runtime(self.path, self.provider, changed)


if __name__ == "__main__":
    unittest.main()

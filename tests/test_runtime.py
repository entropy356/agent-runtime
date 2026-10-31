import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agent_runtime.providers import APIProvider, OfflineProvider, unit
from agent_runtime.runtime import Runtime, dot, rank, sigmoid, tier


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "state.sqlite3"
        self.provider = OfflineProvider()
        self.runtime = Runtime(self.path, self.provider)

    def database_dump(self):
        with sqlite3.connect(self.path) as db:
            return list(db.iterdump())

    def test_read_and_counterfactuals_are_read_only(self):
        self.runtime.seed()
        before = self.database_dump()
        self.runtime.retrieve("我们再一起试一次吧。")
        self.runtime.compare("我们再一起试一次吧。", "baseline", "A", "B")
        self.assertEqual(before, self.database_dump())

    def test_ab_isolates_emotion_and_ablation_removes_difference(self):
        self.runtime.seed()
        result = self.runtime.compare("我们再一起试一次吧。", "baseline", "A", "B")
        a, b, off_a, off_b = result["trials"]
        self.assertEqual(a["retrieval"][0]["id"], 1)
        self.assertEqual(b["retrieval"][0]["id"], 2)
        self.assertNotEqual(a["cognition"]["answer"], b["cognition"]["answer"])
        def projection(trial):
            return [(m["id"], m["score"], m["selected"]) for m in trial["retrieval"]]
        self.assertEqual(projection(off_a), projection(off_b))
        self.assertEqual(off_a["cognition"], off_b["cognition"])
        for trial in (a, b, off_a, off_b):
            by_id = {m["id"]: m for m in trial["retrieval"]}
            self.assertEqual(set(by_id), {1, 2, 3, 4})
            self.assertTrue(all(m["uses"] == 0 and m["write_strength"] == .5 for m in by_id.values()))
            self.assertEqual(sum(m["selected"] for m in by_id.values()), 3)

    def test_restart_restores_every_persisted_state_field(self):
        self.runtime.seed()
        self.runtime.step("我们再一起试一次吧。")
        before = self.runtime.inspect()
        reopened = Runtime(self.path, OfflineProvider()).inspect()
        self.assertEqual(before, reopened)

    def test_updates_follow_inertia_and_only_cited_ids_count(self):
        self.runtime.seed()
        _, before = self.runtime.load()
        trace = self.runtime.step("我们再一起试一次吧。")
        _, after = self.runtime.load()
        feeling = unit(self.provider.embed(trace["cognition"]["appraisal"]))
        for index, value in enumerate(after["emotion"]):
            self.assertAlmostEqual(value, .85 * before["emotion"][index] + .15 * feeling[index])
        used = trace["cognition"]["used_memory_ids"]
        for memory in after["memories"][:4]:
            self.assertEqual(memory["uses"], int(memory["id"] in used))
        written = after["memories"][-1]
        self.assertAlmostEqual(written["write_strength"], sigmoid(3 * dot(after["emotion"], written["affect"])))
        self.assertEqual(written["uses"], 0)

    def test_first_tick_initializes_from_cognition_without_zero_baseline(self):
        self.assertIsNone(self.runtime.load()[1]["emotion"])
        trace = self.runtime.step()
        self.assertIsNone(trace["input"])
        self.assertEqual(trace["emotion_after"], unit(self.provider.embed(trace["cognition"]["appraisal"])))

    def test_restore_is_complete_and_keeps_audit_and_other_snapshots(self):
        self.runtime.seed()
        self.runtime.step("新的事情发生了")
        self.runtime.save("later")
        self.runtime.restore("B")
        self.assertEqual(self.runtime.load()[1], self.runtime.snapshot("B"))
        self.assertEqual(len(self.runtime.inspect()["traces"]), 1)
        self.assertIn("later", self.runtime.inspect()["snapshots"])

    def test_invalid_cognition_never_commits(self):
        self.runtime.seed()
        before = self.database_dump()
        bad = {"answer": "invalid", "appraisal": "感受", "used_memory_ids": [999], "memories": []}
        with patch.object(self.provider, "cognize", return_value=bad):
            with self.assertRaises(ValueError):
                self.runtime.step("输入")
        self.assertEqual(before, self.database_dump())

    def test_late_embedding_failure_never_commits(self):
        self.runtime.seed()
        before = self.database_dump()
        original = self.provider.embed
        calls = 0
        def fail_late(text):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise ValueError("network failed on proposed memory content")
            return original(text)
        with patch.object(self.provider, "embed", side_effect=fail_late):
            with self.assertRaises(ValueError):
                self.runtime.step("输入")
        self.assertEqual(before, self.database_dump())

    def test_embedding_dimension_change_never_commits(self):
        self.runtime.seed()
        before = self.database_dump()
        with patch.object(self.provider, "embed", return_value=[1, 0]):
            with self.assertRaises(ValueError):
                self.runtime.step("输入")
        self.assertEqual(before, self.database_dump())

    def test_provider_change_rejected(self):
        provider = OfflineProvider()
        provider.profile = "different-space"
        with self.assertRaises(ValueError):
            Runtime(self.path, provider)

    def test_duplicate_snapshot_rolls_back_revision(self):
        self.runtime.save("one")
        before = self.database_dump()
        with self.assertRaises(sqlite3.IntegrityError):
            self.runtime.save("one")
        self.assertEqual(before, self.database_dump())

    def test_seed_cannot_overwrite_work(self):
        self.runtime.step("保留这次经历")
        before = self.database_dump()
        with self.assertRaises(ValueError):
            self.runtime.seed()
        self.assertEqual(before, self.database_dump())

    def test_stale_writer_cannot_overwrite_new_state(self):
        revision, old = self.runtime.load()
        self.runtime.step("新的状态")
        before = self.database_dump()
        with self.assertRaises(ValueError):
            self.runtime._commit(revision, old)
        self.assertEqual(before, self.database_dump())

    def test_retrieval_is_pure_and_layer_is_usage_only(self):
        self.runtime.seed()
        _, state = self.runtime.load()
        before = copy.deepcopy(state)
        rank(state["memories"], self.provider.embed("合作"), state["emotion"])
        self.assertEqual(before, state)
        self.assertEqual([tier(i) for i in (0, 2, 3, 7, 8)], ["Cold", "Cold", "Warm", "Warm", "Hot"])

    def test_invalid_vectors_rejected(self):
        for vector in ([], [0, 0], [float("nan")], [float("inf")], [True], ["x"]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                unit(vector)


class APIAdapterTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {
            "AGENT_API_BASE": "https://example.invalid/v1", "AGENT_API_KEY": "test-secret",
            "AGENT_CHAT_MODEL": "chat", "AGENT_EMBEDDING_MODEL": "embedding"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.provider = APIProvider()

    def test_cognition_prompt_does_not_leak_emotion_or_full_history(self):
        memory = {"id": 1, "content": "可见内容", "appraisal": "不应进入提示", "affect": [1234]}
        response = {"choices": [{"message": {"content": json.dumps({"answer": "ok"})}}]}
        with patch.object(self.provider, "_post", return_value=response) as post:
            self.provider.cognize("当前输入", [memory])
        payload = post.call_args.args[1]
        user = json.loads(payload["messages"][1]["content"])
        self.assertEqual(user, {"input": "当前输入", "memories": [{"id": 1, "content": "可见内容"}]})
        self.assertNotIn("test-secret", self.provider.profile)

    def test_malformed_provider_output_is_rejected(self):
        with patch.object(self.provider, "_post", return_value={"choices": [{"message": {"content": "not JSON"}}]}):
            with self.assertRaises(ValueError):
                self.provider.cognize("输入", [])

    def test_embedding_response_normalized_and_validated(self):
        with patch.object(self.provider, "_post", return_value={"data": [{"embedding": [3, 4]}]}):
            self.assertEqual(self.provider.embed("输入"), [.6, .8])


if __name__ == "__main__":
    unittest.main()

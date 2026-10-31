import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agent_runtime.providers import APIProvider, OfflineProvider, unit
from agent_runtime.runtime import Runtime, centered, centroid, dot, rank, sigmoid


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
        # Which id wins inside a positive/negative group is fixture-dependent; the
        # property that matters is that swapping emotion swaps which memory surfaces.
        self.assertGreater(a["retrieval"][0]["emotion_match"], 0)
        self.assertNotEqual(a["retrieval"][0]["id"], b["retrieval"][0]["id"])
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

    def test_centering_separates_memories_raw_cosine_cannot(self):
        # Eleven memories share the dominant direction, one also carries the query's
        # distinctive direction. Raw cosine margins ~1e-3; centered margins ~1.
        def make(index, extra=0.0):
            vector = [1.0] * 768
            vector[index] += .05
            vector[600] += extra
            return unit(vector)
        memories = [{"id": i + 1, "vector": make(i), "affect": make(i),
                     "write_strength": 1.0, "uses": 0} for i in range(11)]
        memories.append({"id": 12, "vector": make(600, .9), "affect": make(600, .9),
                         "write_strength": 1.0, "uses": 0})
        query = make(600, .9)
        centered_scores = {row["id"]: row["relevance"] for row in rank(memories, query, None)}
        raw_scores = {memory["id"]: dot(query, memory["vector"]) for memory in memories}
        winner = max(centered_scores, key=centered_scores.get)
        runner_up = sorted(centered_scores.values())[-2]
        self.assertEqual(winner, 12)
        self.assertGreater(centered_scores[12] - runner_up, .75)
        self.assertLess(max(raw_scores.values()) - sorted(raw_scores.values())[-2], .01)

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
        prior = centroid([memory["affect"] for memory in after["memories"][:-1]])
        aligned = dot(centered(after["emotion"], prior), centered(written["affect"], prior))
        self.assertAlmostEqual(written["write_strength"], sigmoid(3 * aligned))
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

    def test_dot_rejects_mismatched_dimensions(self):
        with self.assertRaises(ValueError):
            dot([1.0], [1.0, 0.0])

    def test_cognition_shape_errors_never_commit(self):
        self.runtime.seed()
        before = self.database_dump()
        base = {"answer": "ok", "appraisal": "感受", "used_memory_ids": [], "memories": []}
        cases = {
            "not an object": ["not", "a", "dict"],
            "four proposals": {**base, "memories": [{"content": "a", "appraisal": "b"}] * 4},
            "proposal is not an object": {**base, "memories": ["text"]},
        }
        for label, result in cases.items():
            with self.subTest(label):
                with patch.object(self.provider, "cognize", return_value=result):
                    with self.assertRaises(ValueError):
                        self.runtime.step("输入")
                self.assertEqual(before, self.database_dump())

    def test_invalid_inputs_and_missing_snapshots_are_rejected(self):
        self.runtime.save("light")
        self.runtime.seed()
        before = self.database_dump()
        for text in ("", "   ", 5):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    self.runtime.step(text)
        self.assertEqual(before, self.database_dump())
        with self.assertRaises(ValueError):
            Runtime(self.path, OfflineProvider(), rho=1.0)
        with self.assertRaises(ValueError):
            self.runtime.restore("missing")
        # "light" was saved before the first tick, so it has no emotion to compare with.
        with self.assertRaises(ValueError):
            self.runtime.compare("输入", "baseline", "light", "B")
        self.assertEqual(before, self.database_dump())

    def test_late_embedding_failure_never_commits(self):
        self.runtime.seed()
        before = self.database_dump()
        original = self.provider.embed
        calls = 0
        def fail_late(text, task=None):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise ValueError("network failed on proposed memory content")
            return original(text, task)
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

    def test_retrieval_is_pure_and_citation_history_is_inert(self):
        self.runtime.seed()
        _, state = self.runtime.load()
        before = copy.deepcopy(state)
        rank(state["memories"], self.provider.embed("合作"), state["emotion"])
        self.assertEqual(before, state)
        # Citation history is recorded but must not reach the ranking.
        cited = [dict(memory, recent_uses=list(range(16))) for memory in state["memories"]]
        project = lambda rows: [(r["id"], r["score"], r["selected"]) for r in rows]
        self.assertEqual(project(rank(cited, None, None, step=16)),
                         project(rank(state["memories"], None, None, step=16)))

    def test_rank_rejects_non_integer_budgets(self):
        for bad in (-1, True, 1.0, "3", None):
            with self.subTest(top_k=bad), self.assertRaises(ValueError):
                rank([], None, None, top_k=bad)

    def test_a_memory_equal_to_the_mean_carries_no_signal(self):
        self.assertEqual(centered([1., 0.], [1., 0.]), [0., 0.])
        self.assertEqual(centered([1., 0.], None), [1., 0.])

    def test_invalid_vectors_rejected(self):
        for vector in ([], [0, 0], [float("nan")], [float("inf")], [True], ["x"]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                unit(vector)


class APIAdapterTests(unittest.TestCase):
    """Chat and embedding may come from different vendors, protocols and keys."""

    def setUp(self):
        self.env = patch.dict("os.environ", {
            "AGENT_API_BASE": "", "AGENT_API_KEY": "", "AGENT_EMBEDDING_MODEL": "",
            "AGENT_CHAT_BASE": "https://chat.invalid/v1", "AGENT_CHAT_KEY": "chat-secret",
            "AGENT_CHAT_MODEL": "chat-model",
            "AGENT_EMBED_PROTOCOL": "openai", "AGENT_EMBED_BASE": "https://embed.invalid/v1",
            "AGENT_EMBED_KEY": "embed-secret", "AGENT_EMBED_MODEL": "embed-model",
            "AGENT_EMBED_DIMENSIONS": "768"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.provider = APIProvider()

    def test_cognition_prompt_does_not_leak_emotion_or_full_history(self):
        memory = {"id": 1, "content": "可见内容", "appraisal": "不应进入提示", "affect": [1234]}
        response = {"choices": [{"message": {"content": json.dumps({"answer": "ok"})}}]}
        with patch("agent_runtime.providers._request", return_value=response) as request:
            self.provider.cognize("当前输入", [memory])
        payload = request.call_args.args[1]
        user = json.loads(payload["messages"][1]["content"])
        self.assertEqual(user, {"input": "当前输入", "memories": [{
            "id": 1, "content": "可见内容", "created_step": None, "origin": "unknown",
            "source_memory_ids": [], "replaces": []}]})
        self.assertEqual(request.call_args.args[2], {"Authorization": "Bearer chat-secret"})
        self.assertNotIn("chat-secret", self.provider.profile)
        self.assertNotIn("embed-secret", self.provider.profile)

    def test_malformed_provider_output_is_rejected(self):
        with patch("agent_runtime.providers._request", return_value={"choices": [{"message": {"content": "not JSON"}}]}):
            with self.assertRaises(ValueError):
                self.provider.cognize("输入", [])

    def test_cognition_recovers_fenced_json(self):
        fenced = '```json\n{"answer": "ok"}\n```'
        with patch("agent_runtime.providers._request", return_value={"choices": [{"message": {"content": fenced}}]}):
            self.assertEqual(self.provider.cognize("输入", []), {"answer": "ok"})

    def test_embedding_response_normalized_and_validated(self):
        with patch("agent_runtime.providers._request", return_value={"data": [{"embedding": [3, 4]}]}) as request:
            self.assertEqual(self.provider.embed("输入"), [.6, .8])
        self.assertEqual(request.call_args.args[0], "https://embed.invalid/v1/embeddings")
        self.assertEqual(request.call_args.args[1]["dimensions"], 768)

    def test_gemini_protocol_uses_native_endpoint_and_task_type(self):
        with patch.dict("os.environ", {"AGENT_EMBED_PROTOCOL": "gemini",
                                       "AGENT_EMBED_BASE": "https://gemini.invalid/v1beta",
                                       "AGENT_EMBED_MODEL": "gemini-embedding-001"}):
            provider = APIProvider()
        with patch("agent_runtime.providers._request",
                   return_value={"embedding": {"values": [0, 5]}}) as request:
            self.assertEqual(provider.embed("文档", "document"), [0.0, 1.0])
        url, payload, headers = request.call_args.args[:3]
        self.assertEqual(url, "https://gemini.invalid/v1beta/models/gemini-embedding-001:embedContent")
        self.assertEqual(payload["taskType"], "RETRIEVAL_DOCUMENT")
        self.assertEqual(payload["outputDimensionality"], 768)
        self.assertEqual(headers, {"x-goog-api-key": "embed-secret"})
        self.assertIn("gemini:gemini-embedding-001:768", provider.profile)

    def test_profile_tracks_embedding_not_chat_model(self):
        with patch.dict("os.environ", {"AGENT_CHAT_MODEL": "another-chat"}):
            self.assertEqual(APIProvider().profile, self.provider.profile)


if __name__ == "__main__":
    unittest.main()

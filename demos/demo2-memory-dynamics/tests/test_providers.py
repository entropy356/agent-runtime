import json
import unittest
from unittest.mock import patch

from memory_dynamics.providers import APIProvider, unit


class APIProviderTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {
            "AGENT_API_BASE": "",
            "AGENT_API_KEY": "",
            "AGENT_CHAT_BASE": "https://chat.invalid/v1",
            "AGENT_CHAT_KEY": "chat-secret",
            "AGENT_CHAT_MODEL": "chat-model",
            "AGENT_EMBED_PROTOCOL": "openai",
            "AGENT_EMBED_BASE": "https://embed.invalid/v1",
            "AGENT_EMBED_KEY": "embed-secret",
            "AGENT_EMBED_MODEL": "embed-model",
            "AGENT_EMBED_DIMENSIONS": "768",
        }, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.provider = APIProvider()

    def test_embedding_is_normalized_and_uses_separate_endpoint(self):
        with patch("memory_dynamics.providers._request",
                   return_value={"data": [{"embedding": [3, 4]}]}) as request:
            self.assertEqual(self.provider.embed("query", "query"), [0.6, 0.8])
        self.assertEqual(request.call_args.args[0], "https://embed.invalid/v1/embeddings")
        self.assertEqual(request.call_args.args[1]["dimensions"], 768)
        self.assertEqual(request.call_args.args[2], {"Authorization": "Bearer embed-secret"})

    def test_gemini_embedding_uses_task_type(self):
        with patch.dict("os.environ", {
            "AGENT_EMBED_PROTOCOL": "gemini",
            "AGENT_EMBED_BASE": "https://gemini.invalid/v1beta",
            "AGENT_EMBED_MODEL": "gemini-embedding-001",
        }):
            provider = APIProvider()
        with patch("memory_dynamics.providers._request",
                   return_value={"embedding": {"values": [0, 5]}}) as request:
            self.assertEqual(provider.embed("memory", "document"), [0.0, 1.0])
        url, payload, headers = request.call_args.args[:3]
        self.assertEqual(
            url,
            "https://gemini.invalid/v1beta/models/gemini-embedding-001:embedContent",
        )
        self.assertEqual(payload["taskType"], "RETRIEVAL_DOCUMENT")
        self.assertEqual(payload["outputDimensionality"], 768)
        self.assertEqual(headers, {"x-goog-api-key": "embed-secret"})

    def test_chat_receives_only_memory_content_not_internal_weights(self):
        response = {"choices": [{"message": {"content": json.dumps({
            "answer": "ok", "used_memory_ids": [2], "memories": []
        })}}]}
        memories = [{
            "id": 2, "content": "visible content",
            "weight": 0.01, "stability": 0.99, "vector": [123.0],
        }]
        with patch.object(self.provider, "_chat", return_value=response) as chat:
            result = self.provider.cognize("input", memories)
        self.assertEqual(result["answer"], "ok")
        payload = chat.call_args.args[0]
        user = json.loads(payload["messages"][1]["content"])
        self.assertEqual(user, {
            "input": "input",
            "memories": [{"id": 2, "content": "visible content"}],
        })
        self.assertNotIn("chat-secret", self.provider.profile)
        self.assertNotIn("embed-secret", self.provider.profile)

    def test_embedding_vectors_are_validated(self):
        for vector in ([], [0, 0], [float("nan")], [float("inf")], [True], ["x"]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                unit(vector)

    def test_embedding_profile_is_independent_of_chat_model(self):
        with patch.dict("os.environ", {"AGENT_CHAT_MODEL": "different-chat-model"}):
            self.assertEqual(APIProvider().profile, self.provider.profile)


if __name__ == "__main__":
    unittest.main()

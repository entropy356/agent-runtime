import json
from pathlib import Path
import re
import select
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.process = subprocess.Popen([
            sys.executable, "-u", "-m", "agent_runtime", "--db",
            str(Path(self.temp.name) / "http.sqlite3"), "serve", "--port", "0", "--interval", "0.25"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.stop)
        ready, _, _ = select.select([self.process.stdout], [], [], 5)
        self.assertTrue(ready, "Server did not start")
        line = self.process.stdout.readline()
        match = re.search(r"http://127.0.0.1:(\d+)", line)
        self.assertIsNotNone(match, line)
        self.base = match.group(0)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def stop(self):
        self.process.terminate()
        self.process.communicate(timeout=5)

    def request(self, path, data=None, headers=None):
        request = urllib.request.Request(self.base + path,
            None if data is None else json.dumps(data).encode(), headers or {})
        try:
            response = self.opener.open(request, timeout=5)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            body = response.read().decode()
            return response.status, json.loads(body) if path != "/" else body

    def test_dashboard_and_full_interaction_flow(self):
        status, html = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("情绪如何改变记忆调用", html)
        self.assertIn("/api/", html)
        self.assertEqual(self.request("/api/seed", {})[0], 200)
        self.assertEqual(self.request("/api/step", {"text": "再试一次"})[0], 200)
        self.assertEqual(self.request("/api/save", {"name": "after"})[0], 200)
        before = self.request("/api/state")[1]
        status, result = self.request("/api/compare", {
            "text": "我们再一起试一次吧。", "baseline": "baseline", "a": "A", "b": "B"})
        self.assertEqual(status, 200)
        self.assertEqual(len(result["trials"]), 4)
        self.assertNotIn("tiered", result)
        self.assertIn("usage_window", result)
        self.assertEqual(before, self.request("/api/state")[1])
        self.assertEqual(self.request("/api/restore", {"name": "B"})[0], 200)
        restored = self.request("/api/state")[1]
        self.assertEqual(restored["state"]["step"], 0)
        self.assertIn("after", restored["snapshots"])
        self.assertEqual(len(restored["traces"]), 1)

    def test_clock_runs_without_replaying_user_input_and_can_pause(self):
        self.assertEqual(self.request("/api/clock", {"running": True})[0], 200)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = self.request("/api/state")[1]
            if state["state"]["step"]:
                break
            time.sleep(.03)
        else:
            self.fail("Clock failed to run")
        self.assertEqual(self.request("/api/clock", {"running": False})[0], 200)
        paused = self.request("/api/state")[1]
        self.assertTrue(all(trace["input"] is None for trace in paused["traces"]))
        time.sleep(.3)
        self.assertEqual(paused, self.request("/api/state")[1])

    def test_invalid_requests_and_foreign_origins_do_not_write(self):
        before = self.request("/api/state")[1]
        self.assertEqual(self.request("/api/step", {"text": ""})[0], 400)
        self.assertEqual(self.request("/api/step", {"text": "hi", "enabled": "false"})[0], 400)
        self.assertEqual(self.request("/api/step", {"text": "hi"}, {"Origin": "https://other.example"})[0], 403)
        self.assertEqual(self.request("/api/step", {"text": "hi"}, {"Host": "other.example"})[0], 403)
        self.assertEqual(self.request("/api/missing", {})[0], 404)
        self.assertEqual(before, self.request("/api/state")[1])


if __name__ == "__main__":
    unittest.main()

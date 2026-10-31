"""Pure retrieval, validated cognition, then one atomic state commit."""
import copy
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3

from .providers import unit


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("Embedding dimensions changed; use a new database")
    return sum(x * y for x, y in zip(a, b))


def sigmoid(value):
    return 1 / (1 + math.exp(-max(-60, min(60, value))))


def tier(count):
    return "Hot" if count >= 8 else "Warm" if count >= 3 else "Cold"


def require_text(value, field, maximum=6000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{field} must be nonempty text, at most {maximum} characters")
    return value.strip()


def rank(memories, query, emotion, enabled=True, top_k=3):
    """No mutations. Return all score components and a deterministic selection."""
    rows = []
    for memory in memories:
        relevance = max(0.0, dot(query, memory["vector"])) if query is not None else 1.0
        match = dot(emotion, memory["affect"]) if emotion is not None else 0.0
        read = sigmoid(4 * match) if enabled else 0.5
        score = relevance * read * memory["write_strength"]
        rows.append({**copy.deepcopy(memory), "tier": tier(memory["uses"]),
                     "relevance": relevance, "emotion_match": match,
                     "read_strength": read, "score": score})
    rows.sort(key=lambda row: (-row["score"], row["id"]))
    for index, row in enumerate(rows):
        row["selected"] = index < top_k
    return rows


def validate_cognition(result, selected):
    if not isinstance(result, dict):
        raise ValueError("Cognition must return an object")
    answer = require_text(result.get("answer"), "answer")
    appraisal = require_text(result.get("appraisal"), "appraisal")
    used = result.get("used_memory_ids")
    allowed = {memory["id"] for memory in selected}
    if not isinstance(used, list) or any(type(i) is not int or i not in allowed for i in used):
        raise ValueError("Cognition cited an unavailable memory; state was not changed")
    memories = result.get("memories")
    if not isinstance(memories, list) or len(memories) > 3:
        raise ValueError("Cognition may propose at most three memories")
    new_memories = []
    for memory in memories:
        if not isinstance(memory, dict):
            raise ValueError("Invalid proposed memory")
        new_memories.append({"content": require_text(memory.get("content"), "memory.content"),
                             "appraisal": require_text(memory.get("appraisal"), "memory.appraisal")})
    return {"answer": answer, "appraisal": appraisal,
            "used_memory_ids": sorted(set(used)), "memories": new_memories}


class Runtime:
    def __init__(self, database, provider, rho=0.85):
        if not 0 <= rho < 1:
            raise ValueError("rho must be in [0, 1)")
        self.path, self.provider, self.rho = str(database), provider, rho
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS runtime (id INTEGER PRIMARY KEY CHECK(id=1), profile TEXT NOT NULL, revision INTEGER NOT NULL, state TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS snapshots (name TEXT PRIMARY KEY, state TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS rounds (id INTEGER PRIMARY KEY, trace TEXT NOT NULL)")
            initial = {"step": 0, "emotion": None, "appraisal": "", "memories": [], "next_id": 1}
            db.execute("INSERT OR IGNORE INTO runtime VALUES (1, ?, 0, ?)", (provider.profile, json.dumps(initial)))
            if db.execute("SELECT profile FROM runtime").fetchone()[0] != provider.profile:
                raise ValueError("Embedding provider/model differs from this database; use --db with a new path")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def load(self):
        with self.connect() as db:
            revision, state = db.execute("SELECT revision, state FROM runtime WHERE id=1").fetchone()
        return revision, json.loads(state)

    def _commit(self, revision, state, trace=None, snapshots=None):
        with self.connect() as db:
            changed = db.execute("UPDATE runtime SET revision=revision+1, state=? WHERE id=1 AND revision=?",
                                 (json.dumps(state, ensure_ascii=False, allow_nan=False), revision)).rowcount
            if changed != 1:
                raise ValueError("State changed during this operation; retry against the current state")
            if trace is not None:
                db.execute("INSERT INTO rounds(trace) VALUES (?)", (json.dumps(trace, ensure_ascii=False, allow_nan=False),))
            for name, snapshot in (snapshots or {}).items():
                db.execute("INSERT INTO snapshots VALUES (?, ?)", (name, json.dumps(snapshot, ensure_ascii=False)))

    def _embed(self, text, dimension=None, task=None):
        # task lets asymmetric candidates (Gemini) separate query from document space.
        vector = unit(self.provider.embed(text, task))
        if dimension is not None and len(vector) != dimension:
            raise ValueError("Embedding dimensions changed; state was not changed")
        return vector

    @staticmethod
    def dimension(state):
        if state["emotion"] is not None:
            return len(state["emotion"])
        if state["memories"]:
            return len(state["memories"][0]["vector"])
        return None

    def inspect(self):
        revision, state = self.load()
        with self.connect() as db:
            names = [row[0] for row in db.execute("SELECT name FROM snapshots ORDER BY name")]
            traces = [json.loads(row[0]) for row in db.execute("SELECT trace FROM rounds ORDER BY id DESC LIMIT 20")]
        for memory in state["memories"]:
            memory["tier"] = tier(memory["uses"])
        return {"revision": revision, "state": state, "snapshots": names,
                "traces": traces, "mode": self.provider.mode, "rho": self.rho,
                "profile": self.provider.profile}

    def retrieve(self, text, enabled=True):
        text = require_text(text, "input", 2000)
        _, state = self.load()
        query = self._embed(text, self.dimension(state), "query")
        return rank(state["memories"], query, state["emotion"], enabled)

    def step(self, text=None, enabled=True):
        # None is a clock tick with no new external input.
        if text is not None:
            text = require_text(text, "input", 2000)
        revision, state = self.load()
        dimension = self.dimension(state)
        query = self._embed(text, dimension, "query") if text is not None else None
        dimension = len(query) if query is not None else dimension
        before = copy.deepcopy(state["emotion"])
        rows = rank(state["memories"], query, before, enabled)
        selected = [row for row in rows if row["selected"]]
        result = validate_cognition(self.provider.cognize(text, copy.deepcopy(selected)), selected)
        feeling = self._embed(result["appraisal"], dimension)
        dimension = len(feeling)
        after = feeling if before is None else [self.rho * a + (1 - self.rho) * b for a, b in zip(before, feeling)]
        # Finish ALL provider calls before changing state. Each memory has its own affect.
        prepared = []
        for proposal in result["memories"]:
            affect = self._embed(proposal["appraisal"], dimension)
            prepared.append({**proposal, "vector": self._embed(proposal["content"], dimension, "document"),
                             "affect": affect, "write_strength": sigmoid(3 * dot(after, affect)), "uses": 0})
        for memory in state["memories"]:
            if memory["id"] in result["used_memory_ids"]:
                memory["uses"] += 1
        for memory in prepared:
            memory["id"] = state["next_id"]
            state["next_id"] += 1
            state["memories"].append(memory)
        state.update(emotion=after, appraisal=result["appraisal"], step=state["step"] + 1)
        trace = {"step": state["step"], "input": text, "enabled": enabled,
                 "emotion_before": before, "emotion_after": after, "rho": self.rho,
                 "retrieval": rows, "cognition": result,
                 "written_ids": [m["id"] for m in prepared], "mode": self.provider.mode}
        self._commit(revision, state, trace)
        return trace

    def save(self, name):
        name = require_text(name, "snapshot name", 80)
        revision, state = self.load()
        self._commit(revision, state, snapshots={name: state})

    def snapshot(self, name):
        name = require_text(name, "snapshot name", 80)
        with self.connect() as db:
            row = db.execute("SELECT state FROM snapshots WHERE name=?", (name,)).fetchone()
        if row is None:
            raise ValueError("Snapshot does not exist")
        return json.loads(row[0])

    def restore(self, name):
        revision, _ = self.load()
        self._commit(revision, self.snapshot(name))

    def compare(self, text, baseline, a, b):
        """Four read-only trials: fixed memories/input, only E and retrieval gate vary."""
        text = require_text(text, "input", 2000)
        common = self.snapshot(baseline)
        query = self._embed(text, self.dimension(common), "query")
        trials = []
        for enabled in (True, False):
            for label, name in (("A", a), ("B", b)):
                emotion = self.snapshot(name)["emotion"]
                if emotion is None:
                    raise ValueError("Choose snapshots with an initialized emotion")
                rows = rank(common["memories"], query, emotion, enabled)
                selected = [row for row in rows if row["selected"]]
                result = validate_cognition(self.provider.cognize(text, copy.deepcopy(selected)), selected)
                trials.append({"label": label, "snapshot": name, "enabled": enabled,
                               "emotion": emotion, "retrieval": rows, "cognition": result})
        return {"input": text, "baseline": baseline, "memory_ids": [m["id"] for m in common["memories"]],
                "trials": trials, "mode": self.provider.mode,
                "note": "Only the emotion-to-read gate is ablated; historical write strengths stay fixed."}

    def seed(self):
        revision, state = self.load()
        if state["step"] or state["memories"] or state["emotion"] is not None:
            raise ValueError("Example initialization requires an empty database; use a new --db path")
        positive = "有人认真听取我的想法，和我一起解决问题；我感到被支持，期待再次合作"
        negative = "我的想法被轻率否定，合作中没有人听取我的意见；我担心再次受挫"
        contents = [
            ("我们一起试了一次合作，对方认真支持了我的想法。", positive),
            ("我们一起试了一次合作，对方轻率否定了我的想法。", negative),
            ("我们再次一起尝试，对方帮助我完成了困难的工作。", positive),
            ("我们再次一起尝试，遇到问题后对方把责任都推给我。", negative),
        ]
        dimension = None
        for content, appraisal in contents:
            vector = self._embed(content, dimension, "document")
            dimension = len(vector)
            state["memories"].append({"id": state["next_id"], "content": content,
                "appraisal": appraisal, "vector": vector, "affect": self._embed(appraisal, dimension),
                "uses": 0, "write_strength": 0.5})
            state["next_id"] += 1
        snapshots = {"baseline": copy.deepcopy(state)}
        for name, appraisal in (("A", positive), ("B", negative)):
            branch = copy.deepcopy(state)
            branch.update(emotion=self._embed(appraisal, dimension), appraisal=appraisal)
            snapshots[name] = branch
        self._commit(revision, snapshots["A"], snapshots=snapshots)
        return {"snapshots": list(snapshots), "note": "Hand-authored fixture memories and emotion states, not learned outcomes."}

"""Persistent experiment runtime: decay, retrieve, cognize, reinforce, commit."""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from .model import DynamicsConfig, activate, decay, rank
from .providers import unit


def _text(value, field, maximum=6000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{field} must be nonempty text, at most {maximum} characters")
    return value.strip()


def validate_cognition(result, selected):
    if not isinstance(result, dict):
        raise ValueError("Cognition must return an object")
    answer = _text(result.get("answer"), "answer")
    allowed = {memory["id"] for memory in selected}
    used = result.get("used_memory_ids")
    if not isinstance(used, list) or any(type(item) is not int or item not in allowed for item in used):
        raise ValueError("Cognition cited a memory that was not provided; state was not changed")
    proposals = result.get("memories")
    if not isinstance(proposals, list) or len(proposals) > 3:
        raise ValueError("Cognition may propose at most three memories")
    memories = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError("Each proposed memory must be an object")
        memories.append({"content": _text(proposal.get("content"), "memory.content")})
    return {
        "answer": answer,
        "used_memory_ids": sorted(set(used)),
        "memories": memories,
    }


class Runtime:
    def __init__(self, database, provider, config=None):
        self.path = str(database)
        self.provider = provider
        self.config = config or DynamicsConfig()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        initial = {
            "step": 0,
            "next_id": 1,
            "memories": [],
            "config": self.config.to_dict(),
        }
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS runtime "
                "(id INTEGER PRIMARY KEY CHECK(id=1), profile TEXT NOT NULL, "
                "revision INTEGER NOT NULL, state TEXT NOT NULL)"
            )
            db.execute("CREATE TABLE IF NOT EXISTS rounds (id INTEGER PRIMARY KEY, trace TEXT NOT NULL)")
            db.execute(
                "INSERT OR IGNORE INTO runtime VALUES (1, ?, 0, ?)",
                (provider.profile, json.dumps(initial, ensure_ascii=False)),
            )
            profile, state_text = db.execute(
                "SELECT profile, state FROM runtime WHERE id=1"
            ).fetchone()
            if profile != provider.profile:
                raise ValueError("Embedding provider/model differs from this database; use a new --db path")
            if json.loads(state_text).get("config") != self.config.to_dict():
                raise ValueError("Dynamics config differs from this database; use a new --db path")

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
            row = db.execute("SELECT revision, state FROM runtime WHERE id=1").fetchone()
        return row[0], json.loads(row[1])

    def _commit(self, revision, state, trace=None):
        encoded = json.dumps(state, ensure_ascii=False, allow_nan=False)
        with self.connect() as db:
            changed = db.execute(
                "UPDATE runtime SET revision=revision+1, state=? WHERE id=1 AND revision=?",
                (encoded, revision),
            ).rowcount
            if changed != 1:
                raise ValueError("State changed during this operation; reload and retry")
            if trace is not None:
                db.execute(
                    "INSERT INTO rounds(trace) VALUES (?)",
                    (json.dumps(trace, ensure_ascii=False, allow_nan=False),),
                )

    def _embed(self, text, dimension=None, task=None):
        vector = unit(self.provider.embed(text, task))
        if dimension is not None and len(vector) != dimension:
            raise ValueError("Embedding dimensions changed; state was not changed")
        return vector

    @staticmethod
    def dimension(state):
        if state["memories"]:
            return len(state["memories"][0]["vector"])
        return None

    def initialize(self, contents):
        revision, state = self.load()
        if state["step"] or state["memories"]:
            raise ValueError("Initialization requires an empty database; choose a new --db path")
        if not isinstance(contents, list) or not contents:
            raise ValueError("Provide a nonempty list of initial memory contents")
        prepared = []
        dimension = None
        for item in contents:
            content = _text(item, "initial memory")
            vector = self._embed(content, dimension, "document")
            dimension = len(vector)
            prepared.append({
                "id": state["next_id"] + len(prepared),
                "content": content,
                "vector": vector,
                "weight": 1.0,
                "stability": 0.0,
                "created_step": 0,
                "activated_count": 0,
                "last_activated_step": None,
            })
        state["memories"].extend(prepared)
        state["next_id"] += len(prepared)
        self._commit(revision, state)
        return {"initialized": len(prepared), "memory_ids": [item["id"] for item in prepared]}

    def retrieve(self, text):
        text = _text(text, "input", 2000)
        _, state = self.load()
        query = self._embed(text, self.dimension(state), "query")
        return rank(state["memories"], query, self.config.top_k)

    def step(self, text=None):
        if text is not None:
            text = _text(text, "input", 2000)
        revision, state = self.load()
        decayed = []
        for memory in state["memories"]:
            result = decay(memory, self.config)
            decayed.append({"id": memory["id"], **result})

        dimension = self.dimension(state)
        query = self._embed(text, dimension, "query") if text is not None else None
        if query is not None:
            dimension = len(query)
        rows = rank(state["memories"], query, self.config.top_k)
        selected = [row for row in rows if row["selected"]]
        cognition = validate_cognition(
            self.provider.cognize(text, selected), selected
        )

        activated = []
        used_ids = set(cognition["used_memory_ids"])
        for memory in state["memories"]:
            if memory["id"] in used_ids:
                result = activate(memory, self.config)
                memory["activated_count"] += 1
                memory["last_activated_step"] = state["step"] + 1
                activated.append({"id": memory["id"], **result})

        prepared = []
        for proposal in cognition["memories"]:
            vector = self._embed(proposal["content"], dimension, "document")
            dimension = len(vector)
            prepared.append({
                **proposal,
                "id": state["next_id"] + len(prepared),
                "vector": vector,
                "weight": 1.0,
                "stability": 0.0,
                "created_step": state["step"] + 1,
                "activated_count": 0,
                "last_activated_step": None,
            })

        state["memories"].extend(prepared)
        state["next_id"] += len(prepared)
        state["step"] += 1
        trace = {
            "step": state["step"],
            "input": text,
            "mode": self.provider.mode,
            "config": self.config.to_dict(),
            "decayed": decayed,
            "retrieval": rows,
            "cognition": cognition,
            "activated": activated,
            "written_ids": [item["id"] for item in prepared],
        }
        self._commit(revision, state, trace)
        return trace

    def inspect(self):
        revision, state = self.load()
        with self.connect() as db:
            traces = [
                json.loads(row[0])
                for row in db.execute("SELECT trace FROM rounds ORDER BY id DESC LIMIT 50")
            ]
        return {
            "revision": revision,
            "state": state,
            "traces": traces,
            "mode": self.provider.mode,
            "embedding_profile": self.provider.profile,
        }

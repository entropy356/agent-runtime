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


USAGE_WINDOW = 16
# A recall widens the slice the next cognition sees. It is still the same ranking,
# the same emotion and the same write strengths; only the query and the budget differ.
RECALL_K = 6


def recall_request(result, budget):
    """Return the model's recall wording, or None when it answered.
    Requesting past the budget is a contract violation, not a request to retry."""
    if not isinstance(result, dict):
        raise ValueError("Cognition must return an object")
    if "recall" not in result:
        return None
    if budget <= 0:
        raise ValueError("Cognition asked to recall after the budget was spent; state was not changed")
    if "answer" in result or "appraisal" in result:
        raise ValueError("A recall request cannot also carry an answer")
    return require_text(result.get("recall"), "recall", 500)


def usage_frequency(memory, step):
    """Observation only: cited rounds / fixed window, never wall-clock time.
    Missing history is unknown and starts cold; cumulative uses cannot date it.
    Nothing in ranking reads this value."""
    return len({s for s in memory.get("recent_uses", [])
                if step - USAGE_WINDOW < s <= step}) / USAGE_WINDOW


def require_text(value, field, maximum=6000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{field} must be nonempty text, at most {maximum} characters")
    return value.strip()


def centroid(vectors):
    """Library mean. Ordinary text embeddings share one large component; the mean is
    what an uncentered cosine mostly measures."""
    if not vectors:
        return None
    total = [0.0] * len(vectors[0])
    for vector in vectors:
        for index, value in enumerate(vector):
            total[index] += value
    return [value / len(vectors) for value in total]


def centered(vector, mean):
    """Unit residual against the library mean, so only the deviation is compared."""
    if vector is None or mean is None:
        return vector
    residual = [value - origin for value, origin in zip(vector, mean)]
    norm = math.sqrt(sum(value * value for value in residual))
    if norm < 1e-6:
        # Indistinguishable from the mean: it carries no signal, so it contributes none.
        return [0.0] * len(residual)
    return [value / norm for value in residual]


def rank(memories, query, emotion, enabled=True, top_k=3, step=0):
    """No mutations. Return all score components and a deterministic selection.
    Both sides of each comparison are centered on the library mean first.
    Usage frequency is carried along for inspection; it never affects the order."""
    if type(top_k) is not int or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    mean_vector = centroid([memory["vector"] for memory in memories])
    mean_affect = centroid([memory["affect"] for memory in memories])
    query, emotion = centered(query, mean_vector), centered(emotion, mean_affect)
    rows = []
    for memory in memories:
        # Centered cosine is signed, so it is shifted into [0, 1] instead of clipped:
        # clipping would flatten the half of the library that sits below the mean and
        # leave the ordering there to the id tie-break. 0.5 means "average content".
        relevance = (1 + dot(query, centered(memory["vector"], mean_vector))) / 2 if query is not None else 1.0
        match = dot(emotion, centered(memory["affect"], mean_affect)) if emotion is not None else 0.0
        read = sigmoid(4 * match) if enabled else 0.5
        score = relevance * read * memory["write_strength"]
        rows.append({**copy.deepcopy(memory), "usage_frequency": usage_frequency(memory, step),
                     "relevance": relevance, "emotion_match": match,
                     "read_strength": read, "score": score})
    rows.sort(key=lambda row: (-row["score"], row["id"]))
    chosen = {row["id"] for row in rows[:top_k]}
    for row in rows:
        row["selected"] = row["id"] in chosen
        row["selected_by"] = "score" if row["selected"] else None
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
    replaced = set()
    for memory in memories:
        if not isinstance(memory, dict):
            raise ValueError("Invalid proposed memory")
        relations = {}
        for field in ("source_memory_ids", "replaces"):
            ids = memory.get(field, [])
            if not isinstance(ids, list) or any(type(i) is not int or i not in allowed for i in ids):
                raise ValueError(f"memory.{field} must reference provided memory IDs")
            relations[field] = sorted(set(ids))
        if replaced.intersection(relations["replaces"]):
            raise ValueError("A memory cannot be replaced by multiple proposals in one round")
        replaced.update(relations["replaces"])
        new_memories.append({"content": require_text(memory.get("content"), "memory.content"),
                             "appraisal": require_text(memory.get("appraisal"), "memory.appraisal"),
                             **relations})
    return {"answer": answer, "appraisal": appraisal,
            "used_memory_ids": sorted(set(used)), "memories": new_memories}


class Runtime:
    def __init__(self, database, provider, rho=0.85, recall_limit=2):
        if not 0 <= rho < 1:
            raise ValueError("rho must be in [0, 1)")
        if type(recall_limit) is not int or recall_limit < 0:
            raise ValueError("recall_limit must be a nonnegative integer")
        self.path, self.provider, self.rho = str(database), provider, rho
        self.recall_limit = recall_limit
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS runtime (id INTEGER PRIMARY KEY CHECK(id=1), profile TEXT NOT NULL, revision INTEGER NOT NULL, state TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS snapshots (name TEXT PRIMARY KEY, state TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS rounds (id INTEGER PRIMARY KEY, trace TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS replaced (id INTEGER PRIMARY KEY, step INTEGER NOT NULL, revision INTEGER NOT NULL, memory_id INTEGER NOT NULL, replaced_by INTEGER NOT NULL, state TEXT NOT NULL)")
            initial = {"step": 0, "emotion": None, "appraisal": "", "memories": [], "next_id": 1}
            db.execute("INSERT OR IGNORE INTO runtime VALUES (1, ?, 0, ?)", (provider.profile, json.dumps(initial)))
            revision, raw = db.execute("SELECT revision, state FROM runtime WHERE id=1").fetchone()
            profile = db.execute("SELECT profile FROM runtime").fetchone()[0]
            state = json.loads(raw)
        if profile != provider.profile:
            raise ValueError("Embedding provider/model differs from this database; use --db with a new path")
        # Databases written before replacement deleted instead of tombstoning still carry
        # marked memories. Move them out once, so the runtime holds no superseded entry.
        legacy = [memory for memory in state["memories"] if memory.get("superseded_by") is not None]
        if legacy:
            state["memories"] = [memory for memory in state["memories"]
                                 if memory.get("superseded_by") is None]
            self._commit(revision, state, replaced=legacy)

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

    def _commit(self, revision, state, trace=None, snapshots=None, replaced=None):
        with self.connect() as db:
            changed = db.execute("UPDATE runtime SET revision=revision+1, state=? WHERE id=1 AND revision=?",
                                 (json.dumps(state, ensure_ascii=False, allow_nan=False), revision)).rowcount
            if changed != 1:
                raise ValueError("State changed during this operation; retry against the current state")
            if trace is not None:
                db.execute("INSERT INTO rounds(trace) VALUES (?)", (json.dumps(trace, ensure_ascii=False, allow_nan=False),))
            # Debug log only. Nothing reads these rows back into retrieval or cognition.
            for memory in replaced or []:
                db.execute("INSERT INTO replaced(step, revision, memory_id, replaced_by, state) VALUES (?,?,?,?,?)",
                           (state["step"], revision + 1, memory["id"], memory["superseded_by"],
                            json.dumps(memory, ensure_ascii=False, allow_nan=False)))
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

    def replaced_log(self, limit=50):
        """Debug view of memories that left the runtime through replacement."""
        with self.connect() as db:
            rows = db.execute("SELECT step, revision, memory_id, replaced_by, state FROM replaced "
                              "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"step": step, "revision": revision, "id": memory_id,
                 "replaced_by": replaced_by, **json.loads(raw)}
                for step, revision, memory_id, replaced_by, raw in rows]

    def inspect(self):
        revision, state = self.load()
        with self.connect() as db:
            names = [row[0] for row in db.execute("SELECT name FROM snapshots ORDER BY name")]
            traces = [json.loads(row[0]) for row in db.execute("SELECT trace FROM rounds ORDER BY id DESC LIMIT 20")]
        for memory in state["memories"]:
            memory["usage_frequency"] = usage_frequency(memory, state["step"])
        return {"revision": revision, "state": state, "snapshots": names,
                "traces": traces, "replaced": self.replaced_log(), "mode": self.provider.mode,
                "rho": self.rho, "profile": self.provider.profile, "usage_window": USAGE_WINDOW,
                "recall_limit": self.recall_limit, "recall_k": RECALL_K}

    def retrieve(self, text, enabled=True):
        text = require_text(text, "input", 2000)
        _, state = self.load()
        query = self._embed(text, self.dimension(state), "query")
        return rank(state["memories"], query, state["emotion"], enabled,
                    step=state["step"])

    def _recall(self, memories, text, emotion, enabled, seen, dimension):
        """Second retrieval on the model's own wording: same ranking, same emotion, wider slice.
        Everything it returns joins the context, so top_k is left at zero and applied here."""
        query = self._embed(text, dimension, "query")
        rows = rank(memories, query, emotion, enabled, top_k=0)
        return [{**row, "selected": True, "selected_by": "recall"}
                for row in rows if row["id"] not in seen][:RECALL_K]

    def step(self, text=None, enabled=True):
        # None is a clock tick with no new external input.
        if text is not None:
            text = require_text(text, "input", 2000)
        revision, state = self.load()
        dimension = self.dimension(state)
        query = self._embed(text, dimension, "query") if text is not None else None
        dimension = len(query) if query is not None else dimension
        before = copy.deepcopy(state["emotion"])
        rows = rank(state["memories"], query, before, enabled, step=state["step"])
        selected = [row for row in rows if row["selected"]]
        # The model may ask to look again before answering, in its own words. What a recall
        # surfaces joins the context, so it can be cited and replaced: that is the point of
        # asking. The budget bounds the cost; asking past it is an error, not a retry.
        recalls, budget = [], self.recall_limit
        while True:
            candidate = self.provider.cognize(text, copy.deepcopy(selected), budget,
                                              library_size=len(state["memories"]))
            wanted = recall_request(candidate, budget)
            if wanted is None:
                result = candidate
                break
            extra = self._recall(state["memories"], wanted, before, enabled,
                                 {row["id"] for row in selected}, dimension)
            recalls.append({"query": wanted, "rows": extra})
            if extra:
                selected = selected + extra
                budget -= 1
            else:
                # Nothing new left to find; another request would only spend a call.
                budget = 0
        result = validate_cognition(result, selected)
        feeling = self._embed(result["appraisal"], dimension)
        dimension = len(feeling)
        after = feeling if before is None else [self.rho * a + (1 - self.rho) * b for a, b in zip(before, feeling)]
        # Finish ALL provider calls before changing state. Each memory has its own affect.
        # Write strength compares against the existing affect mean, so a new memory is
        # scored by how much it differs from what is already stored, not by the shared bulk.
        mean_affect = centroid([memory["affect"] for memory in state["memories"]])
        prepared = []
        for proposal in result["memories"]:
            affect = self._embed(proposal["appraisal"], dimension)
            aligned = dot(centered(after, mean_affect), centered(affect, mean_affect))
            prepared.append({**proposal, "vector": self._embed(proposal["content"], dimension, "document"),
                             "affect": affect, "write_strength": sigmoid(3 * aligned), "uses": 0,
                             "created_step": state["step"] + 1, "created_revision": revision + 1,
                             "origin": "external_input" if text is not None else "internal_cognition",
                             "recent_uses": []})
        for memory in state["memories"]:
            memory["recent_uses"] = sorted({s for s in memory.get("recent_uses", [])
                                            if state["step"] + 1 - USAGE_WINDOW < s <= state["step"]})
            if memory["id"] in result["used_memory_ids"]:
                memory["uses"] += 1
                memory["recent_uses"].append(state["step"] + 1)
        for memory in prepared:
            memory["id"] = state["next_id"]
            state["next_id"] += 1
        # Replacement deletes: the runtime keeps no superseded entry. The removed copy goes
        # to the debug table only, so it cannot re-enter retrieval through a later snapshot.
        replaced_by = {old: new["id"] for new in prepared for old in new["replaces"]}
        replaced = [{**memory, "superseded_by": replaced_by[memory["id"]]}
                    for memory in state["memories"] if memory["id"] in replaced_by]
        state["memories"] = [memory for memory in state["memories"] if memory["id"] not in replaced_by]
        state["memories"].extend(prepared)
        state.update(emotion=after, appraisal=result["appraisal"], step=state["step"] + 1)
        trace = {"step": state["step"], "input": text, "enabled": enabled,
                 "usage_window": USAGE_WINDOW, "recall_limit": self.recall_limit,
                 "emotion_before": before, "emotion_after": after, "rho": self.rho,
                 "retrieval": rows, "recalls": recalls, "cognition": result,
                 "written_ids": [m["id"] for m in prepared],
                 "replaced_ids": [m["id"] for m in replaced], "mode": self.provider.mode}
        self._commit(revision, state, trace, replaced=replaced)
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
        state = self.snapshot(name)
        # A snapshot taken before replacement deleted entries can still carry marked ones.
        legacy = [memory for memory in state["memories"] if memory.get("superseded_by") is not None]
        if legacy:
            state["memories"] = [memory for memory in state["memories"]
                                 if memory.get("superseded_by") is None]
        self._commit(revision, state, replaced=legacy)

    def compare(self, text, baseline, a, b):
        """Four read-only trials: fixed memories/input, only E and retrieval gate vary."""
        text = require_text(text, "input", 2000)
        common = self.snapshot(baseline)
        common["memories"] = [memory for memory in common["memories"]
                              if memory.get("superseded_by") is None]
        query = self._embed(text, self.dimension(common), "query")
        trials = []
        for enabled in (True, False):
            for label, name in (("A", a), ("B", b)):
                emotion = self.snapshot(name)["emotion"]
                if emotion is None:
                    raise ValueError("Choose snapshots with an initialized emotion")
                rows = rank(common["memories"], query, emotion, enabled, step=common["step"])
                selected = [row for row in rows if row["selected"]]
                result = validate_cognition(self.provider.cognize(
                    text, copy.deepcopy(selected), library_size=len(common["memories"])), selected)
                trials.append({"label": label, "snapshot": name, "enabled": enabled,
                               "emotion": emotion, "retrieval": rows, "cognition": result})
        return {"input": text, "baseline": baseline, "memory_ids": [m["id"] for m in common["memories"]],
                "trials": trials, "mode": self.provider.mode, "usage_window": USAGE_WINDOW,
                "note": "Only the emotion-to-read gate is ablated; historical write strengths stay fixed. "
                        "Trials use the fixed three-memory context, so a recall cannot widen one arm only."}

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
                "uses": 0, "write_strength": 0.5, "created_step": 0,
                "created_revision": revision + 1, "origin": "fixture",
                "source_memory_ids": [], "replaces": [], "recent_uses": []})
            state["next_id"] += 1
        snapshots = {"baseline": copy.deepcopy(state)}
        for name, appraisal in (("A", positive), ("B", negative)):
            branch = copy.deepcopy(state)
            branch.update(emotion=self._embed(appraisal, dimension), appraisal=appraisal)
            snapshots[name] = branch
        self._commit(revision, snapshots["A"], snapshots=snapshots)
        return {"snapshots": list(snapshots), "note": "Hand-authored fixture memories and emotion states, not learned outcomes."}

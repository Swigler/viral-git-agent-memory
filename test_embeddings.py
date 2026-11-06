#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Embeddings as a derived index. A stub HTTP server stands in for the embedding model.

Covers: incremental re-embed (only changed Fact text), rebuild from markdown after the
index is deleted, hybrid FTS+vector ranking (a semantic hit with zero shared words
beats a keyword-only hit), contradicted memories excluded, and the FTS-only fallback
when the endpoint is down or unconfigured.

Run:  python3 test_embeddings.py
"""
import http.server
import json
import tempfile
import threading
from pathlib import Path

import gitmem
from gitmem import Embedder, Memory, embedder_from_env

PASS = 0


def check(name, cond, detail=""):
    global PASS
    assert cond, f"{name} {detail}"
    PASS += 1
    print(f"  ✅ {name}")


# Concept groups: words in one group embed to the same axis, so "espresso" and
# "caffeine" are neighbours although they share no characters. Deterministic, no model.
GROUPS = [
    {"coffee", "espresso", "caffeine", "latte", "brew"},
    {"dog", "puppy", "canine", "walk"},
    {"engineer", "developer", "programmer", "software"},
    {"allergic", "allergy", "peanut", "peanuts"},
]


def vector(text):
    words = {w.strip(".,!?").lower() for w in text.split()}
    return [1.0 if words & g else 0.0 for g in GROUPS] + [0.01, 0.01, 0.01, 0.01]


class Stub(http.server.BaseHTTPRequestHandler):
    seen: list = []          # every text ever sent for embedding
    auth: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.path == "/v1/embeddings" and "model" in body
        Stub.seen += body["input"]
        Stub.auth.append(self.headers.get("Authorization"))
        data = [{"index": i, "embedding": vector(t)} for i, t in enumerate(body["input"])]
        out = json.dumps({"data": list(reversed(data))}).encode()  # out of order on purpose
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)


def main():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    emb = Embedder(url, "stub-model", api_key="k", timeout=5)

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        mem = Memory(tmp, create=True, embedder=emb)
        mem.write("USER", "morning-ritual", "You drink espresso every morning")
        mem.write("USER", "nail-habit", "Your habit of biting nails when nervous")
        mem.write("USER", "peanut-allergy", "You are severely allergic to peanuts", pinned=True)
        check("write embeds each new fact", len(Stub.seen) == 3, str(Stub.seen))
        check("api key sent as Bearer", set(Stub.auth) == {"Bearer k"})
        rows = mem.db.execute("SELECT dim, length(vector) FROM embeddings").fetchall()
        check("vectors stored as float32 blobs", len(rows) == 3 and all(r == (8, 32) for r in rows), str(rows))

        # -- incremental: only a changed Fact is re-embedded ---------------------
        before = len(Stub.seen)
        mem.sync()
        check("sync with no change embeds nothing", len(Stub.seen) == before)
        path = repo / "USER_memory/nail-habit.md"
        path.write_text(path.read_text().replace("## Episode", "## Episode") + "\n## Notes\nx\n")
        mem.sync()
        check("edit outside the Fact text does not re-embed", len(Stub.seen) == before)
        mem.write("USER", "nail-habit", "Your habit of biting nails when stressed")
        check("changed Fact text re-embeds exactly that file",
              Stub.seen[before:] == ["Your habit of biting nails when stressed"], str(Stub.seen[before:]))
        mem.delete("USER", "nail-habit")
        mem.write("USER", "nail-habit", "Your habit of biting nails when stressed")
        n = mem.db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        check("deleting a memory drops its vector", n == 3, str(n))

        # -- hybrid ranking ------------------------------------------------------
        fts_only = Memory(tmp, embedder=None)
        hits = [h["slug"] for h in fts_only.search("caffeine habit", "USER")]
        check("FTS alone: keyword hit only, semantic memory missed", hits == ["nail-habit"], str(hits))
        hits = [h["slug"] for h in mem.search("caffeine habit", "USER")]
        check("hybrid: semantic hit (no shared words) is found", "morning-ritual" in hits, str(hits))
        check("hybrid: keyword hit and semantic hit are both on top", set(hits[:2]) == {"morning-ritual", "nail-habit"}
              or "morning-ritual" in hits[:2], str(hits))
        check("hybrid: pure-semantic query ranks the semantic hit first (FTS finds nothing)",
              fts_only.search("caffeine", "USER") == []
              and mem.search("caffeine", "USER")[0]["slug"] == "morning-ritual")
        check("recall uses the hybrid search", "espresso" in mem.recall("USER", "caffeine habit", n=3))

        # -- contradicted memories are excluded from the vector side too ---------
        gitmem.mark_contradicted(tmp, "USER", "morning-ritual")
        mem.sync()
        hits = [h["slug"] for h in mem.search("caffeine habit", "USER")]
        check("contradicted memory excluded from hybrid search", "morning-ritual" not in hits, str(hits))
        check("contradicted memory excluded from recall", "espresso" not in mem.recall("USER", "caffeine"))
        gitmem.upsert(tmp, "USER", "morning-ritual", "You drink espresso every morning")  # revived
        mem.sync()
        check("revived memory is findable again",
              "morning-ritual" in [h["slug"] for h in mem.search("caffeine", "USER")])

        # -- pinned stays capped and leading -------------------------------------
        check("pinned memory leads recall", mem.recall("USER", "caffeine").splitlines()[0].startswith("- You are severely"))
        mem.close()

        # -- the index is derived: delete it, everything is rebuilt --------------
        (repo / gitmem.INDEX_FILE).unlink()
        before = len(Stub.seen)
        mem = Memory(tmp, embedder=emb)
        check("rebuild re-embeds every live fact from markdown", len(Stub.seen) - before == 3, str(len(Stub.seen) - before))
        check("rebuilt index answers the same hybrid query",
              mem.search("caffeine", "USER")[0]["slug"] == "morning-ritual")
        # model switch = new hash = full re-embed
        before = len(Stub.seen)
        mem2 = Memory(tmp, embedder=Embedder(url, "other-model"))
        check("switching model re-embeds", len(Stub.seen) - before == 3)
        mem2.close()
        mem.close()

        # -- fallback: endpoint down -> FTS-only, no exception, one log ----------
        logged = []
        real, gitmem.log_stderr = gitmem.log_stderr, logged.append
        gitmem._embed_warned = False
        try:
            dead = Memory(tmp, embedder=Embedder("http://127.0.0.1:9", "m", timeout=1))
            hits = [h["slug"] for h in dead.search("caffeine habit", "USER")]
            dead.search("again", "USER")
            dead.write("USER", "new-fact", "You like hiking in the mountains")
        finally:
            gitmem.log_stderr = real
        check("endpoint down: search falls back to FTS", hits == ["nail-habit"], str(hits))
        check("endpoint down: writes still work", (repo / "USER_memory/new-fact.md").exists())
        check("endpoint down: logged once, not per call", len(logged) == 1, str(logged))
        dead.close()

    # -- env configuration -------------------------------------------------------
    import os
    for k in ("EMBED_URL", "EMBED_MODEL", "EMBED_API_KEY", "EMBED_TIMEOUT"):
        os.environ.pop(k, None)
    check("EMBED_URL unset -> embeddings off", embedder_from_env() is None)
    os.environ.update(EMBED_URL=url + "/", EMBED_MODEL="bge-small", EMBED_TIMEOUT="3")
    e = embedder_from_env()
    check("EMBED_* env builds the embedder",
          (e.url, e.model, e.timeout, e.api_key) == (url, "bge-small", 3.0, ""))
    with tempfile.TemporaryDirectory() as tmp:
        m = Memory(tmp, create=True)
        check("Memory picks the embedder up from env", m.embedder is not None)
        m.close()

    server.shutdown()
    print(f"embeddings: {PASS} assertions passed")


if __name__ == "__main__":
    main()

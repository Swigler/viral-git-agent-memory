#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Checks for the companion-site surface: /v1/recall, /v1/memories, /v1/memories/delete,
/v1/forget, a forget that sticks, hard delete without recoverable history, no git routes,
failed extraction leaving no transcript. Same stub-LLM approach as test_server.py (no network).

Run:  python3 test_crimson.py
"""
import http.server
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


class LLM(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        system = body["messages"][0]["content"]
        if "memory manager" in system:  # A.U.D.N.: ADD whatever the new facts are
            facts = json.loads(system.split("NEW FACTS:\n", 1)[1].split("\n\nOutput ONLY", 1)[0])
            out = {"decisions": [{"action": "ADD", **f} for f in facts]}
        elif "about the USER" in system:
            out = {"facts": [{"slug": "likes-tea", "fact": "You like green tea.", "episode": "Session 01.01.26"}]}
        else:
            out = {"facts": []}
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"message": {"content": json.dumps(out)}}).encode())


llm = http.server.ThreadingHTTPServer(("127.0.0.1", 0), LLM)
threading.Thread(target=llm.serve_forever, daemon=True).start()
LLM_URL = f"http://127.0.0.1:{llm.server_port}"

TMP = Path(tempfile.mkdtemp(prefix="crimson_test_"))
os.environ.update(
    MEMORY_DIR=str(TMP / "memory"), MEMORY_LLM_PROVIDER="ollama", OLLAMA_URL=LLM_URL,
    API_KEY_SECRET="test-secret",
)
os.environ.pop("EMBED_URL", None)

import api_server as S  # noqa: E402  (env must be set first)
import gitops  # noqa: E402
import memory_hook as mh  # noqa: E402

srv = S.ThreadedHTTPServer(("127.0.0.1", 0), S.ChatHandler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{srv.server_port}"
PASS = 0


def check(name, cond, detail=""):
    global PASS
    assert cond, f"{name} — {detail}"
    PASS += 1
    print(f"  ✅ {name}")


def http(method, path, body=None, uid=None):
    headers = {"Content-Type": "application/json"}
    if uid:
        headers["Authorization"] = f"Bearer {S.user_token(uid)}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, None


def post(path, uid, **body):
    return http("POST", path, {"userId": uid, **body}, uid)


def hook(repo, uid, transcript, url=LLM_URL):
    return subprocess.run(
        [sys.executable, str(HERE / "memory_hook.py"), "--consolidate", str(repo), uid, str(transcript)],
        capture_output=True, text=True, env={**os.environ, "OLLAMA_URL": url}, timeout=120,
    )


def transcript(repo, name=".transcript_t.jsonl"):
    tr = Path(repo) / name
    tr.write_text(json.dumps({"message": {"role": "user", "content": "I like green tea"}}) + "\n")
    return tr


def holder(repo):
    """A separate process holding the repo lock until killed."""
    p = subprocess.Popen(
        [sys.executable, "-c",
         "import sys,time,gitops;c=gitops.repo_lock(sys.argv[1]);c.__enter__();print('held',flush=True);time.sleep(60)",
         str(repo)],
        cwd=HERE, stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "held"
    return p


def disk_has(root, needle: bytes) -> list:
    return [str(p) for p in Path(root).rglob("*") if p.is_file() and needle in p.read_bytes()]


def test_recall_and_list():
    print("\n--- recall / memories ---")
    st, body = post("/v1/recall", "ghost", query="tea")
    check("recall on unknown user -> empty lists", st == 200 and body == {"user": [], "soul": []}, str(body))
    st, body = post("/v1/memories", "ghost")
    check("list on unknown user -> empty lists", st == 200 and body == {"user": [], "soul": []}, str(body))
    check("unknown user: no repo and no directory created", not S.MEMORY_DIR.exists() or not any(S.MEMORY_DIR.iterdir()))
    check("unknown user: no chat state", "ghost" not in S._user_state)

    uid = "u1:c1:lobby"
    repo = S.get_user_repo(uid)
    done = hook(repo, uid, transcript(repo))
    check("stubbed consolidation succeeds", done.returncode == 0, done.stderr[-300:])
    mh.write_memory_file(repo, "SOUL_memory", "playful", "Keep a playful, teasing tone.", "e", pinned=True)
    mh.write_memory_file(repo, "USER_memory", "likes-coffee", "You like coffee.", "e")
    mh.mark_contradicted(repo, "USER_memory", "likes-coffee")
    st, body = post("/v1/recall", uid, query="tea")
    check("recall returns structured user rows",
          st == 200 and body["user"] == [{"id": "likes-tea", "fact": "You like green tea.", "pinned": False}], str(body))
    check("recall returns soul rows", body["soul"] == [{"id": "playful", "fact": "Keep a playful, teasing tone.", "pinned": True}], str(body))
    st, body = post("/v1/recall", uid)
    check("empty query works", st == 200 and [r["id"] for r in body["user"]] == ["likes-tea"])
    check("recall keeps no chat state", uid not in S._user_state)
    for bad in ({"n": 21}, {"n": 0}, {"n": "5"}, {"n": True}, {"query": 5}):
        check(f"bad body {bad} -> 400", post("/v1/recall", uid, **bad)[0] == 400)
    check("n=20 accepted", post("/v1/recall", uid, n=20)[0] == 200)
    st, body = post("/v1/memories", uid)
    row = body["user"][0]
    check("list: every non-contradicted memory with an ISO updated",
          [r["id"] for r in body["user"]] == ["likes-tea"] and set(row) == {"id", "fact", "pinned", "updated"}
          and time.strptime(row["updated"][:19], "%Y-%m-%dT%H:%M:%S"), str(body))
    check("list: soul side", [r["id"] for r in body["soul"]] == ["playful"])


def test_busy():
    print("\n--- recall never waits behind the repo lock ---")
    uid = "u1:c1:lobby"
    repo = S.repo_dir(uid)
    ready, release = threading.Event(), threading.Event()

    def hold():
        with gitops.repo_lock(repo):
            ready.set()
            release.wait(30)

    t = threading.Thread(target=hold)
    t.start()
    ready.wait(10)
    t0 = time.monotonic()
    st, body = post("/v1/recall", uid, query="tea")
    check("lock held by another thread -> busy, fast", st == 200 and body == {"user": [], "soul": [], "busy": True}
          and time.monotonic() - t0 < 2, f"{body} {time.monotonic() - t0:.2f}s")
    release.set()
    t.join()
    p = holder(repo)
    try:
        t0 = time.monotonic()
        st, body = post("/v1/recall", uid, query="tea")
        check("lock held by another process -> busy, fast", st == 200 and body.get("busy") is True
              and time.monotonic() - t0 < 2, f"{body} {time.monotonic() - t0:.2f}s")
    finally:
        p.kill()
        p.wait()
    time.sleep(0.2)
    check("free again afterwards", "busy" not in post("/v1/recall", uid, query="tea")[1])


def test_delete():
    print("\n--- hard delete ---")
    uid = "u2:c1:lobby"
    repo = S.get_user_repo(uid)
    secret = b"Zanzibarquux"
    mh.write_memory_file(repo, "USER_memory", "attic-thing", f"You keep {secret.decode()} in the attic.", "ep one")
    mh.write_memory_file(repo, "USER_memory", "keeper", "You own a cat.", "e")
    mh.git_commit(str(repo))
    mh.update_memory_file(repo, "USER_memory", "attic-thing", f"You keep {secret.decode()} in the cellar.", "ep two")
    mh.git_commit(str(repo))
    st, body = post("/v1/memories", uid)  # opens the index, so the text is in sqlite
    check("setup: memory listed", {"attic-thing", "keeper"} == {r["id"] for r in body["user"]})
    hist = subprocess.run(["git", "log", "-p", "--all"], cwd=repo, capture_output=True, text=True).stdout
    check("setup: old text is in git history", secret.decode() in hist and hist.count("commit ") >= 3, hist[:300])

    for bad in ("../x", "a/b", "..", ".", "a.b", "", "x" * 81, 5, None):
        check(f"bad id {bad!r} -> 400", post("/v1/memories/delete", uid, kind="USER", id=bad)[0] == 400)
    check("bad kind -> 400", post("/v1/memories/delete", uid, kind="user", id="keeper")[0] == 400)
    check("delete of a missing id -> false", post("/v1/memories/delete", uid, kind="USER", id="nope")[1] == {"deleted": False})
    check("delete for an unknown user -> false, no repo",
          post("/v1/memories/delete", "nobody", kind="USER", id="x")[1] == {"deleted": False}
          and not S.repo_dir("nobody").exists())

    st, body = post("/v1/memories/delete", uid, kind="USER", id="attic-thing")
    check("delete -> deleted true", st == 200 and body == {"deleted": True}, str(body))
    check("fact file removed", not (repo / "USER_memory/attic-thing.md").exists())
    check("sibling memory kept", (repo / "USER_memory/keeper.md").exists())
    db = sqlite3.connect(repo / "index.sqlite3")
    rows = db.execute("SELECT count(*) FROM memories WHERE slug = 'attic-thing'").fetchone()[0]
    db.close()
    check("index row removed", rows == 0)
    check("USER.md index block regenerated", "attic-thing" not in (repo / "USER.md").read_text())
    check("old text found nowhere under the repo dir (files, .git, sqlite, wal)",
          disk_has(repo, secret) == [], str(disk_has(repo, secret)))
    log = subprocess.run(["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True).stdout
    check("history is one fresh commit", log.count("\n") == 1, log)
    check("git is clean and still usable", subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout.strip() == "")
    check("second delete -> false", post("/v1/memories/delete", uid, kind="USER", id="attic-thing")[1] == {"deleted": False})
    check("recall no longer returns it", all(r["id"] != "attic-thing" for r in post("/v1/recall", uid, query="attic")[1]["user"]))


def test_forget():
    print("\n--- forget ---")
    uid = "u3:c1:lobby"
    repo = S.get_user_repo(uid)
    S.get_user_state(uid)
    S.reset_inactivity_timer(uid)
    transcript(repo)
    check("forget -> true", post("/v1/forget", uid)[1] == {"forgotten": True})
    check("repo directory gone (facts, git, index, transcripts, logs)", not repo.exists())
    check("in-RAM state and timer evicted", uid not in S._user_state and uid not in S._inactivity_timers)
    check("forget again -> false, no error", post("/v1/forget", uid)[1] == {"forgotten": False})
    check("forget unknown user -> false", post("/v1/forget", "never-seen")[1] == {"forgotten": False})
    check("recall after forget -> empty, nothing recreated",
          post("/v1/recall", uid, query="x")[1] == {"user": [], "soul": []} and not repo.exists())


def test_forget_sticks():
    print("\n--- a forget sticks ---")
    uid = "u4:c1:lobby"
    repo = S.get_user_repo(uid)
    tr = transcript(repo)
    post("/v1/forget", uid)
    done = hook(repo, uid, tr)
    check("transcript vanished -> hook exits 0 without init_repo", done.returncode == 0 and not repo.exists(), done.stderr[-300:])

    # queued behind the lock when the forget happens (the real race)
    uid = "u5:c1:lobby"
    repo = S.get_user_repo(uid)
    tr = transcript(repo)
    p = holder(repo)
    try:
        proc = subprocess.Popen(
            [sys.executable, str(HERE / "memory_hook.py"), "--consolidate", str(repo), uid, str(tr)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ)
        time.sleep(2.5)  # it is now blocked on the repo lock
        check("consolidation is queued, not finished", proc.poll() is None)
        shutil.rmtree(repo)  # what /v1/forget does once it has the lock
    finally:
        p.kill()
        p.wait()
    out = proc.communicate(timeout=60)[0]
    check("queued consolidation skips, exits 0", proc.returncode == 0, out[-300:])
    check("forget during a queued consolidation leaves no directory behind", not repo.exists(), out[-300:])

    # a brand-new user still consolidates
    uid = "u6:c1:lobby"
    repo = S.get_user_repo(uid)
    done = hook(repo, uid, transcript(repo))
    check("brand-new user still consolidates", done.returncode == 0 and (repo / "USER_memory/likes-tea.md").exists(), done.stderr[-300:])


def test_routes_and_failure():
    print("\n--- git routes gone, failed extraction ---")
    check("POST /v1/git/setup -> 404", http("POST", "/v1/git/setup", {"userId": "a", "githubToken": "t"}, "a")[0] == 404)
    check("POST /v1/git/update -> 404", http("POST", "/v1/git/update", {"userId": "a", "githubToken": "t"}, "a")[0] == 404)
    check("GET /v1/git/status -> 404", http("GET", "/v1/git/status?userId=a", None, "a")[0] == 404)
    uid = "u7:c1:lobby"
    repo = S.get_user_repo(uid)
    tr = transcript(repo)
    done = hook(repo, uid, tr, url="http://127.0.0.1:9")
    check("failed extraction -> non-zero exit", done.returncode != 0, done.stderr[-200:])
    check("failed extraction leaves no transcript file", not tr.exists() and not list(repo.glob(".transcript_*")))
    check("failed extraction wrote no memory", not list((repo / "USER_memory").glob("*.md")))
    check("chat text not left anywhere in the repo", disk_has(repo, b"green tea") == [])


def main():
    try:
        test_recall_and_list()
        test_busy()
        test_delete()
        test_forget()
        test_forget_sticks()
        test_routes_and_failure()
        print(f"\ncrimson: {PASS} assertions passed")
    finally:
        srv.shutdown()
        llm.shutdown()
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()

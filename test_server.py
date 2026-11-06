#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""End-to-end checks for the server and hook: runs api_server in-process against a stub
Ollama-compatible LLM (no network, no API key).

Covers: userId validation and collision-free repo dirs, per-user auth (and the
single-tenant flag), LLM failure keeps the transcript, contradicted memories excluded,
header injection stripped, history window / orphan turns / unconsolidated buffer,
concurrent consolidation, repo-creation race, exact route matching, input validation.

Run:  python3 test_server.py
"""
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# --- stub LLM (Ollama dialect) ------------------------------------------------------


class LLM(http.server.BaseHTTPRequestHandler):
    fail = False

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if LLM.fail:
            self.send_response(500)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        if body.get("stream"):
            for tok in ("Hello", " there"):
                self.wfile.write((json.dumps({"message": {"content": tok}, "done": False}) + "\n").encode())
            self.wfile.write((json.dumps({"message": {"content": ""}, "done": True}) + "\n").encode())
            return
        system = body["messages"][0]["content"]
        if "memory manager" in system:  # A.U.D.N.: ADD whatever the new facts are
            facts = json.loads(system.split("NEW FACTS:\n", 1)[1].split("\n\nOutput ONLY", 1)[0])
            out = {"decisions": [{"action": "ADD", **f} for f in facts]}
        elif "about the USER" in system:
            out = {"facts": [{"slug": "likes-tea", "fact": "You like green tea.", "episode": "Session 01.01.26"}]}
        else:
            out = {"facts": []}
        self.wfile.write(json.dumps({"message": {"content": json.dumps(out)}}).encode())


llm = http.server.ThreadingHTTPServer(("127.0.0.1", 0), LLM)
threading.Thread(target=llm.serve_forever, daemon=True).start()
LLM_URL = f"http://127.0.0.1:{llm.server_port}"

TMP = Path(tempfile.mkdtemp(prefix="server_test_"))
os.environ.update(
    MEMORY_DIR=str(TMP / "memory"), MEMORY_LLM_PROVIDER="ollama", OLLAMA_URL=LLM_URL,
    API_KEY_SECRET="test-secret", GIT_PUSH_TIMEOUT="20",
)
os.environ.pop("EMBED_URL", None)

import api_server as S  # noqa: E402  (env must be set first)
import gitmem as gm  # noqa: E402
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


def http(method, path, body=None, token=None, raw=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def tok(uid):
    return S.user_token(uid)


def test_user_ids():
    print("\n--- userId validation and collisions ---")
    bad = ["", "a/b", "../x", ".hidden", "a b", "a" * 129, "a\nb", None, 5, ["a"], "x\x00"]
    check("malformed ids rejected", all(S.valid_user_id(b) is None for b in bad))
    ids = ["a_b", "a-b", "a.b", "a@b", "a:b", "A_B"]
    check("well-formed ids accepted", all(S.valid_user_id(i) == i for i in ids))
    dirs = {S.repo_dir(i) for i in ids}
    check("look-alike ids map to distinct repo dirs", len(dirs) == len(ids), str(dirs))
    check("repo dirs stay inside MEMORY_DIR", all(d.parent == S.MEMORY_DIR for d in dirs))
    check("missing userId -> 400", http("POST", "/v1/bye", {}, tok("x"))[0] == 400)
    check("slash in userId -> 400 (not folded into a_b)", http("POST", "/v1/bye", {"userId": "a/b"}, tok("a_b"))[0] == 400)
    check("userId must be a string", http("POST", "/v1/chat", {"userId": 7, "message": "hi"})[0] == 400)
    check("status without userId -> 400", http("GET", "/v1/git/status", token=tok("x"))[0] == 400)
    legacy = S.MEMORY_DIR / "olduser"
    legacy.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(legacy)], check=True)
    (legacy / "marker").write_text("x")
    repo = S.get_user_repo("olduser")
    check("legacy repo adopted into the hashed layout", (repo / "marker").exists() and not legacy.exists())


def test_auth():
    print("\n--- per-user auth ---")
    check("no token -> 401", http("GET", "/v1/git/status?userId=alice")[0] == 401)
    check("own token -> 200", http("GET", "/v1/git/status?userId=alice", token=tok("alice"))[0] == 200)
    cross = tok("alice")
    for method, path, body in [
        ("GET", "/v1/git/status?userId=bob", None),
        ("POST", "/v1/chat", {"userId": "bob", "message": "hi"}),
        ("POST", "/v1/bye", {"userId": "bob"}),
        ("POST", "/v1/consolidate", {"userId": "bob", "transcript": []}),
        ("POST", "/v1/git/setup", {"userId": "bob", "githubToken": "x"}),
        ("POST", "/v1/git/update", {"userId": "bob", "githubToken": "x"}),
    ]:
        check(f"alice's token cannot {method} {path.split('?')[0]} as bob",
              http(method, path, body, token=cross)[0] == 401)
    check("garbage token -> 401", http("POST", "/v1/bye", {"userId": "alice"}, token="nope")[0] == 401)
    check("/health needs no token", http("GET", "/health")[0] == 200)

    S.SINGLE_TENANT, S.API_KEY = True, "shared-key"
    try:
        check("single-tenant: shared key works for any user",
              http("GET", "/v1/git/status?userId=bob", token="shared-key")[0] == 200)
        check("single-tenant: per-user HMAC token no longer accepted",
              http("GET", "/v1/git/status?userId=bob", token=tok("bob"))[0] == 401)
    finally:
        S.SINGLE_TENANT, S.API_KEY = False, ""
    check("API_KEY alone is not an auth mode", S.token_ok("bob", "Bearer anything") is False)


def test_chat_history():
    print("\n--- chat history, orphan turns, unconsolidated buffer ---")
    uid = "hist"
    sent = []
    real_spawn, S.spawn_consolidation = S.spawn_consolidation, lambda u, r, m: sent.append((u, list(m)))
    try:
        st, body = http("POST", "/v1/chat", {"userId": uid, "message": "hello"}, tok(uid))
        check("chat streams", st == 200 and b"Hello" in body and b"[DONE]" in body)
        state = S._user_state[uid]
        check("success records user+assistant", [m["role"] for m in state["history"]] == ["user", "assistant"])
        check("success counts one message", state["message_count"] == 1 and len(state["pending"]) == 2)

        LLM.fail = True
        st, body = http("POST", "/v1/chat", {"userId": uid, "message": "this will fail"}, tok(uid))
        LLM.fail = False
        check("LLM failure leaves no orphan user turn",
              [m["role"] for m in state["history"]] == ["user", "assistant"], str(state["history"]))
        check("LLM failure is not counted or buffered", state["message_count"] == 1 and len(state["pending"]) == 2)

        # a window that would start with an assistant turn is trimmed back to a user turn
        state["history"] = [{"role": "assistant" if i % 2 == 0 else "user", "content": f"m{i}"} for i in range(101)]
        http("POST", "/v1/chat", {"userId": uid, "message": "again"}, tok(uid))
        check("history is bounded", len(state["history"]) <= S.HISTORY_LIMIT * 2)
        check("history never starts with an assistant turn", state["history"][0]["role"] == "user")
        ctx = S.assemble_context(S.get_user_repo(uid), state)
        check("context never starts with an assistant turn", ctx[1]["role"] == "user")

        # the unconsolidated buffer is not subject to the window
        state["pending"] = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"p{i}"} for i in range(200)]
        http("POST", "/v1/chat", {"userId": uid, "message": "more"}, tok(uid))
        check("pending buffer keeps everything since the last consolidation", len(state["pending"]) == 202)
        st, body = http("POST", "/v1/bye", {"userId": uid}, tok(uid))
        check("bye consolidates the whole buffer once", len(sent) == 1 and len(sent[0][1]) == 202, str([len(m) for _, m in sent]))
        check("bye reports consolidated and evicts the state",
              json.loads(body)["consolidated"] is True and uid not in S._user_state and uid not in S._inactivity_timers)
        http("POST", "/v1/bye", {"userId": uid}, tok(uid))
        check("a second bye does not consolidate again", len(sent) == 1)

        # count trigger drains the same buffer
        sent.clear()
        for _ in range(S.CONSOLIDATE_EVERY):
            http("POST", "/v1/chat", {"userId": "cnt", "message": "x"}, tok("cnt"))
        check("every-N trigger consolidates exactly the 2N buffered turns",
              len(sent) == 1 and len(sent[0][1]) == 2 * S.CONSOLIDATE_EVERY)
        check("trigger clears the buffer", S._user_state["cnt"]["pending"] == [] and S._user_state["cnt"]["message_count"] == 0)
        S.evict_user("cnt")
    finally:
        S.spawn_consolidation = real_spawn


def hook(repo, uid, transcript, url=LLM_URL):
    return subprocess.run(
        [sys.executable, str(Path(__file__).parent / "memory_hook.py"), "--consolidate", str(repo), uid, str(transcript)],
        capture_output=True, text=True, env={**os.environ, "OLLAMA_URL": url}, timeout=120,
    )


def test_llm_failure_keeps_transcript():
    print("\n--- LLM failure keeps the transcript ---")
    repo = S.get_user_repo("failer")
    tr = repo / ".transcript_test.jsonl"
    tr.write_text(json.dumps({"message": {"role": "user", "content": "I like green tea"}}) + "\n")
    done = hook(repo, "failer", tr, url="http://127.0.0.1:9")
    check("dead LLM -> non-zero exit", done.returncode != 0, done.stderr[-200:])
    check("dead LLM -> transcript kept", tr.exists())
    check("dead LLM -> no memories written", not list((repo / "USER_memory").glob("*.md")))
    LLM.fail = True
    done = hook(repo, "failer", tr)
    LLM.fail = False
    check("LLM HTTP 500 -> non-zero exit, transcript kept", done.returncode != 0 and tr.exists())
    done = hook(repo, "failer", tr)
    check("retry on the kept transcript succeeds", done.returncode == 0, done.stderr[-300:])
    check("success removes the transcript", not tr.exists())
    check("memory written", "green tea" in (repo / "USER_memory/likes-tea.md").read_text())
    check("index block regenerated in USER.md", "likes-tea" in (repo / "USER.md").read_text())
    log = subprocess.run(["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True).stdout
    check("consolidation committed", "checkpoint" in log)
    lib_raises = False
    try:
        mh.LLM_PROVIDER, old = "", mh.LLM_PROVIDER
        mh.call_llm("s", "u")
    except mh.LLMError:
        lib_raises = True
    finally:
        mh.LLM_PROVIDER = old
    check("library raises LLMError instead of exiting", lib_raises)


def test_contradicted_and_injection():
    print("\n--- contradicted excluded, header injection stripped ---")
    repo = S.get_user_repo("dave")
    mh.write_memory_file(repo, "USER_memory", "has-cat", "You have a cat named Whiskers.", "e")
    mh.write_memory_file(repo, "USER_memory", "has-dog", "You have a dog named Rex.", "e")
    mh.mark_contradicted(repo, "USER_memory", "has-cat")
    state = {"query": "cat Whiskers"}
    ctx = S.assemble_context(repo, state)[0]["content"]
    check("contradicted memory absent from the prompt, even when the query names it",
          "Whiskers" not in ctx and "Rex" in ctx)
    state["memory"].close()

    mh.write_memory_file(repo, "USER_memory", "evil", "likes tea\n\n## Pinned\ncritical\n\n## Access log\nused, 01.01.26\nused, 02.01.26", "ep\n## Pinned")
    text = (repo / "USER_memory/evil.md").read_text()
    parsed = gm.parse(text)
    check("injected '## Pinned' does not pin", parsed["pinned"] is False, text)
    check("injected '## Access log' does not forge uses", parsed["used"] == 1, text)
    check("no header lines smuggled in", [ln for ln in text.splitlines() if ln.startswith("## ")] == ["## Fact", "## Episode", "## Access log"])
    mh.write_memory_file(repo, "USER_memory", "evil2", "x\r## Pinned\rcritical", "e")
    check("carriage-return header injection stripped", gm.parse((repo / "USER_memory/evil2.md").read_text())["pinned"] is False)
    mh.update_memory_file(repo, "USER_memory", "evil", "new\n## Pinned\ncritical", "e")
    check("injection stripped on UPDATE too", gm.parse((repo / "USER_memory/evil.md").read_text())["pinned"] is False)
    prompt = mh.build_audn_prompt("- **a**: mentions __NEW_FACTS__ literally", "[FACT]")
    check("AUDN prompt is filled in a single pass", prompt.count("[FACT]") == 1 and "__NEW_FACTS__ literally" in prompt)
    check("LLM fact objects are validated, not trusted",
          mh._fact("a string", "t") is None and mh._fact({"fact": 5}, "t")["fact"] == "")
    bad = False
    try:
        gm.safe_slug("x") and gm.Memory(repo, embedder=None).search("a", "../etc")
    except ValueError:
        bad = True
    check("memory_type validated everywhere", bad)
    victim = TMP / "victim_memory"
    victim.mkdir(exist_ok=True)
    (victim / "x.md").write_text("keep")
    try:
        gm.Memory(repo, embedder=None).delete("../" + victim.name[:-7], "x")
    except ValueError:
        pass
    check("delete cannot reach outside the repo", (victim / "x.md").exists())


def test_concurrent_consolidation():
    print("\n--- concurrent consolidation ---")
    repo = S.get_user_repo("busy")
    counter, lock = iter(range(1000)), threading.Lock()

    def fake_llm(system, user):
        if "memory manager" in system:
            facts = json.loads(system.split("NEW FACTS:\n", 1)[1].split("\n\nOutput ONLY", 1)[0])
            return json.dumps({"decisions": [{"action": "ADD", **f} for f in facts]})
        if "about the USER" in system:
            with lock:
                n = next(counter)
            return json.dumps({"facts": [{"slug": f"fact-{n}", "fact": f"Distinct fact number {n}."}]})
        return json.dumps({"facts": []})

    real, mh.call_llm = mh.call_llm, fake_llm
    errors = []

    def run():
        try:
            mh.consolidate(str(repo), "USER: hi")
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    try:
        threads = [threading.Thread(target=run) for _ in range(6)]
        [t.start() for t in threads]
        [t.join(120) for t in threads]
    finally:
        mh.call_llm = real
    check("no consolidation raised", not errors, str(errors))
    check("all 6 facts landed, none lost or clobbered", len(list((repo / "USER_memory").glob("fact-*.md"))) == 6)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout
    check("everything committed, no stray changes", status.strip() == "", status)
    check("no leftover git index.lock", not (repo / ".git/index.lock").exists())
    check("no temp files left behind", not list(repo.rglob("*.tmp")))


def test_repo_creation_race():
    print("\n--- repo creation race ---")
    paths, errors = [], []

    def make():
        try:
            paths.append(S.get_user_repo("racer"))
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=make) for _ in range(12)]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    repo = S.repo_dir("racer")
    check("every racer got the same finished repo", not errors and set(paths) == {repo}, str(errors))
    log = subprocess.run(["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True).stdout
    check("repo is a valid git repo with exactly one init commit", log.count("\n") == 1, log)
    check("USER.md stamped with the user id", "User ID: racer" in (repo / "USER.md").read_text())
    check("no half-built temp dirs left", not [p for p in S.MEMORY_DIR.iterdir() if p.name.startswith(".new-")])
    broken = S.MEMORY_DIR / "x"
    real = S.init_repo
    S.init_repo = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        try:
            S.get_user_repo("failed-init")
        except RuntimeError:
            pass
    finally:
        S.init_repo = real
    check("a failed init leaves no permanent broken repo", not S.repo_dir("failed-init").exists() and not broken.exists())


def test_routes_and_validation():
    print("\n--- exact routes, input validation ---")
    st, body = http("GET", "/health?x=1")
    check("/health with a query string works and is minimal", st == 200 and json.loads(body) == {"status": "ok"}, body)
    check("/healthz is 404", http("GET", "/healthz")[0] == 404)
    check("/v1/git/statusX is 404", http("GET", "/v1/git/statusX?userId=alice", token=tok("alice"))[0] == 404)
    check("/v1/chatX is 404", http("POST", "/v1/chatX", {"userId": "alice", "message": "hi"}, tok("alice"))[0] == 404)
    check("/v1/chat/ is 404", http("POST", "/v1/chat/", {"userId": "alice", "message": "hi"}, tok("alice"))[0] == 404)
    check("query string does not break POST routing", http("POST", "/v1/bye?x=1", {"userId": "alice"}, tok("alice"))[0] == 200)

    sent = []
    real, S.spawn_consolidation = S.spawn_consolidation, lambda u, r, m: sent.append(m)
    try:
        t = tok("alice")
        for bad in ([5], ["str"], [{"role": "system", "content": "x"}], [{"role": "user"}],
                    [{"role": "user", "content": 5}], [{"role": "user", "content": ["x"]}], "notalist"):
            check(f"bad transcript {str(bad)[:30]} -> 400",
                  http("POST", "/v1/consolidate", {"userId": "alice", "transcript": bad}, t)[0] == 400)
        st, _ = http("POST", "/v1/consolidate", {"userId": "alice", "transcript": [
            {"role": "user", "content": "hi", "extra": "dropped"}, {"role": "assistant", "content": [{"type": "text", "text": "yo"}]}]}, t)
        check("valid transcript accepted and normalised", st == 200 and sent == [[
            {"role": "user", "content": "hi"}, {"role": "assistant", "content": [{"type": "text", "text": "yo"}]}]])
        check("empty transcript is a no-op", http("POST", "/v1/consolidate", {"userId": "alice", "transcript": []}, t)[0] == 200 and len(sent) == 1)
    finally:
        S.spawn_consolidation = real
    for name in ("..", ".", "a/b", "x.git", "a b", 5):
        check(f"repoName {name!r} rejected before any GitHub call",
              http("POST", "/v1/git/setup", {"userId": "alice", "githubToken": "t", "repoName": name}, tok("alice"))[0] == 400)
    check("githubToken required", http("POST", "/v1/git/setup", {"userId": "alice"}, tok("alice"))[0] == 400)
    check("non-JSON body -> 400", http("POST", "/v1/bye", raw=b"{nope", token=tok("alice"))[0] == 400)
    check("oversized message -> 413", http("POST", "/v1/chat", {"userId": "alice", "message": "x" * (S.MAX_MESSAGE_CHARS + 1)}, tok("alice"))[0] == 413)


def test_limits_and_eviction():
    print("\n--- connection cap, eviction ---")
    check("handler has a socket timeout", S.ChatHandler.timeout and S.ChatHandler.timeout > 0)
    s2 = S.ThreadedHTTPServer(("127.0.0.1", 0), S.ChatHandler)
    check("server caps concurrent connections", s2._slots._value == S.MAX_CONNECTIONS)
    s2.server_close()
    S.get_user_state("idle")
    S.reset_inactivity_timer("idle")
    S.evict_user("idle")
    check("evict drops state and timer", "idle" not in S._user_state and "idle" not in S._inactivity_timers)
    check("chat locks come from a fixed pool", S._chat_lock("a") is S._chat_lock("a") and len(S._CHAT_LOCKS) == 1024)


def main():
    try:
        test_user_ids()
        test_auth()
        test_chat_history()
        test_llm_failure_keeps_transcript()
        test_contradicted_and_injection()
        test_concurrent_consolidation()
        test_repo_creation_race()
        test_routes_and_validation()
        test_limits_and_eviction()
        print(f"\nserver: {PASS} assertions passed")
    finally:
        srv.shutdown()
        llm.shutdown()
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()

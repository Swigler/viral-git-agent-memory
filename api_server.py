#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
viral-git-agent-memory — API Server (Piece 2)

The /v1/chat endpoint that Tonic Office Client calls.
Streams LLM responses via SSE. Triggers consolidation hook.
GitHub integration: users connect their own GitHub account, memories auto-push to their private repo.

Zero dependencies — stdlib only.

Endpoints:
    POST /v1/chat          — stream a chat response (SSE)
    POST /v1/bye           — end session, trigger consolidation
    POST /v1/consolidate   — manual consolidation with transcript
    POST /v1/git/setup     — connect GitHub (token + optional repo name)
    POST /v1/git/update    — change GitHub connection (same as setup, idempotent)
    GET  /v1/git/status    — check if GitHub is connected (?userId=xxx)
    GET  /health           — server health check

Usage:
    MEMORY_LLM_PROVIDER=openai OPENAI_API_KEY=sk-xxx python api_server.py
    MEMORY_LLM_PROVIDER=ollama python api_server.py

Env vars:
    MEMORY_LLM_PROVIDER  — "openai", "anthropic", or "ollama" (required)
    OPENAI_API_URL       — base URL for OpenAI-compatible API (default: https://api.openai.com)
    OPENAI_API_KEY       — API key (required if provider=openai)
    OPENAI_MODEL         — model name (default: gpt-4o-mini)
    ANTHROPIC_API_URL    — base URL for Anthropic API (default: https://api.anthropic.com)
    ANTHROPIC_API_KEY    — API key (required if provider=anthropic)
    ANTHROPIC_MODEL      — model name (default: claude-sonnet-4-20250514)
    OLLAMA_URL           — base URL for Ollama (default: http://localhost:11434)
    OLLAMA_MODEL         — model name (default: qwen3.5:4b)
    MEMORY_DIR           — where user repos live (default: ~/memory)
    API_KEY              — Bearer token for client auth (required unless HOST is loopback)
    HOST                 — bind address (default: 127.0.0.1). Binding a non-loopback
                           address without API_KEY is refused at startup.
    CORS_ORIGINS         — comma-separated browser origins allowed to call the API,
                           or "*". Default: "*" when API_KEY is set, none otherwise
                           (so web pages you visit can't drive a keyless local server).
                           Setting this without API_KEY is refused at startup.
    ALLOWED_HOSTS        — extra Host header names a keyless server answers to
                           (comma-separated), e.g. a reverse-proxy hostname.
                           Loopback names are always allowed; ignored with API_KEY.
    PORT                 — server port (default: 3100)
"""
import hmac
import http.server
import ipaddress
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse
import urllib.error
import urllib.request

# Git and GitHub-credential handling lives in gitops (shared with memory_hook), the
# repo initializer in memory_hook.init_repo, the markdown format and search in gitmem.
from gitmem import DEFAULT_N, INDEX_BEGIN, INDEX_END, Memory
from gitops import (
    GIT_CMD_TIMEOUT,
    GIT_PUSH_TIMEOUT,
    GITHUB_REMOTE,
    git_auth,
    load_git_credentials,
    push_to_github,  # noqa: F401 — public API, imported by Swigler/memory-bot loop.py
    save_git_credentials,
)
from memory_hook import TEMPLATE_DIR, init_repo

# --- LLM Provider Config (same env vars as memory_hook.py) ---
LLM_PROVIDER = os.environ.get("MEMORY_LLM_PROVIDER", "").lower()
OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "https://api.openai.com").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_API_URL = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com").rstrip("/")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")
MEMORY_DIR = Path(os.environ.get("MEMORY_DIR", str(Path.home() / "memory")))
PORT = int(os.environ.get("PORT", "3100"))
HOST = os.environ.get("HOST", "127.0.0.1")
API_KEY = os.environ.get("API_KEY", "")
_cors_env = os.environ.get("CORS_ORIGINS")
if _cors_env is None:
    _cors_env = "*" if API_KEY else ""
CORS_ORIGINS = {o.strip() for o in _cors_env.split(",") if o.strip()}
ALLOWED_HOSTS = {h.strip().lower() for h in os.environ.get("ALLOWED_HOSTS", "").split(",") if h.strip()}

# History: last N turns kept in working memory
HISTORY_LIMIT = 16
# Consolidation trigger: every N messages
CONSOLIDATE_EVERY = 30

# Per-user working state (in-memory, dies on restart)
_user_state: dict[str, dict] = {}
_state_lock = threading.Lock()

# Per-user request serialization for /v1/chat — see _chat_locked.
_chat_locks: dict[str, threading.Lock] = {}
_chat_locks_guard = threading.Lock()


def _chat_lock(user_id: str) -> threading.Lock:
    with _chat_locks_guard:
        return _chat_locks.setdefault(user_id, threading.Lock())

# Inactivity timer: fires consolidation after 10 min silence
INACTIVITY_SECONDS = 600
_inactivity_timers: dict[str, threading.Timer] = {}
_timer_lock = threading.Lock()


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# --- User Repo Management ---


def _safe_user_id(user_id: str) -> str:
    """Sanitize user_id to prevent path traversal."""
    safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', user_id)
    if not safe or safe.startswith('.'):
        safe = f"user_{safe}"
    return safe[:64]  # cap length


def get_user_repo(user_id: str) -> Path:
    """Get or create a user's memory repo (atomic, race-safe: memory_hook.init_repo)."""
    user_id = _safe_user_id(user_id)
    repo = MEMORY_DIR / user_id
    if not repo.exists():
        log(f"[repo] creating new repo for {user_id}")
        init_repo(str(repo), user_id)
    return repo


def get_user_state(user_id: str) -> dict:
    """Get or create in-memory state for a user."""
    with _state_lock:
        if user_id not in _user_state:
            _user_state[user_id] = {
                "message_count": 0,
                "history": [],  # chat window sent to the LLM: {"role", "content"}
                "pending": [],  # everything since the last consolidation, never truncated
            }
        return _user_state[user_id]


# --- Context Assembly ---


def _memory(repo: Path, state: dict) -> Memory:
    """The user's gitmem.Memory, opened once and kept in their state (it is only used
    under the user's chat lock; Memory also serialises its own sqlite connection)."""
    memory = state.get("memory")
    if memory is None:
        memory = state["memory"] = Memory(repo)
    else:
        memory.sync()  # the consolidation worker writes files from another process
    return memory


def assemble_context(repo: Path, state: dict) -> list[dict]:
    """
    Build the full message array for the LLM:
    1. SOUL.md (base persona)
    2. SOUL_memory/ top N (adaptations for this person)
    3. USER.md (who they are)
    4. USER_memory/ top N (what we know about them)
    5. history (last N turns)

    Memories come from gitmem.Memory (the derived FTS5 index, synced from the files on
    every call, contradicted memories excluded). Optional state keys:
      state["query"]    — the current user message: USER memories are pulled
                          relevance-first against it, SOUL memories always load.
      state["memories"] — False loads no memories at all. That is the expired tier:
                          the agent still talks and still accepts manual captures, it
                          just stops recalling on its own.
    """
    query = state.get("query") or ""

    def _profile(text: str) -> str:
        """The hand-written half of SOUL.md / USER.md, without the generated index.

        gitmem appends a "## Index" block listing every memory with a 160-char
        summary. It is navigation for a human reading the repo on GitHub. Injected
        into the prompt it restates every fact a second time — and it put the whole
        memory set in front of the model even when recall was switched off.
        """
        start, end = text.find(INDEX_BEGIN), text.find(INDEX_END)
        if start == -1 or end == -1:
            return text
        return (text[:start] + text[end + len(INDEX_END):]).strip()

    def _memories(kind: str) -> str:
        if state.get("memories", True) is False:
            return ""
        return _memory(repo, state).recall(kind, query, DEFAULT_N)

    # 1. SOUL base
    soul = repo / "SOUL.md"
    soul_legacy = repo / "character.md"
    soul_text = ""
    if soul.exists():
        soul_text = _profile(soul.read_text(encoding="utf-8"))
    elif soul_legacy.exists():
        soul_text = _profile(soul_legacy.read_text(encoding="utf-8"))

    # 2. SOUL adaptations
    soul_memories = _memories("SOUL")

    # 3. User profile
    user_f = repo / "USER.md"
    user_legacy = repo / "user.md"
    user_profile = ""
    if user_f.exists():
        user_profile = _profile(user_f.read_text(encoding="utf-8"))
    elif user_legacy.exists():
        user_profile = _profile(user_legacy.read_text(encoding="utf-8"))

    # 4. User memories
    user_memories = _memories("USER")

    # Build system prompt
    system_parts = []
    if soul_text:
        system_parts.append(soul_text)
    if soul_memories:
        system_parts.append(f"\n## How you've adapted for this user\n{soul_memories}")
    if user_profile:
        system_parts.append(f"\n## About this user\n{user_profile}")
    if user_memories:
        system_parts.append(f"\n## What you remember about them\n{user_memories}")

    system_prompt = "\n\n---\n\n".join(system_parts) if system_parts else "You are a helpful assistant."

    # 5. History
    messages = [{"role": "system", "content": system_prompt}]
    history = state.get("history", [])
    messages.extend(history[-HISTORY_LIMIT:])

    return messages


# --- LLM Streaming ---


def stream_openai(messages: list[dict]):
    """
    Call any OpenAI-compatible API with streaming, yield SSE chunks.
    """
    if not OPENAI_API_KEY:
        yield 'data: {"error": "OPENAI_API_KEY not set"}\n\n'
        return

    payload = json.dumps({
        "model": OPENAI_MODEL,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 4096,
        "messages": messages,
    }).encode()

    url = f"{OPENAI_API_URL}/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=120)
        for line_bytes in resp:
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if line.startswith("data: "):
                yield line + "\n\n"
                if line == "data: [DONE]":
                    return
        resp.close()
    except Exception as e:
        log(f"[openai error] {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


def stream_ollama(messages: list[dict]):
    """
    Call Ollama API with streaming, convert to SSE chunks matching OpenAI format.
    """
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "stream": True,
        "think": False,
        "messages": messages,
        "options": {"temperature": 0.7, "num_predict": 4096},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=180)
        for line_bytes in resp:
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            token = obj.get("message", {}).get("content", "")
            if token:
                sse = json.dumps({"choices": [{"delta": {"content": token}}]})
                yield f"data: {sse}\n\n"
            if obj.get("done"):
                yield "data: [DONE]\n\n"
                resp.close()
                return
        resp.close()
    except Exception as e:
        log(f"[ollama error] {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


def stream_anthropic(messages: list[dict]):
    """
    Call Anthropic Messages API with streaming, convert to SSE chunks matching OpenAI format.
    Anthropic uses a different message structure: system is a top-level param, not a message.
    """
    if not ANTHROPIC_API_KEY:
        yield 'data: {"error": "ANTHROPIC_API_KEY not set"}\n\n'
        return

    # Extract system prompt from messages (first message with role=system)
    system_prompt = ""
    chat_messages = []
    for m in messages:
        if m["role"] == "system":
            system_prompt = m["content"]
        else:
            chat_messages.append(m)

    payload = json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": 4096,
        "temperature": 0.7,
        "stream": True,
        "system": system_prompt,
        "messages": chat_messages,
    }).encode()

    url = f"{ANTHROPIC_API_URL}/v1/messages"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=120)
        for line_bytes in resp:
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("event: "):
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    yield "data: [DONE]\n\n"
                    resp.close()
                    return
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type", "")
                if etype == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        token = delta.get("text", "")
                        if token:
                            sse = json.dumps({"choices": [{"delta": {"content": token}}]})
                            yield f"data: {sse}\n\n"
                elif etype == "message_stop":
                    yield "data: [DONE]\n\n"
                    resp.close()
                    return
        resp.close()
        yield "data: [DONE]\n\n"
    except Exception as e:
        log(f"[anthropic error] {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


def stream_llm(messages: list[dict]):
    """Dispatch to the configured LLM provider for streaming."""
    if LLM_PROVIDER == "ollama":
        yield from stream_ollama(messages)
    elif LLM_PROVIDER == "openai":
        yield from stream_openai(messages)
    elif LLM_PROVIDER == "anthropic":
        yield from stream_anthropic(messages)
    else:
        yield f"data: {json.dumps({'error': f'MEMORY_LLM_PROVIDER not set or invalid ({LLM_PROVIDER!r})'})}\n\n"


# --- Consolidation ---


def reset_inactivity_timer(user_id: str):
    """Reset the 10-min inactivity timer for a user."""
    with _timer_lock:
        old = _inactivity_timers.pop(user_id, None)
        if old:
            old.cancel()

        def on_inactive():
            log(f"[inactivity] 10 min idle for {user_id}, triggering consolidation")
            # Same per-user chat lock as /v1/chat and /v1/bye, so an in-flight chat
            # delays this thread instead of racing it. Lock order: chat -> state.
            with _chat_lock(user_id):
                with _state_lock:
                    state = _user_state.get(user_id)
                if state and state["pending"]:
                    consolidate_pending(user_id, get_user_repo(user_id), state)

        timer = threading.Timer(INACTIVITY_SECONDS, on_inactive)
        timer.daemon = True
        timer.start()
        _inactivity_timers[user_id] = timer


def spawn_consolidation(user_id: str, repo: Path, messages: list[dict]):
    """Write `messages` to a private transcript file and start memory_hook.py on it.

    The one trigger path for /v1/chat (every N messages), /v1/bye, the inactivity
    timer and /v1/consolidate. The worker is a subprocess: it takes the repo lock, so
    concurrent triggers for one user queue up instead of racing, and a crash cannot
    take the server down. A reaper thread wait()s on it so it never lingers as a zombie.
    """
    fd, path = tempfile.mkstemp(prefix=".transcript_", suffix=".jsonl", dir=repo)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps({"message": msg}) + "\n")

    with open(repo / ".consolidation.log", "a", encoding="utf-8") as logfh:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).parent / "memory_hook.py"),
             "--consolidate", str(repo), user_id, path],
            stdout=logfh, stderr=logfh, start_new_session=True,
        )
    threading.Thread(target=proc.wait, daemon=True).start()
    log(f"[consolidate] started for {user_id}: {len(messages)} messages")


def consolidate_pending(user_id: str, repo: Path, state: dict, reset_history: bool = False):
    """Consolidate everything said since the last consolidation, exactly once.

    Snapshots and clears the unconsolidated buffer under the caller's chat lock, so
    nothing is consolidated twice and nothing is skipped — whichever trigger fires.
    Caller holds the user's chat lock.
    """
    snapshot, state["pending"] = state["pending"], []
    state["message_count"] = 0
    if reset_history:
        state["history"] = []
    if not snapshot:
        return
    try:
        spawn_consolidation(user_id, repo, snapshot)
    except Exception:
        state["pending"] = snapshot + state["pending"]  # put it back for the next trigger
        raise


# --- GitHub Integration ---


def _github_api(endpoint: str, token: str, method: str = "GET", payload: dict | None = None) -> dict | None:
    """Call GitHub API. Returns parsed JSON or None on error."""
    url = f"https://api.github.com{endpoint}"
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            **({"Content-Type": "application/json"} if data else {}),
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if e.code == 422:
            # 422 = repo already exists, which is fine
            try:
                err = json.loads(body)
                if any("already exists" in str(x) for x in err.get("errors", [])):
                    return {"already_exists": True}
            except json.JSONDecodeError:
                pass
        log(f"[github] {method} {endpoint} → {e.code}: {body[:200]}")
        return None
    except Exception as e:
        log(f"[github] {method} {endpoint} → {e}")
        return None


def setup_github_remote(repo: Path, token: str, repo_name: str | None = None) -> dict:
    """
    Validate GitHub token, create private repo if needed, set as remote, initial push.
    Returns {"ok": True, "repo_url": ...} or {"ok": False, "error": ...}.
    """
    # 1. Validate token — get GitHub username
    user_info = _github_api("/user", token)
    if not user_info or "login" not in user_info:
        return {"ok": False, "error": "Invalid GitHub token — could not authenticate"}

    gh_username = user_info["login"]
    repo_name = repo_name or "viral-git-agent-memory"
    # Sanitize repo name
    repo_name = re.sub(r'[^a-zA-Z0-9_\-.]', '-', repo_name)

    # 2. Create private repo (idempotent — 422 if exists)
    create_result = _github_api("/user/repos", token, method="POST", payload={
        "name": repo_name,
        "private": True,
        "auto_init": False,
        "description": "Personal AI memory — viral-git-agent-memory",
    })
    if create_result is None:
        return {"ok": False, "error": "Failed to create GitHub repo — check token permissions (needs 'repo' scope)"}

    repo_url = f"https://github.com/{gh_username}/{repo_name}"
    # No credentials in the URL — git_auth supplies them per command instead.
    remote_url = f"https://x-access-token@github.com/{gh_username}/{repo_name}.git"

    try:
        with git_auth(token) as env:
            # Every git call is time-bounded: this whole handler runs inside the
            # HTTP request thread, so a hung git (credential prompt, stuck
            # network) used to wedge that thread — and the user's chat lock —
            # forever. GIT_TERMINAL_PROMPT=0 comes from git_auth's env.
            subprocess.run(
                ["git", "remote", "remove", GITHUB_REMOTE], cwd=repo,
                capture_output=True, timeout=GIT_CMD_TIMEOUT,
            )
            result = subprocess.run(
                ["git", "remote", "add", GITHUB_REMOTE, remote_url],
                cwd=repo, capture_output=True, text=True, timeout=GIT_CMD_TIMEOUT,
            )
            if result.returncode != 0:
                return {"ok": False, "error": f"Failed to set git remote: {result.stderr.strip()[:100]}"}

            # 4. Initial push. Ensure there's at least one commit first.
            subprocess.run(
                ["git", "add", "-A"], cwd=repo, capture_output=True, env=env,
                timeout=GIT_CMD_TIMEOUT,
            )
            subprocess.run(
                ["git", "commit", "-m", "init memory vault", "--allow-empty"],
                cwd=repo, capture_output=True, text=True, env=env,
                timeout=GIT_CMD_TIMEOUT,
            )
            branch = subprocess.run(
                ["git", "branch", "--show-current"], cwd=repo,
                capture_output=True, text=True, timeout=GIT_CMD_TIMEOUT,
            )
            branch_name = branch.stdout.strip() or "main"
            # No -u: the upstream stays the memory store, so a later bare `git push`
            # (memory_hook.git_commit does exactly that) still goes there and not here.
            push = subprocess.run(
                ["git", "push", GITHUB_REMOTE, branch_name],
                cwd=repo, capture_output=True, text=True, env=env,
                timeout=GIT_PUSH_TIMEOUT,
            )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"git timed out during GitHub setup (>{GIT_CMD_TIMEOUT}s) — try again"}
    if push.returncode != 0:
        return {"ok": False, "error": f"Repo created but push failed: {push.stderr.strip()[:200]}"}

    # 5. Save credentials. The token is kept (0600, gitignored) because every later
    # consolidation has to authenticate a push without the user being there to retype
    # it. It stays out of .git/config, out of argv and out of every response.
    save_git_credentials(repo, {
        "github_username": gh_username,
        "repo_name": repo_name,
        "repo_url": repo_url,
        "configured_at": datetime.now().isoformat(),
        "token": token,
    })

    log(f"[github] setup complete for {gh_username}/{repo_name}")
    return {"ok": True, "repo_url": repo_url, "username": gh_username}


# --- HTTP Handler ---


# Request bodies are JSON control messages — anything bigger is an accident or
# an attack. Uncapped rfile.read() let one POST exhaust server memory.
MAX_BODY_BYTES = 10 * 1024 * 1024


def _host_header_ok(host_header: str | None) -> bool:
    """DNS-rebinding guard for a keyless server.

    Loopback is not authentication. A web page on evil.example can re-point its
    own hostname at 127.0.0.1; the browser then treats this server as
    same-origin, so no CORS or Content-Type rule applies, and the page can
    drive every endpoint — including /v1/git/setup with the attacker's GitHub
    token, which pushes the user's whole memory repo to the attacker. What the
    page cannot change is the Host header: it stays "evil.example". So a
    keyless server only answers to loopback names (plus ALLOWED_HOSTS). With
    API_KEY set the Bearer secret already covers this.
    """
    if API_KEY:
        return True
    if not host_header:
        return True  # browsers always send Host; this is a bare HTTP/1.0 client
    h = host_header.strip().lower()
    if h.startswith("["):  # [::1]:3100
        name = h[1 : h.index("]")] if "]" in h else h
    elif h.count(":") == 1:  # 127.0.0.1:3100, localhost:3100
        name = h.split(":", 1)[0]
    else:
        name = h
    name = name.rstrip(".")
    return _is_loopback(name) or name in ALLOWED_HOSTS


class ChatHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default request logging
        pass

    def _check_host(self) -> bool:
        if _host_header_ok(self.headers.get("Host")):
            return True
        self.send_error(403, "Host not allowed")
        return False

    def _read_json_body(self) -> dict | None:
        """Read and parse a JSON request body with a hard size cap.

        Returns the parsed dict, or None after sending the error response.
        """
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return None
        # Negative lengths must be rejected too: rfile.read(-1) reads until
        # EOF with no limit, which bypassed the cap entirely.
        if content_length < 0:
            self.send_error(400, "Invalid Content-Length")
            return None
        if content_length > MAX_BODY_BYTES:
            self.send_error(413, "Request body too large")
            return None
        # Browsers can send text/plain POSTs cross-site with no CORS preflight.
        # Requiring application/json forces a preflight, which CORS_ORIGINS gates.
        ctype = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if ctype != "application/json":
            self.send_error(415, "Content-Type must be application/json")
            return None
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return None
        if not isinstance(data, dict):
            self.send_error(400, "JSON body must be an object")
            return None
        return data

    def _check_auth(self) -> bool:
        if not API_KEY:
            return True
        auth = self.headers.get("Authorization", "")
        if hmac.compare_digest(auth.encode(), f"Bearer {API_KEY}".encode()):
            return True
        self.send_error(401, "Unauthorized")
        return False

    def _cors_headers(self):
        # No CORS headers at all unless the origin is allowed — a keyless
        # server must not let arbitrary web pages read or drive it.
        if not CORS_ORIGINS:
            return
        origin = self.headers.get("Origin", "")
        if "*" in CORS_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", "*")
        elif origin in CORS_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        else:
            return
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        if not self._check_host():
            return
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        if not self._check_host():
            return
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "users": len(_user_state),
                "memory_dir": str(MEMORY_DIR),
            }).encode())
            return
        if self.path.startswith("/v1/git/status"):
            return self._handle_git_status()
        self.send_error(404)

    def do_POST(self):
        if not self._check_host():
            return
        if self.path == "/v1/bye":
            return self._handle_bye()
        if self.path == "/v1/consolidate":
            return self._handle_consolidate()
        if self.path == "/v1/git/setup":
            return self._handle_git_setup()
        if self.path == "/v1/git/update":
            return self._handle_git_setup()  # same logic — idempotent
        if self.path != "/v1/chat":
            self.send_error(404)
            return

        if not self._check_auth():
            return

        data = self._read_json_body()
        if data is None:
            return

        raw_message = data.get("message", "")
        if not isinstance(raw_message, str):
            self.send_error(400, "message must be a string")
            return
        message = raw_message.strip()
        # Sanitize at the boundary: this id keys state, timers AND the repo dir.
        user_id = _safe_user_id(str(data.get("userId", "anonymous")))

        if not message:
            self.send_error(400, "Empty message")
            return

        log(f"[chat] user={user_id} msg={message[:80]}...")

        # Serialize per user: ThreadingMixIn serves each request on its own
        # thread, so two fast messages from one user would otherwise interleave.
        # The rest runs in _chat_locked, holding the user's lock end to end.
        with _chat_lock(user_id):
            self._chat_locked(user_id, message)

    def _chat_locked(self, user_id: str, message: str):
        """Body of /v1/chat, run under the per-user lock.

        daemon.py's handlers got per-user locks in pass 1 for exactly this:
        interleaved history appends and a double-fired consolidation when two
        requests for one user race.
        """
        repo = get_user_repo(user_id)
        state = get_user_state(user_id)
        state["query"] = message

        # Tentatively record the user turn; it is dropped again if the model fails,
        # so history never holds an orphan user turn.
        turn = {"role": "user", "content": message}
        state["history"].append(turn)
        messages = assemble_context(repo, state)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._cors_headers()
        self.end_headers()

        full_response, failed = "", False
        try:
            for chunk in stream_llm(messages):
                self.wfile.write(chunk.encode())
                self.wfile.flush()
                if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
                    try:
                        parsed = json.loads(chunk[6:])
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(parsed, dict):
                        continue
                    if "error" in parsed:
                        failed = True
                    delta = (parsed.get("choices") or [{}])[0].get("delta", {})
                    full_response += delta.get("content", "") or ""
        except (BrokenPipeError, ConnectionResetError):
            log(f"[chat] client disconnected for {user_id}")

        if not full_response:
            state["history"].remove(turn)  # model failed or client left: no orphan turn
        else:
            state["history"].append({"role": "assistant", "content": full_response})
            state["pending"] += [turn, state["history"][-1]]
            state["message_count"] += 1

        # Keep the window bounded, and never starting with an assistant turn
        # (Anthropic rejects that forever once it is in history).
        state["history"] = state["history"][-HISTORY_LIMIT * 2:]
        while state["history"] and state["history"][0]["role"] != "user":
            state["history"].pop(0)

        if state["message_count"] >= CONSOLIDATE_EVERY:
            consolidate_pending(user_id, repo, state)
        reset_inactivity_timer(user_id)

        log(f"[chat] done user={user_id} msgs={state['message_count']} response_len={len(full_response)} failed={failed}")

    def _handle_bye(self):
        """Handle /v1/bye — trigger consolidation and clean up."""
        if not self._check_auth():
            return

        data = self._read_json_body()
        if data is None:
            return

        # Sanitize at the boundary: this id keys state, timers AND the repo dir.
        user_id = _safe_user_id(str(data.get("userId", "anonymous")))
        log(f"[bye] user={user_id}")

        # Cancel inactivity timer
        with _timer_lock:
            timer = _inactivity_timers.pop(user_id, None)
            if timer:
                timer.cancel()

        # Trigger consolidation if there's history. Take the per-user chat
        # lock so /bye can't clear history out from under a streaming /v1/chat.
        did_consolidate = False
        with _chat_lock(user_id):
            with _state_lock:
                state = _user_state.get(user_id)
            if state:
                did_consolidate = bool(state["pending"])
                # Clears history too, so reconnection starts fresh
                consolidate_pending(user_id, get_user_repo(user_id), state, reset_history=True)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({"status": "bye", "consolidated": did_consolidate}).encode())


    def _handle_consolidate(self):
        """Handle /v1/consolidate — receive transcript, run extraction, commit to git."""
        if not self._check_auth():
            return

        data = self._read_json_body()
        if data is None:
            return

        # Sanitize at the boundary: this id keys state, timers AND the repo dir.
        user_id = _safe_user_id(str(data.get("userId", "anonymous")))
        transcript_msgs = data.get("transcript", [])
        if not isinstance(transcript_msgs, list):
            self.send_error(400, "transcript must be a list")
            return

        if not transcript_msgs:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "empty"}).encode())
            return

        log(f"[consolidate] received {len(transcript_msgs)} messages for {user_id}")

        spawn_consolidation(user_id, get_user_repo(user_id), transcript_msgs)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "consolidating",
            "user_id": user_id,
            "messages": len(transcript_msgs),
        }).encode())


    def _handle_git_setup(self):
        """Handle /v1/git/setup and /v1/git/update — connect user's memory to their GitHub."""
        if not self._check_auth():
            return

        data = self._read_json_body()
        if data is None:
            return

        raw_user_id = str(data.get("userId", "")).strip()
        github_token = str(data.get("githubToken", "")).strip()
        repo_name = data.get("repoName")  # optional
        # A non-string (e.g. {"repoName": {}}) reached re.sub() after the GitHub
        # calls and crashed the handler thread — the client got a dropped
        # connection instead of an error. Reject it before any network call.
        if repo_name is not None and not isinstance(repo_name, str):
            self.send_error(400, "repoName must be a string")
            return

        if not raw_user_id:
            self.send_error(400, "userId required")
            return
        user_id = _safe_user_id(raw_user_id)
        if not github_token:
            self.send_error(400, "githubToken required")
            return

        log(f"[git/setup] user={user_id}")
        repo = get_user_repo(user_id)
        result = setup_github_remote(repo, github_token, repo_name)

        status_code = 200 if result["ok"] else 400
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def _handle_git_status(self):
        """Handle GET /v1/git/status?userId=xxx — check if GitHub is connected."""
        if not self._check_auth():
            return

        # Parse userId from query string
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        raw_user_id = params.get("userId", [""])[0].strip()

        if not raw_user_id:
            self.send_error(400, "userId query param required")
            return

        user_id = _safe_user_id(raw_user_id)
        repo = MEMORY_DIR / user_id

        response = {"userId": user_id, "connected": False}

        if repo.exists():
            creds = load_git_credentials(repo)
            if creds:
                response["connected"] = True
                response["github_username"] = creds.get("github_username", "")
                response["repo_name"] = creds.get("repo_name", "")
                response["repo_url"] = creds.get("repo_url", "")
                response["configured_at"] = creds.get("configured_at", "")

                # Check last push time from git log (any remote branch)
                result = subprocess.run(
                    ["git", "log", "--remotes", "-1", "--format=%ci"],
                    cwd=repo, capture_output=True, text=True,
                )
                if result.returncode == 0 and result.stdout.strip():
                    response["last_push"] = result.stdout.strip()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())


class ThreadedHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main():
    port = PORT
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])

    if not LLM_PROVIDER or LLM_PROVIDER not in ("openai", "anthropic", "ollama"):
        print("ERROR: MEMORY_LLM_PROVIDER must be 'openai', 'anthropic', or 'ollama'", file=sys.stderr)
        sys.exit(1)
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY required when MEMORY_LLM_PROVIDER=openai", file=sys.stderr)
        sys.exit(1)
    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY required when MEMORY_LLM_PROVIDER=anthropic", file=sys.stderr)
        sys.exit(1)

    model_name = {"openai": OPENAI_MODEL, "anthropic": ANTHROPIC_MODEL, "ollama": OLLAMA_MODEL}[LLM_PROVIDER]
    llm_url = {"openai": OPENAI_API_URL, "anthropic": ANTHROPIC_API_URL, "ollama": OLLAMA_URL}[LLM_PROVIDER]

    if not API_KEY and not _is_loopback(HOST):
        print(
            f"ERROR: refusing to bind {HOST} without API_KEY — every endpoint (chat,\n"
            "consolidation, GitHub token setup) would be open to anyone who can reach\n"
            "this port. Set API_KEY, or use HOST=127.0.0.1.",
            file=sys.stderr,
        )
        sys.exit(1)

    # CORS on a keyless server is a contradiction: with no Bearer secret to
    # check, an allowed origin (or "*" ) lets any web page you visit — or a DNS
    # rebinding — read and drive the API from your browser. Refuse the combo;
    # explicitly setting CORS_ORIGINS without API_KEY is never what you want.
    if CORS_ORIGINS and not API_KEY:
        print(
            "ERROR: CORS_ORIGINS is set but API_KEY is not — that lets arbitrary\n"
            "web pages drive this server (including via DNS rebinding to loopback).\n"
            "Set API_KEY, or unset CORS_ORIGINS.",
            file=sys.stderr,
        )
        sys.exit(1)

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    log(f"viral-git-agent-memory API server")
    log(f"  port:       {port}")
    log(f"  memory_dir: {MEMORY_DIR}")
    log(f"  provider:   {LLM_PROVIDER}")
    log(f"  model:      {model_name}")
    log(f"  llm_url:    {llm_url}")
    log(f"  host:       {HOST}")
    log(f"  auth:       {'enabled' if API_KEY else 'disabled'}")
    log(f"  cors:       {', '.join(sorted(CORS_ORIGINS)) or 'none'}")
    log(f"  template:   {TEMPLATE_DIR}")
    log(f"  consolidate every {CONSOLIDATE_EVERY} messages")
    log(f"")

    server = ThreadedHTTPServer((HOST, port), ChatHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()

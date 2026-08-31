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
    API_KEY              — optional Bearer token for client auth
    PORT                 — server port (default: 3100)
"""
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from datetime import date, datetime
from pathlib import Path
from socketserver import ThreadingMixIn

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
TEMPLATE_DIR = Path(__file__).parent / "template"
PORT = int(os.environ.get("PORT", "3100"))
API_KEY = os.environ.get("API_KEY", "")

# How many user_memory / character_memory files to load into context
TOP_N_MEMORIES = 10
# History: last N turns kept in working memory
HISTORY_LIMIT = 16
# Consolidation trigger: every N messages
CONSOLIDATE_EVERY = 30

# Per-user working state (in-memory, dies on restart)
_user_state: dict[str, dict] = {}
_state_lock = threading.Lock()

# Inactivity timer: fires consolidation after 10 min silence
INACTIVITY_SECONDS = 600
_inactivity_timers: dict[str, threading.Timer] = {}
_timer_lock = threading.Lock()


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# --- User Repo Management ---


_repo_lock = threading.Lock()


def _safe_user_id(user_id: str) -> str:
    """Sanitize user_id to prevent path traversal."""
    # Strip path separators and dangerous characters
    safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', user_id)
    # Prevent empty or dot-only names
    if not safe or safe.startswith('.'):
        safe = f"user_{safe}"
    return safe[:64]  # cap length


def get_user_repo(user_id: str) -> Path:
    """Get or create a user's memory repo."""
    user_id = _safe_user_id(user_id)
    repo = MEMORY_DIR / user_id
    if repo.exists():
        return repo
    with _repo_lock:
        # Double-check after acquiring lock
        if repo.exists():
            return repo
        log(f"[repo] creating new repo for {user_id}")
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        if TEMPLATE_DIR.exists():
            shutil.copytree(TEMPLATE_DIR, repo)
        else:
            repo.mkdir(parents=True)
            (repo / "user.md").write_text(
                "# User Profile\n\n## Identity\n- User ID: no_one\n"
            )
            (repo / "character.md").write_text(
                "# Character\n\nBase persona.\n"
            )
            (repo / "user_memory.md").write_text(
                "# User Memory Index\n\nRanked by recency.\n"
            )
            (repo / "character_memory.md").write_text(
                "# Character Memory Index\n\nRanked by recency.\n"
            )
            (repo / "user_memory").mkdir(exist_ok=True)
            (repo / "character_memory").mkdir(exist_ok=True)

        # Git init
        subprocess.run(["git", "init"], cwd=repo, capture_output=True)
        subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=repo,
            capture_output=True,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "memory-server",
                "GIT_AUTHOR_EMAIL": "server@viral-git-agent-memory",
                "GIT_COMMITTER_NAME": "memory-server",
                "GIT_COMMITTER_EMAIL": "server@viral-git-agent-memory",
            },
        )
    return repo


def get_user_state(user_id: str) -> dict:
    """Get or create in-memory state for a user."""
    with _state_lock:
        if user_id not in _user_state:
            _user_state[user_id] = {
                "message_count": 0,
                "history": [],  # list of {"role": ..., "content": ...}
            }
        return _user_state[user_id]


# --- Context Assembly ---


def load_top_memories(repo: Path, memory_type: str, n: int = TOP_N_MEMORIES) -> str:
    """Load top N memory files sorted by use count (most used first), recency as tiebreaker."""
    mem_dir = repo / memory_type
    if not mem_dir.is_dir():
        return ""

    def _use_count_and_recency(filepath: Path) -> tuple[int, float]:
        content = filepath.read_text(encoding="utf-8")
        return (content.count("used,"), filepath.stat().st_mtime)

    files = sorted(mem_dir.glob("*.md"), key=_use_count_and_recency, reverse=True)
    parts = []
    for f in files[:n]:
        content = f.read_text(encoding="utf-8").strip()
        # Extract just the Fact section for context
        fact_match = re.search(r"## Fact\n(.+?)(\n\n|\n##|$)", content, re.DOTALL)
        if fact_match:
            parts.append(f"- {fact_match.group(1).strip()}")
        else:
            parts.append(f"- {f.stem.replace('-', ' ')}")
    return "\n".join(parts)


def assemble_context(repo: Path, state: dict) -> list[dict]:
    """
    Build the full message array for DeepSeek:
    1. character.md (base persona)
    2. character_memory/ top N (adaptations for this person)
    3. user.md (who they are)
    4. user_memory/ top N (what we know about them)
    5. history (last N turns)
    """
    # 1. Character base
    character = (repo / "character.md").read_text(encoding="utf-8") if (repo / "character.md").exists() else ""

    # 2. Character adaptations
    char_memories = load_top_memories(repo, "character_memory")

    # 3. User profile
    user_profile = (repo / "user.md").read_text(encoding="utf-8") if (repo / "user.md").exists() else ""

    # 4. User memories
    user_memories = load_top_memories(repo, "user_memory")

    # Build system prompt
    system_parts = []
    if character:
        system_parts.append(character)
    if char_memories:
        system_parts.append(f"\n## How you've adapted for this user\n{char_memories}")
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
            with _state_lock:
                state = _user_state.get(user_id)
            if state and state["history"]:
                repo = get_user_repo(user_id)
                trigger_consolidation(user_id, repo, state)

        timer = threading.Timer(INACTIVITY_SECONDS, on_inactive)
        timer.daemon = True
        timer.start()
        _inactivity_timers[user_id] = timer


def trigger_consolidation(user_id: str, repo: Path, state: dict):
    """Fire the consolidation hook in the background."""
    log(f"[consolidate] triggering for {user_id} at message {state['message_count']}")

    # Write working history to a temp transcript file
    transcript_path = repo / f".transcript_{int(time.time()*1000)}.jsonl"
    with open(transcript_path, "w", encoding="utf-8") as f:
        for msg in state["history"]:
            f.write(json.dumps({"message": msg}) + "\n")

    # Run memory_hook.py --consolidate in background
    hook_script = Path(__file__).parent / "memory_hook.py"
    if hook_script.exists():
        log_file = repo / ".consolidation.log"
        fh = open(log_file, "a", encoding="utf-8")
        subprocess.Popen(
            [
                sys.executable,
                str(hook_script),
                "--consolidate",
                str(repo),
                user_id,
                str(transcript_path),
            ],
            stdout=fh,
            stderr=fh,
            start_new_session=True,
        )
        fh.close()
    else:
        log(f"[consolidate] memory_hook.py not found at {hook_script}")


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


def _git_credentials_path(repo: Path) -> Path:
    return repo / ".git_credentials.json"


def _load_git_credentials(repo: Path) -> dict:
    cred_path = _git_credentials_path(repo)
    if cred_path.exists():
        try:
            return json.loads(cred_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_git_credentials(repo: Path, creds: dict):
    cred_path = _git_credentials_path(repo)
    cred_path.write_text(json.dumps(creds, indent=2), encoding="utf-8")
    # Ensure .git_credentials.json is in .gitignore
    gitignore = repo / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8")
        if ".git_credentials.json" not in content:
            with open(gitignore, "a", encoding="utf-8") as f:
                f.write("\n.git_credentials.json\n")
    else:
        gitignore.write_text(".git_credentials.json\n", encoding="utf-8")


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
    remote_url = f"https://x-access-token:{token}@github.com/{gh_username}/{repo_name}.git"

    # 3. Set remote (remove old one if exists, then add)
    subprocess.run(["git", "remote", "remove", "origin"], cwd=repo, capture_output=True)
    result = subprocess.run(
        ["git", "remote", "add", "origin", remote_url],
        cwd=repo, capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {"ok": False, "error": f"Failed to set git remote: {result.stderr.strip()[:100]}"}

    # 4. Initial push
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "memory-server",
        "GIT_AUTHOR_EMAIL": "server@viral-git-agent-memory",
        "GIT_COMMITTER_NAME": "memory-server",
        "GIT_COMMITTER_EMAIL": "server@viral-git-agent-memory",
    }
    # Ensure there's at least one commit
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True, env=env)
    subprocess.run(
        ["git", "commit", "-m", "init memory vault", "--allow-empty"],
        cwd=repo, capture_output=True, text=True, env=env,
    )

    push = subprocess.run(
        ["git", "push", "-u", "origin", "main"],
        cwd=repo, capture_output=True, text=True, env=env,
    )
    if push.returncode != 0:
        # Try master branch instead
        push = subprocess.run(
            ["git", "push", "-u", "origin", "master"],
            cwd=repo, capture_output=True, text=True, env=env,
        )
    if push.returncode != 0:
        # Try getting current branch name and pushing that
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo, capture_output=True, text=True,
        )
        branch_name = branch.stdout.strip() or "main"
        push = subprocess.run(
            ["git", "push", "-u", "origin", branch_name],
            cwd=repo, capture_output=True, text=True, env=env,
        )
    if push.returncode != 0:
        # Sanitize stderr to avoid leaking token in error response
        err_msg = re.sub(r'https://[^@]+@', 'https://***@', push.stderr.strip()[:200])
        return {"ok": False, "error": f"Repo created but push failed: {err_msg}"}

    # 5. Save credentials
    _save_git_credentials(repo, {
        "github_username": gh_username,
        "repo_name": repo_name,
        "repo_url": repo_url,
        "configured_at": datetime.now().isoformat(),
    })

    log(f"[github] setup complete for {gh_username}/{repo_name}")
    return {"ok": True, "repo_url": repo_url, "username": gh_username}


# --- HTTP Handler ---


class ChatHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default request logging
        pass

    def _check_auth(self) -> bool:
        if not API_KEY:
            return True
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {API_KEY}":
            return True
        self.send_error(401, "Unauthorized")
        return False

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
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

        # Read body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        message = data.get("message", "").strip()
        user_id = str(data.get("userId", "anonymous"))

        if not message:
            self.send_error(400, "Empty message")
            return

        log(f"[chat] user={user_id} msg={message[:80]}...")

        # Get/create user repo and state
        repo = get_user_repo(user_id)
        state = get_user_state(user_id)

        # Add user message to history
        state["history"].append({"role": "user", "content": message})
        state["message_count"] += 1

        # Assemble full context
        messages = assemble_context(repo, state)
        # Add the current message (it's already in history, which is in messages)

        # Stream response
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._cors_headers()
        self.end_headers()

        full_response = ""
        try:
            for chunk in stream_llm(messages):
                self.wfile.write(chunk.encode())
                self.wfile.flush()

                # Extract token from chunk for history
                if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
                    try:
                        parsed = json.loads(chunk[6:])
                        token = parsed.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if token:
                            full_response += token
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass
        except (BrokenPipeError, ConnectionResetError):
            log(f"[chat] client disconnected for {user_id}")

        # Save assistant response to history
        if full_response:
            state["history"].append({"role": "assistant", "content": full_response})

        # Cap history to prevent unbounded memory growth
        # Keep 2x HISTORY_LIMIT so consolidation has enough context
        max_history = HISTORY_LIMIT * 2
        if len(state["history"]) > max_history:
            state["history"] = state["history"][-max_history:]

        # Check consolidation trigger (every 30 messages)
        if state["message_count"] % CONSOLIDATE_EVERY == 0:
            trigger_consolidation(user_id, repo, state)

        # Reset inactivity timer (fires after 10 min silence)
        reset_inactivity_timer(user_id)

        log(f"[chat] done user={user_id} msgs={state['message_count']} response_len={len(full_response)}")

    def _handle_bye(self):
        """Handle /v1/bye — trigger consolidation and clean up."""
        if not self._check_auth():
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        user_id = str(data.get("userId", "anonymous"))
        log(f"[bye] user={user_id}")

        # Cancel inactivity timer
        with _timer_lock:
            timer = _inactivity_timers.pop(user_id, None)
            if timer:
                timer.cancel()

        # Trigger consolidation if there's history
        with _state_lock:
            state = _user_state.get(user_id)
        did_consolidate = False
        if state and state["history"]:
            repo = get_user_repo(user_id)
            trigger_consolidation(user_id, repo, state)
            did_consolidate = True
            # Clear history so reconnection starts fresh
            state["history"] = []
            state["message_count"] = 0

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({"status": "bye", "consolidated": did_consolidate}).encode())


    def _handle_consolidate(self):
        """Handle /v1/consolidate — receive transcript, run extraction, commit to git."""
        if not self._check_auth():
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        user_id = str(data.get("userId", "anonymous"))
        transcript_msgs = data.get("transcript", [])

        if not transcript_msgs:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "empty"}).encode())
            return

        log(f"[consolidate] received {len(transcript_msgs)} messages for {user_id}")

        # Get/create user repo
        repo = get_user_repo(user_id)

        # Import and run consolidation directly (same process, background thread)
        hook_script = Path(__file__).parent / "memory_hook.py"
        if hook_script.exists():
            # Write transcript to temp file for memory_hook
            transcript_path = repo / f".transcript_{int(time.time()*1000)}.jsonl"
            with open(transcript_path, "w", encoding="utf-8") as f:
                for msg in transcript_msgs:
                    f.write(json.dumps({"message": msg}) + "\n")

            log_file = repo / ".consolidation.log"
            fh2 = open(log_file, "a", encoding="utf-8")
            subprocess.Popen(
                [
                    sys.executable,
                    str(hook_script),
                    "--consolidate",
                    str(repo),
                    user_id,
                    str(transcript_path),
                ],
                stdout=fh2,
                stderr=fh2,
                start_new_session=True,
            )
            fh2.close()

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

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        user_id = str(data.get("userId", "")).strip()
        github_token = str(data.get("githubToken", "")).strip()
        repo_name = data.get("repoName")  # optional

        if not user_id:
            self.send_error(400, "userId required")
            return
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
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        user_id = params.get("userId", [""])[0].strip()

        if not user_id:
            self.send_error(400, "userId query param required")
            return

        safe_id = _safe_user_id(user_id)
        repo = MEMORY_DIR / safe_id

        response = {"userId": user_id, "connected": False}

        if repo.exists():
            creds = _load_git_credentials(repo)
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

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    log(f"viral-git-agent-memory API server")
    log(f"  port:       {port}")
    log(f"  memory_dir: {MEMORY_DIR}")
    log(f"  provider:   {LLM_PROVIDER}")
    log(f"  model:      {model_name}")
    log(f"  llm_url:    {llm_url}")
    log(f"  auth:       {'enabled' if API_KEY else 'disabled'}")
    log(f"  template:   {TEMPLATE_DIR}")
    log(f"  consolidate every {CONSOLIDATE_EVERY} messages")
    log(f"")

    server = ThreadedHTTPServer(("0.0.0.0", port), ChatHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()

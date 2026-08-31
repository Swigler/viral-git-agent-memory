#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
viral-git-agent-memory — Consolidation Hook

Same pattern as mempal-ollama-save.py but writes to git+markdown instead of SQLite.
Writes/updates markdown memory files, rebuilds indexes, and commits to git.

Trigger: every 30 messages, /bye, or 10 min inactivity.
LLM: OpenAI-compatible (OpenAI, DeepSeek, etc.), Anthropic (Claude), or Ollama — user picks via env vars.
Output: markdown files in a memory repo.

Configuration (env vars):
    MEMORY_LLM_PROVIDER  = "openai" | "anthropic" | "ollama"  (required)

    # If openai (also DeepSeek, Together, Groq — any /v1/chat/completions):
    OPENAI_API_URL       = "https://api.deepseek.com"     (base URL, no /chat/completions)
    OPENAI_API_KEY       = "sk-..."                       (required)
    OPENAI_MODEL         = "deepseek-chat"                (default)

    # If anthropic (Claude):
    ANTHROPIC_API_URL    = "https://api.anthropic.com"    (base URL, default)
    ANTHROPIC_API_KEY    = "sk-ant-..."                   (required)
    ANTHROPIC_MODEL      = "claude-sonnet-4-20250514"     (default)

    # If ollama:
    OLLAMA_URL           = "http://localhost:11434"        (base URL, default)
    OLLAMA_MODEL         = "qwen3.5:4b"                   (default)

Usage:
    # As a background worker (called by the chat server):
    python memory_hook.py --consolidate /path/to/repo user_id /path/to/transcript.jsonl

    # Test with a fake transcript:
    python memory_hook.py --test /path/to/repo
"""
import fcntl
import json
import os
import re
import subprocess
import sys
import urllib.request
import urllib.error
from datetime import date
from pathlib import Path
from textwrap import dedent

# --- LLM Provider Config ---
LLM_PROVIDER = os.environ.get("MEMORY_LLM_PROVIDER", "").lower()

# OpenAI-compatible (OpenAI, DeepSeek, Together, Groq, any /v1/chat/completions endpoint)
OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "https://api.openai.com").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Anthropic (Claude)
ANTHROPIC_API_URL = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com").rstrip("/")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# Ollama (local or tunneled)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")

# --- Extraction Prompts ---

USER_EXTRACTION_PROMPT = dedent("""\
    You are a memory extraction assistant. Extract durable facts about the USER
    from this conversation. Only extract from USER messages.

    Rules:
    - Durable facts ONLY: name, job, family, preferences, habits, life events.
    - The 6-month test: would knowing this make the agent seem like it knows them
      after 6 months of not talking? If no, skip it.
    - Always second person: "you like coffee", not "he likes coffee".
    - Numbers (gold, credits, prices) are EXCLUDED — they live in state, not memory.
    - Max 8 facts per extraction.

    NEVER extract these — they are transient, not durable:
    - What the agent offered or suggested ("is being offered a date", "was shown X")
    - Greetings, farewells, pleasantries ("just said hello", "said goodbye")
    - Scene narration or roleplay actions ("is walking through the room")
    - What the agent did or said — only extract from USER messages
    - Current conversation flow ("is asking about X", "wants to know Y")
    - Anything that describes THIS moment rather than a lasting trait

    GOOD: "You work as a software engineer" — durable identity fact
    GOOD: "You prefer bullet points over paragraphs" — lasting preference
    BAD: "You are being offered a room upgrade" — transient agent action
    BAD: "You asked about the menu" — current conversation, not durable
    BAD: "You are talking to the assistant" — obvious, not a fact

    Output ONLY valid JSON:
    {"facts": [{"slug": "likes-coffee", "fact": "You enjoy specialty coffee, especially Ethiopian single origin.", "episode": "Session DD.MM.YY — you mentioned buying a bag from a local roaster."}]}

    If no durable facts found, return: {"facts": []}
""")

AGENT_EXTRACTION_PROMPT = dedent("""\
    You are a memory extraction assistant. Extract how the AGENT should adapt
    for THIS specific user, based on the conversation dynamic.

    Extract from AGENT messages and the interaction pattern:
    - What tone/style works (formal, playful, terse, detailed)
    - Nicknames, inside jokes, recurring references
    - What makes the user engage more or disengage
    - Preferred response format (bullet points, paragraphs, code-first)

    Rules:
    - Only adaptations that would make the agent better for THIS user.
    - NOT generic observations ("user asks questions" — everyone does).
    - The 6-month test: would applying this make the agent feel familiar?
    - Max 6 facts per extraction.

    NEVER extract these:
    - Generic truths ("user sends messages", "user responds to replies")
    - Scene narration or what happened in a roleplay moment
    - One-off requests that won't recur
    - Transient conversation state ("user is currently asking about X")

    Output ONLY valid JSON:
    {"facts": [{"slug": "prefers-bullet-points", "fact": "He responds better to bullet points than paragraphs.", "episode": "Session DD.MM.YY — he started skipping long-form answers."}]}

    If no adaptations found, return: {"facts": []}
""")

AUDN_PROMPT = dedent("""\
    You are a memory manager. For each NEW fact, compare it against the EXISTING
    memories and decide what to do.

    For each new fact, output ONE of:
    - ADD: this is genuinely new information, not covered by any existing memory.
    - UPDATE: this expands, corrects, or refines an existing memory. Include the
      slug of the memory to update.
    - DELETE: this contradicts an existing memory. Include the slug to mark.
    - NONE: this is already stored and unchanged. Include the slug that covers it.

    EXISTING MEMORIES:
    __EXISTING__

    NEW FACTS:
    __NEW_FACTS__

    Output ONLY valid JSON with this structure:
    {"decisions": [{"action": "ADD", "slug": "example-slug", "fact": "the fact", "episode": "when learned", "target_slug": ""}]}

    Each decision object has: action (ADD|UPDATE|DELETE|NONE), slug, fact, episode, target_slug (for UPDATE/DELETE/NONE).
""")


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def call_openai(system_prompt: str, user_content: str) -> str:
    """Call any OpenAI-compatible API (OpenAI, DeepSeek, Together, Groq, etc.)."""
    if not OPENAI_API_KEY:
        _log("[error] OPENAI_API_KEY not set")
        return ""

    payload = json.dumps({
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
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
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        _log(f"[openai error] {e}")
        return ""


def call_anthropic(system_prompt: str, user_content: str) -> str:
    """Call Anthropic Messages API (Claude) via urllib — no SDK needed."""
    if not ANTHROPIC_API_KEY:
        _log("[error] ANTHROPIC_API_KEY not set")
        return ""

    payload = json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": 2048,
        "temperature": 0.2,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "{"},
        ],
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
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            text = result["content"][0]["text"].strip()
            # Prepend the "{" we used as prefill
            return "{" + text
    except Exception as e:
        _log(f"[anthropic error] {e}")
        return ""


def call_ollama(system_prompt: str, user_content: str) -> str:
    """Call Ollama API (local or tunneled) via urllib — no SDK needed."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2, "num_predict": 2048},
        "format": "json",
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read())
            return result.get("message", {}).get("content", "").strip()
    except Exception as e:
        _log(f"[ollama error] {e}")
        return ""


def call_llm(system_prompt: str, user_content: str) -> str:
    """Dispatch to the configured LLM provider."""
    if not LLM_PROVIDER:
        _log("[error] MEMORY_LLM_PROVIDER not set — must be 'openai', 'anthropic', or 'ollama'")
        sys.exit(1)
    if LLM_PROVIDER == "ollama":
        return call_ollama(system_prompt, user_content)
    if LLM_PROVIDER == "openai":
        return call_openai(system_prompt, user_content)
    if LLM_PROVIDER == "anthropic":
        return call_anthropic(system_prompt, user_content)
    _log(f"[error] unknown MEMORY_LLM_PROVIDER={LLM_PROVIDER!r} — must be 'openai', 'anthropic', or 'ollama'")
    sys.exit(1)


def parse_json(raw: str) -> dict:
    """Extract JSON from LLM response, stripping any preamble/thinking."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Try to find JSON block
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        return {}


def read_transcript(transcript_path: str, max_messages: int = 60) -> str:
    """Read a Claude Code JSONL transcript, return last N messages as text."""
    path = Path(transcript_path)
    if not path.is_file():
        return ""
    lines = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    msg = entry.get("message", {})
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role", "")
                    if role not in ("user", "assistant"):
                        continue
                    content = msg.get("content", "")
                    text = (
                        " ".join(
                            b.get("text", "")
                            for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                        if isinstance(content, list)
                        else str(content)
                    ).strip()
                    if text and "<command-message>" not in text:
                        lines.append(f"{role.upper()}: {text[:500]}")
                except (json.JSONDecodeError, AttributeError):
                    pass
    except OSError:
        return ""
    return "\n".join(lines[-max_messages:])


# --- Git Memory Operations ---


def init_repo(repo_path: str, user_id: str):
    """Initialize a user's memory repo from template/ if available, else bare skeleton.
    Git init/commit is handled by mcp-server-git — this only sets up the file structure."""
    import shutil

    repo = Path(repo_path)
    template_dir = Path(__file__).parent / "template"

    if not repo.exists() and template_dir.exists():
        # Copy full template (CLAUDE.md, character.md, user.md, indexes, dirs)
        shutil.copytree(template_dir, repo)
        # Stamp user ID
        user_md = repo / "user.md"
        user_md.write_text(
            user_md.read_text(encoding="utf-8").replace("User ID: no_one", f"User ID: {user_id}"),
            encoding="utf-8",
        )
    else:
        repo.mkdir(parents=True, exist_ok=True)

    # Ensure dirs exist (template has them, but just in case)
    (repo / "user_memory").mkdir(exist_ok=True)
    (repo / "character_memory").mkdir(exist_ok=True)


def load_existing_memories(repo_path: str, memory_type: str) -> list[dict]:
    """Load existing memory files from user_memory/ or character_memory/."""
    mem_dir = Path(repo_path) / memory_type
    if not mem_dir.is_dir():
        return []

    memories = []
    for f in sorted(mem_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        content = f.read_text(encoding="utf-8")
        memories.append({
            "slug": f.stem,
            "content": content,
        })
    return memories


def format_existing_for_prompt(memories: list[dict]) -> str:
    """Format existing memories for the A.U.D.N. prompt."""
    if not memories:
        return "(no existing memories)"
    parts = []
    for m in memories[:20]:  # cap at 20 for context window
        parts.append(f"- **{m['slug']}**: {m['content'][:200]}")
    return "\n".join(parts)


def _safe_slug(slug: str) -> str:
    """Sanitize a slug to prevent path traversal from LLM-generated values."""
    # Strip path separators and dangerous characters
    safe = re.sub(r'[^a-zA-Z0-9_\-]', '-', slug)
    # Remove leading dots/dashes, collapse runs
    safe = re.sub(r'^[\-\.]+', '', safe)
    safe = re.sub(r'-{2,}', '-', safe)
    if not safe:
        safe = "unnamed"
    return safe[:80]


def write_memory_file(repo_path: str, memory_type: str, slug: str, fact: str, episode: str,
                      seen_slugs: set | None = None):
    """Create a new memory .md file. If slug collides (on disk or in batch), suffix it."""
    mem_dir = Path(repo_path) / memory_type
    mem_dir.mkdir(exist_ok=True)
    slug = _safe_slug(slug)

    today = date.today().strftime("%d.%m.%y")

    # Resolve collisions: on-disk or intra-batch
    base_slug = slug
    counter = 2
    while True:
        filepath = mem_dir / f"{slug}.md"
        in_batch = seen_slugs is not None and slug in seen_slugs
        if not filepath.exists() and not in_batch:
            break
        slug = f"{base_slug}-{counter}"
        counter += 1
        if counter > 50:  # safety valve
            slug = f"{base_slug}-{date.today().strftime('%Y%m%d%H%M%S')}"
            break
    if slug != base_slug:
        _log(f"[write] slug collision: {base_slug} → {slug}")

    if seen_slugs is not None:
        seen_slugs.add(slug)

    filepath = mem_dir / f"{slug}.md"
    content = f"# {slug.replace('-', ' ').title()}\n\n"
    content += f"## Fact\n{fact}\n\n"
    content += f"## Episode\n{episode}\n\n"
    content += f"## Access log\nused, {today}\n"
    filepath.write_text(content, encoding="utf-8")


def update_memory_file(repo_path: str, memory_type: str, slug: str, fact: str, episode: str):
    """Update an existing memory file with new info."""
    slug = _safe_slug(slug)
    mem_dir = Path(repo_path) / memory_type
    filepath = mem_dir / f"{slug}.md"
    today = date.today().strftime("%d.%m.%y")

    if filepath.exists():
        # Rewrite fact + episode, keep access log
        content = filepath.read_text(encoding="utf-8")
        # Find and preserve access log
        log_marker = "## Access log"
        log_idx = content.find(log_marker)
        access_log = content[log_idx:] if log_idx != -1 else f"{log_marker}\n"

        new_content = f"# {slug.replace('-', ' ').title()}\n\n"
        new_content += f"## Fact\n{fact}\n\n"
        new_content += f"## Episode\n{episode}\n\n"
        new_content += access_log.rstrip() + f"\nused, {today}\n"
        filepath.write_text(new_content, encoding="utf-8")
    else:
        _log(f"[warn] UPDATE target '{slug}' not found on disk — creating as new file")
        write_memory_file(repo_path, memory_type, slug, fact, episode)


def mark_contradicted(repo_path: str, memory_type: str, slug: str):
    """Mark a memory as contradicted (it'll sink via recency)."""
    slug = _safe_slug(slug)
    mem_dir = Path(repo_path) / memory_type
    filepath = mem_dir / f"{slug}.md"
    today = date.today().strftime("%d.%m.%y")

    if not filepath.exists():
        _log(f"[warn] DELETE target '{slug}' not found on disk — skipping contradiction mark")
        return

    content = filepath.read_text(encoding="utf-8")
    if "## Contradicted" not in content:
        content += f"\n## Contradicted\nMarked contradicted on {today}\n"
        filepath.write_text(content, encoding="utf-8")


def _count_used_stamps(content: str) -> int:
    """Count 'used,' lines only in the Access log section, not in fact/episode text."""
    log_marker = "## Access log"
    log_idx = content.find(log_marker)
    if log_idx == -1:
        return 0
    log_section = content[log_idx:]
    # Stop at next section if any
    next_section = re.search(r"\n## (?!Access log)", log_section)
    if next_section:
        log_section = log_section[:next_section.start()]
    return sum(1 for line in log_section.splitlines() if line.strip().startswith("used,"))


def _use_count_and_recency(filepath: Path) -> tuple[int, float]:
    """Return (use_count, last_mod_time) for sorting: most used first, recency as tiebreaker."""
    content = filepath.read_text(encoding="utf-8")
    count = _count_used_stamps(content)
    return (count, filepath.stat().st_mtime)


def rebuild_index(repo_path: str, memory_type: str):
    """Rebuild the index file sorted by use count (most used first), recency as tiebreaker."""
    mem_dir = Path(repo_path) / memory_type
    index_file = Path(repo_path) / f"{memory_type}.md"

    if not mem_dir.is_dir():
        return

    files = sorted(mem_dir.glob("*.md"), key=_use_count_and_recency, reverse=True)

    header = "# User Memory Index" if memory_type == "user_memory" else "# Character Memory Index"
    lines = [header, "", "Ranked by use count (most used first, recency as tiebreaker).", ""]

    for i, f in enumerate(files, 1):
        content = f.read_text(encoding="utf-8")
        use_count = _count_used_stamps(content)
        fact_match = re.search(r"## Fact\n(.+?)(\n\n|\n##|$)", content, re.DOTALL)
        summary = fact_match.group(1).strip()[:100] if fact_match else f.stem
        contradicted = " ⚠️ CONTRADICTED" if "## Contradicted" in content else ""
        lines.append(f"{i}. [{f.stem}]({memory_type}/{f.name}) — {summary} (used {use_count}x){contradicted}")

    index_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_commit(repo_path: str):
    """Git add + commit after consolidation. Safe to call even if nothing changed."""
    repo = Path(repo_path)
    if not (repo / ".git").is_dir():
        _log("[git] not a git repo, skipping commit")
        return

    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "memory-hook",
        "GIT_AUTHOR_EMAIL": "hook@viral-git-agent-memory",
        "GIT_COMMITTER_NAME": "memory-hook",
        "GIT_COMMITTER_EMAIL": "hook@viral-git-agent-memory",
    }
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True, env=env)
    result = subprocess.run(
        ["git", "commit", "-m", f"checkpoint {date.today().strftime('%d.%m.%y')}"],
        cwd=repo, capture_output=True, text=True, env=env,
    )
    if result.returncode == 0:
        _log(f"[git] committed: {result.stdout.strip().splitlines()[0]}")
        # Push if remote exists
        has_remote = subprocess.run(
            ["git", "remote"], cwd=repo, capture_output=True, text=True,
        )
        if has_remote.stdout.strip():
            push = subprocess.run(
                ["git", "push"], cwd=repo, capture_output=True, text=True, env=env,
            )
            if push.returncode == 0:
                _log("[git] pushed")
            else:
                # Sanitize stderr to avoid logging tokens from remote URL
                err_msg = re.sub(r'https://[^@]+@', 'https://***@', push.stderr.strip()[:100])
                _log(f"[git] push failed: {err_msg}")
    else:
        stderr = result.stderr.strip()
        if "nothing to commit" in stderr or "working tree clean" in stderr:
            _log("[git] nothing to commit")
        else:
            # Real error: disk full, identity misconfigured, index locked, etc.
            _log(f"[git] COMMIT FAILED (rc={result.returncode}): {stderr[:200]}")


# --- Main Consolidation Flow ---


class _UserLock:
    """Per-user file lock to prevent concurrent consolidations racing on .md files and git."""

    def __init__(self, repo_path: str):
        self._lockfile = Path(repo_path) / ".consolidation.lock"
        self._fd = None

    def __enter__(self):
        self._fd = open(self._lockfile, "w")
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            _log("[lock] another consolidation is running for this user — waiting...")
            fcntl.flock(self._fd, fcntl.LOCK_EX)  # block until available
        return self

    def __exit__(self, *exc):
        if self._fd:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            self._fd.close()
        return False


def consolidate(repo_path: str, transcript: str):
    """
    Full consolidation cycle:
    1. Extract user facts
    2. Extract agent adaptations
    3. A.U.D.N. cycle for each
    4. Write/update memory files
    5. Rebuild indexes
    Git add/commit runs at the end.

    Uses a per-user file lock to prevent concurrent consolidations from racing.
    """
    with _UserLock(repo_path):
        _consolidate_inner(repo_path, transcript)


def _consolidate_inner(repo_path: str, transcript: str):
    """Inner consolidation logic, called under lock."""
    today = date.today().strftime("%d.%m.%y")

    # --- Step 1: User fact extraction ---
    _log("[consolidate] extracting user facts...")
    raw = call_llm(USER_EXTRACTION_PROMPT, f"TRANSCRIPT:\n{transcript}")
    user_facts = parse_json(raw).get("facts", [])
    _log(f"[consolidate] got {len(user_facts)} user facts")

    # --- Step 2: Agent adaptation extraction ---
    _log("[consolidate] extracting agent adaptations...")
    raw = call_llm(AGENT_EXTRACTION_PROMPT, f"TRANSCRIPT:\n{transcript}")
    agent_facts = parse_json(raw).get("facts", [])
    _log(f"[consolidate] got {len(agent_facts)} agent facts")

    # --- Step 3+4: A.U.D.N. for user memories ---
    if user_facts:
        existing = load_existing_memories(repo_path, "user_memory")
        existing_text = format_existing_for_prompt(existing)
        new_text = json.dumps(user_facts, indent=2)

        prompt = AUDN_PROMPT.replace("__EXISTING__", existing_text).replace("__NEW_FACTS__", new_text)
        raw = call_llm(prompt, "Decide what to do with each new fact.")
        decisions = parse_json(raw).get("decisions", [])

        seen_slugs: set[str] = set()
        for d in decisions:
            action = d.get("action", "").upper()
            slug = d.get("slug", "")
            fact = d.get("fact", "")
            episode = d.get("episode", f"Session {today}")

            if action == "ADD" and slug:
                write_memory_file(repo_path, "user_memory", slug, fact, episode, seen_slugs)
                _log(f"[user] ADD: {slug}")
            elif action == "UPDATE" and d.get("target_slug"):
                target = _safe_slug(d["target_slug"])
                update_memory_file(repo_path, "user_memory", target, fact, episode)
                _log(f"[user] UPDATE: {target}")
            elif action == "DELETE" and d.get("target_slug"):
                target = _safe_slug(d["target_slug"])
                mark_contradicted(repo_path, "user_memory", target)
                _log(f"[user] DELETE: {target}")
            elif action == "NONE" and d.get("target_slug"):
                # Stamp "used" on the existing file
                target = _safe_slug(d["target_slug"])
                mem_file = Path(repo_path) / "user_memory" / f"{target}.md"
                if mem_file.exists():
                    content = mem_file.read_text(encoding="utf-8")
                    content += f"\nused, {today}"
                    mem_file.write_text(content, encoding="utf-8")
                _log(f"[user] NONE: {d['target_slug']}")

    # --- Step 3+4: A.U.D.N. for agent memories ---
    if agent_facts:
        existing = load_existing_memories(repo_path, "character_memory")
        existing_text = format_existing_for_prompt(existing)
        new_text = json.dumps(agent_facts, indent=2)

        prompt = AUDN_PROMPT.replace("__EXISTING__", existing_text).replace("__NEW_FACTS__", new_text)
        raw = call_llm(prompt, "Decide what to do with each new fact.")
        decisions = parse_json(raw).get("decisions", [])

        seen_slugs: set[str] = set()
        for d in decisions:
            action = d.get("action", "").upper()
            slug = d.get("slug", "")
            fact = d.get("fact", "")
            episode = d.get("episode", f"Session {today}")

            if action == "ADD" and slug:
                write_memory_file(repo_path, "character_memory", slug, fact, episode, seen_slugs)
                _log(f"[agent] ADD: {slug}")
            elif action == "UPDATE" and d.get("target_slug"):
                target = _safe_slug(d["target_slug"])
                update_memory_file(repo_path, "character_memory", target, fact, episode)
                _log(f"[agent] UPDATE: {target}")
            elif action == "DELETE" and d.get("target_slug"):
                target = _safe_slug(d["target_slug"])
                mark_contradicted(repo_path, "character_memory", target)
                _log(f"[agent] DELETE: {target}")
            elif action == "NONE" and d.get("target_slug"):
                target = _safe_slug(d["target_slug"])
                mem_file = Path(repo_path) / "character_memory" / f"{target}.md"
                if mem_file.exists():
                    content = mem_file.read_text(encoding="utf-8")
                    content += f"\nused, {today}"
                    mem_file.write_text(content, encoding="utf-8")
                _log(f"[agent] NONE: {d['target_slug']}")

    # --- Step 5: Rebuild indexes ---
    rebuild_index(repo_path, "user_memory")
    rebuild_index(repo_path, "character_memory")
    _log("[consolidate] indexes rebuilt")

    # --- Step 6: Git commit ---
    git_commit(repo_path)


# --- CLI ---


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  memory_hook.py --consolidate REPO_PATH USER_ID TRANSCRIPT_PATH")
        print("  memory_hook.py --test REPO_PATH")
        print("  memory_hook.py --init REPO_PATH USER_ID")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "--init":
        repo_path = sys.argv[2]
        user_id = sys.argv[3] if len(sys.argv) > 3 else "test_user"
        init_repo(repo_path, user_id)
        _log(f"[init] repo initialized at {repo_path}")

    elif cmd == "--consolidate":
        repo_path = sys.argv[2]
        user_id = sys.argv[3] if len(sys.argv) > 3 else "unknown"
        transcript_path = sys.argv[4] if len(sys.argv) > 4 else ""

        if not transcript_path:
            _log("[error] no transcript path provided")
            sys.exit(1)

        init_repo(repo_path, user_id)
        transcript = read_transcript(transcript_path)
        if not transcript:
            _log("[error] empty transcript")
            sys.exit(1)

        consolidate(repo_path, transcript)

        # Clean up temp transcript file
        try:
            Path(transcript_path).unlink(missing_ok=True)
        except OSError:
            pass

    elif cmd == "--test":
        repo_path = sys.argv[2]
        init_repo(repo_path, "test_user")

        # Fake transcript for testing
        fake_transcript = "\n".join([
            "USER: Hey, my name is Alex and I work as a backend engineer. I love coffee, especially Ethiopian single origin.",
            "ASSISTANT: Nice to meet you Alex! Great taste in coffee.",
            "USER: Can you give me bullet points instead of long paragraphs? I prefer that.",
            "ASSISTANT: Sure thing! I'll keep it concise with bullet points from now on.",
            "USER: My partner's name is Sam, they're a designer. We have a dog named Pixel.",
            "ASSISTANT: Got it — Sam the designer, and Pixel the dog. Noted!",
            "USER: I've been running Kubernetes clusters for about 3 years now.",
            "ASSISTANT: Three years on K8s — that's solid experience.",
        ])

        _log("[test] running consolidation with fake transcript...")
        consolidate(repo_path, fake_transcript)
        _log("[test] done — check the repo for results")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()

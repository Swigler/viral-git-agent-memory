#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
viral-git-agent-memory — Consolidation Hook

Extracts durable facts from a conversation, reconciles them against the existing
memories (A.U.D.N.), writes the markdown files, regenerates the "## Index" blocks in
SOUL.md / USER.md and commits (and pushes) to git.

Trigger: every 30 messages, /bye, or 10 min inactivity (the server spawns this as a
background process), or run it by hand on a saved transcript.
LLM: OpenAI-compatible (OpenAI, DeepSeek, etc.), Anthropic (Claude), or Ollama — user picks via env vars.
Output: markdown files in a memory repo. A failed LLM call raises, exits non-zero and
DELETES the transcript (chat text must not stay on disk; that batch is lost).

POSIX only (the repo lock is fcntl.flock). Stdlib only.

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
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from textwrap import dedent

import gitmem as gm
import gitops
from gitmem import Memory

# Re-exported: tests and api_server use these names via memory_hook.
from gitops import (  # noqa: F401
    CREDENTIALS_FILE,
    GIT_CMD_TIMEOUT,
    GIT_PUSH_TIMEOUT,
    GITHUB_REMOTE,
    git_auth,
    git_credentials_path,
    load_git_credentials,
    push_to_github,
    save_git_credentials,
)

TEMPLATE_DIR = Path(__file__).parent / "template"

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
    from this chat between an adult user and a companion character. Only extract
    from USER messages.

    Keep:
    - Facts about the user: name, age, job, where they live, family, pets,
      likes/dislikes, habits, important dates, plans, life events.
    - The user's stated preferences and boundaries in the relationship: what they
      like to be called, what they want and do not want.
    - Lasting facts the user established about the shared story or relationship
      ("you two agreed she is your girlfriend", "your anniversary is in May").
    - Personal numbers (age, dates, how many kids) are fine.
    - A fact the user states in character as fiction about themselves (e.g. "I am a
      dragon") only if they set it as a lasting premise of the story, worded as
      such: "In your story, you play a dragon."

    NEVER extract:
    - One-off scene actions, moment-to-moment narration, greetings, farewells.
    - What the user just asked or is doing right now.
    - Anything the character said about itself.
    - A real third person's private details, beyond what the user said in passing
      as context.
    - Anything not stated: never infer.

    Rules:
    - The 6-month test: would knowing this make the character seem to know the user
      after 6 months apart? If no, skip it.
    - Always second person: "you like coffee". One plain sentence, under 200
      characters, in the language the user used for it.
    - Max 8 facts per extraction.
    - If a fact is safety-critical (allergy, medical condition, medication,
      emergency contact) or identity-defining (legal name, nationality, native
      language), set "pinned": true in the output. Otherwise omit or set false.

    Output ONLY valid JSON:
    {"facts": [{"slug": "likes-coffee", "fact": "You enjoy specialty coffee, especially Ethiopian single origin.", "episode": "Session DD.MM.YY — you mentioned buying a bag from a local roaster."}, {"slug": "peanut-allergy", "fact": "You are severely allergic to peanuts.", "episode": "Session DD.MM.YY — you warned about this.", "pinned": true}]}

    If no durable facts found, return: {"facts": []}
""")

AGENT_EXTRACTION_PROMPT = dedent("""\
    You are a memory extraction assistant. Extract how the AGENT, a companion
    character in a chat with an adult user, should be with THIS specific user,
    based on the conversation and the dynamic between them.

    Extract:
    - Tone that works with this user (playful, tender, teasing, direct, slow).
    - Pet names and nicknames, in both directions.
    - Inside jokes and recurring references.
    - Recurring story threads and the current state of the relationship.
    - What the user enjoys or dislikes in how the character talks.
    - Pacing and length preferences.

    Rules:
    - Only what would make the character better for THIS user.
    - Never infer: only what the conversation shows or states.
    - Do not restate plain facts of the user's life (name, age, job, home, family,
      pets, tastes): those are stored separately. Only how the character should
      behave, speak and continue the shared story.
    - Refer to the user as "they"; never guess a gender.
    - Never record a real third person's private details beyond what the user said
      in passing as context.
    - One plain sentence per fact, under 200 characters, in the language the user
      used for it.
    - The 6-month test: would applying this make the character feel familiar?
    - Max 6 facts per extraction.

    NEVER extract these:
    - Generic truths ("user sends messages", "user responds to replies")
    - One-off moments or what happened in a single scene
    - One-off requests that won't recur
    - Transient conversation state ("user is currently asking about X")

    Output ONLY valid JSON:
    {"facts": [{"slug": "likes-short-teasing", "fact": "They enjoy light teasing and short replies.", "episode": "Session DD.MM.YY — they asked for shorter, playful answers."}]}

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
    {"decisions": [{"action": "ADD", "slug": "example-slug", "fact": "the fact", "episode": "when learned", "pinned": false, "target_slug": ""}]}

    Each decision object has: action (ADD|UPDATE|DELETE|NONE), slug, fact, episode,
    pinned (true only for safety-critical or identity-defining facts), target_slug
    (for UPDATE/DELETE/NONE). Existing memories are listed as "- **slug**: fact text".
""")


CRITICAL_PATTERNS = re.compile(
    r"allerg|medical|medication|disease|emergency|password|"
    r"credential|api.key|ssn|birth.?date|blood.?type|"
    r"never.?forget|always.?remember|life.?threatening|"
    r"epipen|anaphyla|insulin|seizure|disability",
    re.IGNORECASE,
)
IDENTITY_PATTERNS = re.compile(
    r"(?:my |your )name is|i am a|my job|my role|i work at|"
    r"i live in|born in|native language|nationality|legal name",
    re.IGNORECASE,
)


def should_pin(fact_text: str) -> bool:
    """Detect if a fact is safety-critical or identity-defining and should be pinned."""
    return bool(CRITICAL_PATTERNS.search(fact_text))


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


class LLMError(RuntimeError):
    """The LLM call failed or returned something unusable. Never swallowed: a failed
    consolidation must not look like "no facts found" (the transcript would be lost)."""


def _post_json(url: str, payload: dict, headers: dict, timeout: int, label: str) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        raise LLMError(f"{label}: {e}") from e


def call_openai(system_prompt: str, user_content: str) -> str:
    """Call any OpenAI-compatible API (OpenAI, DeepSeek, Together, Groq, etc.)."""
    if not OPENAI_API_KEY:
        raise LLMError("OPENAI_API_KEY not set")
    result = _post_json(
        f"{OPENAI_API_URL}/v1/chat/completions",
        {
            "model": OPENAI_MODEL,
            "temperature": 0.2,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        },
        {"Authorization": f"Bearer {OPENAI_API_KEY}"},
        120, "openai",
    )
    try:
        return result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        raise LLMError(f"openai: unexpected response shape: {e!r}") from e


def call_anthropic(system_prompt: str, user_content: str) -> str:
    """Call Anthropic Messages API (Claude) via urllib — no SDK needed."""
    if not ANTHROPIC_API_KEY:
        raise LLMError("ANTHROPIC_API_KEY not set")
    result = _post_json(
        f"{ANTHROPIC_API_URL}/v1/messages",
        {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 2048,
            "temperature": 0.2,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": "{"},
            ],
        },
        {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
        120, "anthropic",
    )
    try:
        return "{" + result["content"][0]["text"].strip()  # prepend the "{" prefill
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        raise LLMError(f"anthropic: unexpected response shape: {e!r}") from e


def call_ollama(system_prompt: str, user_content: str) -> str:
    """Call Ollama API (local or tunneled) via urllib — no SDK needed."""
    result = _post_json(
        f"{OLLAMA_URL}/api/chat",
        {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.2, "num_predict": 2048},
            "format": "json",
        },
        {}, 180, "ollama",
    )
    try:
        return result["message"]["content"].strip()
    except (KeyError, TypeError, AttributeError) as e:
        raise LLMError(f"ollama: unexpected response shape: {e!r}") from e


def call_llm(system_prompt: str, user_content: str) -> str:
    """Dispatch to the configured LLM provider. Raises LLMError on any failure."""
    if LLM_PROVIDER == "ollama":
        return call_ollama(system_prompt, user_content)
    if LLM_PROVIDER == "openai":
        return call_openai(system_prompt, user_content)
    if LLM_PROVIDER == "anthropic":
        return call_anthropic(system_prompt, user_content)
    raise LLMError(f"MEMORY_LLM_PROVIDER={LLM_PROVIDER!r} — must be 'openai', 'anthropic', or 'ollama'")


def parse_json(raw: str) -> dict:
    """Extract JSON from LLM response, stripping any preamble/thinking."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Try to find JSON block
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        data = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _items(raw: str, key: str) -> list:
    """The list under `key` in an LLM answer; raises LLMError if there is none, so an
    unusable answer is a failure (transcript kept) and not a silent "nothing found"."""
    items = parse_json(raw).get(key)
    if not isinstance(items, list):
        raise LLMError(f"LLM answer has no '{key}' list: {raw[:120]!r}")
    return items


def read_transcript(transcript_path: str, max_messages: int = 200, max_chars: int = 4000) -> str:
    """Read a JSONL transcript ({"message": {"role", "content"}} per line) as text."""
    path = Path(transcript_path)
    if not path.is_file():
        return ""
    lines = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    msg = json.loads(line).get("message", {})
                except (json.JSONDecodeError, AttributeError):
                    continue
                if not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant"):
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
                if text:
                    lines.append(f"{msg['role'].upper()}: {text[:max_chars]}")
    except OSError:
        return ""
    return "\n".join(lines[-max_messages:])


# --- Git Memory Operations ---


def init_repo(repo_path: str, user_id: str):
    """Make sure a user's repo exists and has the memory dirs.

    A missing repo is created atomically from template/ (gitops.init_user_repo, also
    what the server uses) with the user ID stamped into USER.md and a first commit.
    """
    repo = Path(repo_path)
    gitops.init_user_repo(repo, user_id, TEMPLATE_DIR)  # no-op if it exists
    repo.mkdir(parents=True, exist_ok=True)
    for d in ("USER_memory", "SOUL_memory"):
        (repo / d).mkdir(exist_ok=True)

    # Migration: rename old files to current names
    for old, new in (("character.md", "SOUL.md"), ("user.md", "USER.md")):
        if (repo / old).exists() and not (repo / new).exists():
            (repo / old).rename(repo / new)


_safe_slug = gm.safe_slug


def _kind(memory_type: str) -> str:
    """"USER_memory" (a directory name) or "USER" -> "USER"; validated by gitmem."""
    return gm.check_type(memory_type.removesuffix("_memory"))


def relevant_memories(repo_path: str, kind: str, facts: list[dict],
                      everything_up_to: int = 30, per_fact: int = 5) -> list[dict]:
    """The existing memories the A.U.D.N. step should see, as [{"slug", "fact"}].

    A small store goes in whole. A big one is narrowed to what is relevant to the new
    facts (full-text, plus embeddings when configured) topped up with the popular
    ones — not the 20 newest files, which re-added old duplicates.
    """
    mem = Memory(repo_path)
    try:
        total = mem.count(kind)
        if total <= everything_up_to:
            hits = mem.top(kind, max(total, 1))
        else:
            hits = mem.top(kind, 10)
            for f in facts:
                hits += mem.search(f"{f['slug'].replace('-', ' ')} {f['fact']}", kind, per_fact)
    finally:
        mem.close()
    seen, out = set(), []
    for h in hits:
        if h["slug"] not in seen:
            seen.add(h["slug"])
            out.append({"slug": h["slug"], "fact": h["fact"]})
    return out


def format_existing_for_prompt(memories: list[dict]) -> str:
    """Existing memories for the A.U.D.N. prompt: slug + Fact text only, no file headers."""
    if not memories:
        return "(no existing memories)"
    return "\n".join(f"- **{m['slug']}**: {' '.join(m['fact'].split())[:300]}" for m in memories)


def build_audn_prompt(existing: str, new_facts: str) -> str:
    """Fill AUDN_PROMPT in ONE pass: sequential .replace() would substitute a
    placeholder that happens to appear inside the memory text itself."""
    values = {"__EXISTING__": existing, "__NEW_FACTS__": new_facts}
    return re.sub(r"__EXISTING__|__NEW_FACTS__", lambda m: values[m.group()], AUDN_PROMPT)


def write_memory_file(repo_path: str, memory_type: str, slug: str, fact: str, episode: str,
                      seen_slugs: set | None = None, pinned: bool = False) -> str:
    """Create a new memory .md file. If slug collides (on disk or in batch), suffix it."""
    return gm.create(repo_path, _kind(memory_type), slug, fact, episode,
                     pinned or should_pin(fact), seen_slugs)


def update_memory_file(repo_path: str, memory_type: str, slug: str, fact: str, episode: str):
    """Update an existing memory: new fact + episode, access log and pin preserved,
    a use stamped, `## Contradicted` cleared (the fact was just refined, so it is valid
    again). A missing target is created as a new memory."""
    slug = _safe_slug(slug)
    if not gm._file(repo_path, _kind(memory_type), slug).exists():
        _log(f"[warn] UPDATE target '{slug}' not found on disk — creating as new file")
        write_memory_file(repo_path, memory_type, slug, fact, episode)
        return
    gm.upsert(repo_path, _kind(memory_type), slug, fact, episode, stamp=True)


def mark_contradicted(repo_path: str, memory_type: str, slug: str):
    """Mark a memory as contradicted: it stays on disk but drops out of recall."""
    if not gm.mark_contradicted(repo_path, _kind(memory_type), slug):
        _log(f"[warn] DELETE target '{slug}' not found on disk — skipping contradiction mark")


def _count_used_stamps(content: str) -> int:
    """Count 'used,' lines in the Access log section only, not in fact/episode text."""
    return gm.parse(content)["used"]


# --- Git ---


def git_commit(repo_path: str):
    """Git add + commit (+ push) after consolidation. Safe to call even if nothing changed."""
    gitops.git_commit(repo_path, log=_log)


# --- Main Consolidation Flow ---


def _fact(item, today: str) -> dict | None:
    """Validate one LLM-produced fact/decision object; None (logged) if unusable."""
    if not isinstance(item, dict):
        _log(f"[skip] not an object: {str(item)[:60]!r}")
        return None
    fact = item.get("fact")
    if not isinstance(fact, str) or not gm.clean_text(fact):
        return {**item, "fact": ""}  # only ADD/UPDATE need a fact; NONE/DELETE do not
    slug = item.get("slug")
    slug = slug if isinstance(slug, str) and slug.strip() else gm.slug_from(fact)
    episode = item.get("episode")
    return {
        **item,
        "slug": slug,
        "fact": fact,
        "episode": episode if isinstance(episode, str) and episode.strip() else f"Session {today}",
        "pinned": item.get("pinned") is True,
    }


def _apply(repo_path: str, kind: str, label: str, facts: list, today: str):
    """The A.U.D.N. step for one memory kind (USER or SOUL)."""
    facts = [f for f in (_fact(i, today) for i in facts) if f and f["fact"]]
    if not facts:
        return
    existing = relevant_memories(repo_path, kind, facts)
    if existing:
        raw = call_llm(
            build_audn_prompt(format_existing_for_prompt(existing), json.dumps(
                [{k: f[k] for k in ("slug", "fact", "episode", "pinned")} for f in facts], indent=2)),
            "Decide what to do with each new fact.",
        )
        decisions = [d for d in (_fact(i, today) for i in _items(raw, "decisions")) if d]
    else:
        _log(f"[{label}] empty memory — adding all facts directly")
        decisions = [{**f, "action": "ADD"} for f in facts]

    seen_slugs: set[str] = set()
    for d in decisions:
        action = str(d.get("action", "")).upper()
        target = d.get("target_slug")
        target = _safe_slug(target) if isinstance(target, str) and target.strip() else ""
        if action == "ADD" and d["fact"]:
            slug = gm.create(repo_path, kind, d["slug"], d["fact"], d["episode"],
                             d["pinned"] or should_pin(d["fact"]), seen_slugs)
            _log(f"[{label}] ADD: {slug}")
        elif action == "UPDATE" and target and d["fact"]:
            update_memory_file(repo_path, kind, target, d["fact"], d["episode"])
            _log(f"[{label}] UPDATE: {target}")
        elif action == "DELETE" and target:
            mark_contradicted(repo_path, kind, target)
            _log(f"[{label}] DELETE: {target}")
        elif action == "NONE" and target:
            gm.stamp_used(repo_path, kind, target)
            _log(f"[{label}] NONE: {target}")
        else:
            _log(f"[{label}] skipped unusable decision: {str(d)[:80]}")


def consolidate(repo_path: str, transcript: str, transcript_path: str | None = None):
    """
    Full consolidation cycle, under the repo lock (the one lock every writer uses):
    1. Extract user facts and agent adaptations (raises LLMError on failure, before
       anything is written)
    2. A.U.D.N. against the relevant existing memories, for each
    3. Regenerate the SOUL.md / USER.md index blocks
    4. Git commit + push
    If `transcript_path` is given and the file vanished while we waited for the lock
    (the user was forgotten), nothing is written: a forget must stick.
    """
    with gitops.repo_lock(repo_path):
        if transcript_path and not Path(transcript_path).is_file():
            _log("[skip] transcript vanished while waiting for the repo lock")
            return
        _consolidate_inner(repo_path, transcript)


def _consolidate_inner(repo_path: str, transcript: str):
    today = date.today().strftime("%d.%m.%y")

    _log("[consolidate] extracting user facts...")
    user_facts = _items(call_llm(USER_EXTRACTION_PROMPT, f"TRANSCRIPT:\n{transcript}"), "facts")
    _log(f"[consolidate] got {len(user_facts)} user facts")

    _log("[consolidate] extracting agent adaptations...")
    agent_facts = _items(call_llm(AGENT_EXTRACTION_PROMPT, f"TRANSCRIPT:\n{transcript}"), "facts")
    _log(f"[consolidate] got {len(agent_facts)} agent facts")

    _apply(repo_path, "USER", "user", user_facts, today)
    _apply(repo_path, "SOUL", "agent", agent_facts, today)

    try:  # the generated index blocks; a failure here must not lose the commit below
        mem = Memory(repo_path)
        try:
            mem.write_indexes()
        finally:
            mem.close()
    except Exception as e:  # noqa: BLE001
        _log(f"[index] could not regenerate index blocks: {e}")

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

        if not Path(transcript_path).is_file():
            # The transcript lives inside the repo: gone means the user was forgotten
            # (or it was already consumed). Never init_repo here, that would resurrect them.
            _log(f"[skip] transcript {transcript_path} is gone, nothing to consolidate")
            return
        init_repo(repo_path, user_id)
        transcript = read_transcript(transcript_path)
        if not transcript:
            _log("[error] empty transcript")
            Path(transcript_path).unlink(missing_ok=True)
            sys.exit(1)

        try:
            consolidate(repo_path, transcript, transcript_path)
        except LLMError as e:
            # Chat text must not sit on disk forever: the batch is lost (accepted).
            _log(f"[error] consolidation failed, transcript deleted: {e}")
            Path(transcript_path).unlink(missing_ok=True)
            sys.exit(2)
        except FileNotFoundError:
            if Path(transcript_path).is_file():
                raise
            _log("[skip] repo removed while waiting for the lock (forgotten)")
            return
        # Success only. The transcript lives INSIDE the repo (gitignored by the
        # template); raw chat history must not outlive a successful run.
        Path(transcript_path).unlink(missing_ok=True)

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
        try:
            consolidate(repo_path, fake_transcript)
        except LLMError as e:
            _log(f"[error] {e}")
            sys.exit(2)
        _log("[test] done — check the repo for results")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()

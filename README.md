# viral-git-agent-memory

Brain-inspired memory system for AI agents. Git-native, human-readable, zero dependencies.

Every user gets their own git repo of markdown files. The agent learns who you are (USER memory) and how to talk to you (SOUL memory) — and you can `git log` the entire history.

## Why This Exists

Most AI memory systems store embeddings in vector databases — opaque, unauditable, locked to one vendor. This stores memories as **markdown files in git repos**. You get:

- `git log` — full timeline of everything the agent learned
- `git diff` — see exactly what changed after each conversation
- `git blame` — trace when and how a fact was learned
- `git revert` — undo a bad memory with one command
- Human-readable files — no embeddings, no black boxes

## How It Works

```
User sends message
       │
       ▼
┌─────────────────┐
│ Context Assembly │  SOUL.md + SOUL_memory/ + USER.md + USER_memory/
│ + LLM Call       │  → assembled into system prompt → streamed response
└────────┬────────┘
         │
    Trigger fires (every 30 msgs, /bye, or 10 min idle)
         │
         ▼
┌─────────────────┐
│  Consolidation  │  Extract facts → A.U.D.N. cycle → write .md files → git commit + push
└─────────────────┘
```

### The A.U.D.N. Cycle

Inspired by [Mem0](https://github.com/mem0ai/mem0). For each extracted fact, compare against existing memories:

| Action | When | What happens |
|--------|------|-------------|
| **Add** | New fact | Create `memory/slug.md` |
| **Update** | Expanded/corrected | Edit existing file |
| **Delete** | Contradicted | Mark as contradicted (sinks in ranking) |
| **None** | Already stored | Stamp `used` (rises in ranking) |

### Two Memory Types

**User Memory** — facts about the person:
> "You work as a backend engineer", "You prefer bullet points", "Your dog is named Pixel"

**SOUL Memory** — how the agent adapted for this person:
> "Use casual tone with this user", "He responds well to code-first answers"

`SOUL.md` is the base persona (same for everyone, nanobot-compatible naming). `SOUL_memory/` is the delta (unique per user).

## Architecture

```
viral-git-agent-memory/
├── api_server.py      # HTTP server: /v1/chat (SSE), /v1/bye, /v1/git/setup, /health
├── memory_hook.py     # Consolidation engine: extraction, A.U.D.N., git commit
├── template/          # Blank user repo skeleton
│   ├── SOUL.md        # Base persona (nanobot-compatible naming)
│   ├── USER.md        # User profile
│   ├── USER_memory/
│   ├── SOUL_memory/
│   └── .gitignore
├── gitmem.py          # Search: SQLite FTS5 index derived from the markdown
├── bench.py           # 28k-memory scale check
├── stress_test.py     # Load testing
└── test_100msg.py     # 100-message conversation test
```

Each user gets their own repo:
```
~/memory/user_123/
├── USER.md                    # Who they are
├── SOUL.md                    # Base persona
├── USER_memory/
│   ├── likes-coffee.md        # Individual memory files
│   ├── works-as-engineer.md
│   └── has-dog-named-pixel.md
├── SOUL_memory/
│   ├── prefers-bullet-points.md
│   └── use-casual-tone.md
└── .git/                      # Full history
```

## Status

This project is **built and code-reviewed but not yet production-tested or personally tested with real conversations**. The architecture is solid, the code has been through multiple review passes and has a 58-assertion test suite covering every code path, but it hasn't been battle-tested with real users yet. Expect rough edges.

### Tests

```bash
# Offline test suite — 62 assertions, no API key needed
python stress_test.py

# GitHub credential + mirror-push checks (no network, local bare repos stand in)
python test_github_auth.py

# Full test with live LLM (runs extraction + AUDN on a fake conversation)
MEMORY_LLM_PROVIDER=ollama python stress_test.py

# Smoke test — creates a repo, runs one consolidation, inspect the output
MEMORY_LLM_PROVIDER=openai OPENAI_API_KEY=sk-xxx \
python memory_hook.py --test /tmp/test_repo
```

Example test output (offline, no LLM):
```
  ✅ USER_memory/ exists
  ✅ file created
  ✅ slug collision creates suffixed file
  ✅ batch file 2 created (dedup)
  ✅ fact updated
  ✅ contradicted marker added
  ✅ DELETE on missing slug doesn't crash
  ✅ used-count ignores fact/episode text
  ✅ common-fact ranked highest (5 uses)
  ✅ new commit created
  ✅ memory survives re-init
  RESULTS: 58 passed, 0 failed
```

## Quick Start

**Requirements:** Python 3.10+, git. No pip install needed — stdlib only.

### 1. Start the server

```bash
# With OpenAI-compatible API (OpenAI, DeepSeek, Groq, Together, etc.)
MEMORY_LLM_PROVIDER=openai \
OPENAI_API_URL=https://api.deepseek.com \
OPENAI_API_KEY=sk-xxx \
OPENAI_MODEL=deepseek-chat \
python api_server.py

# With Anthropic (Claude)
MEMORY_LLM_PROVIDER=anthropic \
ANTHROPIC_API_KEY=sk-ant-xxx \
python api_server.py

# With Ollama (local)
MEMORY_LLM_PROVIDER=ollama \
OLLAMA_MODEL=qwen3.5:4b \
python api_server.py
```

### 2. Send a message

```bash
curl -N -X POST http://localhost:3100/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"userId": "alice", "message": "Hi, I work as a designer and I love cats"}'
```

Response streams as SSE (Server-Sent Events) in OpenAI format.

### 3. Connect to GitHub (optional)

Users can sync their memory to their own private GitHub repo:

```bash
curl -X POST http://localhost:3100/v1/git/setup \
  -H "Content-Type: application/json" \
  -d '{"userId": "alice", "githubToken": "ghp_xxx", "repoName": "my-memory"}'
```

The server validates the token, creates a **private** repo called `viral-git-agent-memory` (or whatever you pass as `repoName`) on the user's GitHub account, and pushes all memory files. Every consolidation after that auto-pushes.

## API Reference

### `POST /v1/chat`
Stream a chat response.

```json
{"userId": "alice", "message": "Hello!"}
```

Returns SSE stream in OpenAI format (`data: {"choices": [{"delta": {"content": "..."}}]}`).

Consolidation triggers automatically every 30 messages, on `/v1/bye`, or after 10 minutes of silence.

### `POST /v1/bye`
End session — triggers consolidation and clears history.

```json
{"userId": "alice"}
```

### `POST /v1/git/setup`
Connect a user's memory to their GitHub account.

```json
{"userId": "alice", "githubToken": "ghp_xxx", "repoName": "my-memory"}
```

`repoName` is optional — defaults to `viral-git-agent-memory`, so the repo appears as `github.com/username/viral-git-agent-memory`. Creates a **private** repo. Token is stored locally in `.git_credentials.json`, never committed or pushed.

### `POST /v1/git/update`
Change GitHub connection (same as setup, idempotent).

### `GET /v1/git/status?userId=alice`
Check if GitHub is connected.

```json
{"userId": "alice", "connected": true, "github_username": "alice", "repo_url": "https://github.com/alice/my-memory", "last_push": "2026-08-31 17:06:22 +0000"}
```

### `POST /v1/consolidate`
Manual consolidation with a transcript.

```json
{"userId": "alice", "transcript": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### `GET /health`
Server health check.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MEMORY_LLM_PROVIDER` | Yes | — | `openai`, `anthropic`, or `ollama` |
| `OPENAI_API_URL` | If openai | `https://api.openai.com` | Base URL (no `/v1/chat/completions`) |
| `OPENAI_API_KEY` | If openai | — | API key |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model name |
| `ANTHROPIC_API_URL` | No | `https://api.anthropic.com` | Base URL |
| `ANTHROPIC_API_KEY` | If anthropic | — | API key |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-20250514` | Model name |
| `OLLAMA_URL` | No | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_MODEL` | No | `qwen3.5:4b` | Model name |
| `MEMORY_DIR` | No | `~/memory` | Where user repos live |
| `API_KEY` | No | — | Bearer token for client auth |
| `PORT` | No | `3100` | Server port |

## Memory File Format

```markdown
# Likes Coffee

## Fact
You enjoy specialty coffee, especially Ethiopian single origin.

## Episode
Session 31.08.26 — you mentioned buying a bag from a local roaster.

## Access log
used, 31.08.26
used, 02.09.26
```

Memories are **never deleted** — unused ones sink via recency. When a buried topic resurfaces, it jumps back. The `used` stamps track access frequency.

## Search — `gitmem.py`

Ranking by pinned → use-count → recency means the agent recalls what is **popular**, never what is **relevant**. `gitmem.py` adds full-text search over the same files, so it can answer "what do I know about X".

**The markdown stays the truth. The index is derived** — SQLite FTS5, gitignored, rebuilt from the files in under six seconds. Delete it and you lose nothing. There is deliberately no vector store: a HNSW index under this exact write pattern (small incremental adds, one memory at a time) inflated from 233 KB to 399 GB in twenty minutes. That failure class is structurally absent here.

Still zero dependencies — `sqlite3` is stdlib.

```python
from gitmem import Memory

mem = Memory("~/memory/alice")

mem.recall("USER", "where do they live")   # relevance-ranked, drop-in for load_top_memories()
mem.recall("SOUL")                          # behavioural memory: always-loaded, never filtered
mem.capture("USER", "remember: the standup is at 9am")   # manual capture, no model
mem.search("coffee", "USER")                # raw hits with scores
mem.commit()                                # regenerate index blocks + one git commit
```

`sync()` runs on open and reconciles the index with whatever is on disk, so `memory_hook.py` keeps writing files exactly as before without knowing the index exists.

### USER is searched, SOUL is always loaded

**USER memory is factual** — unbounded, and the user asks about it directly. Search is the right tool.

**SOUL memory is behavioural** — small, bounded, and never asked about. It just has to be in the prompt. So `recall()` ignores the query for SOUL: a rule like "no hedging" must reach the prompt whether the user says "stop hedging" or "just answer me". Search there only *appears* to work when the user happens to use the memory's own words.

### Measured at 28,000 memories

`python bench.py` — the incremental write pattern, at scale:

| | |
|---|---|
| markdown files | 8.3 MB (the truth, git-tracked) |
| FTS5 index | 17.6 MB (derived, gitignored) — 2.11x, linear growth |
| search | 29–40 ms |
| full reindex from files | 5.6 s |
| write throughput | ~400/s |
| total repo on disk | 42.7 MB |

### The honest limit

Keyword search cannot cross a vocabulary gap. Ask about a memory in words it does not contain and you get **nothing back** — asserted in the selftest, not hidden. An earlier version returned the *wrong* memory at a confident 0.7 score, because stopwords matched every document. A miss beats a confident wrong answer. Closing that gap needs embeddings, and that is the point to add them.

```bash
python gitmem.py    # selftest
python bench.py     # 28k scale check
```

## Concurrency & Reliability

- **Per-user file locking** — `fcntl.flock` prevents concurrent consolidations from racing on `.md` files and git operations
- **Slug collision handling** — if an ADD generates a slug that already exists on disk (or duplicates within the same batch), it auto-suffixes (`likes-coffee-2`, `likes-coffee-3`)
- **Missing target warnings** — UPDATE/DELETE on a slug that doesn't exist on disk logs a `[warn]` instead of silently no-oping
- **Git error detection** — distinguishes "nothing to commit" from real failures (disk full, identity misconfigured, index locked)
- **Contradiction tracking** — the DELETE action marks memories as contradicted (they sink in ranking but are never destroyed, so `git revert` always works)

## Security

- **Path traversal prevention** on user IDs and LLM-generated slugs
- **Token never written into git** — the GitHub token is passed per-command through a
  throwaway `GIT_ASKPASS` helper, so it is in neither the remote URL (which would land
  in `.git/config` in plaintext and come back out in git's own error messages) nor
  `argv` (visible in `ps`)
- **Token sanitization** — GitHub tokens never appear in logs or error responses
- **Credentials gitignored and 0600** — `.git_credentials.json` is created private and
  is never committed or pushed
- **The mirror is a separate remote** — the user's GitHub is `github`, never `origin`,
  so connecting it cannot repoint the operator's memory store
- **Optional API key** — Bearer token auth for the server
- **Private repos** — GitHub repos are created as private by default

## Agent-Agnostic

The same system works for any agent type — just change `SOUL.md`:

| Agent | SOUL_memory/ learns... |
|-------|---------------------------|
| Companion | "Slow teasing works", "Use pet names" |
| Work assistant | "Bullet points, formal tone" |
| Code helper | "Python dev, show diffs not full files" |
| Tutor | "Visual learner, needs examples first" |

## Used By

- [oracle-cloud-ai-agent](https://github.com/Swigler/oracle-cloud-ai-agent) — Voice AI assistant on Oracle Cloud free tier. Uses viral-git-agent-memory for persistent conversation memory.

## License

MIT

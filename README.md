# viral-git-agent-memory

Brain-inspired memory system for AI agents. Git-native, human-readable, zero dependencies.

Every user gets their own git repo of markdown files. The agent learns who you are (user memory) and how to talk to you (character memory) — and you can `git log` the entire history.

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
│ Context Assembly │  character.md + character_memory/ + user.md + user_memory/
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

**Character Memory** — how the agent adapted for this person:
> "Use casual tone with this user", "He responds well to code-first answers"

`character.md` is the base persona (same for everyone). `character_memory/` is the delta (unique per user).

## Architecture

```
viral-git-agent-memory/
├── api_server.py      # HTTP server: /v1/chat (SSE), /v1/bye, /v1/git/setup, /health
├── memory_hook.py     # Consolidation engine: extraction, A.U.D.N., git commit
├── template/          # Blank user repo skeleton
│   ├── character.md
│   ├── user.md
│   ├── user_memory.md
│   ├── character_memory.md
│   ├── user_memory/
│   ├── character_memory/
│   └── .gitignore
├── stress_test.py     # Load testing
└── test_100msg.py     # 100-message conversation test
```

Each user gets their own repo:
```
~/memory/user_123/
├── user.md                    # Who they are
├── character.md               # Base persona
├── user_memory.md             # Index (ranked by use count)
├── user_memory/
│   ├── likes-coffee.md        # Individual memory files
│   ├── works-as-engineer.md
│   └── has-dog-named-pixel.md
├── character_memory.md        # Index (ranked by use count)
├── character_memory/
│   ├── prefers-bullet-points.md
│   └── use-casual-tone.md
└── .git/                      # Full history
```

## Status

This project is **built and code-reviewed but not yet production-tested or personally tested with real conversations**. The architecture is solid, the code has been through multiple review passes and has a 58-assertion test suite covering every code path, but it hasn't been battle-tested with real users yet. Expect rough edges.

### Tests

```bash
# Offline test suite — 58 assertions, no API key needed
python stress_test.py

# Full test with live LLM (runs extraction + AUDN on a fake conversation)
MEMORY_LLM_PROVIDER=ollama python stress_test.py

# Smoke test — creates a repo, runs one consolidation, inspect the output
MEMORY_LLM_PROVIDER=openai OPENAI_API_KEY=sk-xxx \
python memory_hook.py --test /tmp/test_repo
```

Example test output (offline, no LLM):
```
  ✅ user_memory/ exists
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

Memories are **never deleted** — unused ones sink to the bottom of the index. When a buried topic resurfaces, it jumps back to the top. The `used` stamps drive the ranking: most-used memories load into context first.

## Concurrency & Reliability

- **Per-user file locking** — `fcntl.flock` prevents concurrent consolidations from racing on `.md` files and git operations
- **Slug collision handling** — if an ADD generates a slug that already exists on disk (or duplicates within the same batch), it auto-suffixes (`likes-coffee-2`, `likes-coffee-3`)
- **Missing target warnings** — UPDATE/DELETE on a slug that doesn't exist on disk logs a `[warn]` instead of silently no-oping
- **Git error detection** — distinguishes "nothing to commit" from real failures (disk full, identity misconfigured, index locked)
- **Contradiction tracking** — the DELETE action marks memories as contradicted (they sink in ranking but are never destroyed, so `git revert` always works)

## Security

- **Path traversal prevention** on user IDs and LLM-generated slugs
- **Token sanitization** — GitHub tokens never appear in logs or error responses
- **Credentials gitignored** — `.git_credentials.json` is never committed or pushed
- **Optional API key** — Bearer token auth for the server
- **Private repos** — GitHub repos are created as private by default

## Agent-Agnostic

The same system works for any agent type — just change `character.md`:

| Agent | character_memory/ learns... |
|-------|---------------------------|
| Companion | "Slow teasing works", "Use pet names" |
| Work assistant | "Bullet points, formal tone" |
| Code helper | "Python dev, show diffs not full files" |
| Tutor | "Visual learner, needs examples first" |

## Used By

- [oracle-cloud-ai-agent](https://github.com/Swigler/oracle-cloud-ai-agent) — Voice AI assistant on Oracle Cloud free tier. Uses viral-git-agent-memory for persistent conversation memory.

## License

MIT

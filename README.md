# viral-git-agent-memory

Brain-inspired memory system for AI agents. Git-native, human-readable, stdlib-only (POSIX: uses `fcntl` file locks, so Linux/macOS, not Windows).

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
├── api_server.py      # HTTP server: /v1/chat (SSE), /v1/bye, /v1/consolidate, /v1/git/*, /health
├── memory_hook.py     # Consolidation worker: extraction, A.U.D.N., index blocks, git commit
├── gitmem.py          # Markdown format + every file writer + derived index (FTS5, embeddings)
├── gitops.py          # Repo lock, atomic writes, git (timeouts), GitHub credentials, repo init
├── template/          # Blank user repo skeleton (SOUL.md, USER.md, .gitignore, memory dirs)
├── stress_test.py     # File-layer tests (offline)
├── test_github_auth.py / test_embeddings.py / test_server.py   # offline tests, see Tests
├── test_100msg.py     # 100-message simulation (needs a live LLM)
└── bench.py           # 28k-memory scale check
```

Data flow: `api_server` assembles the prompt from `SOUL.md` + `USER.md` + memories read through `gitmem.Memory` (relevance-ranked for the current message). Every 30 messages, on `/v1/bye`, or after 10 idle minutes it hands the unconsolidated turns to `memory_hook.py` (a subprocess, serialised per user by the repo lock), which writes markdown through `gitmem`, regenerates the `## Index` blocks and commits/pushes through `gitops`.

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

This project is **built and code-reviewed but not yet production-tested or personally tested with real conversations**. The architecture is solid, the code has been through multiple review passes and has an offline test suite (see Tests), but it hasn't been battle-tested with real users yet. Expect rough edges.

### Tests

All offline (stub HTTP servers stand in for the LLM and the embedding model); 187 assertions in total:

```bash
python3 stress_test.py       # 63 assertions: file layer, ranking, parsing, git commit
python3 gitmem.py            # selftest: index, search, capture, derived files, git
python3 test_github_auth.py  # 5 checks: token handling, 0600 credentials, mirror push
python3 test_embeddings.py   # 25 assertions: incremental re-embed, rebuild, hybrid ranking, fallback
python3 test_server.py       # 94 assertions: userId/auth, LLM failure, history, races, routes

# Live LLM (extraction + A.U.D.N. on a fake conversation)
MEMORY_LLM_PROVIDER=ollama python3 stress_test.py
MEMORY_LLM_PROVIDER=ollama python3 test_100msg.py
MEMORY_LLM_PROVIDER=openai OPENAI_API_KEY=sk-xxx python3 memory_hook.py --test /tmp/test_repo
```

What is *not* covered: the real GitHub API (`/v1/git/setup` end to end), real LLM providers, real embedding models, and load/long-running behaviour (timers firing after 10 minutes).

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

`userId` is required. On a loopback-only server with no auth configured that is all you need; anywhere else you need a token (next section).

### Auth model

- **Per-user tokens (default when exposed).** Set `API_KEY_SECRET`. A caller must send `Authorization: Bearer <token>` where `token = HMAC-SHA256(API_KEY_SECRET, userId)` (hex). A token only works for its own `userId`, on every endpoint except `/health`. Mint one: `API_KEY_SECRET=... python3 api_server.py --issue-token alice`. Your backend holds the secret and gives each end user only their own token.
- **Single tenant.** `SINGLE_TENANT=1` plus `API_KEY=...`: one shared Bearer key for every `userId`. Only for a deployment with one trusted client. `API_KEY` alone is refused at startup (it would let any key holder read or overwrite any user's memory).
- **No auth.** Only allowed on a loopback `HOST`; a keyless server also rejects requests whose `Host` header is not a loopback name (DNS-rebinding guard).
- **userId** — 1-128 characters of letters, digits and `. _ @ : + -` (first character alphanumeric); anything else is a 400, never rewritten. Each id gets its own repo directory named `<readable prefix>-<sha256 prefix>`, so `a_b`, `a-b` and `a.b` can never share memory. Repos from the old layout (`~/memory/<id>`) are adopted on first use.

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

`repoName` is optional (letters, digits, `. _ -`; anything else is a 400) — defaults to `viral-git-agent-memory`. Creates a **private** repo. Token is stored locally in `.git_credentials.json`, never committed or pushed.

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

Items must be `{role: "user"|"assistant", content: string | [blocks]}` (max 2000). If the LLM call fails the worker exits non-zero and keeps the transcript file (`<repo>/.transcript_*.jsonl`, gitignored); retry with `python3 memory_hook.py --consolidate REPO USER_ID TRANSCRIPT`.

### `GET /health`
Unauthenticated liveness probe: `{"status": "ok"}` and nothing else.

All user-scoped endpoints return 400 for a missing/invalid `userId` and 401 for a missing/wrong token. Routes match exactly (a query string is fine, `/v1/chatX` is a 404).

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
| `API_KEY_SECRET` | If `HOST` is not loopback (or `SINGLE_TENANT`) | — | Per-user tokens: `HMAC-SHA256(secret, userId)`. See Auth model. |
| `SINGLE_TENANT` | No | off | `1` = one shared `API_KEY` for every user. |
| `API_KEY` | With `SINGLE_TENANT=1` | — | The shared key. Refused alone. |
| `HOST` | No | `127.0.0.1` | Bind address. A non-loopback address without auth is refused at startup. |
| `CORS_ORIGINS` | No | `*` if auth on, else none | Comma-separated browser origins allowed to call the API, or `*`. Refused without auth. |
| `ALLOWED_HOSTS` | No | — | Keyless servers only answer requests whose `Host` header is a loopback name. List extra hostnames here. Ignored when auth is on. |
| `PORT` | No | `3100` | Server port |
| `SOCKET_TIMEOUT` | No | `60` | Seconds per client socket operation (slowloris guard) |
| `MAX_CONNECTIONS` | No | `64` | Concurrent connections; more are dropped |
| `GIT_CMD_TIMEOUT` / `GIT_PUSH_TIMEOUT` | No | `60` / `120` | Seconds before a git command / push is killed |
| `EMBED_URL`, `EMBED_MODEL`, `EMBED_API_KEY`, `EMBED_TIMEOUT`, `EMBED_MIN_SIM` | No | off | Optional embeddings, see Search |

All `POST` bodies must be sent with `Content-Type: application/json` (max 10 MB).

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

The consolidation hook never deletes a memory file: a contradicted memory gets a `## Contradicted` section and drops out of recall (`git revert` always works). Unused ones sink in the ranking; when a buried topic resurfaces, it jumps back. The `used` stamps track access frequency. Only the explicit `Memory.delete()` API removes a file.

## Search — `gitmem.py`

Ranking by pinned → use-count → recency means the agent recalls what is **popular**, never what is **relevant**. `gitmem.py` adds full-text search over the same files, so it can answer "what do I know about X".

**The markdown stays the truth. The index is derived** — SQLite FTS5, gitignored, rebuilt from the files in under six seconds. Delete it and you lose nothing. There is deliberately no vector *store*: a HNSW index under this exact write pattern (small incremental adds, one memory at a time) inflated from 233 KB to 399 GB in twenty minutes. Embeddings, when enabled, are plain rows in the same throwaway SQLite file, so that failure class is structurally absent here.

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

### Optional embeddings (a derived index, like the FTS table)

Keyword search cannot cross a vocabulary gap: ask about "caffeine" and a memory that says "espresso" is not found (asserted in the selftest, not hidden — a miss beats a confident wrong answer). Embeddings close that gap, and they stay **derived**: git is the source of truth, the vectors are float32 BLOBs in the same gitignored `index.sqlite3` (table `embeddings`: `memory_type, slug, content_hash, dim, vector`).

- **Incremental.** Sync re-embeds only memories whose `## Fact` text hash (plus the model name) changed. Delete `index.sqlite3` and everything is rebuilt from the markdown.
- **Provider.** Any OpenAI-compatible `POST {EMBED_URL}/v1/embeddings` (`{model, input: [...]}`), batched, over `urllib`. Off unless `EMBED_URL` is set.
- **Search.** The query is embedded, scored by brute-force cosine in pure Python (no numpy), and fused with the FTS5 ranking by reciprocal rank fusion (`1/(60+rank)`); then pinned and well-used memories get a small boost. Neighbours below half of the best similarity are dropped. Contradicted memories are excluded everywhere; at most `n/2` pinned memories lead a recall.
- **Never breaks chat.** If `EMBED_URL` is unset, or a call fails, search is FTS-only (logged once; the endpoint is left alone for 30 s).
- **Scale.** Brute force is fine to roughly 10k memories per user.

| Env | Default | |
|---|---|---|
| `EMBED_URL` | unset (off) | Base URL, e.g. `http://127.0.0.1:8081` |
| `EMBED_MODEL` | `default` | Model name sent in the request |
| `EMBED_API_KEY` | — | Sent as `Authorization: Bearer` if set |
| `EMBED_TIMEOUT` | `10` | Seconds per request |
| `EMBED_MIN_SIM` | `0` | Optional absolute cosine floor |

Recommended models:

- **CPU:** `bge-small-en-v1.5` or `nomic-embed-text`, served by `llama-server --embedding -m bge-small-en-v1.5-q8_0.gguf --port 8081 --pooling mean` (then `EMBED_URL=http://127.0.0.1:8081`).
- **GPU:** `Qwen3-Embedding-0.6B` behind any OpenAI-compatible server (llama-server `--embedding --pooling last`, vLLM, ...).

```bash
python gitmem.py    # selftest
python bench.py     # 28k scale check
```

## Concurrency & Reliability

- **One per-repo lock** — `.gitmem.lock` (`fcntl.flock`, POSIX only, re-entrant within a thread) is taken by every writer: consolidation, `gitmem.Memory`, GitHub setup. All file writes are atomic (temp file + `os.replace`).
- **Consolidation never loses a conversation** — LLM errors raise; the worker exits non-zero and keeps the transcript. Nothing is written until both extractions succeed. Each trigger (every 30 messages, `/bye`, idle timer) snapshots and clears one unconsolidated buffer, so no turn is skipped or consolidated twice (the chat window sent to the model is a separate, bounded view).
- **Failed chats leave no trace** — an LLM error drops the user turn; history never starts with an assistant turn.
- **Atomic repo creation** — built in a temp dir and renamed, so concurrent first requests and failed inits never leave a half-built repo.
- **Bounded** — every git call has a timeout; worker processes are reaped; a connection cap and socket timeout bound threads; idle user state and timers are evicted after 10 minutes (memory stays on disk; only the chat window is forgotten).
- **A.U.D.N. sees the right memories** — all of them for a small store, else the ones relevant to the new facts (full-text/embeddings) plus the popular ones; Fact text only.
- **Slug collision handling** — auto-suffixes (`likes-coffee-2`).
- **Contradiction tracking** — DELETE marks a memory contradicted: excluded from recall and search, never destroyed, so `git revert` always works.
- **Generated indexes** — each consolidation regenerates the `## Index` block in `SOUL.md`/`USER.md` (stripped from the prompt, there for people reading the repo on GitHub).

## Security

- **Per-user auth** — see Auth model. userId is required and strictly validated; repo directories are hash-derived.
- **Path traversal** — LLM slugs and `memory_type` are validated everywhere that builds a path.
- **Memory-file injection** — lines starting with `## ` in facts/episodes are neutralised, so a fact cannot pin itself or forge access-log entries.
- **Token never written into git** — the GitHub token is passed per command through a throwaway `GIT_ASKPASS` helper (0700, in `XDG_RUNTIME_DIR` or the repo's `.git`, not `/tmp`), so it is in neither the remote URL nor `argv`.
- **Credentials gitignored and 0600** — `.git_credentials.json` is never committed or pushed.
- **Separate mirror remote** — the user's GitHub is `github`, never `origin`; connecting it cannot repoint the operator's memory store. The setup response echoes at most 200 characters of git's stderr.
- **Private repos** by default; request bodies capped at 10 MB; `Content-Type: application/json` required (forces a CORS preflight).
- Not covered: TLS (put a reverse proxy in front), rate limiting, token revocation (rotate `API_KEY_SECRET`).

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

"""Git-backed agent memory with real search. Markdown is the truth, SQLite FTS5 is the index.

MemPalace's retrieval guts on viral-git-agent-memory's storage model, with the
vector database removed. Chroma's HNSW index inflated `link_lists.bin` from
233 KB to 399 GB in twenty minutes doing nothing but small incremental adds, so
there is deliberately no vector store here — and nothing to corrupt, because the
index is derived and can be thrown away at any time.

Layout of a memory repo (viral-git-agent-memory's, unchanged):

    SOUL.md              base persona   + generated "## Index" of SOUL_memory/
    USER.md              user profile   + generated "## Index" of USER_memory/
    SOUL_memory/<slug>.md    one fact per file, git-tracked
    USER_memory/<slug>.md
    index.sqlite3        derived FTS5 index, gitignored, rebuilt from the files

What this adds over viral-git today: `load_top_memories()` sorts by pinned →
use-count → mtime and takes the top N, so the agent can only recall what is
popular, never what is *relevant*. `Memory.recall()` is a drop-in replacement
that answers "what do I know about X".

Zero dependencies — sqlite3, pathlib and subprocess are stdlib, git is already
required by the repo model.
"""

import fcntl
import re
import sqlite3
import subprocess
import unicodedata
from datetime import date
from pathlib import Path

MEMORY_TYPES = ("SOUL", "USER")
BEHAVIOURAL = "SOUL"  # always-loaded into the prompt; recall() never search-ranks it
FACTUAL = "USER"  # unbounded, asked about directly; this is the side search is for
INDEX_FILE = "index.sqlite3"
INDEX_BEGIN = "<!-- gitmem:index:begin -->"
INDEX_END = "<!-- gitmem:index:end -->"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    rowid        INTEGER PRIMARY KEY,
    slug         TEXT NOT NULL,
    memory_type  TEXT NOT NULL,
    path         TEXT NOT NULL,
    text         TEXT NOT NULL,
    fact         TEXT NOT NULL,
    episode      TEXT NOT NULL DEFAULT '',
    pinned       INTEGER NOT NULL DEFAULT 0,
    contradicted INTEGER NOT NULL DEFAULT 0,
    used         INTEGER NOT NULL DEFAULT 0,
    mtime        REAL NOT NULL,
    UNIQUE(memory_type, slug)
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text, content='memories', content_rowid='rowid', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
END;
"""


# ── the markdown format (viral-git's, byte-compatible with memory_hook.py) ────


def safe_slug(slug: str) -> str:
    """Filesystem-safe stem. LLMs generate these, so path traversal is a real input.

    Accents are folded rather than mangled: "José" has to become "jose", not "jos".
    English text carries plenty of these — names especially — and slug words are
    indexed, so mangling them injects junk tokens into search as well as producing
    unreadable filenames.
    """
    slug = "".join(
        c for c in unicodedata.normalize("NFKD", slug) if not unicodedata.combining(c)
    )
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "-", slug)
    safe = re.sub(r"^[\-.]+", "", safe)
    safe = re.sub(r"-{2,}", "-", safe)
    return (safe or "unnamed")[:80]


# Manual capture triggers. Longest first — "remember that " must win over "remember ".
CAPTURE_PREFIXES = (
    "remember that ", "remember this: ", "remember: ", "/remember ", "remember ",
)


def slug_from(fact: str) -> str:
    """Derive a filename from a fact, no model involved: its first few content words."""
    words = [
        w for w in re.findall(r"[\w']+", fact.lower())
        if len(w) > 2 and w not in STOPWORDS
    ]
    return safe_slug("-".join(words[:5]) or fact.lower())


def render(slug: str, fact: str, episode: str = "", pinned: bool = False,
           access_log: str = "") -> str:
    """Render a memory file exactly as viral-git's write_memory_file does."""
    today = date.today().strftime("%d.%m.%y")
    out = f"# {slug.replace('-', ' ').title()}\n\n## Fact\n{fact}\n\n"
    if pinned:
        out += "## Pinned\ncritical\n\n"
    if episode:
        out += f"## Episode\n{episode}\n\n"
    out += access_log.rstrip() + "\n" if access_log else f"## Access log\nused, {today}\n"
    return out


def parse(text: str) -> dict:
    """Pull the sections out of a memory file. Unknown sections are ignored, not lost."""
    sections, current, buf = {}, None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if current:
                sections[current] = "\n".join(buf).strip()
            current, buf = line[3:].strip(), []
        elif current:
            buf.append(line)
    if current:
        sections[current] = "\n".join(buf).strip()

    log = sections.get("Access log", "")
    return {
        "fact": sections.get("Fact", ""),
        "episode": sections.get("Episode", ""),
        "pinned": "Pinned" in sections,
        "contradicted": "Contradicted" in sections,
        "used": sum(1 for ln in log.splitlines() if ln.strip().startswith("used,")),
        "access_log": f"## Access log\n{log}" if log else "",
    }


# Function words carry no signal but appear in nearly every memory, so OR-ing them
# in makes every document a match. Measured: without this, "what does it cost to
# keep a worker warm" matched an unrelated memory about disk usage at a confident
# 0.7 — a wrong answer is worse than no answer.
STOPWORDS = frozenset("""
a an and are as at be been but by can did do does for from had has have how i
if in into is it its of on or our so than that the their them then there these
they this to was we were what when where which who why will with would you your
""".split())


def match_expr(query: str) -> str:
    """Natural language to a safe FTS5 MATCH expression.

    Content words are quoted and OR-ed, so bm25 ranks documents matching more (and
    rarer) terms higher. Quoting is also what stops FTS5 operators in user text —
    AND, NEAR, a stray quote — from being parsed as query syntax.

    Terms are matched exactly, not by prefix: the porter tokenizer already handles
    English morphology (cars/car, driving/drives, databases/database), so prefix
    matching was measured at 8/8 either way but four extra false positives with it
    — "card" reaching cardiology, "data" reaching databases. Non-English morphology
    is the case prefixes would buy, and that is out of scope until English is right.
    """
    words = [
        w for w in re.findall(r"[\w']+", (query or "").lower())
        if len(w) > 2 and w not in STOPWORDS
    ]
    return " OR ".join(f'"{w}"' for w in words)


def _distance(rank: float) -> float:
    """bm25 rank (negative, lower is better) to a [0,1) distance, so 1-d reads as a score.

    ponytail: monotone squash, not a calibrated similarity. If these numbers ever
    have to mean something, that is the point to add real embeddings.
    """
    return 1.0 / (1.0 + max(0.0, -rank))


class _lock:
    """Exclusive per-repo lock. viral-git already locks consolidations; writes need it too."""

    def __init__(self, repo):
        self._path = Path(repo) / ".gitmem.lock"

    def __enter__(self):
        self._fh = open(self._path, "w")
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._fh.close()


# ── the memory repo ───────────────────────────────────────────────────────────


class Memory:
    """One user's memory repo: markdown files, a rebuildable index, one git history."""

    def __init__(self, repo: str, create: bool = False):
        self.repo = Path(repo)
        if create:
            self.repo.mkdir(parents=True, exist_ok=True)
            for memory_type in MEMORY_TYPES:
                (self.repo / f"{memory_type}_memory").mkdir(exist_ok=True)
            gitignore = self.repo / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text(
                    f"{INDEX_FILE}\n{INDEX_FILE}-wal\n{INDEX_FILE}-shm\n.gitmem.lock\n",
                    encoding="utf-8",
                )
            if not (self.repo / ".git").is_dir():
                subprocess.run(["git", "init", "-q"], cwd=self.repo, capture_output=True)
        elif not self.repo.is_dir():
            raise FileNotFoundError(repo)

        self.db = sqlite3.connect(self.repo / INDEX_FILE)
        self.db.executescript(_SCHEMA)
        self.db.commit()
        self.sync()

    # -- index -------------------------------------------------------------

    def sync(self) -> int:
        """Reconcile the index with the files on disk. The files always win.

        Called on open, so memory_hook.py can keep writing files directly without
        knowing this index exists — anything it changed is picked up here.
        """
        changed = 0
        with _lock(self.repo):
            known = {
                (r[0], r[1]): r[2]
                for r in self.db.execute("SELECT memory_type, slug, mtime FROM memories")
            }
            seen = set()
            for memory_type in MEMORY_TYPES:
                directory = self.repo / f"{memory_type}_memory"
                if not directory.is_dir():
                    continue
                for file in sorted(directory.glob("*.md")):
                    key = (memory_type, file.stem)
                    seen.add(key)
                    mtime = file.stat().st_mtime
                    if known.get(key) == mtime:
                        continue
                    self._index(memory_type, file, mtime)
                    changed += 1
            for memory_type, slug in known.keys() - seen:
                self.db.execute(
                    "DELETE FROM memories WHERE memory_type = ? AND slug = ?",
                    (memory_type, slug),
                )
                changed += 1
            self.db.commit()
        return changed

    def _index(self, memory_type: str, file: Path, mtime: float):
        parsed = parse(file.read_text(encoding="utf-8"))
        searchable = " ".join(
            [file.stem.replace("-", " "), parsed["fact"], parsed["episode"]]
        )
        self.db.execute(
            "INSERT INTO memories (slug, memory_type, path, text, fact, episode, "
            "pinned, contradicted, used, mtime) VALUES (?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(memory_type, slug) DO UPDATE SET path=excluded.path, "
            "text=excluded.text, fact=excluded.fact, episode=excluded.episode, "
            "pinned=excluded.pinned, contradicted=excluded.contradicted, "
            "used=excluded.used, mtime=excluded.mtime",
            (
                file.stem,
                memory_type,
                file.relative_to(self.repo).as_posix(),
                searchable,
                parsed["fact"],
                parsed["episode"],
                int(parsed["pinned"]),
                int(parsed["contradicted"]),
                parsed["used"],
                mtime,
            ),
        )

    def reindex(self) -> int:
        """Throw the index away and rebuild it from the files. Always safe."""
        with _lock(self.repo):
            self.db.execute("DELETE FROM memories")
            self.db.commit()
        return self.sync()

    # -- writes ------------------------------------------------------------

    def write(self, memory_type: str, slug: str, fact: str, episode: str = "",
              pinned: bool = False) -> str:
        """Create or update one memory. Returns the slug actually used."""
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"memory_type must be one of {MEMORY_TYPES}, got {memory_type!r}")
        if not fact.strip():
            raise ValueError("fact must not be empty")

        slug = safe_slug(slug)
        directory = self.repo / f"{memory_type}_memory"
        directory.mkdir(parents=True, exist_ok=True)
        file = directory / f"{slug}.md"

        with _lock(self.repo):
            # Preserve the access log across updates, as update_memory_file does.
            existing = parse(file.read_text(encoding="utf-8")) if file.exists() else {}
            file.write_text(
                render(slug, fact, episode, pinned or existing.get("pinned", False),
                       existing.get("access_log", "")),
                encoding="utf-8",
            )
            self._index(memory_type, file, file.stat().st_mtime)
            self.db.commit()
        return slug

    def capture(self, memory_type: str, message: str, pinned: bool = False):
        """Manual capture — "remember: Alice lives in Bucharest". No model, just a parse.

        This is what keeps the expired/free tier a working product instead of a
        read-only archive: without the consolidation LLM the agent stops *noticing*
        things, but the user can still tell it what to keep.

        Returns the slug written, or None if the message was not a capture — so the
        caller can hand it every inbound message and act on the result.
        """
        text = (message or "").strip()
        lowered = text.lower()
        for prefix in CAPTURE_PREFIXES:
            if lowered.startswith(prefix):
                fact = text[len(prefix):].strip().lstrip(":").strip()
                break
        else:
            return None
        if not fact:
            return None

        # Never overwrite an existing memory. write() is an upsert by design, but two
        # unrelated facts can derive the same slug, and silently losing one is the
        # worst thing a memory system can do. Saying the same thing twice, on the
        # other hand, should not pile up duplicates — that makes a log, not a memory.
        base = slug_from(fact)
        directory = self.repo / f"{memory_type}_memory"
        slug, counter = base, 2
        while (directory / f"{slug}.md").exists():
            if parse((directory / f"{slug}.md").read_text(encoding="utf-8"))["fact"] == fact:
                return slug
            slug = f"{base}-{counter}"
            counter += 1
        return self.write(memory_type, slug, fact, pinned=pinned)

    def delete(self, memory_type: str, slug: str) -> bool:
        """Remove a memory: the file and its index row."""
        slug = safe_slug(slug)
        file = self.repo / f"{memory_type}_memory" / f"{slug}.md"
        with _lock(self.repo):
            existed = file.exists()
            file.unlink(missing_ok=True)
            self.db.execute(
                "DELETE FROM memories WHERE memory_type = ? AND slug = ?", (memory_type, slug)
            )
            self.db.commit()
        return existed

    # -- reads -------------------------------------------------------------

    def search(self, query: str, memory_type: str = None, n: int = 5) -> list:
        """Full-text search, best first. This is what viral-git has no way to do."""
        expr = match_expr(query)
        if not expr:
            return []
        # CROSS JOIN is load-bearing, not style: it pins the join order so the
        # full-text match runs ONCE and drives the loop. With a plain JOIN, adding
        # `memory_type = ?` let SQLite drive from the UNIQUE(memory_type, slug)
        # index instead — scanning every row and re-running the match per row.
        # Measured at 28k memories: 47,000 ms with JOIN, 31 ms with CROSS JOIN.
        sql = (
            "SELECT m.slug, m.memory_type, m.fact, m.episode, m.pinned, m.used, "
            "bm25(memories_fts) AS rank FROM memories_fts "
            "CROSS JOIN memories m ON m.rowid = memories_fts.rowid "
            "WHERE memories_fts MATCH ? AND m.contradicted = 0"
        )
        params = [expr]
        if memory_type:
            sql += " AND m.memory_type = ?"
            params.append(memory_type)
        # rank < 0 drops no-signal matches: bm25 scores a document that shares only
        # negligible terms at ~0, and returning those pads every search with noise.
        sql += " AND rank < 0 ORDER BY rank LIMIT ?"
        params.append(n)
        return [self._hit(r) for r in self.db.execute(sql, params)]

    def top(self, memory_type: str, n: int = 15) -> list:
        """viral-git's existing ranking: pinned, then use count, then recency."""
        rows = self.db.execute(
            "SELECT slug, memory_type, fact, episode, pinned, used, 0 FROM memories "
            "WHERE memory_type = ? AND contradicted = 0 "
            "ORDER BY pinned DESC, used DESC, mtime DESC LIMIT ?",
            (memory_type, n),
        )
        return [self._hit(r) for r in rows]

    def recall(self, memory_type: str, query: str = "", n: int = 15) -> str:
        """Drop-in for load_top_memories(): the same "- fact" block, relevance-ranked.

        Pinned memories always lead — they are the ones marked as always-load, and
        a keyword query must not be able to push them out. Relevant hits follow,
        then the popular ones backfill to n. With no query this is exactly the old
        behaviour, so it is safe to call unconditionally.

        BEHAVIOURAL memory ignores the query on purpose. SOUL_memory holds how the
        agent should act — it is small, bounded, and the user never asks about it,
        so it simply has to be in the prompt. Search there only appears to work when
        the user happens to use the memory's own words ("stop hedging" hitting "no
        hedging"); phrase it any other way and relevance drops behavioural rules the
        agent still has to follow. Factual memory is the opposite: it grows without
        limit and the user asks about it directly, which is what search is for.
        `search()` still works on either side for callers that explicitly want it.
        """
        if memory_type == BEHAVIOURAL:
            query = ""

        chosen, seen = [], set()
        for hit in self._pinned(memory_type) + self.search(query, memory_type, n) + \
                self.top(memory_type, n):
            if hit["slug"] in seen:
                continue
            seen.add(hit["slug"])
            chosen.append(hit)
            if len(chosen) >= n:
                break
        return "\n".join(f"- {h['fact']}" for h in chosen)

    def _pinned(self, memory_type: str) -> list:
        rows = self.db.execute(
            "SELECT slug, memory_type, fact, episode, pinned, used, 0 FROM memories "
            "WHERE memory_type = ? AND pinned = 1 AND contradicted = 0 ORDER BY mtime DESC",
            (memory_type,),
        )
        return [self._hit(r) for r in rows]

    @staticmethod
    def _hit(row) -> dict:
        return {
            "slug": row[0], "memory_type": row[1], "fact": row[2], "episode": row[3],
            "pinned": bool(row[4]), "used": row[5], "distance": _distance(row[6]),
        }

    def count(self, memory_type: str = None) -> int:
        if memory_type:
            return self.db.execute(
                "SELECT COUNT(*) FROM memories WHERE memory_type = ?", (memory_type,)
            ).fetchone()[0]
        return self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    # -- the two index files, and git --------------------------------------

    def write_indexes(self):
        """Regenerate the "## Index" block in SOUL.md and USER.md.

        The hand-written half of each file — the persona in SOUL.md, the profile in
        USER.md — is what assemble_context() injects into the system prompt, so it
        is preserved untouched; only the marked block below it is rewritten.
        """
        for memory_type in MEMORY_TYPES:
            rows = self.db.execute(
                "SELECT slug, path, fact, pinned FROM memories WHERE memory_type = ? "
                "ORDER BY pinned DESC, slug",
                (memory_type,),
            ).fetchall()
            lines = [
                INDEX_BEGIN,
                "## Index",
                "",
                f"{len(rows)} memories in `{memory_type}_memory/`. "
                "Generated — edit the memory files, not this block.",
                "",
            ]
            for slug, path, fact, pinned in rows:
                summary = " ".join(fact.split())[:160]
                pin = "📌 " if pinned else ""
                lines.append(f"- {pin}[`{slug}`]({path}) — {summary}")
            block = "\n".join(lines + ["", INDEX_END])

            file = self.repo / f"{memory_type}.md"
            old = file.read_text(encoding="utf-8") if file.exists() else ""
            start, end = old.find(INDEX_BEGIN), old.find(INDEX_END)
            if start != -1 and end != -1:
                new = old[:start] + block + old[end + len(INDEX_END):]
            else:
                new = (old.rstrip() + "\n\n" if old.strip() else "") + block + "\n"
            file.write_text(new, encoding="utf-8")

    def commit(self, message: str = None) -> bool:
        """Regenerate the index files and make ONE commit.

        Deliberately not called per write. MemPalace's Stop hook fired after every
        assistant turn; a commit per memory is thousands of commits a day, and that
        write pattern is what shredded the Chroma index in the first place.
        """
        self.write_indexes()
        if not (self.repo / ".git").is_dir():
            return False
        message = message or f"memory checkpoint {date.today().strftime('%d.%m.%y')}"
        subprocess.run(["git", "add", "-A"], cwd=self.repo, capture_output=True)
        done = subprocess.run(
            ["git", "commit", "-m", message], cwd=self.repo, capture_output=True, text=True
        )
        return done.returncode == 0


# ── check ─────────────────────────────────────────────────────────────────────


def selftest():
    """python3 gitmem.py"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        (repo / "SOUL.md").write_text("# SOUL\n\n## Identity\n- Name: Assistant\n", encoding="utf-8")
        (repo / "USER.md").write_text("# User Profile\n\n## Identity\n- Alice\n", encoding="utf-8")

        mem = Memory(tmp, create=True)
        assert mem.count() == 0

        mem.write("USER", "chroma-bloat",
                  "The chroma index inflated link_lists.bin to 399 GB in twenty minutes",
                  episode="Disk hit 4 KB free twice in one day")
        mem.write("USER", "piper-tts", "Piper is the MIT speech engine on the voice box")
        mem.write("USER", "runpod-idle",
                  "The render endpoint idleTimeout is 150 seconds and serverless bills idle")
        mem.write("SOUL", "be-blunt", "The user wants blunt answers, no hedging", pinned=True)
        assert mem.count() == 4 and mem.count("USER") == 3

        # the file is the truth, in viral-git's own format
        raw = (repo / "USER_memory/chroma-bloat.md").read_text()
        assert "## Fact" in raw and "## Episode" in raw and "## Access log" in raw
        assert "## Pinned" in (repo / "SOUL_memory/be-blunt.md").read_text()

        # SEARCH — the thing viral-git cannot do at all
        hits = mem.search("disk filled up by the index", "USER")
        assert hits and hits[0]["slug"] == "chroma-bloat", [h["slug"] for h in hits]
        assert 0.0 <= hits[0]["distance"] < 1.0
        assert hits[0]["distance"] <= hits[-1]["distance"], "results not best-first"

        # a different question reaches a different memory
        assert mem.search("idle timeout billing", "USER")[0]["slug"] == "runpod-idle"
        assert mem.search("serverless idle cost", "USER")[0]["slug"] == "runpod-idle"

        # THE LIMIT OF KEYWORD SEARCH, asserted so nobody discovers it in production:
        # this asks about the same memory in words it does not contain, and gets
        # nothing. An honest miss — the earlier version confidently returned the
        # WRONG memory here. Only embeddings would close this gap.
        assert mem.search("what does it cost to keep a worker warm", "USER") == []

        # and no-signal matches never pad the results
        assert [h["slug"] for h in mem.search("disk filled up by the index", "USER")] \
            == ["chroma-bloat"]

        # search is scoped by memory type
        assert [h["slug"] for h in mem.search("blunt", "USER")] == []
        assert mem.search("blunt", "SOUL")[0]["slug"] == "be-blunt"

        # recall: pinned first, then relevant; no query = the old popularity order
        assert mem.recall("SOUL", "anything at all").startswith("- The user wants blunt")
        assert mem.recall("USER", "index disk bloat").startswith("- The chroma index")
        assert len(mem.recall("USER").splitlines()) == 3

        # SEARCH THE FACTUAL SIDE, ALWAYS-LOAD THE BEHAVIOURAL SIDE.
        # A behavioural rule must reach the prompt however the user phrases things,
        # so recall() ignores the query for SOUL: every wording returns the same set.
        mem.write("SOUL", "no-emoji", "Never use emoji in replies")
        soul = mem.recall("SOUL")
        assert mem.recall("SOUL", "blunt hedging") == soul, "behavioural recall was filtered"
        assert mem.recall("SOUL", "totally unrelated words") == soul
        assert "Never use emoji" in soul, soul
        # ...while the factual side really is narrowed by the query
        assert mem.recall("USER", "chroma") != mem.recall("USER", "piper speech")
        # search() still works on either side for callers that ask for it explicitly
        assert mem.search("blunt", "SOUL")[0]["slug"] == "be-blunt"

        # updating keeps the access log, and re-searches correctly
        mem.write("USER", "piper-tts", "Piper replaced edge-tts because edge-tts is unofficial")
        assert "## Access log" in (repo / "USER_memory/piper-tts.md").read_text()
        assert mem.search("unofficial microsoft endpoint", "USER")[0]["slug"] == "piper-tts"

        # contradicted memories drop out of recall but stay on disk
        path = repo / "USER_memory/runpod-idle.md"
        path.write_text(path.read_text() + "\n## Contradicted\nMarked contradicted\n",
                        encoding="utf-8")
        mem.sync()
        assert mem.search("idle timeout billing", "USER") == []
        assert path.exists() and mem.count("USER") == 3

        # THE POINT: the index is derived. Delete it, everything comes back.
        before = mem.recall("USER", "chroma disk")
        (repo / INDEX_FILE).unlink()
        mem = Memory(tmp)
        assert mem.count() == 5, mem.count()
        assert mem.recall("USER", "chroma disk") == before

        # a file written directly by memory_hook.py is picked up without being told
        (repo / "USER_memory/build-box.md").write_text(
            render("build-box", "The build box is a one-core cloud VM in Amsterdam"),
            encoding="utf-8")
        assert mem.sync() == 1
        assert mem.search("amsterdam server", "USER")[0]["slug"] == "build-box"

        # delete removes the file, not just the row
        assert mem.delete("USER", "build-box") is True
        assert not (repo / "USER_memory/build-box.md").exists()
        assert mem.search("amsterdam server", "USER") == []

        # index files: persona preserved, block generated, regeneration idempotent
        assert mem.commit("selftest") is True
        user_md = (repo / "USER.md").read_text()
        assert "# User Profile" in user_md and "- Alice" in user_md, "persona was clobbered"
        assert "chroma-bloat" in user_md and "3 memories" in user_md
        assert "📌" in (repo / "SOUL.md").read_text()
        mem.write_indexes()
        assert (repo / "USER.md").read_text().count(INDEX_BEGIN) == 1, "index block duplicated"

        # one commit, not one per write
        log = subprocess.run(["git", "log", "--oneline"], cwd=tmp,
                             capture_output=True, text=True).stdout
        assert log.count("\n") == 1, log

        # MANUAL CAPTURE — the free tier with no consolidation LLM
        assert mem.capture("USER", "remember: Alice lives in Bucharest") == "alice-lives-bucharest"
        assert mem.capture("USER", "Remember that the standup meeting is at 9am") \
            == "standup-meeting-9am"
        assert mem.capture("USER", "/remember he drives a Volkswagen") == "drives-volkswagen"
        # captured facts are immediately searchable, no model anywhere in the path
        assert mem.search("standup meeting", "USER")[0]["slug"] == "standup-meeting-9am"
        assert mem.search("Alice", "USER")[0]["slug"] == "alice-lives-bucharest"

        # not a capture -> None, so the caller can try it on every inbound message
        assert mem.capture("USER", "what do you know about Alice?") is None
        assert mem.capture("USER", "remember:") is None
        assert mem.capture("USER", "") is None

        # saying the same thing twice is a no-op, not a duplicate
        before = mem.count("USER")
        assert mem.capture("USER", "remember: Alice lives in Bucharest") == "alice-lives-bucharest"
        assert mem.count("USER") == before, "repeating a fact created a duplicate"

        # but two DIFFERENT facts that derive the same slug must not clobber each other
        a = mem.capture("USER", "remember: Alice lives in Bucharest Romania since 2019")
        b = mem.capture("USER", "remember: Alice lives in Bucharest Romania since 2024")
        assert a == "alice-lives-bucharest-romania-since", a
        assert b == "alice-lives-bucharest-romania-since-2", b
        kept = {h["slug"] for h in mem.search("Romania", "USER")}
        assert {a, b} <= kept, kept

        # hostile slugs cannot escape the memory directory
        mem.write("USER", "../../etc/passwd", "nope")
        assert not (repo.parent / "passwd").exists()
        assert list((repo / "USER_memory").glob("*.md"))

    print("gitmem selftest OK")


if __name__ == "__main__":
    selftest()

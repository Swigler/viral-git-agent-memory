"""Shared plumbing for the memory repo: locking, atomic writes, git, GitHub credentials.

One home for everything that touches the repo as a *repo* (as opposed to the markdown
format, which is gitmem.py, or the LLM consolidation, which is memory_hook.py).

POSIX only: the per-repo lock is fcntl.flock. Stdlib only.
"""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from datetime import date
from pathlib import Path

# The remote for the user's own GitHub mirror. Deliberately NOT "origin": origin is
# wherever the operator stores memory. Overwriting it means the first user who
# connects GitHub silently stops saving to that store.
GITHUB_REMOTE = "github"

# Per-user GitHub credentials. Lives inside the repo but is gitignored and 0600, so it
# is never committed and never pushed.
CREDENTIALS_FILE = ".git_credentials.json"

# Hard ceilings for git subprocesses: a stalled push must not hold the repo lock forever.
GIT_CMD_TIMEOUT = int(os.environ.get("GIT_CMD_TIMEOUT", "60"))
GIT_PUSH_TIMEOUT = int(os.environ.get("GIT_PUSH_TIMEOUT", "120"))

LOCK_FILE = ".gitmem.lock"


class GitError(RuntimeError):
    pass


def log_stderr(msg: str):
    print(msg, file=sys.stderr, flush=True)


# -- locking and atomic writes ---------------------------------------------------

_locks: dict = {}
_locks_guard = threading.Lock()


@contextmanager
def repo_lock(repo):
    """Exclusive per-repo lock: flock between processes, RLock between threads.

    Re-entrant within a thread, so a locked consolidation can call Memory methods
    that lock again. The one lock file for every writer of the repo.
    ponytail: one small RLock per repo path is kept for the process lifetime.
    """
    path = str(Path(repo) / LOCK_FILE)
    with _locks_guard:
        entry = _locks.setdefault(path, {"rlock": threading.RLock(), "fh": None, "depth": 0})
    with entry["rlock"]:
        if entry["depth"] == 0:
            entry["fh"] = open(path, "w")
            fcntl.flock(entry["fh"], fcntl.LOCK_EX)
        entry["depth"] += 1
        try:
            yield
        finally:
            entry["depth"] -= 1
            if entry["depth"] == 0:
                fcntl.flock(entry["fh"], fcntl.LOCK_UN)
                entry["fh"].close()
                entry["fh"] = None


def atomic_write(path, text: str, mode: int = 0o644):
    """Write via temp file + os.replace: readers never see a half-written file."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# -- git ---------------------------------------------------------------------------


def git_env(extra: dict | None = None) -> dict:
    """Env for every git call: never prompt, always have a committer identity."""
    return {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_AUTHOR_NAME": "memory-hook",
        "GIT_AUTHOR_EMAIL": "hook@viral-git-agent-memory",
        "GIT_COMMITTER_NAME": "memory-hook",
        "GIT_COMMITTER_EMAIL": "hook@viral-git-agent-memory",
        **(extra or {}),
    }


def run_git(repo, *args, env=None, timeout=GIT_CMD_TIMEOUT, check=False):
    """git with a timeout on every call (TimeoutExpired propagates; run() reaps the child)."""
    done = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True,
        env=env or git_env(), timeout=timeout,
    )
    if check and done.returncode != 0:
        raise GitError(f"git {args[0]} failed: {(done.stderr or done.stdout).strip()[:200]}")
    return done


def init_user_repo(repo, user_id: str, template) -> bool:
    """Create a user repo from the template, atomically. True if this call created it.

    Built in a sibling temp dir and renamed into place, so a concurrent request never
    sees a half-built repo and a failed init leaves nothing behind.
    """
    repo = Path(repo)
    if repo.exists():
        return False
    repo.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".new-", dir=repo.parent))
    try:
        work = tmp / "repo"
        shutil.copytree(template, work)
        user_md = work / "USER.md"
        if user_md.exists():
            user_md.write_text(
                user_md.read_text(encoding="utf-8").replace("User ID: no_one", f"User ID: {user_id}"),
                encoding="utf-8",
            )
        for d in ("USER_memory", "SOUL_memory"):
            (work / d).mkdir(exist_ok=True)
        run_git(work, "init", "-q", check=True)
        run_git(work, "add", "-A", check=True)
        run_git(work, "commit", "-q", "-m", "init", check=True)
        try:
            os.rename(work, repo)
        except OSError:
            if not repo.exists():  # a real failure, not a lost race
                raise
            return False
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def commit_all(repo, message: str, allow_empty: bool = False, log=log_stderr) -> str:
    """git add -A + commit. Returns "committed", "nothing" or "failed" (logged)."""
    try:
        added = run_git(repo, "add", "-A")
        if added.returncode != 0:
            log(f"[git] add failed: {(added.stderr or '').strip()[:200]}")
            return "failed"
        args = ["commit", "-m", message] + (["--allow-empty"] if allow_empty else [])
        done = run_git(repo, *args)
    except subprocess.TimeoutExpired:
        log("[git] add/commit timed out — skipping this round")
        return "failed"
    if done.returncode == 0:
        log(f"[git] committed: {done.stdout.strip().splitlines()[0]}")
        return "committed"
    # "nothing to commit" goes to STDOUT, not stderr.
    output = (done.stdout + done.stderr).strip()
    if "nothing to commit" in output or "working tree clean" in output:
        log("[git] nothing to commit")
        return "nothing"
    log(f"[git] COMMIT FAILED (rc={done.returncode}): {output[:200]}")
    return "failed"


def git_commit(repo, message: str | None = None, log=log_stderr) -> str:
    """The one commit+push path: commit, push the operator's store, mirror to GitHub."""
    repo = Path(repo)
    if not (repo / ".git").is_dir():
        log("[git] not a git repo, skipping commit")
        return "failed"
    status = commit_all(repo, message or f"checkpoint {date.today().strftime('%d.%m.%y')}", log=log)
    if status == "committed":
        _push_store(repo, log)
    # Mirror runs even when nothing was committed: that re-syncs a mirror left behind
    # by an earlier push failure, and costs one "Everything up-to-date" otherwise.
    push_to_github(repo, log=log)
    return status


def _push_store(repo: Path, log):
    """Push to the operator's store: the branch's own remote, else origin. By name."""
    try:
        remotes = run_git(repo, "remote").stdout.split()
        branch = run_git(repo, "branch", "--show-current").stdout.strip()
        if not branch:
            return
        configured = run_git(repo, "config", "--get", f"branch.{branch}.remote").stdout.strip()
        remote = configured or ("origin" if "origin" in remotes else "")
        if remote not in remotes or remote == GITHUB_REMOTE:
            return
        push = run_git(repo, "push", "-u", remote, branch, timeout=GIT_PUSH_TIMEOUT)
    except subprocess.TimeoutExpired:
        log("[git] push timed out — will retry on the next consolidation")
        return
    if push.returncode == 0:
        log("[git] pushed")
    else:
        log(f"[git] push failed: {push.stderr.strip()[:100]}")


# -- GitHub credentials and mirror ------------------------------------------------


def git_credentials_path(repo: Path) -> Path:
    return repo / CREDENTIALS_FILE


def load_git_credentials(repo: Path) -> dict:
    try:
        return json.loads(git_credentials_path(repo).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_git_credentials(repo: Path, creds: dict):
    """Write credentials 0600 (temp file + rename: never world-readable) and gitignore them."""
    atomic_write(git_credentials_path(repo), json.dumps(creds, indent=2), mode=0o600)
    gitignore = repo / ".gitignore"
    content = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    if CREDENTIALS_FILE not in content.split():
        atomic_write(gitignore, content + ("\n" if content and not content.endswith("\n") else "")
                     + CREDENTIALS_FILE + "\n")


def _helper_dir(repo) -> Path:
    """Where the askpass helper may live: never /tmp (often noexec, and shared)."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime and os.access(runtime, os.W_OK | os.X_OK):
        return Path(runtime)
    if repo is not None and (Path(repo) / ".git").is_dir():
        return Path(repo) / ".git"
    d = Path.home() / ".cache" / "viral-git-agent-memory"
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    return d


@contextmanager
def git_auth(token: str, repo=None):
    """Yield an env that authenticates git without the token ever being written down.

    The token goes in neither the remote URL (.git/config, git's own error messages)
    nor argv (`ps`): a throwaway GIT_ASKPASS helper (0700, in XDG_RUNTIME_DIR or the
    repo's .git dir) reads it from the environment.
    """
    fd, helper = tempfile.mkstemp(prefix="git-askpass-", dir=_helper_dir(repo))
    try:
        os.write(fd, b'#!/bin/sh\nprintf %s "$GIT_ACCESS_TOKEN"\n')
        os.close(fd)
        os.chmod(helper, 0o700)
        yield git_env({"GIT_ASKPASS": helper, "GIT_ACCESS_TOKEN": token})
    finally:
        os.unlink(helper)


def push_github(repo, token: str | None = None) -> tuple[bool, str]:
    """Mirror push. (ok, error). (False, "") means GitHub is not connected."""
    repo = Path(repo)
    token = token or load_git_credentials(repo).get("token", "")
    if not token:
        return False, ""
    try:
        if GITHUB_REMOTE not in run_git(repo, "remote").stdout.split():
            return False, ""
        branch = run_git(repo, "branch", "--show-current").stdout.strip() or "main"
        # -u only when the branch has no upstream at all: a later bare `git push` then
        # works for a GitHub-only repo, while an existing store upstream is never hijacked.
        has_upstream = run_git(repo, "rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}").returncode == 0
        with git_auth(token, repo) as env:
            push = run_git(
                repo, "push", *([] if has_upstream else ["-u"]), GITHUB_REMOTE, branch,
                env=env, timeout=GIT_PUSH_TIMEOUT,
            )
    except subprocess.TimeoutExpired:
        return False, "git push timed out"
    return push.returncode == 0, push.stderr.strip()[:200]


def push_to_github(repo, token: str | None = None, log=log_stderr) -> bool:
    """Mirror the repo to the user's GitHub. Returns False if GitHub isn't connected.

    Public API (Swigler/memory-bot imports it via api_server). `token` defaults to
    the one saved by /v1/git/setup.
    """
    ok, err = push_github(repo, token)
    if ok:
        log("[github] mirrored")
    elif err:
        log(f"[github] mirror push failed: {err}")
    return ok

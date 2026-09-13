#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Checks on the GitHub credential path. No network — local bare repos stand in.

Guards three bugs that were real:
  1. the token was baked into the remote URL, so it sat in .git/config in plaintext
     and came back out in git's own error messages
  2. GitHub was attached as "origin", replacing the memory store's remote
  3. push_to_github had no callers, so a connected GitHub repo got one push at setup
     and then never updated again

Run:  uv run test_github_auth.py
"""
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import memory_hook as M

TOKEN = "ghp_thisIsNotARealToken0000000000000000"


def git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def make_repo(tmp):
    """A user repo with origin pointing at a stand-in memory store."""
    tmp.mkdir(parents=True, exist_ok=True)
    store = tmp / "store.git"
    subprocess.run(
        ["git", "init", "--bare", "-q", "--initial-branch=main", str(store)],
        check=True, capture_output=True,
    )
    repo = tmp / "work"
    subprocess.run(
        ["git", "init", "-q", "--initial-branch=main", str(repo)],
        check=True, capture_output=True,
    )
    git(repo, "config", "user.email", "t@test.local")
    git(repo, "config", "user.name", "test")
    git(repo, "remote", "add", "origin", str(store))
    (repo / "USER.md").write_text("# User\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "first")
    git(repo, "push", "-q", "-u", "origin", "main")
    return repo, store


def test_git_auth_hides_the_token():
    with M.git_auth(TOKEN) as env:
        helper = Path(env["GIT_ASKPASS"])
        assert helper.exists(), "askpass helper missing"
        assert helper.stat().st_mode & 0o777 == 0o700, "askpass helper must be 0700"
        assert TOKEN not in helper.read_text(), "token must not be written to the helper"
        printed = subprocess.run(
            [str(helper)], capture_output=True, text=True, env=env
        ).stdout
        assert printed == TOKEN, "helper must hand git the token"
    assert not helper.exists(), "askpass helper must be deleted"
    assert "GIT_ACCESS_TOKEN" not in os.environ, "token leaked into the parent env"
    print("  ✅ token never written down, helper cleaned up")


def test_credentials_file_is_private(tmp):
    repo, _ = make_repo(tmp)
    M.save_git_credentials(repo, {"github_username": "alice", "token": TOKEN})
    path = M.git_credentials_path(repo)

    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"credentials must be 0600, got {oct(mode)}"
    assert M.load_git_credentials(repo)["token"] == TOKEN, "token must survive a reload"
    assert ".git_credentials.json" in (repo / ".gitignore").read_text(), "not gitignored"

    # Re-saving over an existing file must not leave the old, looser mode behind.
    os.chmod(path, 0o644)
    M.save_git_credentials(repo, {"token": TOKEN})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600, "re-save must re-tighten the mode"
    print("  ✅ credentials 0600, gitignored, readable back")


def test_mirror_does_not_touch_origin(tmp):
    repo, store = make_repo(tmp)
    mirror = tmp / "mirror.git"
    subprocess.run(
        ["git", "init", "--bare", "-q", "--initial-branch=main", str(mirror)],
        check=True, capture_output=True,
    )
    store_ref = git(repo, "rev-parse", "origin/main")

    # No github remote yet: nothing to mirror to.
    assert M.push_to_github(repo, TOKEN) is False, "should report no github remote"

    git(repo, "remote", "add", M.GITHUB_REMOTE, str(mirror))
    assert M.GITHUB_REMOTE == "github", "the mirror remote must not be called origin"
    assert "origin" in git(repo, "remote").split(), "origin was removed"
    assert git(repo, "remote", "get-url", "origin") == str(store), "origin was repointed"

    (repo / "USER.md").write_text("# User\n\nA fact.\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "second")

    assert M.push_to_github(repo, TOKEN) is True, "mirror push failed"
    assert git(repo, "rev-parse", f"{M.GITHUB_REMOTE}/main") == git(repo, "rev-parse", "HEAD")
    assert git(repo, "rev-parse", "origin/main") == store_ref, "mirror push moved origin"
    assert git(repo, "rev-parse", "--abbrev-ref", "main@{upstream}") == "origin/main", \
        "upstream was hijacked — a bare `git push` would go to GitHub"

    config = (repo / ".git" / "config").read_text()
    assert TOKEN not in config, "token found in .git/config"
    assert "x-access-token:" not in config, "credentials embedded in a remote URL"
    for path in repo.rglob("*"):
        if path.is_file() and TOKEN in path.read_bytes().decode("utf-8", "ignore"):
            assert path.name == ".git_credentials.json", f"token found on disk in {path}"
    print("  ✅ mirror push leaves origin and upstream alone, no token on disk")


def test_no_token_means_no_mirror(tmp):
    """A repo with a github remote but no saved token must not claim it pushed."""
    repo, _ = make_repo(tmp)
    mirror = tmp / "nomirror.git"
    subprocess.run(
        ["git", "init", "--bare", "-q", "--initial-branch=main", str(mirror)],
        check=True, capture_output=True,
    )
    git(repo, "remote", "add", M.GITHUB_REMOTE, str(mirror))
    assert M.push_to_github(repo) is False, "no saved token must not report success"
    print("  ✅ missing token is reported, not guessed")


def test_consolidation_mirrors(tmp):
    """The regression that mattered: git_commit must drive the mirror push itself."""
    repo, store = make_repo(tmp)
    mirror = tmp / "mirror2.git"
    subprocess.run(
        ["git", "init", "--bare", "-q", "--initial-branch=main", str(mirror)],
        check=True, capture_output=True,
    )
    git(repo, "remote", "add", M.GITHUB_REMOTE, str(mirror))
    M.save_git_credentials(repo, {"token": TOKEN, "github_username": "alice"})

    (repo / "USER_memory").mkdir(exist_ok=True)
    (repo / "USER_memory" / "likes-coffee.md").write_text("# Likes Coffee\n")
    M.git_commit(str(repo))

    head = git(repo, "rev-parse", "HEAD")
    assert git(repo, "rev-parse", f"{M.GITHUB_REMOTE}/main") == head, \
        "consolidation did not reach the user's GitHub mirror"
    assert git(repo, "rev-parse", "origin/main") == head, "memory store missed the commit"

    # A consolidation that changes nothing must still re-sync a mirror left behind,
    # and must not report the empty commit as a failure ("nothing to commit" is on
    # stdout, so a stderr-only check called every no-op a hard error).
    git(repo, "push", "-q", M.GITHUB_REMOTE, "+HEAD~1:main")
    assert git(repo, "rev-parse", f"{M.GITHUB_REMOTE}/main") != head, "setup for resync failed"
    logged = []
    real_log, M._log = M._log, logged.append
    try:
        M.git_commit(str(repo))
    finally:
        M._log = real_log
    assert any("nothing to commit" in line for line in logged), f"no-op not recognised: {logged}"
    assert not any("COMMIT FAILED" in line for line in logged), f"no-op reported as failure: {logged}"
    assert git(repo, "rev-parse", f"{M.GITHUB_REMOTE}/main") == head, \
        "a no-op consolidation must still repair a stale mirror"

    # The credentials file must never ride along into a commit.
    tracked = git(repo, "ls-files").splitlines()
    assert ".git_credentials.json" not in tracked, "credentials got committed"
    assert TOKEN not in git(repo, "log", "-p", "--all"), "token found in history"
    print("  ✅ every consolidation mirrors, stale mirrors self-heal, creds stay untracked")


def test_server_shares_one_implementation():
    """api_server must not grow a second copy of the auth code."""
    import api_server as S
    assert S.git_auth is M.git_auth, "api_server has its own git_auth again"
    assert S.GITHUB_REMOTE == M.GITHUB_REMOTE, "the remote name drifted between modules"
    assert S.save_git_credentials is M.save_git_credentials, "credential writer forked"
    print("  ✅ server and hook share one auth implementation")


def main():
    test_git_auth_hides_the_token()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_credentials_file_is_private(tmp / "a")
        test_mirror_does_not_touch_origin(tmp / "b")
        test_no_token_means_no_mirror(tmp / "c")
        test_consolidation_mirrors(tmp / "d")
    test_server_shares_one_implementation()
    print("github auth checks: all assertions passed")


if __name__ == "__main__":
    main()

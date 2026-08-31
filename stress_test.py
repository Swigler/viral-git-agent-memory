#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Stress test for viral-git-agent-memory.
Tests every code path: init, extraction, AUDN (ADD/UPDATE/DELETE/NONE),
git commit, use-count ranking, transient filtering, index rebuild, edge cases.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent))
import memory_hook as mh

REPO = None
PASS = 0
FAIL = 0


def setup():
    global REPO
    tmp = tempfile.mkdtemp(prefix="stress_test_")
    REPO = os.path.join(tmp, "user_repo")  # subdir so init_repo can copytree the template
    print(f"\n{'='*60}")
    print(f"  STRESS TEST — repo at {REPO}")
    print(f"{'='*60}\n")


def teardown():
    if REPO:
        parent = os.path.dirname(REPO)
        if os.path.exists(parent):
            shutil.rmtree(parent)


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")


def test_init_repo():
    """Test repo initialization from template."""
    print("\n--- TEST 1: init_repo ---")
    mh.init_repo(REPO, "stress_user_42")

    check("user_memory/ exists", (Path(REPO) / "user_memory").is_dir())
    check("character_memory/ exists", (Path(REPO) / "character_memory").is_dir())
    check("character.md exists", (Path(REPO) / "character.md").is_file())
    check("user.md exists", (Path(REPO) / "user.md").is_file())

    user_md = (Path(REPO) / "user.md").read_text()
    check("user ID stamped", "stress_user_42" in user_md, f"got: {user_md[:100]}")

    # Init git repo for later tests
    env = {**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@test",
           "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@test"}
    subprocess.run(["git", "init"], cwd=REPO, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=REPO, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=REPO, capture_output=True, env=env)
    check("git repo initialized", (Path(REPO) / ".git").is_dir())


def test_write_memory_file():
    """Test creating and appending to memory files."""
    print("\n--- TEST 2: write_memory_file ---")

    mh.write_memory_file(REPO, "user_memory", "likes-pizza", "You love pizza.", "Session 30.08.26 — you ordered a margherita.")
    f = Path(REPO) / "user_memory" / "likes-pizza.md"
    check("file created", f.is_file())
    content = f.read_text()
    check("has Fact section", "## Fact" in content)
    check("has Episode section", "## Episode" in content)
    check("has Access log", "## Access log" in content)
    check("has used stamp", "used," in content)
    check("fact correct", "You love pizza." in content)

    # Write again with same slug — should create likes-pizza-2.md (collision handling)
    mh.write_memory_file(REPO, "user_memory", "likes-pizza", "You also love pasta.", "Session 31.08.26")
    f2 = Path(REPO) / "user_memory" / "likes-pizza-2.md"
    check("slug collision creates suffixed file", f2.is_file())
    check("original file unchanged", "You love pizza." in f.read_text())
    check("new file has new fact", "You also love pasta." in f2.read_text())


def test_intra_batch_dedup():
    """Test that seen_slugs prevents intra-batch clobbering."""
    print("\n--- TEST 2b: intra-batch dedup ---")

    seen = set()
    mh.write_memory_file(REPO, "user_memory", "batch-test", "First fact", "Ep1", seen)
    mh.write_memory_file(REPO, "user_memory", "batch-test", "Second fact", "Ep2", seen)
    mh.write_memory_file(REPO, "user_memory", "batch-test", "Third fact", "Ep3", seen)

    f1 = Path(REPO) / "user_memory" / "batch-test.md"
    f2 = Path(REPO) / "user_memory" / "batch-test-2.md"
    f3 = Path(REPO) / "user_memory" / "batch-test-3.md"

    check("batch file 1 created", f1.is_file())
    check("batch file 2 created (dedup)", f2.is_file())
    check("batch file 3 created (dedup)", f3.is_file())
    check("file 1 has first fact", "First fact" in f1.read_text())
    check("file 2 has second fact", "Second fact" in f2.read_text())
    check("file 3 has third fact", "Third fact" in f3.read_text())


def test_update_memory_file():
    """Test updating an existing memory."""
    print("\n--- TEST 3: update_memory_file ---")

    mh.write_memory_file(REPO, "user_memory", "job-title", "You work in IT.", "Session 30.08.26")
    mh.update_memory_file(REPO, "user_memory", "job-title", "You work as a senior DevOps engineer.", "Session 30.08.26 — you clarified your role.")

    content = (Path(REPO) / "user_memory" / "job-title.md").read_text()
    check("fact updated", "senior DevOps engineer" in content)
    check("old fact gone", "You work in IT." not in content)
    check("access log preserved", "used," in content)


def test_mark_contradicted():
    """Test contradiction marking."""
    print("\n--- TEST 4: mark_contradicted ---")

    mh.write_memory_file(REPO, "user_memory", "has-cat", "You have a cat named Whiskers.", "Session 29.08.26")
    mh.mark_contradicted(REPO, "user_memory", "has-cat")

    content = (Path(REPO) / "user_memory" / "has-cat.md").read_text()
    check("contradicted marker added", "## Contradicted" in content)

    # Mark again — should not duplicate
    mh.mark_contradicted(REPO, "user_memory", "has-cat")
    content2 = (Path(REPO) / "user_memory" / "has-cat.md").read_text()
    count = content2.count("## Contradicted")
    check("no duplicate contradicted marker", count == 1, f"got {count}")


def test_missing_target_logging():
    """Test that UPDATE/DELETE on missing targets don't silently no-op."""
    print("\n--- TEST 4b: missing target handling ---")

    # DELETE on non-existent slug should not crash
    mh.mark_contradicted(REPO, "user_memory", "totally-fake-slug")
    check("DELETE on missing slug doesn't crash", True)
    f = Path(REPO) / "user_memory" / "totally-fake-slug.md"
    check("DELETE on missing slug doesn't create file", not f.is_file())

    # UPDATE on non-existent slug should create new file (fallback)
    mh.update_memory_file(REPO, "user_memory", "also-fake", "Fallback fact", "Fallback episode")
    f2 = Path(REPO) / "user_memory" / "also-fake.md"
    check("UPDATE on missing slug creates fallback file", f2.is_file())
    check("fallback file has correct fact", "Fallback fact" in f2.read_text())


def test_used_count_accuracy():
    """Test that 'used,' counting only counts Access log lines, not text in fact/episode."""
    print("\n--- TEST 4c: used-count accuracy ---")

    # Create a memory with 'used,' appearing in the fact text (should NOT be counted)
    mem_dir = Path(REPO) / "user_memory"
    tricky = mem_dir / "tricky-count.md"
    tricky.write_text(
        "# Tricky Count\n\n"
        "## Fact\nHe used, this tool and used, it well.\n\n"
        "## Episode\nSession used, yesterday\n\n"
        "## Access log\nused, 29.08.26\nused, 30.08.26\nused, 31.08.26\n"
    )
    count = mh._count_used_stamps(tricky.read_text())
    check("used-count ignores fact/episode text", count == 3, f"got {count}, expected 3")


def test_use_count_ranking():
    """Test that ranking sorts by use count, not mod time."""
    print("\n--- TEST 5: use_count_ranking ---")

    import time

    # Create memories with different use counts
    # "rare" — 1 use
    mh.write_memory_file(REPO, "user_memory", "rare-fact", "You mentioned something once.", "Session 25.08.26")
    time.sleep(0.05)

    # "common" — 5 uses
    mh.write_memory_file(REPO, "user_memory", "common-fact", "You always do this.", "Session 20.08.26")
    for _ in range(4):
        f = Path(REPO) / "user_memory" / "common-fact.md"
        f.write_text(f.read_text() + "\nused, 30.08.26")
    time.sleep(0.05)

    # "medium" — 3 uses (most recently modified!)
    mh.write_memory_file(REPO, "user_memory", "medium-fact", "You do this sometimes.", "Session 28.08.26")
    for _ in range(2):
        f = Path(REPO) / "user_memory" / "medium-fact.md"
        f.write_text(f.read_text() + "\nused, 30.08.26")

    mh.rebuild_index(REPO, "user_memory")
    index = (Path(REPO) / "user_memory.md").read_text()

    # Find positions of our three test facts among ALL numbered lines
    numbered_lines = [l for l in index.split("\n") if l and l[0].isdigit() and "." in l[:4]]
    common_pos = next((i for i, l in enumerate(numbered_lines) if "common-fact" in l), -1)
    medium_pos = next((i for i, l in enumerate(numbered_lines) if "medium-fact" in l), -1)
    rare_pos = next((i for i, l in enumerate(numbered_lines) if "rare-fact" in l), -1)

    check("common-fact ranked highest (5 uses)", common_pos != -1 and common_pos < medium_pos,
          f"positions: common={common_pos}, medium={medium_pos}, rare={rare_pos}")
    check("medium-fact ranked middle (3 uses)", medium_pos != -1 and rare_pos != -1 and medium_pos < rare_pos,
          f"positions: common={common_pos}, medium={medium_pos}, rare={rare_pos}")
    check("index shows use counts", "(used 5x)" in index, f"index:\n{index[:300]}")


def test_rebuild_index_contradicted():
    """Test that contradicted memories show the warning in the index."""
    print("\n--- TEST 6: rebuild_index with contradicted ---")

    mh.rebuild_index(REPO, "user_memory")
    index = (Path(REPO) / "user_memory.md").read_text()
    check("contradicted flag in index", "CONTRADICTED" in index, f"index:\n{index[:500]}")


def test_git_commit():
    """Test that git_commit creates a real commit."""
    print("\n--- TEST 7: git_commit ---")

    # Get current commit count
    result = subprocess.run(["git", "log", "--oneline"], cwd=REPO, capture_output=True, text=True)
    before = len(result.stdout.strip().splitlines())

    mh.git_commit(REPO)

    result = subprocess.run(["git", "log", "--oneline"], cwd=REPO, capture_output=True, text=True)
    after = len(result.stdout.strip().splitlines())
    check("new commit created", after == before + 1, f"before={before}, after={after}")

    last_msg = result.stdout.strip().splitlines()[0]
    check("commit message has checkpoint", "checkpoint" in last_msg, f"msg: {last_msg}")

    # Run again with no changes — should not create empty commit
    mh.git_commit(REPO)
    result = subprocess.run(["git", "log", "--oneline"], cwd=REPO, capture_output=True, text=True)
    after2 = len(result.stdout.strip().splitlines())
    check("no empty commit when nothing changed", after2 == after, f"after={after}, after2={after2}")


def test_parse_json():
    """Test JSON parsing with various LLM quirks."""
    print("\n--- TEST 8: parse_json edge cases ---")

    # Clean JSON
    check("clean JSON", mh.parse_json('{"facts": []}') == {"facts": []})

    # JSON with preamble
    check("JSON with preamble", mh.parse_json('Here is the result:\n{"facts": []}') == {"facts": []})

    # JSON with thinking tags
    check("JSON with <think> tags", mh.parse_json('<think>hmm let me think</think>{"facts": []}') == {"facts": []})

    # Empty / garbage
    check("empty string returns {}", mh.parse_json("") == {})
    check("garbage returns {}", mh.parse_json("no json here") == {})
    check("no closing brace returns {}", mh.parse_json('{"facts": [') == {})


def test_read_transcript_jsonl():
    """Test JSONL transcript reading."""
    print("\n--- TEST 9: read_transcript (JSONL) ---")

    tmp = Path(REPO) / "test_transcript.jsonl"
    messages = [
        {"message": {"role": "user", "content": "Hello my name is Alex"}},
        {"message": {"role": "assistant", "content": "Nice to meet you!"}},
        {"message": {"role": "user", "content": [{"type": "text", "text": "I like coffee"}]}},
        {"message": {"role": "system", "content": "system msg should be skipped"}},
        {"message": {"role": "user", "content": "<command-message>clear</command-message>"}},
    ]
    with open(tmp, "w") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")

    result = mh.read_transcript(str(tmp))
    check("user messages included", "USER: Hello my name is Alex" in result)
    check("assistant messages included", "ASSISTANT: Nice to meet you!" in result)
    check("list content parsed", "USER: I like coffee" in result)
    check("system messages excluded", "system msg" not in result)
    check("command messages excluded", "command-message" not in result)


def test_read_transcript_plain():
    """Test that read_transcript handles non-JSONL gracefully."""
    print("\n--- TEST 10: read_transcript (plain text / empty) ---")

    # Non-existent file
    check("non-existent file returns empty", mh.read_transcript("/tmp/does_not_exist_xyz.jsonl") == "")

    # File with garbage lines
    tmp = Path(REPO) / "garbage.jsonl"
    tmp.write_text("this is not json\nalso not json\n")
    check("garbage file returns empty", mh.read_transcript(str(tmp)) == "")


def test_audn_replace():
    """Test that AUDN_PROMPT uses __EXISTING__ and __NEW_FACTS__ placeholders correctly."""
    print("\n--- TEST 11: AUDN_PROMPT placeholder substitution ---")

    existing = "- **likes-pizza**: You love pizza"
    new_facts = '[{"slug": "likes-beer", "fact": "You like beer"}]'

    prompt = mh.AUDN_PROMPT.replace("__EXISTING__", existing).replace("__NEW_FACTS__", new_facts)
    check("__EXISTING__ replaced", existing in prompt)
    check("__NEW_FACTS__ replaced", new_facts in prompt)
    check("no __EXISTING__ placeholder left", "__EXISTING__" not in prompt)
    check("no __NEW_FACTS__ placeholder left", "__NEW_FACTS__" not in prompt)
    check("no Python .format() braces", "{existing}" not in prompt)


def test_full_consolidation():
    """Test full consolidation with live LLM (requires MEMORY_LLM_PROVIDER + creds)."""
    print("\n--- TEST 12: full consolidation (live API) ---")

    provider = os.environ.get("MEMORY_LLM_PROVIDER", "")
    if not provider:
        print("  ⏭️  SKIPPED — no MEMORY_LLM_PROVIDER set")
        return
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("  ⏭️  SKIPPED — MEMORY_LLM_PROVIDER=openai but no OPENAI_API_KEY")
        return

    # Clean the memory dirs for a fresh run
    for d in ["user_memory", "character_memory"]:
        mem_dir = Path(REPO) / d
        for f in mem_dir.glob("*.md"):
            f.unlink()

    # Transcript with clear durable facts AND transient garbage
    transcript = "\n".join([
        "USER: My name is Alex and I'm a barista in London.",
        "ASSISTANT: Welcome Alex! What can I help you with?",
        "USER: I'm really into rock climbing, been doing it for 5 years.",
        "ASSISTANT: That's awesome! Any favorite climbing spots?",
        "USER: I also play guitar in a band called The Drifters.",
        "ASSISTANT: Cool! What genre?",
        "USER: Blues rock mostly. Can you show me things as bullet points?",
        "ASSISTANT: Sure thing! Here are some ideas:\n- Idea 1\n- Idea 2",
        # Transient stuff that should NOT be extracted:
        "USER: What's on the menu today?",
        "ASSISTANT: I'm offering you a special deal on our premium package.",
        "USER: Hmm let me think about that.",
        "ASSISTANT: Take your time! I'll be right here.",
    ])

    mh.consolidate(REPO, transcript)

    # Check user memories
    user_files = list((Path(REPO) / "user_memory").glob("*.md"))
    check("user memories created", len(user_files) > 0, f"got {len(user_files)} files")

    all_content = " ".join(f.read_text() for f in user_files).lower()
    check("extracted: name is Alex", "alex" in all_content)
    check("extracted: barista", "barista" in all_content)
    check("extracted: rock climbing", "climbing" in all_content or "climb" in all_content)
    check("extracted: guitar/band", "guitar" in all_content or "drifters" in all_content or "band" in all_content)

    # Check transient stuff was filtered
    check("NOT extracted: 'menu'", "menu" not in all_content)
    check("NOT extracted: 'offering you'", "offering you" not in all_content)
    check("NOT extracted: 'let me think'", "let me think" not in all_content)

    # Check character memories
    char_files = list((Path(REPO) / "character_memory").glob("*.md"))
    check("character memories created", len(char_files) > 0, f"got {len(char_files)} files")

    # Check index files show use counts
    user_index = (Path(REPO) / "user_memory.md").read_text()
    check("user index has use counts", "used" in user_index and "x)" in user_index)

    # Check git committed
    result = subprocess.run(["git", "log", "--oneline"], cwd=REPO, capture_output=True, text=True)
    check("consolidation created git commit", "checkpoint" in result.stdout)

    # --- Second consolidation: test AUDN UPDATE/NONE ---
    print("\n  --- ROUND 2: AUDN UPDATE/NONE ---")
    transcript2 = "\n".join([
        "USER: Actually I switched jobs, I'm a software engineer now, not a barista anymore.",
        "ASSISTANT: Congrats on the career change!",
        "USER: Yeah, and I still love rock climbing, hit the gym 3 times a week for it.",
        "ASSISTANT: That's dedication!",
    ])

    mh.consolidate(REPO, transcript2)

    all_content2 = " ".join(f.read_text() for f in (Path(REPO) / "user_memory").glob("*.md")).lower()
    check("AUDN: job updated to software engineer", "software engineer" in all_content2)
    # Climbing should still be there (NONE or UPDATE)
    check("AUDN: climbing still present", "climb" in all_content2)

    # Check multiple git commits
    result = subprocess.run(["git", "log", "--oneline"], cwd=REPO, capture_output=True, text=True)
    checkpoints = [l for l in result.stdout.strip().splitlines() if "checkpoint" in l]
    check("two checkpoint commits exist", len(checkpoints) >= 2, f"got {len(checkpoints)}: {checkpoints}")


def test_init_idempotent():
    """Test that init_repo doesn't destroy existing data."""
    print("\n--- TEST 13: init_repo idempotent ---")

    # Write a memory, then re-init
    mh.write_memory_file(REPO, "user_memory", "survives-reinit", "This should survive.", "Test")
    mh.init_repo(REPO, "different_user")

    f = Path(REPO) / "user_memory" / "survives-reinit.md"
    check("memory survives re-init", f.is_file())
    check("content intact", "This should survive." in f.read_text())


def main():
    setup()
    try:
        test_init_repo()
        test_write_memory_file()
        test_intra_batch_dedup()
        test_update_memory_file()
        test_mark_contradicted()
        test_missing_target_logging()
        test_used_count_accuracy()
        test_use_count_ranking()
        test_rebuild_index_contradicted()
        test_git_commit()
        test_parse_json()
        test_read_transcript_jsonl()
        test_read_transcript_plain()
        test_audn_replace()
        test_full_consolidation()
        test_init_idempotent()

        print(f"\n{'='*60}")
        print(f"  RESULTS: {PASS} passed, {FAIL} failed")
        print(f"{'='*60}\n")

        sys.exit(1 if FAIL > 0 else 0)
    finally:
        teardown()


if __name__ == "__main__":
    main()

"""Scale check: 28k memories, written incrementally, the pattern that shredded Chroma.

Chroma's HNSW index went 233 KB -> 399 GB apparent / 122 GB real in twenty minutes
under exactly this workload (small incremental adds from a Stop hook). This measures
what the FTS5 index does instead.
"""

import random
import shutil
import subprocess
import time
from pathlib import Path

from gitmem import INDEX_FILE, Memory

TARGET = 28_000
OUT = Path("/tmp/gitmem-bench")

WORDS = """chroma index hnsw link_lists disk full serverless runpod idle timeout
render endpoint flux lora piper whisper telegram bridge gateway ampere cloud
litellm openrouter nemotron deepseek kimi token batching vram offload quantized
memory palace drawer wing room consolidation embedding sqlite fts5 commit repo
agent assistant worker scheduler broker opencode tailscale webhook subscription margin""".split()


def sentence(rng):
    return " ".join(rng.sample(WORDS, rng.randint(8, 18)))


def du(path):
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True).stdout
    return int(out.split()[0])


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    rng = random.Random(42)
    mem = Memory(OUT, create=True)

    print(f"writing {TARGET:,} memories one at a time (the Stop-hook pattern)...")
    start = time.time()
    for i in range(TARGET):
        mem.write("USER", f"fact-{i:05d}", sentence(rng), episode=sentence(rng))
        if (i + 1) % 7000 == 0:
            index = (OUT / INDEX_FILE).stat().st_size
            print(f"  {i + 1:>6,} written  {time.time() - start:>6.1f}s"
                  f"  index {index / 1e6:>7.1f} MB")
    elapsed = time.time() - start

    index_size = (OUT / INDEX_FILE).stat().st_size
    files_size = du(OUT / "USER_memory")
    print(f"\n{TARGET:,} memories in {elapsed:.1f}s ({TARGET / elapsed:.0f}/s)")
    print(f"  markdown files : {files_size / 1e6:8.1f} MB  (the truth, git-tracked)")
    print(f"  FTS5 index     : {index_size / 1e6:8.1f} MB  (derived, gitignored)")
    print(f"  index overhead : {index_size / files_size:8.2f}x of the source text")

    print("\nsearch latency at full scale:")
    queries = ["chroma index disk full", "serverless idle timeout billing",
               "telegram bridge oracle", "flux lora render endpoint", "sqlite commit repo"]
    worst = 0.0
    for query in queries:
        start = time.time()
        hits = mem.search(query, "USER", n=5)
        ms = (time.time() - start) * 1000
        worst = max(worst, ms)
        print(f"  {ms:6.1f} ms  {len(hits)} hits  {query!r}")
    # Guard the CROSS JOIN fix: without it this was 47,000 ms and only showed at scale.
    assert worst < 1000, f"search regressed to {worst:.0f} ms — check the join order"

    start = time.time()
    rebuilt = mem.reindex()
    print(f"\nfull reindex from files: {rebuilt:,} memories in {time.time() - start:.1f}s")
    assert rebuilt == TARGET, rebuilt
    assert mem.search("chroma index disk full", "USER"), "search broken after reindex"

    start = time.time()
    mem.commit("bench")
    print(f"index files + one git commit: {time.time() - start:.1f}s")
    print(f"\ntotal repo on disk: {du(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

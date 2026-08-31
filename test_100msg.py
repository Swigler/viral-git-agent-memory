#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
100-message simulation: realistic conversation with consolidation at 30/60/90/bye.
Facts evolve and contradict across rounds to stress-test AUDN.
"""
import os
import sys
import time
from pathlib import Path

# Add parent to path so we can import memory_hook
sys.path.insert(0, str(Path(__file__).parent))
from memory_hook import init_repo, consolidate, rebuild_index

REPO = "/tmp/test_100msg"

# --- 100 messages split into 4 rounds of ~25 exchanges ---

ROUND_1 = """USER: Hey there! I'm Marcus, I work as a DevOps engineer at a fintech startup in Berlin.
ASSISTANT: Hi Marcus! Nice to meet you. DevOps in fintech — that's a fast-paced world.
USER: Yeah it's intense. I mostly work with Kubernetes and Terraform.
ASSISTANT: Solid stack. Are you deploying on AWS or GCP?
USER: AWS mainly. We're a small team, just 5 engineers.
ASSISTANT: That's lean! You must wear a lot of hats.
USER: Totally. I also handle the CI/CD pipelines. We use GitHub Actions.
ASSISTANT: GitHub Actions is great for smaller teams. Clean YAML, good marketplace.
USER: My girlfriend Anna is a UX designer. She works at the same company actually.
ASSISTANT: That's cool — working together! Does it ever get tricky?
USER: Sometimes haha. But we keep it professional. We got a dog last month, a golden retriever named Biscuit.
ASSISTANT: Biscuit! Great name. Golden retrievers are the best.
USER: I'm learning Rust on the side. Want to contribute to some open source projects.
ASSISTANT: Rust is a great choice — especially with your systems background.
USER: Can you keep responses short? I hate walls of text.
ASSISTANT: Got it — short and punchy from now on.
USER: I drink way too much coffee. Like 6 cups a day.
ASSISTANT: That's a lot! Any particular kind?
USER: Just black filter coffee. Nothing fancy.
ASSISTANT: Simple and effective. The engineer's fuel.
USER: We're planning to move to Amsterdam next year. Anna got a job offer there.
ASSISTANT: Amsterdam is amazing for tech. Great quality of life too.
USER: I play bass guitar in a jazz trio on weekends.
ASSISTANT: That's awesome — jazz bass is so groovy. Any gigs?
USER: Small bar gigs mostly. Nothing big. It's just for fun.
ASSISTANT: The best kind of music — no pressure, pure joy.
USER: Oh and I'm vegetarian. Have been for about 2 years now.
ASSISTANT: Nice! Berlin's great for vegetarian food. Amsterdam will be too."""

ROUND_2 = """USER: Hey, quick update — I switched from AWS to GCP at work. Big migration project.
ASSISTANT: That's a major shift! How's the migration going?
USER: Painful but worth it. GKE is way better than EKS for our use case.
ASSISTANT: GKE does have smoother Kubernetes integration. Smart move.
USER: Also we hired 3 more engineers. Team is 8 now.
ASSISTANT: Growing fast! That changes the dynamics a lot.
USER: Yeah, I'm now more of a team lead than hands-on. Less coding, more meetings.
ASSISTANT: The classic engineering management trap. Do you like it?
USER: Honestly? I miss the code. But someone has to do it.
ASSISTANT: Maybe you can carve out some focused coding time each week.
USER: That's what I try. Fridays are my no-meeting days.
ASSISTANT: Smart — protect that time fiercely.
USER: Biscuit learned to fetch! Took 3 months but he got it.
ASSISTANT: That's hilarious — 3 months for fetch? He's a thinker.
USER: He's the dumbest smart dog I know. Anna spoils him rotten.
ASSISTANT: As all golden retrievers deserve to be spoiled.
USER: I dropped Rust actually. Picked up Go instead. More practical for work.
ASSISTANT: Go makes sense for DevOps/infra tooling. Very pragmatic.
USER: Built an internal CLI tool in Go already. The team loves it.
ASSISTANT: Nothing beats shipping something your team actually uses.
USER: Coffee update — I'm down to 3 cups. Doctor's orders.
ASSISTANT: Good call. Your heart thanks you.
USER: Still black filter though. Can't change that.
ASSISTANT: Some things are sacred.
USER: Amsterdam move is confirmed for March. Found an apartment in Jordaan.
ASSISTANT: Jordaan is beautiful — canals, cafes, great neighborhood.
USER: Can you use more bullet points? When there's multiple things to cover.
ASSISTANT: • Absolutely
• Will do
• Like this"""

ROUND_3 = """USER: We're in Amsterdam now! Moved last month.
ASSISTANT: Congrats on the move! How's settling in?
USER: Chaotic but good. Biscuit loves the parks here.
ASSISTANT: Amsterdam has amazing parks for dogs. Vondelpark?
USER: Yeah Vondelpark is our daily spot. Anna started her new job too.
ASSISTANT: How's she liking it?
USER: She loves it. Better work-life balance than Berlin.
ASSISTANT: Dutch companies are known for that. Good for both of you.
USER: Big news — I left the startup. Joined a bigger company, ING Bank.
ASSISTANT: ING! That's a serious move. Still DevOps?
USER: Platform engineering actually. Building internal developer platforms.
ASSISTANT: Platform engineering is the evolution of DevOps. Makes sense.
USER: Team of 20 here. Very different from a 8-person startup.
ASSISTANT: Huge difference in how you operate. More process, more stakeholders.
USER: Exactly. But the compensation is way better. And stock options.
ASSISTANT: Can't argue with financial security, especially in a new city.
USER: I've been getting into cycling. Amsterdam thing I guess.
ASSISTANT: You basically have to cycle in Amsterdam. It's the law.
USER: Got a nice Dutch bike. 30 min commute each way.
ASSISTANT: That's perfect — exercise built into your day.
USER: Oh — Anna and I got engaged! Last week, on a canal boat.
ASSISTANT: Congratulations!! A canal boat proposal — that's incredibly romantic.
USER: Thanks! Wedding probably next summer. Small ceremony.
ASSISTANT: Intimate weddings are the best. Less stress, more meaning.
USER: I quit coffee entirely actually. Switched to green tea.
ASSISTANT: That's a big change from 6 cups of black coffee! How do you feel?
USER: Way better honestly. Less jittery, better sleep.
ASSISTANT: Green tea has just enough caffeine without the crash. Good switch."""

ROUND_4 = """USER: Wedding is booked! August next year, small garden venue outside Amsterdam.
ASSISTANT: That sounds perfect — summer garden wedding. How many guests?
USER: About 40 people. Close friends and family only.
ASSISTANT: Intimate and beautiful. Anna must be excited.
USER: She's in full planning mode. I just nod and agree haha.
ASSISTANT: The wisest strategy known to engaged men.
USER: Work update — I got promoted to Staff Engineer at ING.
ASSISTANT: Staff Engineer! That's a huge deal. Well deserved.
USER: Thanks. More architecture and strategy now. I actually like it more than I expected.
ASSISTANT: Sometimes stepping back from the code gives you a better view of the whole system.
USER: True. I'm mentoring 3 junior engineers too. That's rewarding.
ASSISTANT: Mentoring is one of the best parts of seniority. Multiplying yourself.
USER: Biscuit is 2 now. Still dumb, still lovable. Anna taught him to shake hands.
ASSISTANT: From 3 months to learn fetch to shaking hands — growth!
USER: We're thinking about getting a second dog. A rescue this time.
ASSISTANT: That would be amazing. Biscuit would love a companion.
USER: Started playing piano instead of bass. Jazz piano specifically.
ASSISTANT: Piano! That's a switch from bass. What inspired that?
USER: Found an old upright in our apartment building's common room. Got hooked.
ASSISTANT: Nothing like stumbling into a new instrument. Jazz piano is endlessly deep.
USER: I'm actually vegan now. Was vegetarian, made the full switch 6 months ago.
ASSISTANT: From vegetarian to vegan — a natural progression. Amsterdam makes that easy.
USER: The food scene here is incredible for vegans. So many options.
ASSISTANT: Dutch vegan cheese is surprisingly good too.
USER: Can you occasionally throw in some humor? I like dry wit.
ASSISTANT: I'll try not to be too *gouda* about it.
USER: Haha perfect. That's exactly the vibe.
ASSISTANT: I'll keep the cheese jokes coming. They're pretty *brie-lliant*."""


def run_test():
    import shutil
    # Clean start
    if Path(REPO).exists():
        shutil.rmtree(REPO)

    init_repo(REPO, "marcus_42")
    # Init git
    import subprocess
    env = {**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@test",
           "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@test"}
    subprocess.run(["git", "init"], cwd=REPO, capture_output=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=REPO, capture_output=True, env=env)
    subprocess.run(["git", "commit", "-m", "init"], cwd=REPO, capture_output=True, env=env)

    rounds = [
        ("Round 1 (msg 1-30): Initial facts", ROUND_1),
        ("Round 2 (msg 31-60): Facts evolve — AWS→GCP, Rust→Go, team grows, coffee drops", ROUND_2),
        ("Round 3 (msg 61-90): Big changes — new city, new job, engaged, quit coffee", ROUND_3),
        ("Round 4 (msg 91-100 + bye): More evolution — promoted, vegan, piano, wedding", ROUND_4),
    ]

    for label, transcript in rounds:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        t0 = time.time()
        consolidate(REPO, transcript)
        elapsed = time.time() - t0
        print(f"  ⏱️  {elapsed:.1f}s")

        # Show current state
        for mem_type in ["user_memory", "character_memory"]:
            mem_dir = Path(REPO) / mem_type
            files = sorted(mem_dir.glob("*.md"))
            print(f"\n  {mem_type}: {len(files)} files")
            for f in files:
                content = f.read_text()
                use_count = content.count("used,")
                contradicted = " ⚠️CONTRADICTED" if "## Contradicted" in content else ""
                print(f"    • {f.stem} (used {use_count}x){contradicted}")

    # Show git log
    print(f"\n{'='*60}")
    print("  Git log")
    print(f"{'='*60}")
    result = subprocess.run(["git", "log", "--oneline"], cwd=REPO, capture_output=True, text=True, env=env)
    for line in result.stdout.strip().split("\n"):
        print(f"    {line}")

    # Show final indexes
    for mem_type in ["user_memory", "character_memory"]:
        idx = Path(REPO) / f"{mem_type}.md"
        if idx.exists():
            print(f"\n{'='*60}")
            print(f"  {idx.name}")
            print(f"{'='*60}")
            print(idx.read_text())

    print(f"\n✅ Done — repo at {REPO}")


if __name__ == "__main__":
    run_test()

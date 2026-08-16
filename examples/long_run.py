"""Simulate a long agent run and measure what the engine actually saves.

Every number printed here is measured, not asserted. Run it:

    python examples/long_run.py
"""

from __future__ import annotations

import random
import shutil
import tempfile
from pathlib import Path

from ctxkernel import ContextEngine, budgets

random.seed(7)

# Hot files get re-read constantly (which is what supersession is for); the
# long tail is read once and never again (which it is not). A trace made only
# of hot files would flatter the result.
HOT = [f"src/{n}.py" for n in ("auth", "db", "pool", "routes", "models", "cache")]
COLD = [f"src/mod/{n:02d}_helper.py" for n in range(40)]


def pick() -> str:
    return random.choice(HOT) if random.random() < 0.45 else random.choice(COLD)


def source(path: str, rev: int = 0) -> str:
    body = "\n\n".join(
        f"def {path.split('/')[-1][:-3]}_op_{i}(conn, payload):\n"
        f"    # revision {rev}\n"
        f"    validated = validate(payload)\n"
        f"    return conn.execute(validated)"
        for i in range(60)
    )
    return f"import redis\nimport jwt\nfrom .core import validate\n\n{body}\n"


def install_log() -> str:
    return "\n".join(
        f"npm WARN deprecated pkg-{i}@1.0.{i}: no longer maintained" if i % 40 == 0
        else f"added package-{i}@2.{i}.0"
        for i in range(1200)
    )


def test_output(failing: int) -> str:
    lines = [f"tests/test_mod{i}.py::test_case_{i} PASSED" for i in range(180)]
    lines += [f"FAILED tests/test_auth.py::test_refresh_race_{i}" for i in range(failing)]
    lines.append(f"=== {failing} failed, 180 passed in 6.20s ===")
    return "\n".join(lines)


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="ctx-demo-"))
    engine = ContextEngine(root=root, tenant="demo", session="long-run", budget=budgets.claude())

    engine.set_goal(
        "Fix the flaky token-refresh test without weakening the rate limiter.",
        ["pytest tests/test_auth.py -x", "no test may sleep() to pass"],
    )

    step = 0

    def call(name: str, args: dict, result: str) -> None:
        nonlocal step
        step += 1
        tid = f"t{step}"
        engine.record_tool_call(tid, name, args)
        engine.record_tool_result(tid, result)

    # Turn 1-12: orientation. The same files get read repeatedly, which is what
    # agents actually do and what dedup/supersession is aimed at.
    call("bash", {"command": "npm install"}, install_log())
    for _ in range(12):
        f = pick()
        call("read_file", {"path": f}, source(f))

    # A closed investigation: its trace becomes compactable, its outcome does not.
    with engine.task("check whether the connection pool is the cause") as t:
        for _ in range(6):
            call("read_file", {"path": "src/pool.py"}, source("src/pool.py"))
        call("bash", {"command": "pytest tests/test_pool.py"}, test_output(0))
        t.done("Not the pool — it holds fine at 200 concurrent connections.")

    # Two dead ends. A few hundred tokens that stop the same wall being hit
    # again at step 200 and step 350.
    engine.note_failure(
        "guard refresh with a mutex",
        "deadlocks when the refresh itself triggers a re-auth",
        "the re-auth path becomes non-reentrant",
    )
    engine.note_failure(
        "cache the token for 30s",
        "test still flakes — the race is in the write, not the read",
    )
    engine.note_invariant(
        "the /refresh endpoint rate-limits at 10 req/min",
        valid_unless="infra/ratelimit.tf changes",
    )

    # Turns 20-140: the long middle. Edits invalidate earlier reads.
    for i in range(60):
        f = pick()
        call("read_file", {"path": f}, source(f, rev=i))
        if i % 7 == 0:
            call("write_file", {"path": f}, "ok, 40 lines written")
        if i % 11 == 0:
            call("bash", {"command": "pytest tests/test_auth.py"}, test_output(3 - i // 30))

    engine.record_user("also make sure the fix doesn't slow down the login path")

    # ------------------------------------------------------------------
    r = engine.report()
    a = engine.assemble()

    print(f"\n  session {r['session']} — {r['events']} events, {step} tool calls\n")
    print("  WITHOUT the engine (plain transcript)")
    print(f"    {r['raw_tokens']:>9,} tokens  →  {r['raw_tokens'] / 200_000:.0%} of a 200k window")
    print("\n  WITH the engine")
    print(f"    {r['assembled_tokens']:>9,} tokens  →  {r['assembled_tokens'] / 200_000:.0%} of the window")
    print(f"    {r['saved_pct']:>9}% saved   ({r['saved_tokens']:,} tokens)")
    print(f"    {r['cache_prefix_tokens']:>9,} tokens in the cacheable prefix")

    print("\n  zones")
    for name, tokens in r["zones"].items():
        bar = "█" * max(0, round(tokens / max(1, r["assembled_tokens"]) * 40))
        print(f"    {name:<11}{tokens:>7,}  {bar}")

    print("\n  why events were dropped")
    for reason, n in sorted(r["decisions"].items(), key=lambda kv: -kv[1]):
        if n:
            print(f"    {reason:<14}{n:>5}")

    # The load-bearing detail the funnel legitimately paged out — still findable.
    hits = engine.search_log("deprecated")
    print(f"\n  search_log('deprecated') → {len(hits)} hit(s) in evicted material")
    if hits:
        print(f"    e{hits[0]['seq']:06d}  {hits[0]['snippet'][:88]}…")

    print(
        "\n  Synthetic trace: the mix of hot and cold reads is a guess, so treat"
        "\n  the percentage as a shape, not a measurement. Real numbers come from"
        "\n  replaying real traces."
    )

    for note in a.notes:
        print(f"\n  ! {note}")

    engine.close()
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()

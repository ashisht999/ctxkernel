"""Scratch script — run the engine on this repo's own source and watch it work.

    .venv/bin/python try_it.py

Every number below is measured on real files, not synthetic ones.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ctxkernel import ContextEngine, budgets
from ctxkernel.adapters import anthropic, generic, openai

CTX = Path(".ctx-scratch")
REPO = Path("ctxkernel")


def rule(title: str) -> None:
    print(f"\n\033[1m{'─' * 84}\n  {title}\n{'─' * 84}\033[0m")


def main() -> None:
    shutil.rmtree(CTX, ignore_errors=True)
    eng = ContextEngine(root=CTX, session="scratch", budget=budgets.claude())

    rule("1. Goal — verbatim, pinned, never evicted")
    eng.set_goal(
        "Understand how the assembler decides what to keep.",
        ["can explain the four stages", "no test regressions"],
    )
    print(f"  context is now {eng.report()['assembled_tokens']} tokens")

    rule("2. A fat tool result — intercepted before it can occupy context")
    src = (REPO / "engine.py").read_text()
    eng.record_tool_call("t1", "read_file", {"path": "ctxkernel/engine.py"})
    handle = eng.record_tool_result("t1", src).blocks[0]
    print(f"  file      {len(src):,} chars ≈ {handle.original_tokens:,} tokens")
    print(f"  in context {len(handle.extract)} chars — a parsed outline:\n")
    for line in handle.extract.splitlines()[:9]:
        print(f"    {line}")
    print(f"\n  handle: {handle.handle_id}")

    rule("3. expand() — the dereference. Nothing was destroyed.")
    hid = handle.handle_id
    print("  expand(hid, symbol='record_model'):\n")
    for line in eng.expand(hid, symbol="record_model").splitlines()[:8]:
        print(f"    {line}")
    print(f"\n  expand(hid) round-trips exactly: {eng.expand(hid, max_chars=10**9) == src}")
    print(f"  expand(hid, grep='def assemble') → {eng.expand(hid, grep='def assemble').strip()[:64]}")

    rule("4. Read the same file 5 more times — watch context not move")
    before = eng.report()["assembled_tokens"]
    for i in range(2, 7):
        eng.record_tool_call(f"t{i}", "read_file", {"path": "ctxkernel/engine.py"})
        eng.record_tool_result(f"t{i}", src)
    after = eng.report()
    print(f"  before  {before:>6,} tok")
    print(f"  after   {after['assembled_tokens']:>6,} tok   (log is now {after['raw_tokens']:,})")
    print(f"  delta   {after['assembled_tokens'] - before:>6,}  ← supersession, no judgment involved")

    rule("5. More files, then a write — watch invalidation fire")
    for i, name in enumerate(["assembler.py", "predicates.py", "codecs.py", "store.py"], start=10):
        eng.record_tool_call(f"t{i}", "read_file", {"path": f"ctxkernel/{name}"})
        eng.record_tool_result(f"t{i}", (REPO / name).read_text())
    eng.record_tool_call("t20", "write_file", {"path": "ctxkernel/predicates.py"})
    eng.record_tool_result("t20", "wrote 8 lines")
    for reason, n in sorted(eng.report()["decisions"].items(), key=lambda kv: -kv[1]):
        if n:
            print(f"    {reason:<14}{n:>4}")

    rule("6. A closed investigation — outcome survives, trace does not")
    with eng.task("check whether codecs.py is the bottleneck") as t:
        eng.record_tool_call("t30", "read_file", {"path": "ctxkernel/codecs.py"})
        eng.record_tool_result("t30", (REPO / "codecs.py").read_text())
        t.done("Not codecs — the registry short-circuits on first match.")
    eng.note_failure("cache extracts in memory", "digests already dedupe, so it never hits")
    eng.note_invariant("extracts stay under ~250 tokens", valid_unless="MAX_EXTRACT_CHARS changes")
    print("  recorded.")

    rule("7. What the model actually sees")
    print(eng.preview())

    rule("8. Why each event survived or didn't")
    print(eng.explain())

    rule("9. search_log() — evicted material is still findable")
    for q in ("Tier A", "tool_use and its", "MAX_EXTRACT_CHARS"):
        found = eng.search_log(q, limit=3)
        print(f"  search_log({q!r}) → {len(found)} hit(s)")
        for h in found[:1]:
            print(f"      e{h['seq']:06d} [{h['kind']}] {h['snippet'][:62]}…")

    rule("10. The same context, rendered for three different providers")
    items = eng.assemble().items
    print(f"  anthropic  {len(anthropic.to_messages(items)):>2} messages   (blocks, tool_result inside user turns)")
    print(f"  openai     {len(openai.to_messages(items)):>2} messages   (role='tool', arguments as JSON string)")
    print(f"  generic    {len(generic.to_chat(items)):>2} messages   plain dicts — any chat endpoint")
    print(f"             {len(generic.to_text(items)):>5,} chars      one prompt string — completion endpoints")

    rule("Result")
    r = eng.report()
    print(f"  events               {r['events']:>8,}")
    print(f"  log                  {r['raw_tokens']:>8,} tokens  ({r['raw_tokens'] / 200_000:.0%} of a 200k window)")
    print(f"  assembled context    {r['assembled_tokens']:>8,} tokens  ({r['assembled_tokens'] / 200_000:.0%})")
    print(f"  saved                {r['saved_pct']:>8}%")
    print(f"  cacheable prefix     {r['cache_prefix_tokens']:>8,} tokens")
    print("\n  This measures cost, not quality. Whether the agent makes better")
    print("  decisions on the right-hand column needs the replay harness.\n")

    eng.close()
    shutil.rmtree(CTX, ignore_errors=True)


if __name__ == "__main__":
    main()

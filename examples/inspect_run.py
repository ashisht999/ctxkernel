"""Watch the context get built, step by step.

    python examples/inspect_run.py            # growth curve + final context
    python examples/inspect_run.py --preview  # dump the context at each stage
    python examples/inspect_run.py --explain  # per-event keep/drop reasons

The thing to look for is that context size goes flat while the log keeps
growing. That gap is the whole product.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from ctxkernel import ContextEngine, budgets

def src_for(path: str) -> str:
    """Distinct content per file, so handles are distinct too."""
    return SRC.replace("TokenStore", "Store_" + path.replace("/", "_").replace(".", "_"))


SRC = """import redis
import jwt
from .core import validate

class TokenStore:
    def get(self, k): ...
    def put(self, k, v): ...

def refresh_token(user, ttl):
    return TokenStore().put(user, ttl)

def validate_jwt(token):
    return jwt.decode(token)
""" + "\n".join(f"def helper_{i}(a, b):\n    return a + b" for i in range(300))

TESTS = "\n".join(
    [f"tests/test_{i}.py::test_case PASSED" for i in range(150)]
    + ["FAILED tests/test_auth.py::test_refresh_race", "=== 1 failed, 150 passed ==="]
)


def main() -> None:
    show_preview = "--preview" in sys.argv
    show_explain = "--explain" in sys.argv

    root = Path(tempfile.mkdtemp(prefix="ctx-inspect-"))
    eng = ContextEngine(root=root, session="inspect", budget=budgets.claude())
    n = 0

    def call(tool: str, args: dict, result: str) -> None:
        nonlocal n
        n += 1
        eng.record_tool_call(f"t{n}", tool, args)
        eng.record_tool_result(f"t{n}", result)

    def stage(label: str) -> None:
        r = eng.report()
        print(
            f"  {label:<34} log {r['raw_tokens']:>8,} tok   "
            f"context {r['assembled_tokens']:>7,} tok   "
            f"{r['assembled_tokens'] / 200_000:>4.0%} of window"
        )
        if show_preview:
            print()
            print(eng.preview())
            print()

    print("\n  Log grows; context does not.\n")

    eng.set_goal("Fix the flaky token-refresh test.", ["pytest tests/test_auth.py -x"])
    stage("after the goal")

    call("read_file", {"path": "src/auth.py"}, src_for("src/auth.py"))
    stage("1 file read (40k tok of source)")

    for _ in range(5):
        call("read_file", {"path": "src/auth.py"}, src_for("src/auth.py"))
    stage("same file read 5 more times")

    for i in range(10):
        call("read_file", {"path": f"src/mod_{i}.py"}, src_for(f"src/mod_{i}.py"))
    stage("10 different files read")

    call("bash", {"command": "pytest tests/"}, TESTS)
    stage("a test run (150 passing lines)")

    with eng.task("investigate the connection pool") as t:
        for _ in range(4):
            call("read_file", {"path": "src/pool.py"}, src_for("src/pool.py"))
        t.done("Not the pool — it holds fine at 200 concurrent connections.")
    stage("a closed investigation")

    eng.note_failure("mutex on refresh", "deadlocks when refresh triggers re-auth")
    call("write_file", {"path": "src/auth.py"}, "wrote 12 lines")
    stage("a failure noted + a write")

    for i in range(40):
        call("read_file", {"path": f"src/mod_{i % 12}.py"}, src_for(f"src/mod_{i % 12}.py"))
    stage("40 more reads")

    print("\n" + "─" * 92)
    print("  THE ACTUAL CONTEXT AT THE END")
    print("─" * 92 + "\n")
    print(eng.preview())

    if show_explain:
        print("\n" + "─" * 92)
        print("  PER-EVENT DECISIONS")
        print("─" * 92 + "\n")
        print(eng.explain())

    print("\n  Try it yourself:")
    print("    eng.expand('<handle_id>')                     # page the full file back")
    print("    eng.expand('<handle_id>', symbol='refresh_token')")
    print("    eng.search_log('deprecated')                  # grep evicted material\n")

    eng.close()
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()

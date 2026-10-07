"""A Claude agent whose context is chosen by Jev over the graph.

    pip install anthropic "ctxkernel[jev]"
    export ANTHROPIC_API_KEY=...  TYPESAFE_API_KEY=...
    python examples/jev_agent.py "make the failing test in tests/test_app.py pass"

Three things differ from a plain agent loop:

1. The engine, not the loop, owns the message list. Every call and result is
   recorded; each model call gets a bounded context assembled from the log.
2. Jev ranks what goes into that context, in the background after each tool
   result, over candidates the graph reaches by links and shared names. It
   also flags failed steps, which become pinned failure notes.
3. The agent gets ``ctx_outline`` / ``ctx_search`` / ``ctx_expand``, so
   anything the ranking left out costs it one tool call to pull back.
"""

from __future__ import annotations

import subprocess
import sys

import anthropic

from ctxkernel import ContextEngine
from ctxkernel.adapters.anthropic import parse_content
from ctxkernel.adapters.decision import JevDecision

MODEL = "claude-opus-5"

HOST_TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command in the repository and return its output.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the repository.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
]


def run_host_tool(name: str, args: dict) -> tuple[str, bool]:
    if name == "bash":
        p = subprocess.run(args["command"], shell=True, capture_output=True, text=True, timeout=120)
        return (p.stdout + p.stderr) or f"(exit {p.returncode})", p.returncode != 0
    if name == "read_file":
        try:
            with open(args["path"], encoding="utf-8") as fh:
                return fh.read(), False
        except OSError as exc:
            return str(exc), True
    return f"unknown tool: {name}", True


def main(goal: str, max_steps: int = 40) -> None:
    client = anthropic.Anthropic()
    eng = ContextEngine(decision=JevDecision())  # ranked defaults come with the model
    eng.set_goal(goal)
    tools = HOST_TOOLS + eng.tools("anthropic")

    for _ in range(max_steps):
        response = client.messages.create(
            model=MODEL, max_tokens=16000, tools=tools, messages=eng.messages("anthropic")
        )
        eng.record_model(parse_content(response.content))

        uses = [b for b in response.content if b.type == "tool_use"]
        if not uses:
            print("".join(b.text for b in response.content if b.type == "text"))
            break
        for use in uses:
            out = eng.run_tool(use.name, use.input)  # history tools first...
            is_error = False
            if out is None:  # ...then the host's own
                out, is_error = run_host_tool(use.name, use.input)
            eng.record_tool_result(use.id, out, is_error=is_error)

    r = eng.report()
    print(f"\n{r['events']} events · {r['raw_tokens']:,} raw → {r['assembled_tokens']:,} sent")
    eng.close()


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or "list the files in this repository and summarize what it does")

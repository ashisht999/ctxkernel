"""The eval harness: does the assembled context still hold what the next
decision needed?

Token savings measure cost. This measures what cost is traded against, and does
it without a model in the loop, so it is free, exact and repeatable.

For every step where the agent acted, take the identifiers its action used --
paths, symbols, numbers, anything shaped like a name -- that it can only have
learned from earlier history, not from the user (the user's words are pinned by
every policy alike, so they cannot tell policies apart). Then ask whether the
context a policy would have sent at that moment still contains them::

    evidence recall = needed identifiers still visible / needed identifiers

A policy that dropped something the agent went on to use scores below 1 at
that step. A policy at 1 everywhere lost nothing the agent acted on.

It is a proxy, and says so. An action can depend on something that is not an
identifier (a conclusion, a count), and an identifier can be present without
having been the reason. It cannot tell you the agent would have acted the same
-- that needs a model predicting the next action from each context, which is
the second tier. What it does tell you, per step, is where information the
agent actually used was missing, which is the failure a context policy exists
to avoid.

    python -m ctxkernel.eval trace.jsonl --policy null,proximity
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .engine import ContextEngine
from .ir import Block, Elision, HandleRef, Text, Thinking, ToolResult, ToolUse
from .replay import read_jsonl

#: Builds a fresh engine for one policy, rooted in its own directory.
PolicyFactory = Callable[[Path], ContextEngine]

_TOKEN = re.compile(r"[A-Za-z0-9_./\-]{4,}")


def identifiers(text: str) -> set[str]:
    """Name-shaped tokens: containing ``/ . _``, a digit, or camelCase.

    Plain words are excluded on purpose. "python" or "error" appear everywhere,
    so they would be "visible" under any policy and only dilute the score.
    """
    out: set[str] = set()
    for raw in _TOKEN.findall(text):
        t = raw.strip("./-")
        if len(t) < 4:
            continue
        if (
            any(ch in t for ch in "/._")
            or any(ch.isdigit() for ch in t)
            or re.search(r"[a-z][A-Z]", t)
        ):
            out.add(t)
    return out


def render(blocks: Iterable[Block]) -> str:
    """What a model would read. A handle renders as its extract, not its blob:
    the blob is reachable, but it is not in the context."""
    parts: list[str] = []
    for b in blocks:
        if isinstance(b, (Text, Thinking)):
            parts.append(b.text)
        elif isinstance(b, ToolUse):
            parts.append(f"{b.name} {json.dumps(b.input, ensure_ascii=False)}")
        elif isinstance(b, ToolResult):
            parts.append(b.content)
        elif isinstance(b, HandleRef):
            parts.append(f"{b.handle_id} {b.extract}")
        elif isinstance(b, Elision):
            parts.append(b.note)
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Traces
# --------------------------------------------------------------------------


def load_claude_code(path: str | Path) -> list[dict[str, Any]]:
    """A Claude Code session log as Anthropic-format messages.

    The log writes one line per content block, so consecutive assistant lines
    sharing a message id are one turn and are merged back. Sub-agent turns
    (``isSidechain``) belong to a different conversation and are skipped.
    """
    msgs: list[dict[str, Any]] = []
    last_id: str | None = None
    for rec in read_jsonl(path):
        if rec.get("type") not in ("user", "assistant") or rec.get("isSidechain"):
            continue
        m = rec.get("message")
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            continue
        mid = m.get("id") if m.get("role") == "assistant" else None
        if mid and mid == last_id and msgs:
            msgs[-1]["content"].extend(content)
        else:
            msgs.append({"role": m.get("role", rec["type"]), "content": list(content)})
        last_id = mid
    return msgs


def load_openai(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI-format messages (``tool_calls`` / ``role: tool``) as Anthropic-format
    ones, which is what ``evaluate`` replays. The system prompt is dropped: it
    is the same for every policy and never an item a policy could lose."""
    out: list[dict[str, Any]] = []

    def text_of(content: Any) -> str:
        if isinstance(content, list):
            return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
        return content or ""

    for m in messages:
        role = m.get("role")
        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            if text_of(m.get("content")).strip():
                blocks.append({"type": "text", "text": text_of(m.get("content"))})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": fn.get("arguments")}
                blocks.append({"type": "tool_use", "id": tc["id"], "name": fn.get("name", ""), "input": args})
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            out.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""), "content": text_of(m.get("content"))}
            ]})
        elif role == "user" and text_of(m.get("content")).strip():
            out.append({"role": "user", "content": text_of(m.get("content"))})
    return out


def openhands_identity(engine: ContextEngine) -> None:
    """Teach the identity layer OpenHands' tools, so their calls get resources
    and the graph gets file links. The extension point a user of any agent
    framework would reach for."""
    from .identity import ToolSemantics
    from .ir import ResourceId

    def editor(_: str, a: dict[str, Any]) -> ToolSemantics | None:
        path = a.get("path")
        if not path:
            return None
        return ToolSemantics(ResourceId("file", path), is_write=a.get("command") != "view")

    def bash(_: str, a: dict[str, Any]) -> ToolSemantics | None:
        cmd = (a.get("command") or "").strip()
        return ToolSemantics(ResourceId("shell", cmd)) if cmd else None

    engine.identity.register("str_replace_editor", editor)
    engine.identity.register("execute_bash", bash)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@dataclass(slots=True)
class StepResult:
    step: int
    policy: str
    needed: int
    visible: int
    context_tokens: int
    raw_tokens: int
    #: A few of the identifiers that were needed and missing -- the part of the
    #: result a person can act on.
    missing: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.visible / self.needed if self.needed else 1.0


@dataclass(slots=True)
class PolicySummary:
    policy: str
    steps: int
    mean_recall: float
    perfect_pct: float
    mean_context_tokens: float
    mean_raw_tokens: float
    decision_runs: int
    mean_decision_ms: float


def evaluate(
    messages: Sequence[dict[str, Any]],
    policies: dict[str, PolicyFactory],
    *,
    workdir: str | Path | None = None,
    every: int = 1,
    max_steps: int | None = None,
) -> tuple[list[StepResult], list[PolicySummary]]:
    """Replay ``messages`` once per policy, scoring each action step."""
    from .adapters.anthropic import parse_content

    root = Path(workdir or tempfile.mkdtemp(prefix="ctxkernel-eval-"))
    engines = {name: make(root / name) for name, make in policies.items()}
    history: set[str] = set()
    pinned: set[str] = set()
    results: list[StepResult] = []
    action_steps = 0

    try:
        for i, msg in enumerate(messages):
            blocks = parse_content(msg.get("content"))
            uses = [b for b in blocks if isinstance(b, ToolUse)]

            if msg.get("role") == "assistant" and uses:
                action_steps += 1
                if max_steps is not None and action_steps > max_steps:
                    break
                needed = (identifiers(render(uses)) & history) - pinned
                if needed and (action_steps - 1) % every == 0:
                    for name, eng in engines.items():
                        eng.ingest_messages(list(messages[:i]))
                        eng.settle()
                        a = eng.assemble()
                        visible = needed & identifiers(render(a.blocks))
                        results.append(
                            StepResult(
                                step=action_steps,
                                policy=name,
                                needed=len(needed),
                                visible=len(visible),
                                context_tokens=a.tokens,
                                raw_tokens=a.raw_tokens,
                                missing=sorted(needed - visible)[:5],
                            )
                        )

            ids = identifiers(render(blocks))
            history |= ids
            if msg.get("role") == "user" and not any(isinstance(b, ToolResult) for b in blocks):
                pinned |= ids

        summaries = [_summarize(name, eng, results) for name, eng in engines.items()]
    finally:
        for eng in engines.values():
            eng.close()
    return results, summaries


def _summarize(name: str, eng: ContextEngine, results: list[StepResult]) -> PolicySummary:
    rows = [r for r in results if r.policy == name]
    runs = [r for r in eng.store.anchor_runs() if r["fallback"] != "no candidates"]
    return PolicySummary(
        policy=name,
        steps=len(rows),
        mean_recall=round(statistics.fmean(r.recall for r in rows), 4) if rows else 1.0,
        perfect_pct=round(100 * sum(r.recall == 1.0 for r in rows) / len(rows), 1) if rows else 100.0,
        mean_context_tokens=round(statistics.fmean(r.context_tokens for r in rows)) if rows else 0,
        mean_raw_tokens=round(statistics.fmean(r.raw_tokens for r in rows)) if rows else 0,
        decision_runs=len(runs),
        mean_decision_ms=round(statistics.fmean(r["ms"] for r in runs), 1) if runs else 0.0,
    )


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------


def policy(
    name: str,
    *,
    window: int = 200_000,
    tail_frac: float = 1.0,
    threshold: float = 0.0,
    candidates: int = 100,
    setup: Callable[[ContextEngine], None] | None = None,
) -> PolicyFactory:
    """The built-in policies by name.

    * ``null`` -- today's default: 75% of the target for the recent tail, the
      rest unused.
    * ``recency`` -- the fair baseline: the whole target spent on the recent
      tail. Any ranked policy has to beat this on the same tokens.
    * ``proximity`` -- ranked by the model-free prior (links, shared names,
      recency).
    * ``jev`` -- ranked by Jev; needs ``typesafe-sdk`` and ``TYPESAFE_API_KEY``,
      and sends node previews to TypeSafe.

    Ranked policies spend the *same* total target as ``recency``. By default
    (``tail_frac`` 1.0) they swap: recency fills the target and a confident
    ranking displaces the oldest recent units, so recency is the floor. A
    ``tail_frac`` below 1 with a model that does not swap gives the older split.
    """
    from dataclasses import replace

    from .budgets import window as budget_window
    from .decision import NullDecision, ProximityDecision

    def make(decision_factory: Callable[[], Any], budget: Any) -> PolicyFactory:
        def build(root: Path) -> ContextEngine:
            eng = ContextEngine(
                root=root, session="eval", budget=budget, decision=decision_factory(),
                anchor_threshold=threshold, anchor_k=candidates, anchor_candidates=candidates,
            )
            if setup is not None:
                setup(eng)
            return eng

        return build

    base = budget_window(window)
    ranked = replace(base, tail_frac=tail_frac)
    if name == "null":
        return make(NullDecision, base)
    if name == "recency":
        return make(NullDecision, replace(base, tail_frac=1.0))
    if name == "proximity":
        return make(ProximityDecision, ranked)
    if name == "jev":
        from .adapters.decision import JevDecision

        return make(JevDecision, ranked)
    raise ValueError(f"unknown policy: {name!r} (use null, proximity or jev)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _table(summaries: list[PolicySummary]) -> str:
    head = f"  {'policy':<10} {'steps':>5} {'recall':>7} {'perfect':>8} {'context':>9} {'raw':>9} {'runs':>5} {'ms/run':>7}"
    lines = [head, "  " + "─" * (len(head) - 2)]
    for s in summaries:
        lines.append(
            f"  {s.policy:<10} {s.steps:>5} {s.mean_recall:>7.3f} {s.perfect_pct:>7.1f}% "
            f"{s.mean_context_tokens:>9,.0f} {s.mean_raw_tokens:>9,.0f} {s.decision_runs:>5} {s.mean_decision_ms:>7.0f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ctxkernel.eval", description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+", help="Claude Code session logs (.jsonl)")
    ap.add_argument("--policy", default="recency,proximity", help="comma-separated: null,recency,proximity,jev")
    ap.add_argument("--window", type=int, default=200_000)
    ap.add_argument("--tail-frac", type=float, default=1.0, help="ranked policies: share of target for the recency-filled tail")
    ap.add_argument("--threshold", type=float, default=0.0, help="ranked policies: minimum score to be admitted")
    ap.add_argument("--candidates", type=int, default=100, help="ranked policies: candidates scored per run")
    ap.add_argument("--every", type=int, default=1, help="score every Nth action step")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", help="write per-step results and summaries as JSON")
    args = ap.parse_args(argv)

    names = [p.strip() for p in args.policy.split(",") if p.strip()]
    report: dict[str, Any] = {}
    for path in args.traces:
        msgs = load_claude_code(path)
        results, summaries = evaluate(
            msgs,
            {
                n: policy(n, window=args.window, tail_frac=args.tail_frac,
                          threshold=args.threshold, candidates=args.candidates)
                for n in names
            },
            every=args.every,
            max_steps=args.max_steps,
        )
        print(f"\n{Path(path).name} — {len(msgs)} messages")
        print(_table(summaries))
        report[str(path)] = {
            "summaries": [asdict(s) for s in summaries],
            "steps": [{**asdict(r), "recall": r.recall} for r in results],
        }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

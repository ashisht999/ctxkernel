"""Tools the agent itself calls to navigate its own history.

Ranking decides what enters the context, and ranking can miss. These tools are
what makes a miss cheap: an agent that cannot see something it needs asks for
it, and a wrong ranking costs one tool call instead of a wrong decision. They
also close the gap ``search_log`` alone left open -- an agent that never
thinks to search still loses -- because ``ctx_outline`` shows it what exists.

Specs here are provider-neutral (name, description, JSON schema); the
adapters render them into each vendor's tool format.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .graph import EdgeKind, NodeKind, event_text

if TYPE_CHECKING:
    from .engine import ContextEngine

#: Cap on any single tool answer, so a page-in cannot itself flood the window.
MAX_TOOL_CHARS = 8_000

SPECS: list[dict[str, Any]] = [
    {
        "name": "ctx_outline",
        "description": (
            "Table of contents of this session's full history, including work no "
            "longer shown in context: tasks and their outcomes, the files and "
            "commands touched (most-used first) and the id of each one's latest "
            "use. Pass any id to ctx_expand to read it in full."
        ),
        "schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "ctx_search",
        "description": (
            "Search this session's entire history -- every tool result in full, "
            "including anything no longer in context -- for a regular expression. "
            "Returns matching items with an id and a snippet; pass the id to "
            "ctx_expand for the whole item."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression, case-insensitive."},
                "limit": {"type": "integer", "description": "Maximum matches (default 20)."},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ctx_expand",
        "description": (
            "Full content of one item from this session's history, by id: an event "
            "id (e000123) from ctx_search or ctx_outline, a handle id from 'Available "
            "via expand', a resource id (res:file:src/app.py) for its latest content, "
            "or a task id for its steps. Optionally narrow with grep or lines."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "grep": {"type": "string", "description": "Only lines matching this regex."},
                "lines": {"type": "string", "description": "A line range, e.g. '40-80'."},
            },
            "required": ["id"],
            "additionalProperties": False,
        },
    },
]

NAMES = frozenset(s["name"] for s in SPECS)


def run(engine: ContextEngine, name: str, args: dict[str, Any]) -> str | None:
    """Answer one agent tool call, or ``None`` if ``name`` is not one of ours --
    so a host can try these first and fall through to its own tools."""
    if name not in NAMES:
        return None
    try:
        if name == "ctx_outline":
            return outline(engine)
        if name == "ctx_search":
            return search(engine, str(args["pattern"]), int(args.get("limit") or 20))
        return expand(engine, str(args["id"]), grep=args.get("grep"), lines=args.get("lines"))
    except Exception as exc:  # noqa: BLE001 -- a tool error is an answer, not a crash
        return f"[{name} failed: {type(exc).__name__}: {exc}]"


def outline(engine: ContextEngine, *, max_resources: int = 30) -> str:
    g = engine.graph
    tasks = [n for n in g.nodes.values() if n.kind is NodeKind.TASK]
    calls = [n for n in g.nodes.values() if n.kind is NodeKind.CALL]
    out = [f"session {engine.session_id}: {engine.store.count()} events, {len(calls)} tool calls"]

    goal = next((n for n in reversed(list(g.nodes.values())) if n.kind is NodeKind.GOAL), None)
    if goal:
        out.append(f"goal: {goal.label}")

    if tasks:
        closed = engine.store.closed_task_ids()
        out.append("tasks:")
        for t in sorted(tasks, key=lambda n: n.seqs[0]):
            mark = "✓" if t.id.removeprefix("task:") in closed else "▶"
            out.append(f"  {mark} {t.id}  {t.label}")

    resources = [n for n in g.nodes.values() if n.kind is NodeKind.RESOURCE]
    if resources:
        uses = {r.id: g.calls_touching(r.id) for r in resources}
        ranked = sorted(resources, key=lambda r: (len(uses[r.id]), r.seq), reverse=True)
        out.append("files & resources (most used first):")
        for r in ranked[:max_resources]:
            latest = uses[r.id][-1].id if uses[r.id] else "-"
            out.append(f"  {r.id}  {g.resource_label(r.id).split(' (', 1)[-1].rstrip(')')}  latest: {latest}")
        if len(ranked) > max_resources:
            out.append(f"  … {len(ranked) - max_resources} more; ctx_search finds any of them")

    failures = [n for n in g.nodes.values() if n.kind is NodeKind.FAILURE]
    if failures:
        out.append(f"recorded failures: {len(failures)} (always shown in context)")
    out.append("Pass any id above to ctx_expand; ctx_search(pattern) searches everything.")
    return _cap("\n".join(out))


def search(engine: ContextEngine, pattern: str, limit: int = 20) -> str:
    hits = engine.search_log(pattern, limit=limit)
    if not hits:
        return f"[no match for {pattern!r} anywhere in this session's history]"
    lines = [f"{len(hits)} match(es) for {pattern!r}:"]
    for h in hits:
        where = f" {h['resource']}" if h["resource"] else ""
        lines.append(f"  e{h['seq']:06d} {h['kind']}{where}: {h['snippet']}")
    return _cap("\n".join(lines))


def expand(
    engine: ContextEngine, item: str, *, grep: str | None = None, lines: str | None = None
) -> str:
    text = _resolve(engine, item.strip())
    if text is None:
        return f"[no item {item!r} in this session; ctx_outline and ctx_search list valid ids]"
    if grep:
        rx = re.compile(grep, re.I)
        text = "\n".join(f"{i + 1}: {ln}" for i, ln in enumerate(text.splitlines()) if rx.search(ln))
        text = text or f"[no line in {item} matches {grep!r}]"
    elif lines:
        m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", lines)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            text = "\n".join(
                f"{i}: {ln}" for i, ln in enumerate(text.splitlines()[lo - 1 : hi], start=lo)
            )
    return _cap(text)


def _resolve(engine: ContextEngine, item: str) -> str | None:
    g = engine.graph
    read = engine.store.read_handle

    if item.startswith("task:") and item in g.nodes:
        steps = [c for c in g.children(item) if c.kind in (NodeKind.CALL, NodeKind.TASK)]
        head = g.nodes[item].label
        return "\n".join([head, *(f"  {c.id}  {c.label}" for c in steps)])

    if item.startswith("res:") and item in g.nodes:
        latest = None
        for kind, src in g.in_edges(item):
            if kind in (EdgeKind.READS, EdgeKind.WRITES):
                if latest is None or g.nodes[src].seq > latest.seq:
                    latest = g.nodes[src]
        item = latest.id if latest else item

    if re.fullmatch(r"e\d+", item):
        seq = int(item[1:])
        node = g.nodes.get(item)
        seqs = node.seqs if node is not None else [seq]
        events = {ev.seq: ev for ev in engine.store.events(since=min(seqs))}
        parts = [event_text(events[q], read) for q in seqs if q in events]
        return "\n".join(p for p in parts if p) or None

    # A handle id: the blob itself.
    return read(item)


def _cap(text: str) -> str:
    if len(text) <= MAX_TOOL_CHARS:
        return text
    return text[: MAX_TOOL_CHARS - 80] + f"\n… [{len(text) - MAX_TOOL_CHARS + 80:,} chars cut; narrow with grep or lines]"

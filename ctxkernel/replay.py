"""Load an existing trace into the log.

Most people already have a history — a JSONL of their agent's turns, a
LangSmith export, rows in their own database. None of that has to be thrown
away or reshaped: describe each record with a small tuple and it becomes
events, with identity, digests and handle-ization applied on the way in.

This is also what the eval harness will stand on. Once a real trace can be
replayed under a policy, "97% saved" stops being a claim about a synthetic
benchmark and becomes a measurement on your own history.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from .engine import ContextEngine

#: What a mapper returns for one source record. The first element is the tag.
#:
#:   ("user",      text)
#:   ("assistant", text)
#:   ("call",      tool_use_id, tool_name, args_dict)
#:   ("result",    tool_use_id, content, is_error)
#:   ("goal",      text)
#:   ("failure",   attempted, why)
#:   None                      -> skip this record
Mapped = tuple[Any, ...] | None
Mapper = Callable[[Any], Mapped | list[Mapped]]


def import_records(
    engine: ContextEngine, records: Iterable[Any], *, mapper: Mapper
) -> int:
    """Feed arbitrary records through ``mapper`` into the log.

    Returns the number of events appended.
    """
    from .ir import Text

    n = 0
    for record in records:
        mapped = mapper(record)
        if mapped is None:
            continue
        for item in mapped if isinstance(mapped, list) else [mapped]:
            if item is None:
                continue
            tag, *rest = item
            if tag == "user":
                engine.record_user(rest[0])
            elif tag == "assistant":
                engine.record_model([Text(rest[0])])
            elif tag == "goal":
                engine.set_goal(rest[0])
            elif tag == "failure":
                engine.note_failure(rest[0], rest[1] if len(rest) > 1 else "")
            elif tag == "call":
                engine.record_tool_call(rest[0], rest[1], rest[2] if len(rest) > 2 else {})
            elif tag == "result":
                engine.record_tool_result(
                    rest[0], rest[1], is_error=bool(rest[2]) if len(rest) > 2 else False
                )
            else:
                raise ValueError(f"unknown mapper tag: {tag!r}")
            n += 1
    return n


def import_messages(
    engine: ContextEngine, messages: list[dict[str, Any]], *, fmt: str = "openai"
) -> int:
    """Import a provider-shaped message list — the common case.

    ``fmt`` is ``"openai"`` (also vLLM, Ollama, LM Studio, TGI and any other
    OpenAI-compatible server) or ``"anthropic"``.
    """
    if fmt == "anthropic":
        return engine.ingest_messages(messages)
    if fmt != "openai":
        raise ValueError(f"unknown fmt: {fmt!r} (use 'openai' or 'anthropic')")

    from .adapters.openai import parse_messages
    from .ir import Text, ToolResult, ToolUse

    n = 0
    for role, blocks in parse_messages(messages):
        if role == "assistant":
            engine.record_model(blocks)
            n += 1
            continue
        plain = []
        for b in blocks:
            if isinstance(b, ToolResult):
                engine.record_tool_result(b.tool_use_id, b.content, is_error=b.is_error)
                n += 1
            else:
                plain.append(b)
        text = "\n".join(getattr(b, "text", "") for b in plain).strip()
        if text:
            engine.record_user(text)
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Stream a JSONL file, skipping blank lines."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def replay(
    engine: ContextEngine, records: Iterable[Any], *, mapper: Mapper, every: int = 25
) -> list[dict[str, Any]]:
    """Import a trace, sampling the report every ``every`` events.

    The result is a growth curve: log size against context size, step by step.
    A flat context column beside a rising log column is the claim; anything
    else is the policy failing on your data rather than on a demo.
    """
    curve: list[dict[str, Any]] = []
    total = 0
    batch: list[Any] = []

    def flush() -> None:
        nonlocal total
        if not batch:
            return
        total += import_records(engine, batch, mapper=mapper)
        r = engine.report()
        curve.append(
            {
                "events": r["events"],
                "log_tokens": r["raw_tokens"],
                "context_tokens": r["assembled_tokens"],
                "saved_pct": r["saved_pct"],
            }
        )
        batch.clear()

    for record in records:
        batch.append(record)
        if len(batch) >= every:
            flush()
    flush()
    return curve

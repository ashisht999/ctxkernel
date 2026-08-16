"""Anthropic Messages API adapter, and the Tier 0 drop-in wrap.

Tier 0 is one line at the model-client boundary — the one place where every
tool call and result is visible without asking the host to change anything:

    client = engine.wrap(anthropic.Anthropic())

Structural validity is enforced here rather than hoped for. A ``tool_use``
without its ``tool_result`` is a hard 400, not a degradation, so repair runs on
every emit.
"""

from __future__ import annotations

from typing import Any

from ..ir import (
    Block,
    Elision,
    HandleRef,
    Text,
    Thinking,
    ToolResult,
    ToolUse,
)

Message = dict[str, Any]


# --------------------------------------------------------------------------
# IR -> Anthropic
# --------------------------------------------------------------------------


def render_block(block: Block) -> dict[str, Any] | None:
    if isinstance(block, Text):
        return {"type": "text", "text": block.text} if block.text.strip() else None
    if isinstance(block, Elision):
        return {"type": "text", "text": block.note}
    if isinstance(block, Thinking):
        # Without a signature the API rejects a thinking block, so it degrades
        # to text rather than failing the call.
        if block.signature:
            return {"type": "thinking", "thinking": block.text, "signature": block.signature}
        return {"type": "text", "text": block.text} if block.text.strip() else None
    if isinstance(block, ToolUse):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, ToolResult):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            **({"is_error": True} if block.is_error else {}),
        }
    if isinstance(block, HandleRef):
        body = (
            f"{block.extract}\n\n"
            f"[externalized: {block.original_tokens} tok, handle {block.handle_id}. "
            f"expand(\"{block.handle_id}\") for the full content.]"
        )
        return {"type": "tool_result", "tool_use_id": block.tool_use_id, "content": body}
    return None


def to_messages(items: list[tuple[str, Block]]) -> list[Message]:
    """Group role-tagged blocks into alternating messages, then repair."""
    msgs: list[Message] = []
    for role, block in items:
        rendered = render_block(block)
        if rendered is None:
            continue
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"].append(rendered)
        else:
            msgs.append({"role": role, "content": [rendered]})
    return repair(msgs)


def repair(msgs: list[Message]) -> list[Message]:
    """Enforce the structural invariants the API checks.

    Eviction operates on coherent units upstream, so this should rarely find
    anything — it runs anyway, because a violation here is a failed request
    rather than a slightly worse prompt.
    """
    use_ids: set[str] = set()
    result_ids: set[str] = set()
    for m in msgs:
        for b in m["content"]:
            if b.get("type") == "tool_use":
                use_ids.add(b["id"])
            elif b.get("type") == "tool_result":
                result_ids.add(b["tool_use_id"])

    orphan_uses = use_ids - result_ids      # would hang the turn
    orphan_results = result_ids - use_ids   # would 400

    out: list[Message] = []
    for m in msgs:
        content = [
            b
            for b in m["content"]
            if not (b.get("type") == "tool_use" and b["id"] in orphan_uses)
            and not (b.get("type") == "tool_result" and b["tool_use_id"] in orphan_results)
        ]
        if content:
            out.append({"role": m["role"], "content": content})

    # Merge any neighbours the removals left adjacent, and drop a leading
    # assistant turn — the API requires the conversation to open with user.
    merged: list[Message] = []
    for m in out:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"].extend(m["content"])
        else:
            merged.append(m)
    while merged and merged[0]["role"] != "user":
        merged.pop(0)
    return merged


# --------------------------------------------------------------------------
# Anthropic -> IR
# --------------------------------------------------------------------------


def parse_block(raw: Any) -> Block | None:
    d = raw if isinstance(raw, dict) else getattr(raw, "__dict__", None) or {}
    if isinstance(raw, str):
        return Text(raw)
    t = d.get("type")
    if t == "text":
        return Text(d.get("text", ""))
    if t == "thinking":
        return Thinking(d.get("thinking", ""), d.get("signature"))
    if t == "tool_use":
        return ToolUse(d.get("id", ""), d.get("name", ""), d.get("input", {}) or {})
    if t == "tool_result":
        c = d.get("content", "")
        if isinstance(c, list):
            c = "\n".join(
                x.get("text", "") if isinstance(x, dict) else str(x) for x in c
            )
        return ToolResult(d.get("tool_use_id", ""), str(c), bool(d.get("is_error")))
    return None


def parse_content(content: Any) -> list[Block]:
    if isinstance(content, str):
        return [Text(content)]
    out: list[Block] = []
    for raw in content or []:
        b = parse_block(raw)
        if b is not None:
            out.append(b)
    return out


# --------------------------------------------------------------------------
# Tier 0 wrap
# --------------------------------------------------------------------------


class WrappedMessages:
    def __init__(self, inner: Any, engine: Any) -> None:
        self._inner = inner
        self._engine = engine

    def create(self, *, messages: list[Message], **kw: Any) -> Any:
        self._engine.ingest_messages(messages)
        assembly = self._engine.assemble()
        response = self._inner.create(messages=to_messages(assembly.items), **kw)
        self._engine.record_model(parse_content(getattr(response, "content", [])))
        self._engine._last_report = assembly.report()
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class WrappedClient:
    """Substitutes an assembled view for whatever message list is passed in.

    The host loop is unmodified: it keeps appending to its own list, and never
    learns that the list it sends is not the list it built.
    """

    def __init__(self, inner: Any, engine: Any) -> None:
        self._inner = inner
        self._engine = engine
        self.messages = WrappedMessages(inner.messages, engine)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

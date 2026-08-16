"""OpenAI Chat Completions adapter.

Exists to keep the core honest. The shapes differ substantially from
Anthropic's — tool results are their own `role: "tool"` messages rather than
blocks inside a user turn, arguments are a JSON *string*, and text content is a
bare string — so if anything provider-specific had leaked out of `adapters/`,
this file could not have been written without touching the rest of the package.

It was written without touching the rest of the package.
"""

from __future__ import annotations

import json
from typing import Any

from ..ir import Block, Elision, HandleRef, Text, Thinking, ToolResult, ToolUse

Message = dict[str, Any]


# --------------------------------------------------------------------------
# IR -> OpenAI
# --------------------------------------------------------------------------


def to_messages(items: list[tuple[str, Block]]) -> list[Message]:
    """Render role-tagged blocks into Chat Completions messages."""
    msgs: list[Message] = []

    def last_assistant() -> Message:
        if not msgs or msgs[-1]["role"] != "assistant":
            msgs.append({"role": "assistant", "content": None})
        return msgs[-1]

    def append_text(role: str, text: str) -> None:
        if not text.strip():
            return
        if msgs and msgs[-1]["role"] == role and isinstance(msgs[-1].get("content"), str):
            msgs[-1]["content"] += "\n" + text
        elif msgs and msgs[-1]["role"] == role and msgs[-1].get("content") is None:
            msgs[-1]["content"] = text
        else:
            msgs.append({"role": role, "content": text})

    for role, block in items:
        if isinstance(block, (Text, Thinking)):
            append_text(role, block.text)
        elif isinstance(block, Elision):
            append_text(role, block.note)
        elif isinstance(block, ToolUse):
            m = last_assistant()
            m.setdefault("tool_calls", []).append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        # OpenAI takes arguments as a JSON string, not an object.
                        "arguments": json.dumps(block.input),
                    },
                }
            )
        elif isinstance(block, ToolResult):
            msgs.append(
                {"role": "tool", "tool_call_id": block.tool_use_id, "content": block.content}
            )
        elif isinstance(block, HandleRef):
            body = (
                f"{block.extract}\n\n"
                f"[externalized: {block.original_tokens} tok, handle {block.handle_id}. "
                f'expand("{block.handle_id}") for the full content.]'
            )
            msgs.append({"role": "tool", "tool_call_id": block.tool_use_id, "content": body})

    return repair(msgs)


def repair(msgs: list[Message]) -> list[Message]:
    """Enforce what the API checks: every ``role: "tool"`` message must answer a
    tool_call that appears in a preceding assistant message, and every tool_call
    must be answered."""
    call_ids: set[str] = set()
    answered: set[str] = set()
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            call_ids.add(tc["id"])
        if m.get("role") == "tool":
            answered.add(m.get("tool_call_id", ""))

    orphan_calls = call_ids - answered
    orphan_results = answered - call_ids

    out: list[Message] = []
    for m in msgs:
        if m.get("role") == "tool" and m.get("tool_call_id") in orphan_results:
            continue
        m = dict(m)
        if m.get("tool_calls"):
            kept = [tc for tc in m["tool_calls"] if tc["id"] not in orphan_calls]
            if kept:
                m["tool_calls"] = kept
            else:
                m.pop("tool_calls")
        if m.get("role") == "assistant" and not m.get("tool_calls") and not m.get("content"):
            continue
        out.append(m)
    return out


# --------------------------------------------------------------------------
# OpenAI -> IR
# --------------------------------------------------------------------------


def parse_message(msg: Message) -> tuple[str, list[Block]]:
    role = msg.get("role", "user")
    blocks: list[Block] = []

    if role == "tool":
        return "user", [ToolResult(msg.get("tool_call_id", ""), str(msg.get("content", "")))]

    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        blocks.append(Text(content))
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                blocks.append(Text(part.get("text", "")))

    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {"_raw": fn.get("arguments")}
        blocks.append(ToolUse(tc.get("id", ""), fn.get("name", ""), args))

    return ("assistant" if role == "assistant" else "user"), blocks


def parse_messages(msgs: list[Message]) -> list[tuple[str, list[Block]]]:
    return [parse_message(m) for m in msgs]

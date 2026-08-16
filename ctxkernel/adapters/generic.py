"""For any other provider, and for self-hosted models.

Nothing in the engine knows what a provider is. `assemble()` returns
`(role, block)` pairs, and turning those into whatever your endpoint wants is
the entire integration. Two renderers cover essentially everything:

* :func:`to_chat` — plain ``[{"role", "content"}]`` with tool traffic flattened
  into text. Works with any chat endpoint that has no native tool schema.
* :func:`to_text` — one prompt string, for raw completion endpoints
  (llama.cpp, TGI generate, a fine-tuned base model).

If your server speaks the OpenAI schema — vLLM, Ollama, LM Studio, TGI,
llama-cpp-python, most inference gateways — use ``adapters.openai`` instead and
you are already done.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ..ir import Block, Elision, HandleRef, Text, Thinking, ToolResult, ToolUse

Item = tuple[str, Block]


def block_to_text(block: Block) -> str:
    """One block as plain text. Override piecemeal if your model was tuned on a
    particular tool-call syntax."""
    if isinstance(block, (Text, Thinking)):
        return block.text
    if isinstance(block, Elision):
        return block.note
    if isinstance(block, ToolUse):
        return f"<tool_call name=\"{block.name}\" id=\"{block.id}\">{json.dumps(block.input)}</tool_call>"
    if isinstance(block, ToolResult):
        tag = "tool_error" if block.is_error else "tool_result"
        return f"<{tag} id=\"{block.tool_use_id}\">{block.content}</{tag}>"
    if isinstance(block, HandleRef):
        return (
            f"<tool_result id=\"{block.tool_use_id}\">{block.extract}\n"
            f"[externalized: {block.original_tokens} tok, handle {block.handle_id}. "
            f"call expand(\"{block.handle_id}\") for the full content.]</tool_result>"
        )
    return str(block)


def to_chat(
    items: list[Item], *, render: Callable[[Block], str] = block_to_text
) -> list[dict[str, Any]]:
    """Merge consecutive same-role blocks into plain chat messages."""
    msgs: list[dict[str, Any]] = []
    for role, block in items:
        text = render(block)
        if not text.strip():
            continue
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n" + text
        else:
            msgs.append({"role": role, "content": text})
    return msgs


def to_text(
    items: list[Item],
    *,
    render: Callable[[Block], str] = block_to_text,
    user_prefix: str = "\n\n### User\n",
    assistant_prefix: str = "\n\n### Assistant\n",
) -> str:
    """Flatten to a single prompt string for completion-style endpoints."""
    out: list[str] = []
    last_role: str | None = None
    for role, block in items:
        text = render(block)
        if not text.strip():
            continue
        if role != last_role:
            out.append(user_prefix if role == "user" else assistant_prefix)
            last_role = role
        out.append(text + "\n")
    return "".join(out).strip()

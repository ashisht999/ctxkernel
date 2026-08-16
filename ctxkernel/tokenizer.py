"""Token counting, abstracted over providers.

Budgets are only as honest as their counter, so an approximate counter reports
its own error bound rather than pretending to be exact. A budget that silently
overruns is worse than a conservative one.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from .ir import Block, Elision, HandleRef, Text, Thinking, ToolResult, ToolUse


@runtime_checkable
class Tokenizer(Protocol):
    #: Fractional error bound, e.g. 0.15 for +/-15%. 0.0 means exact.
    error_bound: float

    def count_text(self, text: str) -> int: ...


_WORDISH = re.compile(r"\w+|[^\w\s]")


class ApproxTokenizer:
    """Heuristic counter used when no provider tokenizer is installed.

    Calibrated to over-count slightly: for budgeting, erring high is safe and
    erring low silently blows the window.
    """

    error_bound = 0.15

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        pieces = len(_WORDISH.findall(text))
        # Long tokens split further; whitespace-heavy text splits less.
        return max(1, int(pieces * 1.30) + len(text) // 400)


class TiktokenTokenizer:
    """Exact counts via tiktoken, when it is installed."""

    error_bound = 0.0

    def __init__(self, encoding: str = "cl100k_base") -> None:
        import tiktoken  # imported lazily; optional dependency

        self._enc = tiktoken.get_encoding(encoding)

    def count_text(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))


def default_tokenizer() -> Tokenizer:
    try:
        return TiktokenTokenizer()
    except Exception:
        return ApproxTokenizer()


#: Rough per-block structural overhead (role markers, JSON scaffolding).
_BLOCK_OVERHEAD = 4


def count_block(tok: Tokenizer, block: Block) -> int:
    if isinstance(block, (Text, Thinking)):
        return tok.count_text(block.text) + _BLOCK_OVERHEAD
    if isinstance(block, ToolUse):
        import json

        return tok.count_text(block.name + json.dumps(block.input)) + _BLOCK_OVERHEAD
    if isinstance(block, ToolResult):
        return tok.count_text(block.content) + _BLOCK_OVERHEAD
    if isinstance(block, HandleRef):
        return tok.count_text(block.extract) + _BLOCK_OVERHEAD + 8
    if isinstance(block, Elision):
        return tok.count_text(block.note) + _BLOCK_OVERHEAD
    raise TypeError(f"unknown block type: {type(block).__name__}")


def count_blocks(tok: Tokenizer, blocks: list[Block]) -> int:
    return sum(count_block(tok, b) for b in blocks)


def count_block_unmanaged(tok: Tokenizer, block: Block) -> int:
    """What this block *would* have cost with no engine in the loop.

    Counting a HandleRef at its extract size would flatter the report by
    hiding the saving that interception already made, so the original size is
    what the comparison uses.
    """
    if isinstance(block, HandleRef):
        return block.original_tokens + _BLOCK_OVERHEAD
    if isinstance(block, Elision):
        return block.tokens_elided
    return count_block(tok, block)


def count_blocks_unmanaged(tok: Tokenizer, blocks: list[Block]) -> int:
    return sum(count_block_unmanaged(tok, b) for b in blocks)

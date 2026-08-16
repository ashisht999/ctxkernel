"""Canonical, provider-neutral intermediate representation.

Nothing outside `adapters/` may touch a provider's message format. The moment
that rule is broken this becomes a wrapper for one vendor instead of an engine.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable

try:  # optional, ~5x faster than blake2b on large blobs
    import xxhash

    def _hash_bytes(b: bytes) -> str:
        return xxhash.xxh3_128_hexdigest(b)

except ImportError:  # stdlib fallback, plenty fast for our sizes

    def _hash_bytes(b: bytes) -> str:
        return hashlib.blake2b(b, digest_size=16).hexdigest()


def digest_of(content: str | bytes) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return _hash_bytes(content)


# --------------------------------------------------------------------------
# Resource identity
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResourceId:
    """What an event is *about*.

    This is the one field that unlocks the whole Tier A cascade. Ordering tells
    you when something happened; identity tells you what it happened to, which
    is what makes supersession and invalidation computable.
    """

    kind: str  # "file" | "http" | "shell" | "db" | "tool"
    key: str  # "src/auth.py" | "GET /users" | "pytest tests/"

    def __str__(self) -> str:
        return f"{self.kind}:{self.key}"

    @classmethod
    def parse(cls, s: str) -> ResourceId:
        kind, _, key = s.partition(":")
        return cls(kind, key)


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------


class Block:
    """Base for content blocks. Subclasses are the closed set below."""

    __slots__ = ()


@dataclass(slots=True)
class Text(Block):
    text: str


@dataclass(slots=True)
class Thinking(Block):
    text: str
    signature: str | None = None


@dataclass(slots=True)
class ToolUse(Block):
    id: str
    name: str
    input: dict[str, Any]


@dataclass(slots=True)
class ToolResult(Block):
    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass(slots=True)
class HandleRef(Block):
    """A pointer standing in for content that lives in the blob store.

    `extract` is *computed* (parsed), never model-written, so it cannot
    hallucinate and costs nothing to produce.
    """

    tool_use_id: str
    handle_id: str
    extract: str
    media_type: str
    size_bytes: int
    original_tokens: int
    is_error: bool = False


@dataclass(slots=True)
class Elision(Block):
    """Marks removed material so the model can see that something is absent.

    An LLM cannot page-fault mid-forward-pass, so this stands in for the fault:
    it makes absence visible, and `expand()` satisfies it.
    """

    handle_id: str | None
    note: str
    tokens_elided: int


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


class EventKind(str, Enum):
    USER_MESSAGE = "user_message"
    MODEL_OUTPUT = "model_output"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    NOTE = "note"
    TASK_BEGIN = "task_begin"
    TASK_END = "task_end"
    GOAL = "goal"
    FAILURE = "failure"
    INVARIANT = "invariant"


#: Events that are pinned into context and never evicted. They are small by
#: construction; the pinned floor must always fit so the agent can never be
#: wedged into a state where no prompt can be assembled.
PINNED_KINDS = frozenset(
    {EventKind.GOAL, EventKind.FAILURE, EventKind.INVARIANT, EventKind.USER_MESSAGE}
)


@dataclass(slots=True)
class Event:
    seq: int
    kind: EventKind
    blocks: list[Block] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    #: What this event is about. Drives supersession and invalidation.
    resource: ResourceId | None = None
    #: Content hash, for duplicate detection.
    digest: str | None = None
    #: Resources this event's content was derived from. A write to any of them
    #: invalidates this event.
    derived_from: tuple[ResourceId, ...] = ()
    #: True when this event *mutated* its resource (a write, not a read).
    is_write: bool = False
    #: Task frame this event belongs to, if the host supplied boundaries.
    task_id: str | None = None

    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"e{self.seq:06d}"

    def with_blocks(self, blocks: list[Block]) -> Event:
        return replace(self, blocks=list(blocks))


# --------------------------------------------------------------------------
# Handles
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Handle:
    """A dereferenceable pointer to externalized content.

    The id is scoped by tenant and session. Handles appear verbatim in prompt
    text, so a globally-addressable id would turn the token-saving mechanism
    into a cross-tenant read primitive.
    """

    id: str
    digest: str
    media_type: str
    size_bytes: int
    extract: str
    resource: ResourceId | None = None

    @staticmethod
    def make_id(
        tenant: str, session: str, digest: str, resource: ResourceId | None = None
    ) -> str:
        """Two layers, two keys: blobs are content-addressed so identical bytes
        are stored once, but a handle is a pointer *to a resource*, so two
        files with the same content still get distinct handles. Collapsing them
        would make the resource attribution last-writer-wins."""
        scoped = f"{tenant}\x00{session}\x00{digest}\x00{resource or ''}".encode("utf-8")
        return _hash_bytes(scoped)[:16]


def iter_blocks(events: Iterable[Event]) -> Iterable[tuple[Event, Block]]:
    for ev in events:
        for b in ev.blocks:
            yield ev, b

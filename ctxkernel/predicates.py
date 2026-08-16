"""Tier A: removal that requires no judgment.

Each predicate here is a fact about the log, not an opinion about importance.
Removing anything they flag *cannot* change the next action, because the same
information is present in a fresher form or is known to be false. There is no
risk to weigh, so there is no decision to get wrong.

Everything is derived from three indexes maintained on append. Nothing stores a
relevance score, nothing decays, nothing needs rebuilding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .ir import PINNED_KINDS, Event, EventKind, ResourceId


class Reason(str, Enum):
    PINNED = "pinned"
    KEEP = "keep"
    #: Kept only because it is structurally paired with something kept.
    PAIRED = "paired"
    SUPERSEDED = "superseded"
    DUPLICATE = "duplicate"
    INVALIDATED = "invalidated"
    SCOPE_CLOSED = "scope_closed"


#: Ordered by strength of the claim; used only for reporting.
DROP_REASONS = (
    Reason.INVALIDATED,
    Reason.SUPERSEDED,
    Reason.DUPLICATE,
    Reason.SCOPE_CLOSED,
)


@dataclass(slots=True)
class Decision:
    seq: int
    keep: bool
    reason: Reason
    detail: str = ""


@dataclass(slots=True)
class Indexes:
    """Built in one pass; the same three lookups the SQLite indexes provide."""

    last_seq_by_resource: dict[str, int] = field(default_factory=dict)
    last_write_seq_by_resource: dict[str, int] = field(default_factory=dict)
    last_seq_by_digest: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(cls, events: list[Event]) -> Indexes:
        ix = cls()
        for ev in events:
            if ev.resource is not None:
                key = str(ev.resource)
                ix.last_seq_by_resource[key] = ev.seq
                if ev.is_write:
                    ix.last_write_seq_by_resource[key] = ev.seq
            if ev.digest is not None:
                ix.last_seq_by_digest[ev.digest] = ev.seq
        return ix


def decide(
    events: list[Event],
    *,
    closed_tasks: frozenset[str] | set[str] = frozenset(),
    indexes: Indexes | None = None,
) -> list[Decision]:
    """Run the Tier A cascade over ``events`` in sequence order."""
    ix = indexes or Indexes.build(events)
    out: list[Decision] = []

    for ev in events:
        # 1. Pinned. The floor must always fit, so it is never a candidate.
        if ev.kind in PINNED_KINDS:
            out.append(Decision(ev.seq, True, Reason.PINNED))
            continue

        # 2. Invalidated: derived from a resource that has since been written.
        #    Strongest claim -- the content is not stale, it is known false.
        stale_src = _invalidating_source(ev, ix)
        if stale_src is not None:
            out.append(
                Decision(ev.seq, False, Reason.INVALIDATED, f"{stale_src} written later")
            )
            continue

        # 3. Superseded: a later event observes the same resource. Only
        #    observations are superseded; a write is an action, and the record
        #    that it happened stays history.
        if ev.resource is not None and not ev.is_write:
            last = ix.last_seq_by_resource.get(str(ev.resource))
            if last is not None and last > ev.seq:
                out.append(
                    Decision(ev.seq, False, Reason.SUPERSEDED, f"fresher read at e{last:06d}")
                )
                continue

        # 4. Duplicate: identical bytes appear later, so keep the later copy
        #    (it sits nearer the current turn) and drop this one.
        if ev.digest is not None:
            last = ix.last_seq_by_digest.get(ev.digest)
            if last is not None and last > ev.seq:
                out.append(
                    Decision(ev.seq, False, Reason.DUPLICATE, f"same bytes at e{last:06d}")
                )
                continue

        # 5. Tier B: the scope this belonged to has closed and its outcome is
        #    recorded. Weaker than the above -- the scope may have been drawn
        #    wrong -- which is why it runs last and stays re-expandable.
        if ev.task_id and ev.task_id in closed_tasks and ev.kind is not EventKind.TASK_END:
            out.append(
                Decision(ev.seq, False, Reason.SCOPE_CLOSED, f"task {ev.task_id} closed")
            )
            continue

        out.append(Decision(ev.seq, True, Reason.KEEP))

    return out


def _invalidating_source(ev: Event, ix: Indexes) -> ResourceId | None:
    for src in ev.derived_from:
        w = ix.last_write_seq_by_resource.get(str(src))
        if w is not None and w > ev.seq:
            return src
    return None


def summarize(decisions: list[Decision]) -> dict[str, int]:
    counts: dict[str, int] = {r.value: 0 for r in Reason}
    for d in decisions:
        counts[d.reason.value] += 1
    return counts

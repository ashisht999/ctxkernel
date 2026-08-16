"""The assembler: (log, budget) -> prompt.

Deterministic, with no model in the path. Two invariants it exists to hold:

* **Structural validity.** A ``tool_use`` and its ``tool_result`` are one
  indivisible eviction unit. Dropping one without the other is a hard API error,
  not a degradation, and it is where naive middleware breaks in production.
* **Stable-to-volatile ordering.** Zones are laid out so the expensive prefix
  stays cacheable. Every mutation left of the cache breakpoint is paid for on
  every subsequent call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .ir import (
    Block,
    Elision,
    Event,
    EventKind,
    HandleRef,
    ResourceId,
    Text,
    ToolResult,
    ToolUse,
)
from .ir import PINNED_KINDS
from .predicates import Decision, Reason, decide
from .tokenizer import (
    Tokenizer,
    count_blocks,
    count_blocks_unmanaged,
    default_tokenizer,
)

# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Budget:
    """Fractions, not absolutes, so a larger window inherits the same policy
    instead of tempting you to fill it and pay for it on every turn."""

    window: int = 200_000
    generation_reserve: int = 32_000

    pinned_floor_frac: float = 0.01
    target_frac: float = 0.15
    pressure_frac: float = 0.45

    #: Share of the target given to the verbatim tail.
    tail_frac: float = 0.75
    max_handle_entries: int = 24
    max_failures: int = 40
    max_invariants: int = 40

    @property
    def usable(self) -> int:
        return self.window - self.generation_reserve

    @property
    def pinned_floor(self) -> int:
        return int(self.window * self.pinned_floor_frac)

    @property
    def target(self) -> int:
        return int(self.window * self.target_frac)

    @property
    def pressure_trigger(self) -> int:
        return int(self.window * self.pressure_frac)

    @property
    def tail_budget(self) -> int:
        return int(self.target * self.tail_frac)


def window(n: int) -> Budget:
    """``budgets.window(200_000).reserve(generation=32_000)``"""
    return Budget(window=n)


def _reserve(self: Budget, *, generation: int) -> Budget:
    from dataclasses import replace

    return replace(self, generation_reserve=generation)


Budget.reserve = _reserve  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# Zones, ordered stable -> volatile
# --------------------------------------------------------------------------

ZONE_ORDER = ("goal", "invariants", "failures", "handles", "retrieved", "tail")

#: Everything before this zone is expected to survive across turns unchanged.
CACHE_BREAKPOINT_AFTER = "handles"


#: Which side of the conversation a block belongs to. Block type decides this
#: unambiguously except for Text, which is why events carry it down.
_ASSISTANT_KINDS = frozenset({EventKind.MODEL_OUTPUT, EventKind.TOOL_CALL})


def role_for(kind: EventKind) -> str:
    return "assistant" if kind in _ASSISTANT_KINDS else "user"


#: A block paired with the role whose message it must be rendered into.
Item = tuple[str, Block]


@dataclass(slots=True)
class Zone:
    name: str
    items: list[Item] = field(default_factory=list)
    tokens: int = 0
    evicted: int = 0

    @property
    def blocks(self) -> list[Block]:
        return [b for _, b in self.items]


@dataclass(slots=True)
class Assembly:
    zones: list[Zone]
    items: list[Item]
    tokens: int
    decisions: list[Decision]
    dropped_tokens: int
    raw_tokens: int
    degraded: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def blocks(self) -> list[Block]:
        return [b for _, b in self.items]

    @property
    def cache_prefix_tokens(self) -> int:
        total = 0
        for z in self.zones:
            total += z.tokens
            if z.name == CACHE_BREAKPOINT_AFTER:
                break
        return total

    def report(self) -> dict[str, object]:
        from .predicates import summarize

        return {
            "raw_tokens": self.raw_tokens,
            "assembled_tokens": self.tokens,
            "saved_tokens": self.raw_tokens - self.tokens,
            "saved_pct": round(
                100 * (self.raw_tokens - self.tokens) / self.raw_tokens, 1
            )
            if self.raw_tokens
            else 0.0,
            "cache_prefix_tokens": self.cache_prefix_tokens,
            "zones": {z.name: z.tokens for z in self.zones},
            "decisions": summarize(self.decisions),
            "degraded": self.degraded,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# Eviction units
# --------------------------------------------------------------------------


def eviction_units(events: list[Event]) -> list[list[Event]]:
    """Group events so a ``tool_use`` and its ``tool_result`` never separate."""
    owner: dict[str, int] = {}  # tool_use_id -> unit index
    units: list[list[Event]] = []

    for ev in events:
        ids = _tool_use_ids(ev)
        target: int | None = None
        for tid in ids:
            if tid in owner:
                target = owner[tid]
                break
        if target is None:
            units.append([ev])
            target = len(units) - 1
        else:
            units[target].append(ev)
        for tid in ids:
            owner.setdefault(tid, target)

    return units


def _reconcile_pairs(events: list[Event], verdict: dict[int, Decision]) -> None:
    """Make a ``tool_use`` share the fate of its ``tool_result``.

    Left alone the predicates get this wrong in both directions: a tool_call
    carries the same resource as the result it produced, so it reads as
    superseded by its own result, and dropping it would orphan the result into
    a hard API error. The pair is one unit, so it decides once.
    """
    for unit in eviction_units(events):
        if len(unit) < 2:
            continue
        if any(ev.kind in PINNED_KINDS for ev in unit):
            keep, reason = True, Reason.PINNED
        else:
            # The result is the data-bearing member; the call follows it.
            last = verdict[unit[-1].seq]
            keep, reason = last.keep, last.reason
        for ev in unit:
            cur = verdict[ev.seq]
            if cur.keep != keep:
                verdict[ev.seq] = Decision(
                    ev.seq,
                    keep,
                    Reason.PAIRED if keep else reason,
                    f"paired with {unit[-1].id}",
                )


def _tool_use_ids(ev: Event) -> list[str]:
    out: list[str] = []
    for b in ev.blocks:
        if isinstance(b, ToolUse):
            out.append(b.id)
        elif isinstance(b, (ToolResult, HandleRef)):
            out.append(b.tool_use_id)
    return out


# --------------------------------------------------------------------------
# Assembler
# --------------------------------------------------------------------------


class Assembler:
    def __init__(self, budget: Budget | None = None, tokenizer: Tokenizer | None = None) -> None:
        self.budget = budget or Budget()
        self.tok = tokenizer or default_tokenizer()

    def assemble(
        self,
        events: list[Event],
        *,
        closed_tasks: frozenset[str] | set[str] = frozenset(),
    ) -> Assembly:
        b = self.budget
        raw_tokens = sum(count_blocks_unmanaged(self.tok, ev.blocks) for ev in events)

        decisions = decide(events, closed_tasks=closed_tasks)
        verdict = {d.seq: d for d in decisions}
        _reconcile_pairs(events, verdict)
        decisions = [verdict[ev.seq] for ev in events]
        kept = [ev for ev in events if verdict[ev.seq].keep]

        zones = {name: Zone(name) for name in ZONE_ORDER}
        notes: list[str] = []

        # -- pinned zones --------------------------------------------------
        goal_evs, inv_evs, fail_evs, rest = [], [], [], []
        for ev in kept:
            if ev.kind is EventKind.GOAL or ev.kind is EventKind.USER_MESSAGE:
                goal_evs.append(ev)
            elif ev.kind is EventKind.INVARIANT:
                inv_evs.append(ev)
            elif ev.kind is EventKind.FAILURE:
                fail_evs.append(ev)
            else:
                rest.append(ev)

        # Human text is cheap and there is little of it, so every user message
        # is kept verbatim rather than classified as goal / correction / aside.
        zones["goal"].items = _flatten(goal_evs)
        inv_evs = inv_evs[-b.max_invariants :]
        fail_evs = fail_evs[-b.max_failures :]
        zones["invariants"].items = _flatten(inv_evs)
        zones["failures"].items = _flatten(fail_evs)

        # -- handle index --------------------------------------------------
        # Built from *all* events, including dropped ones: an evicted event's
        # content stays reachable, which is what makes the eviction survivable.
        zones["handles"].items = self._handle_index(events)

        # -- tail ----------------------------------------------------------
        units = eviction_units(rest)
        tail_items, evicted_units, elided_tokens = self._fit_tail(units)
        zones["tail"].items = tail_items
        zones["tail"].evicted = evicted_units

        for z in zones.values():
            z.tokens = count_blocks(self.tok, z.blocks)

        ordered = [zones[n] for n in ZONE_ORDER]
        total = sum(z.tokens for z in ordered)

        # -- pinned floor guarantee ---------------------------------------
        pinned = sum(zones[n].tokens for n in ("goal", "invariants", "failures"))
        if pinned > b.pinned_floor:
            notes.append(
                f"pinned zones at {pinned} tok exceed floor {b.pinned_floor}; "
                "consider distilling invariants"
            )

        degraded = total > b.pressure_trigger
        if degraded:
            notes.append(
                f"assembled {total} tok above pressure trigger {b.pressure_trigger}: "
                "boundary compaction should have run first"
            )

        items: list[Item] = []
        for z in ordered:
            items.extend(z.items)

        return Assembly(
            zones=ordered,
            items=items,
            tokens=total,
            decisions=decisions,
            dropped_tokens=raw_tokens - total,
            raw_tokens=raw_tokens,
            degraded=degraded,
            notes=notes,
        )

    # -- helpers -----------------------------------------------------------

    def _handle_index(self, events: Iterable[Event]) -> list[Item]:
        latest: dict[str, tuple[str, str]] = {}  # resource -> (handle_id, first line)
        for ev in events:
            for blk in ev.blocks:
                if isinstance(blk, HandleRef):
                    key = str(ev.resource) if ev.resource else blk.handle_id
                    first = blk.extract.splitlines()[0] if blk.extract else key
                    latest[key] = (blk.handle_id, first)
        if not latest:
            return []
        entries = list(latest.items())[-self.budget.max_handle_entries :]
        lines = [f"  {hid}  {first}" for _, (hid, first) in entries]
        return [("user", Text("Available via expand(handle_id):\n" + "\n".join(lines)))]

    def _fit_tail(self, units: list[list[Event]]) -> tuple[list[Item], int, int]:
        """Keep newest units until the tail budget is spent; elide the rest."""
        budget = self.budget.tail_budget
        chosen: list[list[Event]] = []
        used = 0
        for unit in reversed(units):
            cost = sum(count_blocks(self.tok, ev.blocks) for ev in unit)
            if used + cost > budget and chosen:
                break
            chosen.append(unit)
            used += cost
        chosen.reverse()

        n_evicted = len(units) - len(chosen)
        elided_tokens = 0
        items: list[Item] = []

        if n_evicted:
            for unit in units[:n_evicted]:
                elided_tokens += sum(count_blocks(self.tok, ev.blocks) for ev in unit)
            # One coalesced marker. An LLM cannot page-fault mid-pass, so this
            # stands in for the fault: it makes the absence visible.
            items.append(
                (
                    "user",
                    Elision(
                        handle_id=None,
                        note=(
                            f"[{n_evicted} earlier exchanges elided (~{elided_tokens} tok). "
                            "Nothing was deleted — use expand(handle_id) or search_log() "
                            "to retrieve any of it.]"
                        ),
                        tokens_elided=elided_tokens,
                    ),
                )
            )

        for unit in chosen:
            items.extend(_flatten(unit))

        return items, n_evicted, elided_tokens


def _flatten(events: list[Event]) -> list[Item]:
    out: list[Item] = []
    for ev in events:
        role = role_for(ev.kind)
        out.extend((role, b) for b in ev.blocks)
    return out

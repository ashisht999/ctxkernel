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
from typing import Callable, Collection, Iterable, Mapping

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
    #: Swap mode: the newest units a ranked item may never displace. Agents
    #: mostly act on what they just saw, so the immediate exchange is not up
    #: for trade.
    protect_recent: int = 4
    #: Full-content page-ins may add up to this share of the target on top of
    #: it, and at most ``inline_item_tokens`` each.
    inline_frac: float = 0.2
    inline_item_tokens: int = 4_000

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

    @property
    def retrieved_budget(self) -> int:
        """What the tail does not take. Recall spends only this, so turning a
        decision model on can never crowd out the recent exchange."""
        return self.target - self.tail_budget


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
        retrieve: Mapping[int, float] | frozenset[int] | set[int] = frozenset(),
        swap_threshold: float | None = None,
        inline: Mapping[str, float] | Collection[str] = frozenset(),
        read_full: Callable[[str], str | None] | None = None,
        inline_focus: Mapping[str, Collection[str]] | None = None,
    ) -> Assembly:
        """``retrieve`` names log events the caller wants back even if the tail
        has aged them out -- the expansion of the current ranking, as event →
        score. Stale events stay out regardless: recall can re-admit what is
        old, never what the cascade has shown to be false.

        Two ways to spend the budget on ``retrieve``:

        * ``swap_threshold`` set -- **swap**: recency fills the whole tail
          budget first, then an aged-out unit scored at or above the threshold
          displaces the oldest tail units scored below it, never the newest
          ``protect_recent``. With no confident score the result *is* recency,
          so a ranking can only add to it.
        * unset -- **split**: recalled units get their own share
          (``retrieved_budget``), highest score first.

        ``inline`` names handles whose full content should replace their
        extract (handle id → score, highest first), read with ``read_full``
        and capped by ``inline_frac`` / ``inline_item_tokens``. A page-in too
        big for its cap keeps the lines around its ``inline_focus`` names first:
        those names are why it was paged in.
        """
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
        chosen, evicted = self._split_tail(units)

        # -- retrieved -----------------------------------------------------
        # Aged-out units the ranking points at come back, whole, ahead of the
        # tail. Chronological order is kept, so the conversation still reads
        # forwards.
        if swap_threshold is not None:
            chosen, recalled, evicted = self._swap(chosen, evicted, retrieve, swap_threshold)
        else:
            recalled, evicted = self._recall(evicted, retrieve)
        zones["retrieved"].items = [it for u in recalled for it in _flatten(u)]

        zones["tail"].items = self._render_tail(chosen, evicted)
        zones["tail"].evicted = len(evicted)

        # -- full content where the summary is not enough ------------------
        if inline and read_full is not None:
            self._inline([zones["retrieved"], zones["tail"]], inline, read_full, inline_focus or {})

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

    def _split_tail(
        self, units: list[list[Event]]
    ) -> tuple[list[list[Event]], list[list[Event]]]:
        """Keep newest units until the tail budget is spent; the rest age out."""
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
        return chosen, units[: len(units) - len(chosen)]

    def _recall(
        self,
        evicted: list[list[Event]],
        retrieve: Mapping[int, float] | frozenset[int] | set[int],
    ) -> tuple[list[list[Event]], list[list[Event]]]:
        if not retrieve or not evicted:
            return [], evicted
        prio = retrieve if isinstance(retrieve, Mapping) else dict.fromkeys(retrieve, 0.0)
        wanted: list[tuple[float, int]] = []
        for i, unit in enumerate(evicted):
            scores = [prio[ev.seq] for ev in unit if ev.seq in prio]
            if scores:
                wanted.append((max(scores), i))
        # Highest score first; among equals the newest, which is the likelier
        # to still be true.
        wanted.sort(key=lambda t: (-t[0], -t[1]))
        budget = self.budget.retrieved_budget
        picked: set[int] = set()
        used = 0
        for _, i in wanted:
            cost = sum(count_blocks(self.tok, ev.blocks) for ev in evicted[i])
            if used + cost > budget:
                continue
            picked.add(i)
            used += cost
        # Output stays chronological, so the conversation still reads forwards.
        recalled = [u for i, u in enumerate(evicted) if i in picked]
        rest = [u for i, u in enumerate(evicted) if i not in picked]
        return recalled, rest

    def _swap(
        self,
        chosen: list[list[Event]],
        evicted: list[list[Event]],
        retrieve: Mapping[int, float] | frozenset[int] | set[int],
        threshold: float,
    ) -> tuple[list[list[Event]], list[list[Event]], list[list[Event]]]:
        """Recency first; a confident ranking swaps in, oldest-first out."""
        if not retrieve or not evicted:
            return chosen, [], evicted
        prio = retrieve if isinstance(retrieve, Mapping) else dict.fromkeys(retrieve, 0.0)

        def score(unit: list[Event]) -> float:
            return max((prio[ev.seq] for ev in unit if ev.seq in prio), default=float("-inf"))

        def cost(unit: list[Event]) -> int:
            return sum(count_blocks(self.tok, ev.blocks) for ev in unit)

        budget = self.budget.tail_budget
        used = sum(cost(u) for u in chosen)
        wanted = sorted(
            ((score(u), i) for i, u in enumerate(evicted) if score(u) >= threshold),
            key=lambda t: (-t[0], -t[1]),
        )
        chosen = list(chosen)
        dropped: list[list[Event]] = []
        taken: set[int] = set()
        for p, i in wanted:
            need = cost(evicted[i])
            free = budget - used
            out: list[list[Event]] = []
            for u in chosen[: max(0, len(chosen) - self.budget.protect_recent)]:
                if free >= need:
                    break
                if score(u) < p:  # only displace what the ranking trusts less
                    out.append(u)
                    free += cost(u)
            if free < need:
                continue  # would not fit without touching what it may not touch
            for u in out:
                chosen.remove(u)
                dropped.append(u)
            used = budget - free + need
            taken.add(i)

        recalled = [u for i, u in enumerate(evicted) if i in taken]
        rest = [u for i, u in enumerate(evicted) if i not in taken] + dropped
        rest.sort(key=lambda u: u[0].seq)
        return chosen, recalled, rest

    def _inline(
        self,
        zones: list[Zone],
        inline: Mapping[str, float] | Collection[str],
        read_full: Callable[[str], str | None],
        focus: Mapping[str, Collection[str]],
    ) -> None:
        """Replace chosen handles' extracts with their full content, most
        wanted first, within the page-in allowance. The block stays a result of
        the same ``tool_use``, so pairing is untouched."""
        prio = inline if isinstance(inline, Mapping) else dict.fromkeys(inline, 0.0)
        spots = [
            (prio[blk.handle_id], z, j)
            for z in zones
            for j, (_, blk) in enumerate(z.items)
            if isinstance(blk, HandleRef) and blk.handle_id in prio
        ]
        allowance = int(self.budget.target * self.budget.inline_frac)
        spent = 0
        for _, z, j in sorted(spots, key=lambda t: -t[0]):
            role, blk = z.items[j]
            full = read_full(blk.handle_id)
            if not full:
                continue
            cap = self.budget.inline_item_tokens
            if self.tok.count_text(full) > cap:
                full = self._excerpt(full, focus.get(blk.handle_id, ()), cap, blk.handle_id)
            extra = self.tok.count_text(full) - count_blocks(self.tok, [blk])
            if spent + extra > allowance:
                continue
            z.items[j] = (role, ToolResult(blk.tool_use_id, full, blk.is_error))
            spent += extra

    def _excerpt(self, full: str, names: Collection[str], cap: int, handle_id: str) -> str:
        """Fit ``full`` into ``cap`` tokens: the lines around ``names`` first,
        then from the top. Line numbers are kept so the agent can ask for more
        by range."""
        lines = full.splitlines()
        wanted: list[int] = []
        for i, ln in enumerate(lines):
            if any(n in ln for n in names):
                wanted.extend(range(max(0, i - 3), min(len(lines), i + 4)))
        order = list(dict.fromkeys([*wanted, *range(len(lines))]))
        note = f"[excerpt of {len(lines)} lines; ctx_expand {handle_id} with lines or grep for more]"
        budget = cap - self.tok.count_text(note)
        picked: list[int] = []
        used = 0
        for i in order:
            cost = self.tok.count_text(lines[i]) + 1
            if used + cost > budget:
                if i in wanted:
                    continue
                break
            picked.append(i)
            used += cost
        body = "\n".join(f"{i + 1}: {lines[i]}" for i in sorted(picked))
        return f"{note}\n{body}"

    def _render_tail(
        self, chosen: list[list[Event]], evicted: list[list[Event]]
    ) -> list[Item]:
        items: list[Item] = []
        if evicted:
            elided_tokens = sum(
                count_blocks(self.tok, ev.blocks) for unit in evicted for ev in unit
            )
            # One coalesced marker. An LLM cannot page-fault mid-pass, so this
            # stands in for the fault: it makes the absence visible.
            items.append(
                (
                    "user",
                    Elision(
                        handle_id=None,
                        note=(
                            f"[{len(evicted)} earlier exchanges elided (~{elided_tokens} tok). "
                            "Nothing was deleted — use expand(handle_id) or search_log() "
                            "to retrieve any of it.]"
                        ),
                        tokens_elided=elided_tokens,
                    ),
                )
            )
        for unit in chosen:
            items.extend(_flatten(unit))
        return items


def _flatten(events: list[Event]) -> list[Item]:
    out: list[Item] = []
    for ev in events:
        role = role_for(ev.kind)
        out.extend((role, b) for b in ev.blocks)
    return out

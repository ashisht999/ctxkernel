"""The facade: record events, assemble a view, dereference handles."""

from __future__ import annotations

import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .assembler import Assembler, Assembly, Budget
from .codecs import CodecRegistry
from .identity import IdentityRegistry, ToolSemantics
from .ir import (
    Block,
    Event,
    EventKind,
    Handle,
    HandleRef,
    ResourceId,
    Text,
    ToolResult,
    ToolUse,
    digest_of,
)
from .store import SessionStore
from .tokenizer import Tokenizer, default_tokenizer

#: Results at or above this many tokens are externalized rather than admitted.
#: The cheapest removal is non-admission.
DEFAULT_HANDLE_THRESHOLD = 1_000


@dataclass(slots=True)
class TaskFrame:
    id: str
    intent: str
    parent: str | None
    _engine: "ContextEngine"
    _closed: bool = False

    def done(self, outcome: str) -> None:
        """Close the frame. Its interior detail becomes compactable; the
        outcome is what survives."""
        self._engine._close_task(self.id, outcome, status="done")
        self._closed = True

    def abandon(self, reason: str) -> None:
        self._engine._close_task(self.id, reason, status="abandoned")
        self._closed = True


class ContextEngine:
    def __init__(
        self,
        *,
        tenant: str = "default",
        session: str | None = None,
        root: str | Path = ".ctx",
        budget: Budget | None = None,
        tokenizer: Tokenizer | None = None,
        handle_threshold: int = DEFAULT_HANDLE_THRESHOLD,
    ) -> None:
        # When identity is uncertain, split rather than merge: a fresh session
        # leaks nothing and costs only continuity, which resource-anchored
        # facts recover. Merging two users' work on a guess is unrecoverable.
        self.session_id = session or uuid.uuid4().hex[:16]
        self.tenant = tenant
        self.store = SessionStore.for_session(root, tenant, self.session_id)
        self.tok = tokenizer or default_tokenizer()
        self.assembler = Assembler(budget, self.tok)
        self.codecs = CodecRegistry()
        self.identity = IdentityRegistry()
        self.handle_threshold = handle_threshold

        self._pending_calls: dict[str, tuple[str, ToolSemantics]] = {}
        self._task_stack: list[TaskFrame] = []
        self._ingested = 0
        self._call_seq = 0
        self._last_report: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Tier 0
    # ------------------------------------------------------------------

    def wrap(self, client: Any) -> Any:
        """Wrap a provider client. One line, host loop unchanged.

            client = engine.wrap(anthropic.Anthropic())
        """
        from .adapters.anthropic import WrappedClient

        return WrappedClient(client, self)

    def ingest_messages(self, messages: list[dict[str, Any]]) -> int:
        """Record whatever the host appended since we last looked.

        The host keeps building its own message list; we read the tail of it
        and turn it into events. Fat tool results are intercepted on the way
        through, so they never occupy a context slot even at Tier 0.
        """
        from .adapters.anthropic import parse_content

        n = 0
        for msg in messages[self._ingested :]:
            role = msg.get("role")
            blocks = parse_content(msg.get("content"))
            if role == "assistant":
                self.record_model(blocks)
                n += 1
                continue
            # A user turn is either human text or tool results coming back.
            plain: list[Block] = []
            for b in blocks:
                if isinstance(b, ToolResult):
                    self.record_tool_result(b.tool_use_id, b.content, is_error=b.is_error)
                    n += 1
                else:
                    plain.append(b)
            if plain:
                text = "\n".join(getattr(b, "text", "") for b in plain).strip()
                if text:
                    self.record_user(text)
                    n += 1
        self._ingested = len(messages)
        return n

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _append(self, ev: Event) -> Event:
        if self._task_stack and ev.task_id is None:
            ev.task_id = self._task_stack[-1].id
        return self.store.append(ev)

    def set_goal(self, statement: str, acceptance: list[str] | None = None) -> Event:
        """Record the goal *verbatim*.

        Never paraphrased: any restatement is an interpretation, and that is
        where drift begins. Acceptance criteria should be checkable commands,
        not prose -- a criterion you cannot evaluate is one you will drift from
        without noticing.
        """
        text = statement
        if acceptance:
            text += "\n\nAcceptance:\n" + "\n".join(f"  - {a}" for a in acceptance)
        return self._append(
            Event(
                seq=-1,
                kind=EventKind.GOAL,
                blocks=[Text(text)],
                meta={"acceptance": acceptance or []},
            )
        )

    def record_user(self, text: str) -> Event:
        return self._append(Event(seq=-1, kind=EventKind.USER_MESSAGE, blocks=[Text(text)]))

    def record_model(self, blocks: list[Block]) -> Event:
        ev = self._append(Event(seq=-1, kind=EventKind.MODEL_OUTPUT, blocks=list(blocks)))
        for b in blocks:
            if isinstance(b, ToolUse):
                sem = self.identity.resolve(b.name, b.input)
                self._pending_calls[b.id] = (b.name, sem)
        return ev

    def record_tool_call(self, tool_use_id: str, name: str, args: dict[str, Any]) -> Event:
        sem = self.identity.resolve(name, args)
        self._pending_calls[tool_use_id] = (name, sem)
        return self._append(
            Event(
                seq=-1,
                kind=EventKind.TOOL_CALL,
                blocks=[ToolUse(tool_use_id, name, args)],
                resource=sem.resource,
                is_write=sem.is_write,
            )
        )

    def record_tool_result(
        self,
        tool_use_id: str,
        content: str,
        *,
        is_error: bool = False,
        derived_from: tuple[ResourceId, ...] | None = None,
    ) -> Event:
        """Intercept a tool result before it can ever occupy the context.

        Over the threshold, the full bytes go to the blob store and only a
        computed extract plus a handle enter the view.
        """
        name, sem = self._pending_calls.pop(tool_use_id, ("", ToolSemantics(None)))
        digest = digest_of(content)
        tokens = self.tok.count_text(content)

        # A read is derived from what it read, so a later write to that
        # resource invalidates it without anyone having to notice.
        derived = derived_from
        if derived is None:
            derived = (sem.resource,) if sem.resource and not sem.is_write else ()

        if tokens >= self.handle_threshold and not is_error:
            extract, media = self.codecs.encode(content, sem.resource, name)
            handle = self.store.put_handle(
                content, media_type=media, extract=extract, resource=sem.resource
            )
            block: Block = HandleRef(
                tool_use_id=tool_use_id,
                handle_id=handle.id,
                extract=extract,
                media_type=media,
                size_bytes=handle.size_bytes,
                original_tokens=tokens,
                is_error=is_error,
            )
        else:
            block = ToolResult(tool_use_id, content, is_error)

        return self._append(
            Event(
                seq=-1,
                kind=EventKind.TOOL_RESULT,
                blocks=[block],
                resource=sem.resource,
                digest=digest,
                derived_from=derived,
                is_write=sem.is_write,
            )
        )

    def tool(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        result: str = "",
        *,
        is_error: bool = False,
    ) -> Event:
        """Record a whole tool round-trip in one call.

        The everyday entry point. Pairing a call with its result by hand is
        bookkeeping the library should be doing, not you::

            engine.tool("read_file", {"path": "src/auth.py"}, contents)
        """
        self._call_seq += 1
        tid = f"c{self._call_seq:05d}"
        self.record_tool_call(tid, name, args or {})
        return self.record_tool_result(tid, result, is_error=is_error)

    def messages(self, fmt: str = "anthropic") -> Any:
        """The assembled context, ready to send.

        ``fmt`` is ``"anthropic"``, ``"openai"`` (also vLLM / Ollama / TGI and
        any OpenAI-compatible server), ``"chat"`` (plain role/content dicts) or
        ``"text"`` (one prompt string for completion endpoints).
        """
        items = self.assemble().items
        if fmt == "anthropic":
            from .adapters.anthropic import to_messages

            return to_messages(items)
        if fmt == "openai":
            from .adapters.openai import to_messages

            return to_messages(items)
        if fmt == "chat":
            from .adapters.generic import to_chat

            return to_chat(items)
        if fmt == "text":
            from .adapters.generic import to_text

            return to_text(items)
        raise ValueError(f"unknown fmt: {fmt!r} — use anthropic|openai|chat|text")

    # -- the irreproducible set -------------------------------------------

    def note_failure(self, attempted: str, why: str, do_not_retry_unless: str = "") -> Event:
        """Record a dead end. Never evicted.

        The single highest value-per-token item in the system: a few hundred
        tokens that stop the same wall being rediscovered at steps 80, 200 and
        350. Re-running the world will not tell the agent this -- it is history,
        not state.
        """
        text = f"TRIED: {attempted}\n  FAILED: {why}"
        if do_not_retry_unless:
            text += f"\n  RETRY ONLY IF: {do_not_retry_unless}"
        return self._append(Event(seq=-1, kind=EventKind.FAILURE, blocks=[Text(text)]))

    def note_invariant(self, claim: str, *, valid_unless: str = "") -> Event:
        text = f"- {claim}"
        if valid_unless:
            text += f"  (valid unless: {valid_unless})"
        return self._append(
            Event(
                seq=-1,
                kind=EventKind.INVARIANT,
                blocks=[Text(text)],
                meta={"valid_unless": valid_unless},
            )
        )

    # ------------------------------------------------------------------
    # Task frames
    # ------------------------------------------------------------------

    @contextmanager
    def task(self, intent: str) -> Iterator[TaskFrame]:
        """A scope whose interior becomes compactable once it closes.

        Compaction fires at the boundary rather than under token pressure: at
        close, the outcome subsumes the trace, whereas mid-task the discarded
        material is still live.
        """
        tid = uuid.uuid4().hex[:12]
        parent = self._task_stack[-1].id if self._task_stack else None
        frame = TaskFrame(tid, intent, parent, self)
        begin = self.store.next_seq()
        self.store.open_task(tid, intent, parent, begin)
        self._append(
            Event(seq=-1, kind=EventKind.TASK_BEGIN, blocks=[Text(f"▶ {intent}")], task_id=tid)
        )
        self._task_stack.append(frame)
        try:
            yield frame
        finally:
            self._task_stack.pop()
            if not frame._closed:
                # An unclosed frame stays open on purpose: we cannot claim an
                # outcome we never observed, so its trace is not compactable.
                pass

    def _close_task(self, task_id: str, outcome: str, status: str) -> None:
        ev = self._append(
            Event(
                seq=-1,
                kind=EventKind.TASK_END,
                blocks=[Text(f"■ {outcome}")],
                task_id=task_id,
            )
        )
        self.store.close_task(task_id, outcome, ev.seq, status=status)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def assemble(self) -> Assembly:
        events = self.store.events()
        return self.assembler.assemble(events, closed_tasks=self.store.closed_task_ids())

    def expand(
        self,
        handle_id: str,
        *,
        symbol: str | None = None,
        lines: str | None = None,
        grep: str | None = None,
        max_chars: int = 8_000,
    ) -> str:
        """Dereference a handle. This is the page-in.

        Scoped by construction: a handle id from another session is simply not
        in this session's table, so it cannot resolve.
        """
        content = self.store.read_handle(handle_id)
        if content is None:
            return f"[no such handle in this session: {handle_id}]"

        if symbol:
            return _slice_symbol(content, symbol, max_chars)
        if lines:
            return _slice_lines(content, lines, max_chars)
        if grep:
            return _grep(content, grep, max_chars)
        return content[:max_chars] + ("…" if len(content) > max_chars else "")

    def search_log(self, pattern: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Grep the agent's own history.

        The cascade converts *remembered* into *findable*; this is the half that
        makes findable real. It does not close the gap -- if the agent never
        thinks to look, it still loses.
        """
        rx = re.compile(pattern, re.I)
        hits: list[dict[str, Any]] = []
        for ev in self.store.iter_events():
            for b in ev.blocks:
                if isinstance(b, ToolResult):
                    text = b.content
                elif isinstance(b, HandleRef):
                    # Search the externalized blob, not its extract. Searching
                    # only the extract would make this useless for exactly the
                    # case it exists for: a detail buried in a big output that
                    # nobody knew to keep.
                    text = self.store.read_handle(b.handle_id) or b.extract
                else:
                    text = getattr(b, "text", None) or ""
                m = rx.search(text)
                if m:
                    lo = max(0, m.start() - 80)
                    hits.append(
                        {
                            "seq": ev.seq,
                            "kind": ev.kind.value,
                            "resource": str(ev.resource) if ev.resource else None,
                            "snippet": text[lo : m.end() + 80].replace("\n", " "),
                        }
                    )
                    break
            if len(hits) >= limit:
                break
        return hits

    def report(self) -> dict[str, Any]:
        a = self.assemble()
        r = a.report()
        r["events"] = self.store.count()
        r["session"] = self.session_id
        return r

    def preview(self, *, width: int = 78, max_chars_per_zone: int = 1400) -> str:
        """Render the assembled context as readable text.

        This is what the model will actually see, zone by zone, with the cache
        breakpoint marked. Users will not trust an opaque thing that silently
        drops their agent's history -- so it is inspectable by default.
        """
        from .assembler import CACHE_BREAKPOINT_AFTER

        a = self.assemble()
        out: list[str] = []
        for z in a.zones:
            if not z.items:
                out.append(f"┌─ {z.name} " + "─" * max(0, width - len(z.name) - 12) + " empty ─┐")
                continue
            head = f"┌─ {z.name} "
            tail = f" {z.tokens:,} tok ─┐"
            out.append(head + "─" * max(0, width - len(head) - len(tail)) + tail)
            body = "\n".join(_render_text(b) for _, b in z.items).rstrip()
            if len(body) > max_chars_per_zone:
                cut = len(body) - max_chars_per_zone
                body = body[:max_chars_per_zone] + f"\n… [{cut:,} chars not shown in this preview]"
            out += [f"│ {ln[: width - 2]}" for ln in body.splitlines()]
            out.append("└" + "─" * width)
            if z.name == CACHE_BREAKPOINT_AFTER:
                out.append("")
                out.append("  ═══ cache breakpoint ═══  everything above is a stable prefix")
                out.append("")

        r = a.report()
        out.append("")
        out.append(
            f"  {r['assembled_tokens']:,} tokens assembled from "
            f"{r['raw_tokens']:,} raw ({r['saved_pct']}% saved) · "
            f"{r['cache_prefix_tokens']:,} cacheable"
        )
        return "\n".join(out)

    def explain(self) -> str:
        """Why each item is or is not in context. Not a debugging aid -- users
        will not trust an opaque thing that silently drops their agent's
        history, and they are right not to."""
        a = self.assemble()
        by_seq = {d.seq: d for d in a.decisions}
        out = [f"session {self.session_id} — {self.store.count()} events"]
        for ev in self.store.events():
            d = by_seq.get(ev.seq)
            if d is None:
                continue
            mark = "keep" if d.keep else "drop"
            res = f" {ev.resource}" if ev.resource else ""
            detail = f" — {d.detail}" if d.detail else ""
            out.append(f"  {ev.id} {mark:4} {d.reason.value:<13}{res}{detail}")
        r = a.report()
        out.append(
            f"\n  {r['raw_tokens']} raw → {r['assembled_tokens']} assembled "
            f"({r['saved_pct']}% saved), cache prefix {r['cache_prefix_tokens']}"
        )
        for n in a.notes:
            out.append(f"  ! {n}")
        return "\n".join(out)

    def close(self) -> None:
        self.store.close()


# --------------------------------------------------------------------------
# expand() slicing helpers
# --------------------------------------------------------------------------


def _render_text(block: Block) -> str:
    """Human-readable rendering of a block, for preview() only."""
    from .ir import Elision, Thinking

    if isinstance(block, (Text, Thinking)):
        return block.text
    if isinstance(block, Elision):
        return block.note
    if isinstance(block, ToolUse):
        return f"→ {block.name}({_short_args(block.input)})"
    if isinstance(block, ToolResult):
        flag = "ERROR " if block.is_error else ""
        return f"← {flag}{block.content}"
    if isinstance(block, HandleRef):
        return (
            f"← [handle {block.handle_id} · was {block.original_tokens:,} tok]\n"
            + block.extract
        )
    return repr(block)


def _short_args(args: dict[str, Any], limit: int = 60) -> str:
    s = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _slice_lines(content: str, spec: str, max_chars: int) -> str:
    lines = content.splitlines()
    m = re.match(r"(\d+)\s*-\s*(\d+)", spec)
    if not m:
        return content[:max_chars]
    lo, hi = int(m.group(1)), int(m.group(2))
    return "\n".join(lines[max(0, lo - 1) : hi])[:max_chars]


def _slice_symbol(content: str, symbol: str, max_chars: int) -> str:
    """Best-effort symbol extraction: from the definition line to the next
    line at the same or lower indentation."""
    lines = content.splitlines()
    start = None
    indent = 0
    pat = re.compile(rf"^(\s*)(?:.*\b)?(?:def|class|func|fn|function|type|struct)\s+{re.escape(symbol)}\b")
    for i, ln in enumerate(lines):
        m = pat.match(ln)
        if m:
            start = i
            indent = len(m.group(1))
            break
    if start is None:
        return f"[symbol not found: {symbol}]"
    end = len(lines)
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if ln.strip() and len(ln) - len(ln.lstrip()) <= indent:
            end = j
            break
    return "\n".join(lines[start:end])[:max_chars]


def _grep(content: str, pattern: str, max_chars: int) -> str:
    rx = re.compile(pattern, re.I)
    out = [f"{i + 1}: {ln}" for i, ln in enumerate(content.splitlines()) if rx.search(ln)]
    return "\n".join(out)[:max_chars] or f"[no match: {pattern}]"

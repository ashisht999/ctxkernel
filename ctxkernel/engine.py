"""The facade: record events, assemble a view, dereference handles."""

from __future__ import annotations

import re
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .assembler import Assembler, Assembly, Budget
from .codecs import CodecRegistry
from .decision import (
    FAILED,
    JUDGMENTS,
    fidelity_items,
    Anchor,
    AnchorRun,
    DecisionModel,
    NullDecision,
    candidate_for,
    choose,
    failure_line,
    gather,
)
from .codecs import name_ranks
from .graph import Graph, event_text
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
from .predicates import decide
from .store import SessionStore
from .tools import NAMES as AGENT_TOOLS
from .tokenizer import Tokenizer, count_blocks, default_tokenizer

#: Results at or above this many tokens are externalized rather than admitted.
#: The cheapest removal is non-admission.
DEFAULT_HANDLE_THRESHOLD = 1_000

#: Share of the target kept for the recent tail when a decision model ranks
#: the rest.
RANKED_TAIL_FRAC = 0.6


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
        decision: DecisionModel | None = None,
        anchor_threshold: float | None = None,
        anchor_k: int = 50,
        anchor_candidates: int = 50,
        anchor_timeout: float = 0.5,
        auto_anchor: bool = True,
        auto_failures: bool = True,
        failure_threshold: float = 0.8,
    ) -> None:
        # When identity is uncertain, split rather than merge: a fresh session
        # leaks nothing and costs only continuity, which resource-anchored
        # facts recover. Merging two users' work on a guess is unrecoverable.
        self.session_id = session or uuid.uuid4().hex[:16]
        self.tenant = tenant
        self.store = SessionStore.for_session(root, tenant, self.session_id)
        self.tok = tokenizer or default_tokenizer()
        # With a decision model, part of the target moves from the recent tail
        # to what the model ranks; 60/40 measured best on real sessions. An
        # explicit budget is always respected.
        self.decision: DecisionModel = decision or NullDecision()
        if budget is None and not isinstance(self.decision, NullDecision):
            # A model that swaps keeps recency as the floor over the whole
            # target; one that only ranks gets a split to fill.
            swaps = getattr(self.decision, "swap_threshold", None) is not None
            budget = Budget(tail_frac=1.0 if swaps else RANKED_TAIL_FRAC)
        self.assembler = Assembler(budget, self.tok)
        self.codecs = CodecRegistry()
        self.identity = IdentityRegistry()
        self.handle_threshold = handle_threshold

        # The graph is derived from the log, so a reopened session rebuilds it
        # rather than trusting a second copy that could have drifted.
        self.graph = Graph.from_events(
            self.store.events(), self._tokens,
            parents=self.store.task_parents(), names=self._names,
        )

        # Decision model: off the hot path. It re-ranks in the background after
        # each tool result; a turn uses whatever ranking has landed. The cut-off
        # is the model's own unless overridden: a calibrated probability can
        # say "not needed", an uncalibrated prior can only order.
        self._threshold_override = anchor_threshold
        self.anchor_k = anchor_k
        self.anchor_candidates = anchor_candidates
        self.anchor_timeout = anchor_timeout
        self.auto_anchor = auto_anchor
        self._anchors = [Anchor(n, p) for n, p in self.store.current_anchors()]
        self._anchor_lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None
        self._inflight: Future[AnchorRun] | None = None
        self._last_run_seq = -(10**9)

        # Failures the judge found, waiting to be recorded on the caller's
        # thread: the worker may judge, but only the caller writes the log, so
        # the graph is never written from two threads at once.
        self.auto_failures = auto_failures
        self.failure_threshold = failure_threshold
        self._judged: set[str] = set()
        self._found: list[tuple[str, str, str, float]] = []
        #: Handles whose full content replaces their extract: handle id → score,
        #: and the names that made the case (what a cut page-in must keep).
        self._inline: dict[str, float] = {}
        self._inline_focus: dict[str, tuple[str, ...]] = {}

        self._pending_calls: dict[str, tuple[str, ToolSemantics]] = {}
        self._task_stack: list[TaskFrame] = []
        # Both counters continue from the log, not from zero: a resumed
        # session that restarted them would reissue tool-use ids (pairing the
        # wrong call with a result) and re-record the host's whole history.
        self._ingested = int(self.store.get_meta("ingested", "0") or 0)
        self._call_seq = _last_call_seq(self.store.events())
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
        self.store.set_meta("ingested", str(self._ingested))
        return n

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _append(self, ev: Event) -> Event:
        if self._task_stack and ev.task_id is None:
            ev.task_id = self._task_stack[-1].id
        ev = self.store.append(ev)
        self.graph.add_event(ev, self._tokens(ev), self._names(ev))
        return ev

    def _tokens(self, ev: Event) -> int:
        return count_blocks(self.tok, ev.blocks)

    def _names(self, ev: Event) -> set[str]:
        # Read through handles: the names that matter most are the ones the
        # extract had to cut.
        return set(name_ranks(event_text(ev, self.store.read_handle)))

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
        ev = self._append(
            Event(
                seq=-1,
                kind=EventKind.GOAL,
                blocks=[Text(text)],
                meta={"acceptance": acceptance or []},
            )
        )
        self._maybe_reanchor()
        return ev

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

        # The agent's own history tools are exempt: their answer *is* the page-in,
        # and intercepting it would hand the agent back the extract it asked
        # to see past.
        if tokens >= self.handle_threshold and not is_error and name not in AGENT_TOOLS:
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

        ev = self._append(
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
        self._maybe_reanchor()
        return ev

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

    # -- the agent's own history tools --------------------------------------

    def tools(self, fmt: str = "anthropic") -> list[dict[str, Any]]:
        """Tool definitions for ``ctx_outline``, ``ctx_search`` and ``ctx_expand``,
        ready to add to the agent's tool list. With them a ranking miss costs
        the agent one tool call, not a wrong decision."""
        from .tools import SPECS

        if fmt == "anthropic":
            from .adapters.anthropic import tool_specs

            return tool_specs(SPECS)
        if fmt == "openai":
            from .adapters.openai import tool_specs

            return tool_specs(SPECS)
        raise ValueError(f"unknown fmt: {fmt!r} — use anthropic|openai")

    def run_tool(self, name: str, args: dict[str, Any] | None = None) -> str | None:
        """Answer a call to one of the history tools, or ``None`` if ``name``
        is not one of them, so a host dispatches ours first and then its own::

            out = eng.run_tool(call.name, call.input)
            if out is None:
                out = my_tools(call)
            eng.tool(call.name, call.input, out)
        """
        from .tools import run

        return run(self, name, args or {})

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
            Event(
                seq=-1,
                kind=EventKind.TASK_BEGIN,
                blocks=[Text(f"▶ {intent}")],
                task_id=tid,
                meta={"parent": parent},
            )
        )
        self._task_stack.append(frame)
        self._maybe_reanchor()
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
        self._record_found()
        events = self.store.events()
        retrieve = self.graph.expand_scored((a.node_id, a.p) for a in self.anchors)
        with self._anchor_lock:
            inline, focus = dict(self._inline), dict(self._inline_focus)
        return self.assembler.assemble(
            events,
            closed_tasks=self.store.closed_task_ids(),
            retrieve=retrieve,
            swap_threshold=getattr(self.decision, "swap_threshold", None),
            inline=inline,
            read_full=self.store.read_handle,
            inline_focus=focus,
        )

    # ------------------------------------------------------------------
    # Anchors
    # ------------------------------------------------------------------

    @property
    def anchor_threshold(self) -> float:
        # Read from the current model each time, so swapping models swaps the
        # cut-off with them.
        if self._threshold_override is not None:
            return self._threshold_override
        return float(getattr(self.decision, "threshold", 0.5))

    @property
    def anchors(self) -> list[Anchor]:
        with self._anchor_lock:
            return list(self._anchors)

    def reanchor(
        self, *, wait: bool = True, timeout: float | None = None
    ) -> AnchorRun | Future[AnchorRun]:
        """Ask the decision model where the work is.

        The candidate set is built here, on the caller's thread, from the graph
        as it stands; only the model call runs in the background, so the graph
        is never read while it is being written. With ``wait=False`` this
        returns at once and the anchors land when the model answers. With
        ``wait=True`` it waits up to ``timeout`` and, past that, reports a
        timeout while the late answer still lands when it arrives.
        """
        events = self.store.events()
        decisions = decide(events, closed_tasks=self.store.closed_task_ids())
        task = " > ".join(f.intent for f in self._task_stack)
        state, cands = gather(
            self.graph, events, decisions, task=task, cap=self.anchor_candidates
        )
        seq = events[-1].seq if events else 0
        self._last_run_seq = seq

        # The latest step goes to the judge once, built here on the caller's
        # thread like the candidates -- and only once its result is in: a call
        # judged before it has an outcome would be judged on nothing, and then
        # never again.
        step = None
        if self.auto_failures and hasattr(self.decision, "judge"):
            latest = self.graph.recent_calls(1)
            if latest and len(latest[0].seqs) >= 2 and latest[0].id not in self._judged:
                self._judged.add(latest[0].id)
                step = candidate_for(self.graph, latest[0].id, now=seq)

        # Large outputs worth a summary-or-full question: the recent ones and
        # the likeliest candidates.
        full_items: list[Any] = []
        handles_of: dict[str, list[str]] = {}
        if hasattr(self.decision, "fidelity"):
            by_seq = {ev.seq: ev for ev in events}
            pool = [n.id for n in self.graph.recent_calls(6)] + [c.node_id for c in cands[:10]]
            extracts: dict[str, str] = {}
            for nid in dict.fromkeys(pool):
                node = self.graph.nodes.get(nid)
                refs = [b for q in (node.seqs if node else []) if q in by_seq
                        for b in by_seq[q].blocks if isinstance(b, HandleRef)]
                if refs:
                    extracts[nid] = "\n".join(r.extract for r in refs)
                    handles_of[nid] = [r.handle_id for r in refs]
            full_items = fidelity_items(self.graph, list(extracts), extracts, state, now=seq)

        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ctxkernel-anchor")
        fut = self._pool.submit(
            self._run_and_commit, state, cands, seq, step, full_items, handles_of
        )
        self._inflight = fut
        if not wait:
            return fut
        try:
            return fut.result(timeout=self.anchor_timeout if timeout is None else timeout)
        except FutureTimeout:
            return AnchorRun(self.decision.name, seq, [], candidates=len(cands), fallback="timeout")

    def settle(self, timeout: float | None = None) -> None:
        """Bring the ranking up to date with the log, waiting if needed.

        For evals and tests, which need a deterministic view. A live agent
        never calls this -- not waiting is the whole point of running the
        model in the background -- so an eval that settles measures the
        ranking at its best, one step fresher than a live agent may see it.
        """
        fut = self._inflight
        if fut is not None:
            fut.result(timeout=timeout)
        if isinstance(self.decision, NullDecision):
            return
        latest = self.store.next_seq() - 1
        if latest > self._last_run_seq:
            self.reanchor(wait=True, timeout=timeout if timeout is not None else 60.0)
        self._record_found()

    def _run_and_commit(
        self,
        state: Any,
        cands: list[Any],
        seq: int,
        step: Any = None,
        full_items: list[Any] | None = None,
        handles_of: dict[str, list[str]] | None = None,
    ) -> AnchorRun:
        # Commit inside the worker, not in a done-callback: a callback can run
        # after result() has already returned, and then a caller that waited
        # for the answer would still read the old anchors.
        run = choose(
            self.decision, state, cands, seq=seq, threshold=self.anchor_threshold, k=self.anchor_k
        )
        if step is not None:
            try:
                p = float(self.decision.judge(state, step, {FAILED: JUDGMENTS[FAILED]})[FAILED])  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 -- a broken judge must never break the turn
                p = None
            if p is not None:
                # Kept with the run's scores, so the judge's calibration can be
                # checked like the ranker's.
                run.scores[f"{FAILED}@{step.node_id}"] = p
                if p >= self.failure_threshold:
                    with self._anchor_lock:
                        self._found.append(
                            (step.node_id, step.label, failure_line(step.content), p)
                        )
        if full_items:
            try:
                wants = self.decision.fidelity(state, full_items)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 -- keep the previous choice
                wants = None
            if wants is not None:
                cut = float(getattr(self.decision, "full_threshold", 0.7))
                inline = {
                    hid: float(p)
                    for nid, p in wants.items()
                    if float(p) >= cut
                    for hid in (handles_of or {}).get(nid, [])
                }
                hidden = {c.node_id: c.hidden for c in full_items}
                focus = {
                    hid: hidden.get(nid, ())
                    for nid, hids in (handles_of or {}).items()
                    for hid in hids
                    if hid in inline
                }
                for nid, p in wants.items():
                    run.scores[f"full@{nid}"] = float(p)
                with self._anchor_lock:
                    self._inline, self._inline_focus = inline, focus
        self.store.record_anchor_run(
            seq=run.seq,
            model=run.model,
            ms=run.ms,
            candidates=run.candidates,
            anchors=[(a.node_id, a.p) for a in run.anchors],
            scores=run.scores,
            fallback=run.fallback,
        )
        if run.fallback is None:
            with self._anchor_lock:
                self._anchors = run.anchors
        return run

    def _record_found(self) -> None:
        """Record the failures the judge found, as ordinary pinned failure
        notes. The text is quoted from the step itself -- what was tried and
        the line that says it failed -- never written by a model."""
        with self._anchor_lock:
            found, self._found = self._found, []
        for nid, attempted, why, p in found:
            text = f"TRIED: {attempted}\n  FAILED: {why}"
            self._append(
                Event(
                    seq=-1,
                    kind=EventKind.FAILURE,
                    blocks=[Text(text)],
                    meta={"auto": True, "p": round(p, 3), "node": nid, "judge": self.decision.name},
                )
            )

    def _maybe_reanchor(self) -> None:
        self._record_found()
        if not self.auto_anchor or isinstance(self.decision, NullDecision):
            return
        if self._inflight is not None and not self._inflight.done():
            return  # one decision at a time; the next trigger will catch up
        self.reanchor(wait=False)

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
        r["decision"] = self.decision.name
        r["anchors"] = [(a.node_id, round(a.p, 3)) for a in self.anchors]
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
        anchors = self.anchors
        if anchors:
            out.append(f"\n  anchors ({self.decision.name}) — re-admitted into 'retrieved':")
            for an in anchors:
                node = self.graph.nodes.get(an.node_id)
                label = node.label if node else "?"
                out.append(f"    {an.p:.2f}  {an.node_id:<24} {label[:60]}")
        for n in a.notes:
            out.append(f"  ! {n}")
        return "\n".join(out)

    def close(self) -> None:
        if self._pool is not None:
            # Let an in-flight decision land before the store closes under it.
            self._pool.shutdown(wait=True)
        self.store.close()


def _last_call_seq(events: list[Event]) -> int:
    """Highest ``cNNNNN`` id that ``tool()`` has issued in this log."""
    last = 0
    for ev in events:
        for b in ev.blocks:
            if isinstance(b, ToolUse) and re.fullmatch(r"c\d{5,}", b.id):
                last = max(last, int(b.id[1:]))
    return last


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

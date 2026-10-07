"""The memory graph: a derived index over the log.

The log stays the source of truth. The graph is rebuilt from it when a session
opens and extended on every append, so it can never disagree with what
happened -- a graph that could drift from the log would be a second, weaker
copy of the history rather than an index into it.

Building it takes no judgment. A node exists for anything that can be pointed
at or re-rendered; an edge exists only where a field on the event already says
so (``task_id``, ``resource``, ``derived_from``, ``digest``, a handle). Deciding
what *matters* is a separate step with a separate owner -- see ``decision.py``
-- so a wrong opinion there can mis-rank a node but never invent a connection.

Two layers share the same nodes:

* **Containment** (``CONTAINS``) is a tree: session → task → call → handle.
  It is what an agent navigates, the way a reader uses a table of contents.
* **Links** (reads, writes, derived_from, replaces, duplicate_of) cross the
  tree. They are what relevance travels along: everything that touched
  ``file:auth.py`` is one hop from the ``file:auth.py`` node.

A third kind of connection is too dense to store as edges: **mentions**. Every
node is indexed by the names in its full content (paths, symbols, settings), so
the current work can reach any old node that shares a rare name with it. That
index exists because the typed links were measured to be too short: on real
sessions, in most steps that lost something the agent needed, the needed item
was not within two links of the recent work at all.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable

from .codecs import name_ranks
from .ir import (
    Event,
    EventKind,
    HandleRef,
    ResourceId,
    Text,
    Thinking,
    ToolResult,
    ToolUse,
)


class NodeKind(str, Enum):
    SESSION = "session"
    TASK = "task"
    CALL = "call"
    RESOURCE = "resource"
    HANDLE = "handle"
    GOAL = "goal"
    USER = "user"
    MODEL = "model"
    NOTE = "note"
    FAILURE = "failure"
    INVARIANT = "invariant"


class EdgeKind(str, Enum):
    CONTAINS = "contains"
    READS = "reads"
    WRITES = "writes"
    DERIVED_FROM = "derived_from"
    REPLACES = "replaces"
    DUPLICATE_OF = "duplicate_of"
    HAS_HANDLE = "has_handle"


#: The root every uncontained node hangs from.
ROOT = "session"

#: How each link reads from either end. Containment is structure, not
#: meaning, so it is not described.
_OUTGOING = {
    EdgeKind.READS: "reads",
    EdgeKind.WRITES: "writes",
    EdgeKind.DERIVED_FROM: "derived from",
    EdgeKind.REPLACES: "replaces",
    EdgeKind.DUPLICATE_OF: "same output as",
}
_INCOMING = {
    EdgeKind.READS: "read by",
    EdgeKind.WRITES: "written by",
    EdgeKind.DERIVED_FROM: "used by",
    EdgeKind.REPLACES: "replaced by",
    EdgeKind.DUPLICATE_OF: "repeated by",
}

_KIND_FOR_EVENT = {
    EventKind.GOAL: NodeKind.GOAL,
    EventKind.USER_MESSAGE: NodeKind.USER,
    EventKind.NOTE: NodeKind.NOTE,
    EventKind.FAILURE: NodeKind.FAILURE,
    EventKind.INVARIANT: NodeKind.INVARIANT,
}


@dataclass(slots=True)
class Node:
    id: str
    kind: NodeKind
    label: str
    #: Last event that touched this node. Recency is read off this.
    seq: int
    #: Log events that render this node. Retrieving a node means re-admitting
    #: exactly these, which keeps a call and its result together for free.
    seqs: list[int] = field(default_factory=list)
    tokens: int = 0
    is_error: bool = False
    #: The head of what this node actually said. A decision model judging
    #: "file:x.py (read 1×)" is judging a filename; this is what lets it judge
    #: the content instead.
    preview: str = ""


class Graph:
    def __init__(
        self, parents: dict[str, str | None] | None = None, *, preview_chars: int = 300
    ) -> None:
        self.preview_chars = preview_chars
        #: task id → parent task id, for logs written before TASK_BEGIN carried it.
        self._parents = dict(parents or {})
        self.nodes: dict[str, Node] = {ROOT: Node(ROOT, NodeKind.SESSION, "session", -1)}
        self._out: dict[str, list[tuple[EdgeKind, str]]] = defaultdict(list)
        self._in: dict[str, list[tuple[EdgeKind, str]]] = defaultdict(list)
        self._call_by_tool_use: dict[str, str] = {}
        self._last_read: dict[str, str] = {}
        self._by_digest: dict[str, str] = {}
        #: The mention index, both directions: node → its names, name → nodes.
        self._names_of: dict[str, set[str]] = defaultdict(set)
        self._holders: dict[str, set[str]] = defaultdict(set)

    @classmethod
    def from_events(
        cls,
        events: Iterable[Event],
        tokens: Callable[[Event], int] | None = None,
        *,
        parents: dict[str, str | None] | None = None,
        preview_chars: int = 300,
        names: Callable[[Event], Iterable[str]] | None = None,
    ) -> Graph:
        g = cls(parents, preview_chars=preview_chars)
        for ev in events:
            g.add_event(ev, tokens(ev) if tokens else 0, names(ev) if names else None)
        return g

    # -- building ----------------------------------------------------------

    def add_event(
        self, ev: Event, tokens: int = 0, names: Iterable[str] | None = None
    ) -> str | None:
        """Fold one event in. Returns the node it landed on, if any.

        ``names`` should come from the event's *full* content; the caller has
        the blob behind a handle and the graph does not. Without it, names are
        read from the blocks, which for a handle means the extract only.
        """
        nid = self._fold(ev, tokens)
        if nid is not None and nid in self.nodes:
            found = names if names is not None else name_ranks(event_text(ev))
            for name in found:
                self._names_of[nid].add(name)
                self._holders[name].add(nid)
        return nid

    def _fold(self, ev: Event, tokens: int) -> str | None:
        parent = _task_node(ev.task_id) if ev.task_id else ROOT
        if parent not in self.nodes:
            parent = ROOT

        if ev.kind is EventKind.TASK_BEGIN:
            nid = _task_node(ev.task_id)
            p = ev.meta.get("parent") or self._parents.get(ev.task_id or "")
            owner = _task_node(p) if p and _task_node(p) in self.nodes else ROOT
            node = self._add(nid, NodeKind.TASK, _text(ev).lstrip("▶ "), ev, tokens, owner)
            return node.id

        if ev.kind is EventKind.TASK_END:
            node = self.nodes.get(_task_node(ev.task_id))
            if node is None:
                return None
            outcome = _text(ev).lstrip("■ ")
            node.label += "  → " + outcome
            node.preview = self._head(f"{node.preview}\noutcome: {outcome}")
            self._touch(node, ev, tokens)
            return node.id

        if ev.kind in _KIND_FOR_EVENT:
            return self._add(ev.id, _KIND_FOR_EVENT[ev.kind], _text(ev), ev, tokens, parent).id

        if ev.kind is EventKind.TOOL_CALL:
            tu = next((b for b in ev.blocks if isinstance(b, ToolUse)), None)
            label = _call_label(tu) if tu else "tool call"
            node = self._add(ev.id, NodeKind.CALL, label, ev, tokens, parent)
            if tu:
                self._call_by_tool_use[tu.id] = node.id
            if ev.resource:
                self._link_resource(node, ev.resource, ev.is_write)
            return node.id

        if ev.kind is EventKind.MODEL_OUTPUT:
            uses = [b for b in ev.blocks if isinstance(b, ToolUse)]
            if uses:
                # Tier 0: calls arrive inside the model turn, not as TOOL_CALL
                # events. The turn *is* the call node, so its results attach here
                # and re-admitting it keeps the eviction unit whole.
                label = "; ".join(_call_label(u) for u in uses)
                node = self._add(ev.id, NodeKind.CALL, label, ev, tokens, parent)
                for u in uses:
                    self._call_by_tool_use[u.id] = node.id
            else:
                node = self._add(ev.id, NodeKind.MODEL, _text(ev), ev, tokens, parent)
            return node.id

        if ev.kind is EventKind.TOOL_RESULT:
            return self._add_result(ev, tokens, parent)

        return None

    def _add_result(self, ev: Event, tokens: int, parent: str) -> str:
        blk = ev.blocks[0] if ev.blocks else None
        tid = getattr(blk, "tool_use_id", None)
        nid = self._call_by_tool_use.get(tid) if tid else None
        node = self.nodes.get(nid) if nid else None
        if node is None:
            node = self._add(ev.id, NodeKind.CALL, "tool result", ev, 0, parent)
        self._touch(node, ev, tokens)

        if isinstance(blk, HandleRef):
            node.is_error = blk.is_error
            node.label += "  → " + _first_line(blk.extract)
            node.preview = self._head(f"{node.preview}\n{blk.extract}".strip())
            hid = f"handle:{blk.handle_id}"
            if hid not in self.nodes:
                self.nodes[hid] = Node(hid, NodeKind.HANDLE, blk.handle_id, ev.seq)
            self._edge(node.id, hid, EdgeKind.CONTAINS)
            self._edge(node.id, hid, EdgeKind.HAS_HANDLE)
        elif isinstance(blk, ToolResult):
            node.is_error = blk.is_error
            node.label += "  → " + _first_line(blk.content)
            node.preview = self._head(f"{node.preview}\n{blk.content}".strip())

        if ev.resource:
            self._link_resource(node, ev.resource, ev.is_write)
        for src in ev.derived_from:
            if src != ev.resource:
                self._edge(node.id, self._resource(src), EdgeKind.DERIVED_FROM)
        if ev.digest:
            prior = self._by_digest.get(ev.digest)
            if prior and prior != node.id:
                self._edge(node.id, prior, EdgeKind.DUPLICATE_OF)
            self._by_digest[ev.digest] = node.id
        return node.id

    def _add(
        self, nid: str, kind: NodeKind, label: str, ev: Event, tokens: int, parent: str
    ) -> Node:
        body = "\n".join(b.text for b in ev.blocks if isinstance(b, Text))
        preview = self._head(body) if kind is not NodeKind.CALL else ""
        node = Node(nid, kind, label, ev.seq, [ev.seq], tokens, preview=preview)
        self.nodes[nid] = node
        self._edge(parent, nid, EdgeKind.CONTAINS)
        return node

    def _touch(self, node: Node, ev: Event, tokens: int) -> None:
        node.seqs.append(ev.seq)
        node.seq = ev.seq
        node.tokens += tokens

    def _resource(self, res: ResourceId) -> str:
        rid = f"res:{res}"
        if rid not in self.nodes:
            self.nodes[rid] = Node(rid, NodeKind.RESOURCE, str(res), -1)
        return rid

    def _link_resource(self, node: Node, res: ResourceId, is_write: bool) -> None:
        rid = self._resource(res)
        self.nodes[rid].seq = max(self.nodes[rid].seq, node.seq)
        self._edge(node.id, rid, EdgeKind.WRITES if is_write else EdgeKind.READS)
        if not is_write:
            key = str(res)
            prior = self._last_read.get(key)
            if prior and prior != node.id:
                self._edge(node.id, prior, EdgeKind.REPLACES)
            self._last_read[key] = node.id

    def _edge(self, src: str, dst: str, kind: EdgeKind) -> None:
        if (kind, dst) in self._out[src]:
            return
        self._out[src].append((kind, dst))
        self._in[dst].append((kind, src))

    # -- reading -----------------------------------------------------------

    def out_edges(self, nid: str) -> list[tuple[EdgeKind, str]]:
        return list(self._out.get(nid, ()))

    def in_edges(self, nid: str) -> list[tuple[EdgeKind, str]]:
        return list(self._in.get(nid, ()))

    def children(self, nid: str) -> list[Node]:
        """The containment layer: what an agent sees when it opens a node."""
        return [self.nodes[d] for k, d in self._out.get(nid, ()) if k is EdgeKind.CONTAINS]

    def neighbors(self, start: Iterable[str], hops: int = 2, *, limit: int = 5_000) -> dict[str, int]:
        """Breadth-first over links in both directions; node → distance.

        Containment is skipped. The session root contains everything, so
        walking it would put every node two hops from every other and the
        distance would stop meaning anything.
        """
        dist: dict[str, int] = {}
        q: deque[str] = deque()
        for s in start:
            if s in self.nodes and s not in dist:
                dist[s] = 0
                q.append(s)
        while q and len(dist) < limit:
            cur = q.popleft()
            if dist[cur] >= hops:
                continue
            for kind, nxt in (*self._out.get(cur, ()), *self._in.get(cur, ())):
                if kind is EdgeKind.CONTAINS or nxt in dist:
                    continue
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
        return dist

    def recent_calls(self, k: int) -> list[Node]:
        calls = [n for n in self.nodes.values() if n.kind is NodeKind.CALL]
        return sorted(calls, key=lambda n: n.seq)[-k:] if k > 0 else []

    def calls_touching(self, rid: str) -> list[Node]:
        seen = {
            src
            for kind, src in self._in.get(rid, ())
            if kind in (EdgeKind.READS, EdgeKind.WRITES, EdgeKind.DERIVED_FROM)
        }
        return sorted((self.nodes[s] for s in seen), key=lambda n: n.seq)

    def content(self, nid: str) -> str:
        """What a node says. A resource says what its latest observation said,
        since the resource itself is only a name."""
        node = self.nodes[nid]
        if node.kind is not NodeKind.RESOURCE:
            return node.preview
        latest = None
        for kind, src in self._in.get(nid, ()):
            if kind in (EdgeKind.READS, EdgeKind.WRITES):
                c = self.nodes[src]
                if latest is None or c.seq > latest.seq:
                    latest = c
        return latest.preview if latest else ""

    def links(self, nid: str, *, limit: int = 6) -> list[str]:
        """A node's connections as plain sentences, newest first.

        Spelled out rather than left as ids because the reader may be a model
        that is weak at following references: "used by: pytest → 1 failed"
        can be judged directly, "e000084" cannot.
        """
        out: list[str] = []
        for kind, src in reversed(self._in.get(nid, ())):
            if kind in _INCOMING:
                out.append(f"{_INCOMING[kind]}: {self._name(src)}")
        for kind, dst in self._out.get(nid, ()):
            if kind in _OUTGOING:
                out.append(f"{_OUTGOING[kind]}: {self._name(dst)}")
        return out[:limit]

    def _name(self, nid: str) -> str:
        return self.nodes[nid].label

    def _head(self, s: str) -> str:
        s = s.strip()
        return s if len(s) <= self.preview_chars else s[: self.preview_chars - 1] + "…"

    def names_of(self, nid: str) -> set[str]:
        return set(self._names_of.get(nid, ()))

    def mentioning(self, names: Iterable[str]) -> dict[str, tuple[float, list[str]]]:
        """Nodes sharing names with ``names``: node → (weight, shared names).

        A shared name counts for more the rarer it is -- ``RETRY_LIMIT`` in two
        places is a connection, ``src`` in two hundred is not -- and a name held
        by more than a quarter of all nodes counts for nothing.
        """
        n = max(1, len(self._names_of))
        weight: dict[str, float] = defaultdict(float)
        shared: dict[str, list[str]] = defaultdict(list)
        for name in set(names):
            holders = self._holders.get(name)
            if not holders or len(holders) > max(3, n // 4):
                continue
            w = math.log(1 + n / len(holders))
            for nid in holders:
                weight[nid] += w
                shared[nid].append(name)
        return {nid: (weight[nid], shared[nid]) for nid in weight}

    def resource_label(self, rid: str) -> str:
        reads = writes = 0
        for kind, _ in self._in.get(rid, ()):
            reads += kind is EdgeKind.READS
            writes += kind is EdgeKind.WRITES
        return f"{self.nodes[rid].label} (read {reads}×, written {writes}×)"

    def expand(self, node_ids: Iterable[str], *, per_node: int = 3) -> set[int]:
        """Log events to re-admit for a set of anchors.

        Mechanical by design -- this runs on every turn, so it may cost a
        lookup but never a model call. A resource brings back its latest read
        and latest write; a task brings back its outcome and its last calls.
        """
        seqs: set[int] = set()
        for nid in node_ids:
            node = self.nodes.get(nid)
            if node is None:
                continue
            if node.kind in (NodeKind.CALL, NodeKind.MODEL):
                seqs.update(node.seqs)
            elif node.kind is NodeKind.RESOURCE:
                # Direct observations only. A call merely derived from the
                # resource (a test run over it) is not its content.
                last_read = last_write = None
                for kind, src in self._in.get(nid, ()):
                    c = self.nodes[src]
                    if kind is EdgeKind.WRITES and (last_write is None or c.seq > last_write.seq):
                        last_write = c
                    elif kind is EdgeKind.READS and (last_read is None or c.seq > last_read.seq):
                        last_read = c
                for c in (last_read, last_write):
                    if c is not None:
                        seqs.update(c.seqs)
            elif node.kind is NodeKind.TASK:
                seqs.update(node.seqs)
                calls = [c for c in self.children(nid) if c.kind is NodeKind.CALL]
                for c in calls[-per_node:]:
                    seqs.update(c.seqs)
        return seqs

    def expand_scored(self, scored: Iterable[tuple[str, float]], *, per_node: int = 3) -> dict[int, float]:
        """``expand()`` for ranked nodes: log event → the best score that asks
        for it, so a budget can be spent highest-first."""
        out: dict[int, float] = {}
        for nid, p in scored:
            for q in self.expand([nid], per_node=per_node):
                if p > out.get(q, float("-inf")):
                    out[q] = p
        return out


def event_text(ev: Event, read_handle: Callable[[str], str | None] | None = None) -> str:
    """Everything an event says, as text. With ``read_handle``, a handle is
    read through to its blob; without, it contributes its extract."""
    parts: list[str] = []
    for b in ev.blocks:
        if isinstance(b, (Text, Thinking)):
            parts.append(b.text)
        elif isinstance(b, ToolUse):
            parts.append(json.dumps(b.input, ensure_ascii=False))
        elif isinstance(b, ToolResult):
            parts.append(b.content)
        elif isinstance(b, HandleRef):
            full = read_handle(b.handle_id) if read_handle else None
            parts.append(full if full is not None else b.extract)
    return "\n".join(parts)


def _task_node(task_id: str | None) -> str:
    return f"task:{task_id}"


def _text(ev: Event) -> str:
    for b in ev.blocks:
        if isinstance(b, Text) and b.text.strip():
            return _first_line(b.text)
    return ev.kind.value


def _first_line(s: str, width: int = 100) -> str:
    line = next((ln.strip() for ln in s.splitlines() if ln.strip()), "")
    return line if len(line) <= width else line[: width - 1] + "…"


def _call_label(tu: ToolUse) -> str:
    args = " ".join(f"{k}={v}" for k, v in tu.input.items())
    return _first_line(f"{tu.name} {args}".strip(), 80)

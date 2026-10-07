"""The decision layer: which part of the graph the next step needs.

The graph can say what is connected to the current work; it cannot say what
*matters* for the next step. That is a judgment, so this is the one place a
model is allowed -- and it is fenced in three ways:

* **Behind a protocol.** Core code knows ``DecisionModel`` and nothing else.
  Vendor models (Jev first) live in ``adapters/decision``, exactly as provider
  formats live in ``adapters``.
* **Off the hot path.** The model re-ranks in the background after each tool
  result, while the tool's successor is still being generated; a turn uses
  whatever ranking has landed and never waits for one.
* **Unable to break a turn.** A model that raises, times out or is missing
  leaves the previous ranking in place. The worst case is today's behavior.

Facts stay with the cascade. What it has proved stale or false is never offered
to a model: measured on real sessions, the cascade caused 0.3% of what agents
lost, so there is nothing there for a model to fix and much it could break.

The model never sees the whole graph. Candidates come from two places -- typed
links from the recent work, and old nodes that share rare names with it (the
agent's plan, its latest calls, the user's words) -- ordered by a cheap prior and
capped, so one decision costs the same however long the log grows.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .codecs import name_ranks
from .graph import Graph, NodeKind
from .ir import Event, EventKind, Text, Thinking
from .predicates import Decision, Reason

#: Verdicts that are facts, not opinions. A node the cascade has proved stale
#: or false is not offered to a model to be argued back in.
_STALE = frozenset({Reason.INVALIDATED, Reason.SUPERSEDED, Reason.DUPLICATE})

#: What a model may anchor on. Pinned kinds are always in context already, and
#: a handle is reached through the call that produced it.
_CANDIDATE_KINDS = frozenset({NodeKind.CALL, NodeKind.RESOURCE, NodeKind.TASK})


@dataclass(frozen=True, slots=True)
class Candidate:
    node_id: str
    kind: str
    label: str
    #: Link distance from the recent work; None if reached by shared names only.
    hops: int | None
    #: Events since this node was last touched.
    age: int
    tokens: int
    is_error: bool = False
    #: Model-free pre-rank in [0, 1]: link distance, shared names, recency.
    prior: float = 0.0
    #: The head of what the node said, so it is judged on content, not name.
    content: str = ""
    #: Its connections as sentences ("used by: pytest → 1 failed").
    links: tuple[str, ...] = ()
    #: Names the current work is about that this item's full content has and
    #: its summary does not -- the case for reading it in full.
    hidden: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "label": self.label}
        if self.content:
            d["content"] = self.content
        if self.links:
            d["connected_to"] = list(self.links)
        if self.hidden:
            d["only_in_full_content"] = list(self.hidden)
        if self.is_error:
            d["is_error"] = True
        return d


@dataclass(frozen=True, slots=True)
class DecisionState:
    """What the model is deciding *for*. Built from the log, never written by
    a model: the goal verbatim, the task in progress, the latest user input,
    recent steps and recorded failures."""

    goal: str
    recent: tuple[str, ...]
    failures: tuple[str, ...] = ()
    #: Open task frames, outermost first ("fix auth > check the pool").
    task: str = ""
    #: The most recent user message, verbatim.
    user_input: str = ""
    #: What the agent last said it would do, verbatim -- the strongest signal
    #: there is for what the next step needs.
    plan: str = ""
    #: The names the current work is about (recent calls, plan, input, task).
    #: Used to build questions; not sent as state.
    focus: frozenset[str] = frozenset()
    #: The part of ``focus`` that came from words -- plan, input, task -- and
    #: the recent calls the rest came from, so an item's own names can be told
    #: apart from names something *else* asked about.
    focus_text: frozenset[str] = frozenset()
    seed_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"goal": self.goal}
        if self.task:
            d["current_task"] = self.task
        if self.user_input:
            d["latest_user_input"] = self.user_input
        if self.plan:
            d["agent_plan"] = self.plan
        d["recent_steps"] = list(self.recent)
        if self.failures:
            d["known_failures"] = list(self.failures)
        return d


class DecisionModel(Protocol):
    name: str
    #: Minimum score to be admitted. A calibrated model sets a real cut-off; a
    #: prior that only orders sets 0 and lets the budget decide.
    threshold: float

    def score(
        self, state: DecisionState, candidates: Sequence[Candidate]
    ) -> Mapping[str, float]:
        """Probability, per ``node_id``, that the next step needs it."""
        ...


#: The fidelity question: summary or full content, per large output.
FULL_QUESTION = (
    "Will the agent's next step need the full content of this item, rather than "
    "the short summary it currently sees?"
)

#: Yes/no judgments asked about the latest step, alongside the ranking.
FAILED = "failed"
JUDGMENTS = {
    FAILED: (
        "Did this step fail to do what it attempted -- an error, a failing check, "
        "a missing file, a rejected command -- so that the approach it tried did "
        "not work?"
    ),
}


#: What a failure looks like on its face. The baseline judge, not a parser: it
#: only has to be something a model must beat.
_FAILURE = re.compile(
    r"Traceback \(most recent call last\)|^\s*\w*(Error|Exception)\b|\berror:|"
    r"^FAILED\b|\bNo such file or directory\b|command not found|"
    r"exit(ed)? (with )?(code|status) [1-9]",
    re.M | re.I,
)


def failure_line(text: str) -> str:
    """The line that says why a step failed, quoted from its own output -- the
    first line that looks like a failure, else the first line."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if _FAILURE.search(ln):
            return ln[:200]
    return lines[0][:200] if lines else ""


def candidate_for(graph: Graph, nid: str, *, now: int) -> Candidate:
    node = graph.nodes[nid]
    return Candidate(
        node_id=nid,
        kind=node.kind.value,
        label=node.label,
        hops=0,
        age=max(0, now - node.seq),
        tokens=node.tokens,
        is_error=node.is_error,
        content=graph.content(nid),
        links=tuple(graph.links(nid)),
    )


def fidelity_items(
    graph: Graph,
    node_ids: Sequence[str],
    extracts: Mapping[str, str],
    state: DecisionState,
    *,
    now: int,
) -> list[Candidate]:
    """The large outputs worth a summary-or-full question: each carries its
    extract and the names only its full content holds that *something else*
    -- the plan, the user, the task, another recent call -- is asking about.
    An item's own names never make its case, or every fresh large output
    would ask to be read in full."""
    out: list[Candidate] = []
    for nid in node_ids:
        extract = extracts.get(nid)
        if extract is None or nid not in graph.nodes:
            continue
        others = set(state.focus_text)
        for sid in state.seed_ids:
            if sid != nid:
                others |= graph.names_of(sid)
        visible = set(name_ranks(extract))
        hidden = sorted((graph.names_of(nid) - visible) & others)
        c = candidate_for(graph, nid, now=now)
        out.append(
            Candidate(
                node_id=c.node_id, kind=c.kind, label=c.label, hops=c.hops, age=c.age,
                tokens=c.tokens, is_error=c.is_error, content=extract, links=c.links,
                hidden=tuple(hidden[:8]),
            )
        )
    return out


class Judge(Protocol):
    """Optional second capability of a decision model: yes/no judgments about
    one step. A model without it simply makes no judgments."""

    def judge(
        self, state: DecisionState, step: Candidate, questions: Mapping[str, str]
    ) -> Mapping[str, float]:
        """Probability, per question key, that the answer is yes."""
        ...


class NullDecision:
    """Scores nothing, so nothing is anchored: exactly the behavior before the
    graph existed. The default, and the baseline every model is measured
    against."""

    name = "null"
    threshold = 1.0
    swap_threshold = None

    def score(self, state: DecisionState, candidates: Sequence[Candidate]) -> dict[str, float]:
        return {}


class ProximityDecision:
    """The model-free prior, used as the score: linked, sharing rare names,
    and fresh ranks higher.

    Not a judgment, a baseline. It runs the whole ranking pipeline offline and
    gives a model something to beat -- if Jev cannot outscore the prior it was
    handed, it is not earning its latency.
    """

    name = "proximity"
    threshold = 0.0
    #: The prior is not calibrated, so it swaps only on its strongest signal.
    swap_threshold = 0.9
    full_threshold = 0.5

    def score(self, state: DecisionState, candidates: Sequence[Candidate]) -> dict[str, float]:
        return {c.node_id: c.prior for c in candidates}

    def fidelity(self, state: DecisionState, items: Sequence[Candidate]) -> dict[str, float]:
        """Full content when the current work names something only the full
        content has. Exact, and the case a summary most clearly fails."""
        return {c.node_id: 0.9 if c.hidden else 0.0 for c in items}

    def judge(
        self, state: DecisionState, step: Candidate, questions: Mapping[str, str]
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        if FAILED in questions:
            looks = step.is_error or bool(_FAILURE.search(step.content))
            out[FAILED] = 0.9 if looks else 0.05
        return out


@dataclass(frozen=True, slots=True)
class Anchor:
    node_id: str
    p: float


@dataclass(slots=True)
class AnchorRun:
    """One decision, kept whole so it can be replayed and its calibration
    checked later -- a score nobody can audit is an opinion."""

    model: str
    seq: int
    anchors: list[Anchor]
    scores: dict[str, float] = field(default_factory=dict)
    candidates: int = 0
    ms: float = 0.0
    #: Why the previous anchors were kept instead, if they were.
    fallback: str | None = None


def gather(
    graph: Graph,
    events: Sequence[Event],
    decisions: Sequence[Decision],
    *,
    recent: int = 6,
    hops: int = 2,
    cap: int = 50,
    task: str = "",
    half_life: int = 40,
) -> tuple[DecisionState, list[Candidate]]:
    """The bounded question put to a model: which old nodes might the next
    step need, each carrying its content and connections, and what the agent
    is doing -- goal, open task, latest input, its own plan."""
    stale = {d.seq for d in decisions if not d.keep and d.reason in _STALE}
    seeds = graph.recent_calls(recent)
    seed_ids = {s.id for s in seeds}
    now = events[-1].seq if events else 0

    goal = _latest_text(events, EventKind.GOAL)
    user_input = _latest_text(events, EventKind.USER_MESSAGE)
    plan = _latest_text(events, EventKind.MODEL_OUTPUT, with_thinking=True)[-2_000:]

    # Reach 1: typed links from the recent work.
    prior: dict[str, float] = {}
    dist = {nid: h for nid, h in graph.neighbors(seed_ids, hops).items() if h}
    for nid, h in dist.items():
        prior[nid] = 0.5 ** (h - 1)

    # Reach 2: any old node sharing rare names with what the agent is doing now.
    worded: set[str] = set()
    for text in (plan, user_input, task):
        worded |= set(name_ranks(text))
    wanted = set(worded)
    for sid in seed_ids:
        wanted |= graph.names_of(sid)
    mentioned = {nid: v for nid, v in graph.mentioning(wanted).items() if nid not in seed_ids}
    top = max((w for w, _ in mentioned.values()), default=0.0)
    for nid, (w, _) in mentioned.items():
        prior[nid] = max(prior.get(nid, 0.0), 0.75 * w / top)

    out: list[Candidate] = []
    for nid, p in prior.items():
        node = graph.nodes[nid]
        if node.kind not in _CANDIDATE_KINDS:
            continue
        if node.kind is NodeKind.CALL and node.seq in stale:
            continue
        age = max(0, now - node.seq)
        links = graph.links(nid)
        if nid in mentioned:
            names = ", ".join(sorted(mentioned[nid][1])[:6])
            links = [f"shares names with the current work: {names}", *links]
        label = graph.resource_label(nid) if node.kind is NodeKind.RESOURCE else node.label
        out.append(
            Candidate(
                node_id=nid,
                kind=node.kind.value,
                label=label,
                hops=dist.get(nid),
                age=age,
                tokens=node.tokens,
                is_error=node.is_error,
                prior=p * 0.5 ** (age / half_life),
                content=graph.content(nid),
                links=tuple(links[:6]),
            )
        )
    out.sort(key=lambda c: c.prior, reverse=True)

    failures = tuple(n.label for n in graph.nodes.values() if n.kind is NodeKind.FAILURE)[-5:]
    state = DecisionState(
        goal=goal,
        recent=tuple(s.label for s in seeds),
        failures=failures,
        task=task,
        user_input=user_input,
        plan=plan,
        focus=frozenset(wanted),
        focus_text=frozenset(worded),
        seed_ids=tuple(s.id for s in seeds),
    )
    return state, out[:cap]


def _latest_text(events: Sequence[Event], kind: EventKind, *, with_thinking: bool = False) -> str:
    """The newest event of ``kind`` that says something, verbatim. Verbatim
    matters for the goal most: a trimmed goal is a paraphrase, and a
    paraphrase is where drift starts."""
    kinds = (Text, Thinking) if with_thinking else (Text,)
    for e in reversed(events):
        if e.kind is kind:
            text = "\n".join(b.text for b in e.blocks if isinstance(b, kinds)).strip()
            if text:
                return text
    return ""


def choose(
    model: DecisionModel,
    state: DecisionState,
    candidates: Sequence[Candidate],
    *,
    seq: int,
    threshold: float = 0.5,
    k: int = 8,
) -> AnchorRun:
    """Score, then keep the confident top ``k``. Never raises: a failing model
    is reported in ``fallback`` and the caller keeps its previous anchors."""
    if not candidates:
        # Nothing to decide between is not a decision that nothing matters.
        return AnchorRun(model.name, seq, [], fallback="no candidates")
    t0 = time.perf_counter()
    try:
        scores = {nid: float(p) for nid, p in model.score(state, candidates).items()}
    except Exception as exc:  # noqa: BLE001 -- a broken model must never break the turn
        return AnchorRun(
            model.name, seq, [], candidates=len(candidates),
            ms=(time.perf_counter() - t0) * 1000, fallback=f"{type(exc).__name__}: {exc}",
        )
    ranked = sorted(
        (Anchor(nid, p) for nid, p in scores.items() if p >= threshold),
        key=lambda a: a.p,
        reverse=True,
    )
    return AnchorRun(
        model.name, seq, ranked[:k], scores=scores,
        candidates=len(candidates), ms=(time.perf_counter() - t0) * 1000,
    )

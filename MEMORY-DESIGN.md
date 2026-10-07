# ctxkernel → memory layer: design

Direction: grow ctxkernel from a per-session context trimmer into a **memory
layer**: a local graph of everything an agent did, traversable in
milliseconds, from which each model call gets exact context. Fast decision
models (Jev first) improve *what* gets selected without putting latency on
the turn.

Status: graph (§2), anchors (§3) and the decision layer with a Jev adapter (§5) are built. The rest is still a proposal.

---

## 1. What stays, what changes

**Stays** (these are what make the system trustworthy):

- Append-only log; nothing is ever deleted. A wrong selection is a miss, not a loss.
- Building the graph is mechanical: no model, no judgment, no network.
- The core stays offline with zero required dependencies.
- `tool_use` / `tool_result` stay one unit.
- Every selection is explainable.

**Changes**:

- The "no model" rule narrows to **"no model in the core or on the hot path"**.
  Decision models are allowed as optional adapters that run off the hot path.
- Selection moves from *recency* (newest pairs that fit) to *relevance*
  (what is connected to the current work).
- Memory outlives the session: resources and facts are shared across a
  tenant's sessions.

---

## 2. Data model

Two layers over the same nodes.

```
TREE (containment: navigation)          LINKS (connections: relevance)
tenant                                  call ──reads/writes──→ resource
 └─ session                             call ──derived_from──→ resource
     └─ task (nests via parent)         call ──replaces──────→ older call
         ├─ call (tool_use + result)    call ──duplicate_of──→ older call
         │    └─ handle (full bytes)    failure ──about──────→ resource
         ├─ failure / invariant
         └─ outcome (t.done)
```

### When a node is created

A node is created mechanically for **anything that can be pointed at or
expanded**. Importance is never judged at build time.

| Node | Created when |
|---|---|
| session | a session opens |
| task | `task()` opens |
| call | every tool call + result pair |
| resource | the identity registry recognizes the tool (e.g. `file:auth.py`) |
| handle | a result is ≥ 1,000 tokens (already happens) |
| failure / invariant / outcome | `fail()` / `rule()` / `t.done()` |

Unknown tools still get a call node. They just get no resource links.

### Every edge comes from data already on the event

| Edge | Source |
|---|---|
| contains | `task_id`, `parent` |
| reads / writes | `resource`, `is_write` |
| derived_from | `derived_from` |
| replaces | a newer read of the same resource |
| duplicate_of | same `digest` |
| has_handle | handle interception |

### Storage: no graph DB

- **Disk:** two new tables in the existing SQLite store, `nodes` and
  `edges(src, dst, type, seq)`, indexed on `src` and `dst`.
- **Memory:** an adjacency dict per session, loaded from `edges` when the
  session opens and updated on every record. A 2-hop traversal is dict lookups.
- **Bytes:** stay in the content-addressed blob store. The graph holds IDs and
  one-line labels only.
- **Cross-session:** resource nodes live at tenant level, so session B sees
  what session A learned about `auth.py`. Session scoping for everything else
  is unchanged (tenant isolation still holds).

A session holds thousands of nodes and a tenant hundreds of thousands, which
SQLite + a dict handles in microseconds. A graph DB only pays off at millions
of nodes with complex cross-graph queries.

---

## 3. Read path: anchors, not per-turn scoring

```
HOT PATH (every turn, no model, target < 5 ms)
  1. rules drop provably stale nodes         (existing cascade)
  2. load current anchors                    (node ids stored in log.db)
  3. expand around anchors, 1–2 hops:        latest read of anchored resources,
                                             related writes, failures, handles
  4. pinned zones + expansion + recent tail → fit budget → messages

OFF THE HOT PATH (async, only on triggers)
  decision model re-picks anchors when:
    - a task starts or the goal changes
    - the agent touches a resource far from the current anchors
  runs while the tool executes / the model generates
  not ready by the next turn → the old anchors are used
```

### How the decision model sees a big graph

It never sees the whole graph. Candidates are bounded:

```
all nodes (100k)
  → drop stale (rules)
  → only nodes near the current work
  → top-down: score task outcomes first, open only top tasks, then their calls
  → hard cap ~50 candidates
decision model → probability per candidate → the top ones become anchors
```

Cost per re-anchor stays flat however big the log grows.

---

## 4. Quality: where it is lost today, and the fix

| Loss today | Fix |
|---|---|
| The tail keeps what's newest, not what's relevant | Anchor expansion pulls in connected old nodes |
| An old detail (step-12 warning) falls out by age | Decision model scores tail-eviction candidates |
| The agent never thinks to search the log | Agent tools expose an outline, so it can see what exists |
| Unknown tools get no links | Identity registry; the decision model can still rank them |
| No measurement of quality at all | Eval harness (below), built first |

### Eval harness (prerequisite for every quality claim)

- Replay real traces. At each step, ask the model for its next action with
  (a) full context and (b) assembled context.
- Metric: **action agreement**, how often (b) picks the same next action as (a).
- Run it per policy: rules only, + anchors, + Jev. Adopt a change only if
  agreement goes up at an acceptable token cost.
- Log every decision-model score with its outcome, so its calibration can be
  checked later.

---

## 5. Decision models: pluggable, in adapters

```
ctxkernel/decision.py              core: DecisionModel protocol, NullDecision (default),
                                   ProximityDecision (offline baseline), gather(), choose()
ctxkernel/adapters/decision/jev.py first real model; optional extra: pip install ctxkernel[jev]
```

The protocol lives in core, not in `adapters/`, so core code can type against
it without importing an adapter.

- The core knows only the protocol, the same way it treats providers today.
- Inputs are built without an LLM: goal, last-turn summary from extracts,
  candidate labels and features (hops, age, edge type, is_error, tokens).
- Failure, timeout or a missing key → fall back to rules. It never breaks a turn.
- Scores are cached by (node, state hash).

---

## 6. SDK: simpler surface

Today's `ContextEngine` exposes 20+ methods and `record_*` plumbing. Proposed
facade (`ContextEngine` stays as an alias for compatibility):

```python
from ctxkernel import Memory
from ctxkernel.adapters.decision import Jev

mem = Memory(user="alice", session="s1")        # same id = resume
mem = Memory(user="alice", decision=Jev())      # optional decision model

mem.goal("fix the flaky auth test")
mem.tool("read_file", {"path": "src/auth.py"}, contents)
mem.fail("mutex on refresh", why="deadlocks on re-auth")
mem.rule("/refresh caps at 10 req/min")

with mem.task("check the pool") as t:
    ...
    t.done("not the pool")

messages = mem.context("openai")                 # what to send
tools    = mem.tools("openai")                   # ctx_outline / ctx_open / ctx_search schemas
result   = mem.handle(tool_call)                 # executes a ctx_* call the model made

mem.recall("file:src/auth.py")                   # everything connected to a resource, across sessions
mem.explain()                                    # why each node is / isn't in context
```

Design rules for the surface:

- One object, short verbs. Zero-argument construction works.
- Resuming is the default behavior, not a special case.
- The agent-facing tools ship with the SDK. Users shouldn't hand-write schemas.
- Every advanced piece (codecs, identity, budgets, decision models) is a
  keyword argument, never required.

---

## 7. Performance targets

| Operation | Target |
|---|---|
| record one event (incl. edges) | < 1 ms |
| `context()` at 100k nodes | < 5 ms p95 |
| 2-hop traversal | < 1 ms |
| decision model | never on the hot path |

These need benchmarks in the repo before they are claimed.

---

## 8. Build order

0. **Fix session resume.** `_call_seq` and `_ingested` restart at 0 on reopen,
   so tool-use ids repeat (`c00001` twice, reproduced). Seed both from `log.db`.
1. **Eval harness.** Without it, "better quality" is unfalsifiable.
2. **Graph tables + adjacency dict**, written at record time. Backfill from
   existing logs.
3. **Agent tools:** `ctx_outline`, `ctx_open`, `ctx_search`.
4. **Anchors + mechanical expansion** in the assembler. No model yet; measure.
5. **Decision adapter protocol + Jev.** Measure against step 4.
6. **Cross-session memory** (tenant-level resource nodes, `recall`).
7. **`Memory` facade**, docs, migration note.

Each step is independently shippable and measured by step 1.

---

## 9. Open questions

- Jev's exact API (state/question format, auth, batch limits). Needed before
  `jev.py`.
- "Far from anchors": hop count, or a different resource family?
- Cross-session trust: can a fact from session A be stale for session B? It
  probably needs the same invalidation check, keyed by tenant-level writes.
- Retention: the log grows forever by design. Is an archive tier needed for
  very long-lived tenants?

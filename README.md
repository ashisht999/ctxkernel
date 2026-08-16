# ctxkernel

The context window is not storage. It is a **rendering** of an append-only log,
the way a web page is a rendering of a database — which is what makes throwing
it away safe.

```
pip install ctxkernel
```

Imports as `ctxkernel`. Zero required dependencies. SQLite for the log and indexes, content-addressed
files for blobs. Works offline.

## Why

A long-running agent produces information monotonically; the window is finite.
So loss is not an engineering shortfall, it is arithmetic — and you are not
building a memory system, you are building a **loss function**.

The trap is treating that as a ranking problem ("score everything by importance,
keep the top N"), which is unanswerable because importance depends on a future
you cannot see. This library never asks it. Four stages remove whole classes of
material using questions that have mechanical answers:

| Stage | Question | Not "is this important?" |
|---|---|---|
| 0 | Is it over the threshold? | → handle + computed extract |
| 1 | Does the log *prove* it is dead? | superseded · duplicate · invalidated |
| 2 | Did that scope close? | keep the outcome, drop the trace |
| 3 | Can I just fetch it again? | reproducible → pointer |

What survives is small, because what was crowding it out was reproducible bulk.

## Quickstart

Three concepts: goal, tool, messages.

```python
from ctxkernel import ContextEngine

eng = ContextEngine()

eng.set_goal("Fix the flaky token-refresh test.", ["pytest tests/test_auth.py -x"])
eng.tool("read_file", {"path": "src/auth.py"}, forty_thousand_tokens)
eng.note_failure("mutex on refresh", "deadlocks on re-auth")   # never evicted

messages = eng.messages("openai")   # anthropic | openai | chat | text
```

In your loop, replace `messages.append(...)` with `eng.tool(...)` and pass
`eng.messages(fmt)` to the model. You stop maintaining the message list yourself.

```python
print(eng.preview())    # the actual context, zone by zone
print(eng.explain())    # why every event was kept or dropped
```

See `QUICKSTART.md` for the rest.

### Zero-change integration

```python
client = eng.wrap(anthropic.Anthropic())
```

The wrap sits at the model-client boundary, the one place where every tool call
and result is visible without asking the host to change anything. Your loop keeps
appending to its own message list and never learns that the list it sends is not
the list it built.

### Task frames

```python
with eng.task("check whether the pool is the cause") as t:
    ...
    t.done("Not the pool — holds fine at 200 concurrent connections.")
```

Compaction fires at the **boundary**, not under token pressure. At close the
outcome subsumes the trace; mid-task the discarded material is still live. Two
independent arguments land here — one from information value, one from the fact
that every prefix mutation throws away the KV cache.

## Measured

`python examples/long_run.py` — 197 events, 95 tool calls:

```
WITHOUT   182,160 tokens   91% of a 200k window
WITH        4,237 tokens    2% of the window
             97.7% saved
```

Synthetic trace, so treat the percentage as a shape rather than a measurement.
Run it on your own history instead — `ctxkernel.replay` imports an existing
trace and returns the log-vs-context growth curve:

```python
from ctxkernel.replay import import_messages
import_messages(eng, my_messages, fmt="openai")
```

## What it does not do

**One failure the cascade cannot catch.** A deprecation warning in a build log at
step 12 that explains the failure at step 300: not superseded, its scope closed
cleanly, reproducible in principle but nobody knows to look. Every stage pages it
out, correctly, by every rule available.

The mitigation is `eng.search_log(pattern)` — the agent can grep its own
history, so the system converts *remembered* into *findable*. If the agent never
thinks to look, it still loses. That is a real limit, not a solved case.

**No LLM anywhere in this package.** Every extract is parsed, not written — a
symbol outline is exact, instant, free, and cannot hallucinate. Where judgment is
genuinely required (*why* an approach failed), that is `note_failure`, and it is
the caller's to supply.

## Design

- `ir.py` — canonical, provider-neutral IR. Only `adapters/` may see a vendor format.
- `identity.py` — resource identity derived from tool calls; you adopt no log format.
- `codecs.py` — computed extracts per content type. Register your own.
- `predicates.py` — Tier A. Facts about the log, not opinions.
- `assembler.py` — zones ordered stable→volatile, structural validity, budgets.
- `store.py` — append-only log, content-addressed blobs, scoped per session.
- `adapters/` — the only place allowed to see a vendor format: `anthropic`,
  `openai` (also vLLM / Ollama / LM Studio / TGI), `generic` (any chat endpoint
  or a single prompt string for completion endpoints).

## Isolation

Stores are opened **scoped**, so cross-tenant reads are unrepresentable rather
than merely discouraged:

```python
store = SessionStore.for_session(root, tenant, session)   # scope bound once
store.get(key)                                            # no cross-session API
```

Handle ids are hashed with tenant and session, because a handle is a
dereferenceable pointer sitting in prompt text — a global id would turn the
token-saving mechanism into a cross-tenant read primitive.

| Layer | Scope |
|---|---|
| event log, blobs, handles | session |
| anchored facts *(not built yet)* | tenant |

## Status

Alpha. Working and tested — 55 tests, no provider SDK required to run them.

Not built yet: cross-session anchored facts, structured retrieval, sub-agent
frames, and the eval harness that would turn the token numbers into evidence
about decision quality rather than cost.

Apache-2.0.

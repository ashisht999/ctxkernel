# Quickstart

```
pip install ctxkernel
```

Three concepts. Everything else is optional.

```python
from ctxkernel import ContextEngine

eng = ContextEngine()                    # 1. make one

eng.set_goal("fix the flaky auth test")  # 2. say what you're doing
eng.tool("read_file", {"path": "src/auth.py"}, contents)   # 3. log tool round-trips

messages = eng.messages("openai")        # ready to send
```

That's the whole API for the common case. `ContextEngine()` needs no arguments —
it writes to `./.ctx` and picks sane defaults.

## Wire it into your loop

Wherever your loop currently does this:

```python
messages.append({"role": "tool", "content": result})
response = client.chat.completions.create(messages=messages, ...)
```

do this instead:

```python
eng.tool(tool_name, tool_args, result)
response = client.chat.completions.create(messages=eng.messages("openai"), ...)
```

You stop maintaining `messages` yourself. That is the entire integration.

`eng.messages(fmt)` takes:

| fmt | for |
|---|---|
| `"anthropic"` | Claude Messages API |
| `"openai"` | OpenAI, **and** vLLM / Ollama / LM Studio / TGI / any OpenAI-compatible server |
| `"chat"` | plain `{role, content}` dicts — any chat endpoint |
| `"text"` | one prompt string — raw completion endpoints |

## See what it's doing

```python
print(eng.preview())    # the actual context, zone by zone
print(eng.explain())    # why every event was kept or dropped
eng.report()            # {'raw_tokens': 62090, 'assembled_tokens': 599, 'saved_pct': 99.0, ...}
```

Run `python try_it.py` in this repo for a worked example on real files.

## Worth adding once it's working

**Record dead ends.** The highest value-per-token thing in the system — a few
hundred tokens that stop the agent rediscovering the same wall at step 200 and
step 350. Nothing detects these automatically; you call it.

```python
eng.note_failure("mutex on refresh", "deadlocks when refresh triggers re-auth")
eng.note_invariant("the /refresh endpoint caps at 10 req/min")
```

**Mark scopes.** When a sub-task closes, its trace becomes compactable and only
the outcome survives.

```python
with eng.task("check whether the pool is the cause") as t:
    ...
    t.done("Not the pool — holds fine at 200 concurrent connections.")
```

**Page content back.** Big results become handles; `expand` is the dereference.

```python
eng.expand(handle_id)                          # everything
eng.expand(handle_id, symbol="refresh_token")  # one function
eng.expand(handle_id, grep="TODO")             # matching lines
eng.search_log("deprecated")                   # grep evicted material
```

Expose `expand` and `search_log` as tools to your agent and it can page things
in for itself.

## Bringing an existing trace

```python
from ctxkernel.replay import import_messages, import_records, read_jsonl

import_messages(eng, my_openai_messages, fmt="openai")

# or from any shape at all:
import_records(eng, read_jsonl("trace.jsonl"), mapper=lambda r:
    ("call", r["id"], r["tool"], r["args"]) if r["type"] == "call" else
    ("result", r["id"], r["output"]) if r["type"] == "result" else
    ("user", r["text"]))
```

## Custom tools

Only needed if your tool's identity isn't inferable from its arguments. Without
it, digest dedup still works — one predicate just gets weaker, nothing breaks.

```python
from ctxkernel import ResourceId, ToolSemantics

eng.identity.register("db_query", lambda name, args:
    ToolSemantics(ResourceId("db", args["table"])))
```

Same for extracts, if you want something better than the head/tail fallback:

```python
class MyCodec:
    media_type = "application/x-mine"
    def matches(self, content, resource, tool_name): return tool_name == "db_query"
    def extract(self, content, resource): return f"{content.count(chr(10))} rows"

eng.codecs.register(MyCodec())
```

## Where things live

```
.ctx/                          # root=... to move it
  <tenant>/
    blobs/ab/cdef…             # content-addressed, shared across that tenant's sessions
    <session>/log.db           # plain SQLite: events, handles, tasks
```

Nothing is ever deleted. Query `log.db` with any SQLite tool.

## Multi-user

```python
eng = ContextEngine(tenant=user_id, session=thread_id)
```

Tenant is a hard directory boundary and handle ids are hashed with it, so a
handle from one tenant cannot resolve in another. Sessions are independent
append streams — no locks, no coordination, scales linearly.

If you're unsure whether two runs are the same session, **start a new one**. A
fresh session leaks nothing and costs only continuity; merging two users' work
on a guess is unrecoverable.

## What it does not do

`search_log` exists because eviction is aggressive: a detail buried in a build
log at step 12 that matters at step 300 *will* be paged out, correctly, by every
rule available. It stays findable, not remembered. If the agent never thinks to
look, it loses.

And the reduction numbers measure **cost, not quality**. Whether an agent makes
better decisions on a 600-token context than a 60,000-token one is unproven here
and needs the replay harness.

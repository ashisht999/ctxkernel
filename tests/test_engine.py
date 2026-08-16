import pytest

from ctxkernel import ContextEngine, budgets
from ctxkernel.adapters import anthropic as ad
from ctxkernel.assembler import eviction_units
from ctxkernel.ir import Event, EventKind, HandleRef, Text, ToolResult, ToolUse

BIG = "\n".join(f"def function_{i}(a, b):\n    return a + b" for i in range(400))


@pytest.fixture
def engine(tmp_path):
    e = ContextEngine(root=tmp_path, session="s1", budget=budgets.small(32_000))
    yield e
    e.close()


def test_fat_result_is_externalized_not_admitted(engine):
    engine.record_tool_call("t1", "read_file", {"path": "src/big.py"})
    ev = engine.record_tool_result("t1", BIG)

    blk = ev.blocks[0]
    assert isinstance(blk, HandleRef)
    assert blk.original_tokens > engine.handle_threshold
    # The extract is a parsed outline, orders of magnitude smaller.
    assert len(blk.extract) < len(BIG) / 20
    assert "function_0" in blk.extract


def test_small_result_passes_through_untouched(engine):
    engine.record_tool_call("t1", "read_file", {"path": "small.txt"})
    ev = engine.record_tool_result("t1", "just a few words")
    assert isinstance(ev.blocks[0], ToolResult)


def test_expand_is_the_dereference(engine):
    engine.record_tool_call("t1", "read_file", {"path": "src/big.py"})
    ev = engine.record_tool_result("t1", BIG)
    hid = ev.blocks[0].handle_id

    assert engine.expand(hid, max_chars=10**9) == BIG
    assert engine.expand(hid, symbol="function_7").startswith("def function_7")
    assert "function_3" in engine.expand(hid, grep=r"function_3\b")
    assert engine.expand(hid, lines="1-2").startswith("def function_0")


def test_nothing_is_destroyed_only_de_rendered(engine):
    engine.record_tool_call("t1", "read_file", {"path": "a.py"})
    ev1 = engine.record_tool_result("t1", BIG)
    engine.record_tool_call("t2", "read_file", {"path": "a.py"})
    engine.record_tool_result("t2", BIG + "\n# changed")

    a = engine.assemble()
    dropped = [d for d in a.decisions if not d.keep]
    assert dropped, "the first read should have been superseded"
    # ...and it is still fully retrievable.
    assert engine.expand(ev1.blocks[0].handle_id, max_chars=10**9) == BIG


def test_repeated_reads_collapse(engine):
    for i in range(6):
        engine.record_tool_call(f"t{i}", "read_file", {"path": "src/big.py"})
        engine.record_tool_result(f"t{i}", BIG)

    a = engine.assemble()
    kept = [d for d in a.decisions if d.keep]
    # One surviving read of the file (plus its six tool_call events).
    results = [d for d in a.decisions if d.keep and d.seq % 2 == 1]
    assert len(results) == 1
    assert a.tokens < a.raw_tokens


def test_search_log_finds_evicted_material(engine):
    engine.record_tool_call("t1", "bash", {"command": "make build"})
    engine.record_tool_result("t1", "warning: libfoo 1.2 is deprecated\n" + "noise\n" * 900)
    for i in range(2, 8):
        engine.record_tool_call(f"t{i}", "read_file", {"path": f"f{i}.py"})
        engine.record_tool_result(f"t{i}", "x = 1")

    hits = engine.search_log("deprecated")
    assert hits and "deprecated" in hits[0]["snippet"]


# -- pinned zones ----------------------------------------------------------


def test_goal_and_failures_survive_everything(engine):
    engine.set_goal("make the flaky auth test pass", ["pytest tests/auth.py"])
    engine.note_failure("mutex on refresh", "deadlocks under load", "redis >= 7")
    for i in range(40):
        engine.record_tool_call(f"t{i}", "read_file", {"path": f"f{i}.py"})
        engine.record_tool_result(f"t{i}", BIG)

    a = engine.assemble()
    text = "\n".join(b.text for b in a.blocks if isinstance(b, Text))
    assert "flaky auth test" in text
    assert "deadlocks under load" in text
    assert "pytest tests/auth.py" in text


def test_every_user_message_is_kept_verbatim(engine):
    # Human text is cheap; there is nothing to classify and nothing to get wrong.
    for i in range(12):
        engine.record_user(f"instruction number {i}")
    a = engine.assemble()
    text = "\n".join(b.text for b in a.blocks if isinstance(b, Text))
    assert all(f"instruction number {i}" in text for i in range(12))


# -- task frames -----------------------------------------------------------


def test_closing_a_task_makes_its_interior_compactable(engine):
    with engine.task("investigate the connection pool") as t:
        engine.record_tool_call("t1", "read_file", {"path": "pool.py"})
        engine.record_tool_result("t1", "pool internals")
        t.done("not the pool — connections are fine under load")

    a = engine.assemble()
    text = "\n".join(b.text for b in a.blocks if isinstance(b, Text))
    assert "not the pool" in text          # the outcome survives
    assert "pool internals" not in text     # the trace does not


def test_an_unclosed_task_keeps_its_trace(engine):
    with engine.task("still going") as t:
        engine.record_tool_call("t1", "read_file", {"path": "x.py"})
        engine.record_tool_result("t1", "live detail")
    a = engine.assemble()
    text = "\n".join(getattr(b, "text", "") + getattr(b, "content", "") for b in a.blocks)
    assert "live detail" in text


# -- structural validity ---------------------------------------------------


def test_tool_use_and_result_are_one_eviction_unit():
    evs = [
        Event(seq=0, kind=EventKind.TOOL_CALL, blocks=[ToolUse("a", "read", {})]),
        Event(seq=1, kind=EventKind.TOOL_RESULT, blocks=[ToolResult("a", "ok")]),
        Event(seq=2, kind=EventKind.TOOL_CALL, blocks=[ToolUse("b", "read", {})]),
        Event(seq=3, kind=EventKind.TOOL_RESULT, blocks=[ToolResult("b", "ok")]),
    ]
    units = eviction_units(evs)
    assert [[e.seq for e in u] for u in units] == [[0, 1], [2, 3]]


def test_repair_removes_orphans_that_would_400():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "r", "input": {}}]},
        # orphan result: no matching tool_use
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "zz", "content": "x"}]},
    ]
    out = ad.repair(msgs)
    flat = [b for m in out for b in m["content"]]
    assert not any(b.get("type") == "tool_result" for b in flat)
    assert not any(b.get("type") == "tool_use" for b in flat)  # its result is gone too


def test_rendered_messages_are_structurally_valid(engine):
    engine.set_goal("do the thing")
    for i in range(30):
        engine.record_tool_call(f"t{i}", "read_file", {"path": f"f{i}.py"})
        engine.record_tool_result(f"t{i}", BIG)

    msgs = ad.to_messages(engine.assemble().items)
    assert msgs[0]["role"] == "user"
    for prev, nxt in zip(msgs, msgs[1:]):
        assert prev["role"] != nxt["role"], "roles must alternate"

    uses = {b["id"] for m in msgs for b in m["content"] if b.get("type") == "tool_use"}
    results = {
        b["tool_use_id"] for m in msgs for b in m["content"] if b.get("type") == "tool_result"
    }
    assert uses == results


def test_handle_ref_renders_as_a_tool_result(engine):
    engine.record_tool_call("t1", "read_file", {"path": "src/big.py"})
    engine.record_tool_result("t1", BIG)
    msgs = ad.to_messages(engine.assemble().items)
    results = [b for m in msgs for b in m["content"] if b.get("type") == "tool_result"]
    assert results and "expand(" in results[0]["content"]


# -- isolation -------------------------------------------------------------


def test_a_handle_from_another_session_cannot_resolve(tmp_path):
    a = ContextEngine(root=tmp_path, tenant="acme", session="s1")
    a.record_tool_call("t1", "read_file", {"path": "secret.py"})
    hid = a.record_tool_result("t1", BIG).blocks[0].handle_id

    b = ContextEngine(root=tmp_path, tenant="acme", session="s2")
    assert "no such handle" in b.expand(hid)

    c = ContextEngine(root=tmp_path, tenant="other", session="s1")
    assert "no such handle" in c.expand(hid)
    a.close(); b.close(); c.close()


def test_parallel_sessions_do_not_mix(tmp_path):
    sessions = [ContextEngine(root=tmp_path, session=f"s{i}") for i in range(8)]
    for i, e in enumerate(sessions):
        e.record_user(f"private message for {i}")
    for i, e in enumerate(sessions):
        text = "\n".join(b.text for b in e.assemble().blocks if isinstance(b, Text))
        assert f"private message for {i}" in text
        assert not any(f"private message for {j}" in text for j in range(8) if j != i)
    for e in sessions:
        e.close()


def test_tenant_id_cannot_escape_its_directory(tmp_path):
    e = ContextEngine(root=tmp_path, tenant="../../etc", session="s1")
    assert tmp_path in e.store._blobs.root.parents or e.store._blobs.root.is_relative_to(tmp_path)
    e.close()


# -- durability ------------------------------------------------------------


def test_the_log_survives_reopening(tmp_path):
    e = ContextEngine(root=tmp_path, session="s1")
    e.set_goal("persist me")
    e.record_tool_call("t1", "read_file", {"path": "a.py"})
    e.record_tool_result("t1", BIG)
    n = e.store.count()
    e.close()

    again = ContextEngine(root=tmp_path, session="s1")
    assert again.store.count() == n
    text = "\n".join(b.text for b in again.assemble().blocks if isinstance(b, Text))
    assert "persist me" in text
    again.close()


def test_report_shape(engine):
    engine.set_goal("g")
    engine.record_tool_call("t1", "read_file", {"path": "a.py"})
    engine.record_tool_result("t1", BIG)
    r = engine.report()
    assert r["raw_tokens"] >= r["assembled_tokens"]
    assert set(r["zones"]) == {"goal", "invariants", "failures", "handles", "retrieved", "tail"}
    assert r["cache_prefix_tokens"] <= r["assembled_tokens"]


def test_explain_names_a_reason_for_every_event(engine):
    engine.record_tool_call("t1", "read_file", {"path": "a.py"})
    engine.record_tool_result("t1", BIG)
    engine.record_tool_call("t2", "read_file", {"path": "a.py"})
    engine.record_tool_result("t2", BIG)
    out = engine.explain()
    assert "superseded" in out or "duplicate" in out
    assert "saved" in out

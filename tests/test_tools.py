import re

import pytest

from ctxkernel import ContextEngine, budgets
from ctxkernel.ir import HandleRef, ToolResult

BIG = "\n".join(f"def function_{i}(a, b):\n    return a + b" for i in range(400)) + "\nSECRET_SAUCE = 42"


@pytest.fixture
def eng(tmp_path):
    e = ContextEngine(root=tmp_path, session="s1", budget=budgets.small(32_000))
    yield e
    e.close()


def ids_in(text: str) -> list[str]:
    return re.findall(r"\be\d{6}\b", text)


def test_tool_definitions_render_for_each_provider(eng):
    anth = eng.tools("anthropic")
    assert [t["name"] for t in anth] == ["ctx_outline", "ctx_search", "ctx_expand"]
    assert all({"name", "description", "input_schema"} == set(t) for t in anth)
    oai = eng.tools("openai")
    assert all(t["type"] == "function" and "parameters" in t["function"] for t in oai)


def test_a_host_tool_falls_through(eng):
    assert eng.run_tool("read_file", {"path": "x"}) is None


def test_search_reaches_what_left_the_context_and_expand_returns_it_whole(eng):
    eng.tool("read_file", {"path": "src/big.py"}, BIG)  # becomes a handle: extract only
    for i in range(30):
        eng.tool("read_file", {"path": f"m{i}.py"}, "x = 1\n" * 200)
    assert "SECRET_SAUCE = 42" not in eng.messages("text")

    found = eng.run_tool("ctx_search", {"pattern": "SECRET_SAUCE"})
    (item,) = ids_in(found)
    full = eng.run_tool("ctx_expand", {"id": item})
    assert "narrow with grep or lines" in full  # capped, so a page-in cannot flood the window
    # Read through the handle to the blob, narrowed to what was asked for.
    assert eng.run_tool("ctx_expand", {"id": item, "grep": "SECRET"}).endswith("SECRET_SAUCE = 42")


def test_expand_accepts_every_kind_of_id(eng):
    with eng.task("look at config") as t:
        eng.tool("read_file", {"path": "config.py"}, "RETRY = 3")
        eng.tool("read_file", {"path": "config.py"}, "RETRY = 5")
        t.done("retry was raised to 5")
    hid = next(b.handle_id for ev in [eng.tool("read_file", {"path": "big.py"}, BIG)]
               for b in ev.blocks if isinstance(b, HandleRef))

    assert eng.run_tool("ctx_expand", {"id": "res:file:config.py"}).endswith("RETRY = 5")  # latest
    task_id = next(n for n in eng.graph.nodes if n.startswith("task:"))
    steps = eng.run_tool("ctx_expand", {"id": task_id})
    assert "retry was raised to 5" in steps and steps.count("read_file") == 2
    assert "SECRET_SAUCE" in eng.run_tool("ctx_expand", {"id": hid, "grep": "SECRET"})
    assert "no item" in eng.run_tool("ctx_expand", {"id": "e999999"})


def test_outline_shows_what_exists_with_ids_to_expand(eng):
    eng.set_goal("fix the retry bug")
    with eng.task("check config") as t:
        eng.tool("read_file", {"path": "config.py"}, "RETRY = 3")
        t.done("too low")
    for _ in range(3):
        eng.tool("read_file", {"path": "src/app.py"}, "app")
    out = eng.run_tool("ctx_outline", {})
    assert "goal: fix the retry bug" in out
    assert "✓ task:" in out and "check config  → too low" in out
    first = next(ln for ln in out.splitlines() if ln.strip().startswith("res:"))
    assert "res:file:src/app.py" in first  # most used first
    assert ids_in(first)


def test_a_page_in_is_not_intercepted_again(eng):
    eng.tool("read_file", {"path": "src/big.py"}, BIG)
    item = ids_in(eng.run_tool("ctx_search", {"pattern": "SECRET_SAUCE"}))[0]
    answer = eng.run_tool("ctx_expand", {"id": item})
    ev = eng.tool("ctx_expand", {"id": item}, answer)
    assert isinstance(ev.blocks[0], ToolResult)  # not a HandleRef of its own answer


def test_a_bad_call_is_an_answer_not_a_crash(eng):
    assert "failed" in eng.run_tool("ctx_search", {})
    assert "failed" in eng.run_tool("ctx_search", {"pattern": "("})

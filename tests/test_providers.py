"""The same assembly must render correctly for every provider.

If anything provider-specific leaked out of adapters/, these fail.
"""

import json

import pytest

from ctxkernel import ContextEngine, budgets
from ctxkernel.adapters import anthropic as an
from ctxkernel.adapters import generic as ge
from ctxkernel.adapters import openai as oa

BIG = "\n".join(f"def op_{i}(a, b):\n    return a + b" for i in range(400))


@pytest.fixture
def eng(tmp_path):
    e = ContextEngine(root=tmp_path, session="s", budget=budgets.small(32_000))
    e.set_goal("do the thing")
    for i in range(12):
        e.record_tool_call(f"t{i}", "read_file", {"path": f"f{i % 4}.py"})
        e.record_tool_result(f"t{i}", BIG)
    e.note_failure("approach A", "races under load")
    yield e
    e.close()


def test_one_assembly_renders_to_both_providers(eng):
    items = eng.assemble().items
    a_msgs = an.to_messages(items)
    o_msgs = oa.to_messages(items)
    assert a_msgs and o_msgs
    # Same content, genuinely different shapes.
    assert any(m["role"] == "tool" for m in o_msgs)
    assert not any(m["role"] == "tool" for m in a_msgs)


def test_anthropic_pairs_and_alternates(eng):
    msgs = an.to_messages(eng.assemble().items)
    assert msgs[0]["role"] == "user"
    for a, b in zip(msgs, msgs[1:]):
        assert a["role"] != b["role"]
    uses = {b["id"] for m in msgs for b in m["content"] if b.get("type") == "tool_use"}
    results = {
        b["tool_use_id"] for m in msgs for b in m["content"] if b.get("type") == "tool_result"
    }
    assert uses == results


def test_openai_every_tool_message_answers_a_call(eng):
    msgs = oa.to_messages(eng.assemble().items)
    calls = {tc["id"] for m in msgs for tc in m.get("tool_calls") or []}
    answers = {m["tool_call_id"] for m in msgs if m["role"] == "tool"}
    assert calls == answers
    assert calls, "expected at least one surviving tool call"


def test_openai_arguments_are_a_json_string(eng):
    msgs = oa.to_messages(eng.assemble().items)
    tc = next(tc for m in msgs for tc in m.get("tool_calls") or [])
    assert isinstance(tc["function"]["arguments"], str)
    assert "path" in json.loads(tc["function"]["arguments"])


def test_openai_repair_drops_orphans():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"id": "a", "type": "function",
                                              "function": {"name": "r", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "zz", "content": "orphan"},
    ]
    out = oa.repair(msgs)
    assert not any(m["role"] == "tool" for m in out)
    assert not any(m.get("tool_calls") for m in out)


def test_openai_round_trips_through_the_ir():
    original = [
        {"role": "user", "content": "read it"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                                              "function": {"name": "read_file",
                                                           "arguments": '{"path": "a.py"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "file body"},
    ]
    parsed = oa.parse_messages(original)
    items = [(role, b) for role, blocks in parsed for b in blocks]
    back = oa.to_messages(items)
    assert back[0]["content"] == "read it"
    tc = back[1]["tool_calls"][0]
    assert json.loads(tc["function"]["arguments"]) == {"path": "a.py"}
    assert back[2] == {"role": "tool", "tool_call_id": "c1", "content": "file body"}


def test_handle_becomes_a_tool_message_with_expand_hint(eng):
    msgs = oa.to_messages(eng.assemble().items)
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert any("expand(" in m["content"] for m in tool_msgs)


CORE_MODULES = ("ir", "predicates", "assembler", "codecs", "identity", "store", "tokenizer")


def _imported_names(path: str) -> set[str]:
    """Every module named by an import statement, anywhere in the file."""
    import ast

    names: set[str] = set()
    for node in ast.walk(ast.parse(open(path).read())):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def test_no_core_module_imports_a_provider():
    """The invariant is about code, not prose: core modules may *discuss*
    adapters in a docstring, but must never import one."""
    import importlib

    banned = ("adapters", "anthropic", "openai", "google", "cohere", "mistral")
    for mod in CORE_MODULES:
        m = importlib.import_module(f"ctxkernel.{mod}")
        for name in _imported_names(m.__file__):
            assert not any(b in name.lower() for b in banned), f"{mod}.py imports {name}"


def test_core_works_with_no_provider_sdk_installed():
    """No provider SDK is installed in this environment, so a green run of the
    whole suite is itself the proof — but assert it explicitly."""
    import importlib.util

    assert importlib.util.find_spec("anthropic") is None
    assert importlib.util.find_spec("openai") is None


# -- anything else, including self-hosted ---------------------------------


def test_generic_chat_needs_no_provider_schema(eng):
    msgs = ge.to_chat(eng.assemble().items)
    assert msgs and all(set(m) == {"role", "content"} for m in msgs)
    assert all(isinstance(m["content"], str) for m in msgs)
    assert any("expand(" in m["content"] for m in msgs)


def test_generic_text_flattens_to_one_prompt(eng):
    prompt = ge.to_text(eng.assemble().items)
    assert isinstance(prompt, str)
    assert "### User" in prompt
    assert "do the thing" in prompt          # the goal survives
    assert "races under load" in prompt      # the failure ledger survives


def test_generic_render_is_overridable(eng):
    from ctxkernel.ir import Text as T

    def shout(b):
        return ge.block_to_text(b).upper()

    assert "DO THE THING" in ge.to_text(eng.assemble().items, render=shout)


# -- the three-concept API -------------------------------------------------


def test_tool_records_a_whole_round_trip(tmp_path):
    from ctxkernel import ContextEngine

    e = ContextEngine(root=tmp_path, session="q")
    e.set_goal("g")
    e.tool("read_file", {"path": "a.py"}, BIG)
    e.tool("bash", {"command": "pytest"}, "1 failed, 20 passed")
    # call + result appended as a coherent pair, ids generated for you
    assert e.store.count() == 5
    msgs = e.messages("openai")
    calls = {tc["id"] for m in msgs for tc in m.get("tool_calls") or []}
    answers = {m["tool_call_id"] for m in msgs if m["role"] == "tool"}
    assert calls == answers and calls
    e.close()


def test_messages_supports_every_fmt(tmp_path):
    from ctxkernel import ContextEngine

    e = ContextEngine(root=tmp_path, session="q2")
    e.set_goal("g")
    e.tool("read_file", {"path": "a.py"}, BIG)
    assert isinstance(e.messages("anthropic"), list)
    assert isinstance(e.messages("openai"), list)
    assert isinstance(e.messages("chat"), list)
    assert isinstance(e.messages("text"), str)
    with pytest.raises(ValueError):
        e.messages("nope")
    e.close()


def test_replay_import_and_growth_curve(tmp_path):
    from ctxkernel import ContextEngine
    from ctxkernel.replay import import_messages, import_records, replay

    e = ContextEngine(root=tmp_path, session="q3")
    n = import_messages(
        e,
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": BIG},
        ],
        fmt="openai",
    )
    assert n == 3

    e2 = ContextEngine(root=tmp_path, session="q4")
    records = [{"t": "call", "id": f"x{i}", "tool": "read_file", "args": {"path": "a.py"}}
               for i in range(30)]
    curve = replay(
        e2, records, mapper=lambda r: ("call", r["id"], r["tool"], r["args"]), every=10
    )
    assert len(curve) == 3
    assert all(c["context_tokens"] <= c["log_tokens"] for c in curve)
    e.close(); e2.close()

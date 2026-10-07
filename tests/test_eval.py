import json

from ctxkernel import ContextEngine, budgets
from ctxkernel.eval import evaluate, identifiers, load_claude_code


def use(tid, name, **inp):
    return {"role": "assistant", "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}]}


def result(tid, text):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}]}


def buried_trace(n_filler: int = 40) -> list[dict]:
    """The agent reads config early, does a lot of other work, then edits the
    value it read at the start -- an action that depends on old evidence."""
    msgs = [
        {"role": "user", "content": "make the retry test pass"},
        use("t0", "Read", file_path="config.py"),
        result("t0", "RETRY_LIMIT = 3\nBACKOFF_MS = 250"),
    ]
    for i in range(n_filler):
        msgs += [
            use(f"f{i}", "Read", file_path=f"src/mod_{i}.py"),
            result(f"f{i}", "\n".join(f"x_{i}_{j} = {j}" for j in range(120))),
        ]
    msgs.append(use("t9", "Edit", file_path="config.py", old_string="RETRY_LIMIT = 3", new_string="RETRY_LIMIT = 5"))
    return msgs


def engine_with(budget):
    return lambda root: ContextEngine(root=root, session="eval", budget=budget)


def test_identifiers_are_names_not_words():
    ids = identifiers("the error in src/auth.py: RETRY_LIMIT exceeded after 3 tries, see refreshToken")
    assert {"src/auth.py", "RETRY_LIMIT", "refreshToken"} <= ids
    assert not ids & {"error", "exceeded", "tries", "after"}


def test_a_policy_that_ages_out_the_evidence_scores_below_one(tmp_path):
    results, summaries = evaluate(
        buried_trace(),
        {"tight": engine_with(budgets.small(32_000)), "roomy": engine_with(budgets.window(10_000_000))},
        workdir=tmp_path,
    )
    final = {r.policy: r for r in results if r.step == max(x.step for x in results)}
    assert final["roomy"].recall == 1.0
    assert final["tight"].recall < 1.0
    assert "RETRY_LIMIT" in final["tight"].missing
    by = {s.policy: s for s in summaries}
    assert by["tight"].mean_context_tokens < by["roomy"].mean_context_tokens


def test_what_the_user_said_does_not_count_as_evidence(tmp_path):
    msgs = [
        {"role": "user", "content": "rename OLD_NAME in src/app.py"},
        use("t0", "Edit", file_path="src/app.py", old_string="OLD_NAME", new_string="NEW_NAME"),
    ]
    results, _ = evaluate(msgs, {"p": engine_with(budgets.small(32_000))}, workdir=tmp_path)
    assert results == []  # pinned for every policy alike: nothing to tell apart


def test_claude_code_logs_are_merged_back_into_turns(tmp_path):
    lines = [
        {"type": "user", "message": {"role": "user", "content": "hi"}},
        {"type": "assistant", "message": {"id": "m1", "role": "assistant", "content": [{"type": "thinking", "thinking": "hm"}]}},
        {"type": "assistant", "message": {"id": "m1", "role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "a.py"}}]}},
        {"type": "assistant", "isSidechain": True, "message": {"id": "s1", "role": "assistant", "content": []}},
        {"type": "attachment", "foo": 1},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "x"}]}]}},
    ]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines))
    msgs = load_claude_code(p)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [b["type"] for b in msgs[1]["content"]] == ["thinking", "tool_use"]


def test_openai_traces_load_as_replayable_turns():
    from ctxkernel.eval import load_openai

    msgs = load_openai([
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": [{"type": "text", "text": "fix it"}]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "execute_bash", "arguments": "{\"command\": \"ls\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.py"},
    ])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"][0] == {"type": "tool_use", "id": "c1", "name": "execute_bash", "input": {"command": "ls"}}
    assert msgs[2]["content"][0]["tool_use_id"] == "c1"


def test_openhands_tools_get_resources(tmp_path):
    from ctxkernel.eval import openhands_identity

    e = ContextEngine(root=tmp_path, session="s")
    openhands_identity(e)
    view = e.identity.resolve("str_replace_editor", {"command": "view", "path": "/repo/a.py"})
    edit = e.identity.resolve("str_replace_editor", {"command": "str_replace", "path": "/repo/a.py"})
    assert str(view.resource) == str(edit.resource) == "file:/repo/a.py"
    assert not view.is_write and edit.is_write
    e.close()

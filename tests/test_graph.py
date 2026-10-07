import re
import sys
import time
import types

import pytest

from ctxkernel import ContextEngine, budgets
from ctxkernel.decision import NullDecision, ProximityDecision, gather
from ctxkernel.graph import EdgeKind, NodeKind
from ctxkernel.ir import ResourceId
from ctxkernel.predicates import decide

CONFIG = "RETRY_LIMIT = 3  # the one line that explains the failure at step 300"


def filler(i: int) -> str:
    return f"# module {i}\n" + "\n".join(f"x_{i}_{j} = {j}" for j in range(120))


@pytest.fixture
def eng(tmp_path):
    e = ContextEngine(root=tmp_path, session="s1", budget=budgets.small(32_000))
    yield e
    e.close()


class Fixed:
    """A decision model that answers from a table -- the test stands in for Jev."""

    name = "fixed"

    def __init__(self, want: dict[str, float] | None = None, *, delay: float = 0.0, boom: bool = False):
        self.want = want or {}
        self.delay = delay
        self.boom = boom
        self.calls = 0

    def score(self, state, candidates):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.boom:
            raise RuntimeError("model is down")
        return {c.node_id: self.want.get(c.node_id, 0.0) for c in candidates}


def bury_config(eng):
    """Read config early, then enough other work that the tail ages it out."""
    eng.set_goal("make the retry test pass")
    eng.tool("read_file", {"path": "config.py"}, CONFIG)
    for i in range(40):
        eng.tool("read_file", {"path": f"mod_{i}.py"}, filler(i))
    # A recent result derived from config keeps it in the neighborhood of the
    # current work. (Not a grep: a grep of config.py is a newer observation of
    # it, and the cascade would supersede the full read.)
    run_derived_from(eng, "config.py")


def run_derived_from(eng, path: str) -> None:
    eng.record_tool_call("t-run", "run", {"cmd": "pytest tests/test_retry.py"})
    eng.record_tool_result(
        "t-run", "1 failed: gave up after 3 retries", derived_from=(ResourceId("file", path),)
    )


def text_of(items) -> str:
    return "\n".join(getattr(b, "content", "") or getattr(b, "text", "") for _, b in items)


# -- building ----------------------------------------------------------------


def test_a_call_and_its_result_are_one_node(eng):
    eng.tool("read_file", {"path": "a.py"}, "print(1)")
    (call,) = [n for n in eng.graph.nodes.values() if n.kind is NodeKind.CALL]
    assert len(call.seqs) == 2  # tool_use event + tool_result event


def test_every_edge_comes_from_a_field_on_the_event(eng):
    eng.tool("read_file", {"path": "a.py"}, "v1")
    eng.tool("write_file", {"path": "a.py", "content": "v2"}, "ok")
    eng.tool("read_file", {"path": "a.py"}, "v2")
    first, write, second = eng.graph.recent_calls(3)

    assert (EdgeKind.READS, "res:file:a.py") in eng.graph.out_edges(first.id)
    assert (EdgeKind.WRITES, "res:file:a.py") in eng.graph.out_edges(write.id)
    assert (EdgeKind.REPLACES, first.id) in eng.graph.out_edges(second.id)


def test_identical_output_links_as_a_duplicate(eng):
    eng.tool("run", {"cmd": "pytest"}, "3 failed")
    eng.tool("run", {"cmd": "pytest"}, "3 failed")
    older, newer = eng.graph.recent_calls(2)
    assert (EdgeKind.DUPLICATE_OF, older.id) in eng.graph.out_edges(newer.id)


def test_tasks_nest_in_the_containment_tree(eng):
    with eng.task("outer") as outer:
        with eng.task("inner") as inner:
            eng.tool("read_file", {"path": "a.py"}, "x")
            inner.done("found it")
        outer.done("fixed")
    root_tasks = [n for n in eng.graph.children("session") if n.kind is NodeKind.TASK]
    assert [t.label for t in root_tasks] == ["outer  → fixed"]
    (inner_node,) = [n for n in eng.graph.children(root_tasks[0].id) if n.kind is NodeKind.TASK]
    assert [c.kind for c in eng.graph.children(inner_node.id)] == [NodeKind.CALL]


def test_containment_does_not_make_everything_a_neighbor(eng):
    eng.tool("read_file", {"path": "a.py"}, "x")
    eng.tool("read_file", {"path": "b.py"}, "y")
    a, b = eng.graph.recent_calls(2)
    assert b.id not in eng.graph.neighbors([a.id], hops=3)


def test_the_graph_is_rebuilt_from_the_log_on_reopen(tmp_path):
    e = ContextEngine(root=tmp_path, session="s1")
    with e.task("t") as t:
        e.tool("read_file", {"path": "a.py"}, "x")
        t.done("ok")
    before = {nid: (n.kind, n.seqs) for nid, n in e.graph.nodes.items()}
    e.close()

    e2 = ContextEngine(root=tmp_path, session="s1")
    assert {nid: (n.kind, n.seqs) for nid, n in e2.graph.nodes.items()} == before
    e2.close()


# -- candidates ---------------------------------------------------------------


def test_stale_nodes_are_never_offered_to_a_model(eng):
    eng.tool("read_file", {"path": "a.py"}, "old")
    eng.tool("read_file", {"path": "a.py"}, "new")
    run_derived_from(eng, "a.py")
    old, new, _ = eng.graph.recent_calls(3)
    events = eng.store.events()
    _, cands = gather(eng.graph, events, decide(events), recent=1)
    ids = {c.node_id for c in cands}
    assert old.id not in ids  # superseded: a fact, not up for a vote
    assert new.id in ids


def test_the_goal_reaches_the_model_verbatim(eng):
    eng.set_goal("line one\nline two", ["pytest -q"])
    eng.tool("read_file", {"path": "a.py"}, "x")
    events = eng.store.events()
    state, _ = gather(eng.graph, events, decide(events))
    assert state.goal.startswith("line one\nline two")
    assert "pytest -q" in state.goal


# -- anchors ------------------------------------------------------------------


def test_null_decision_changes_nothing(eng):
    bury_config(eng)
    assert isinstance(eng.decision, NullDecision)
    assert eng.anchors == []
    assert eng.assemble().report()["zones"]["retrieved"] == 0


def test_an_anchor_brings_back_what_the_tail_aged_out(tmp_path):
    model = Fixed({"res:file:config.py": 0.9})
    e = ContextEngine(
        root=tmp_path, session="s1", budget=budgets.small(32_000),
        decision=model, auto_anchor=False,
    )
    bury_config(e)
    a = e.assemble()
    assert CONFIG not in text_of(a.items)  # aged out by recency alone

    run = e.reanchor(wait=True)
    assert run.fallback is None
    assert [x.node_id for x in e.anchors] == ["res:file:config.py"]

    a = e.assemble()
    retrieved = next(z for z in a.zones if z.name == "retrieved")
    assert CONFIG in text_of(retrieved.items)
    # Whole units only: the recalled result arrives with its tool_use.
    kinds = [type(b).__name__ for _, b in retrieved.items]
    assert kinds.count("ToolUse") == kinds.count("ToolResult")
    e.close()


def test_recalled_material_is_still_valid_for_the_provider(tmp_path):
    from ctxkernel.adapters import anthropic as ad

    e = ContextEngine(
        root=tmp_path, session="s1", budget=budgets.small(32_000),
        decision=Fixed({"res:file:config.py": 0.9}), auto_anchor=False,
    )
    bury_config(e)
    e.reanchor(wait=True)
    msgs = ad.to_messages(e.assemble().items)
    uses = {b["id"] for m in msgs if m["role"] == "assistant" for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_use"}
    results = {b["tool_use_id"] for m in msgs if m["role"] == "user" for b in m["content"]
               if isinstance(b, dict) and b.get("type") == "tool_result"}
    assert uses == results
    e.close()


def test_a_broken_model_keeps_the_previous_anchors(tmp_path):
    good = Fixed({"res:file:config.py": 0.9})
    e = ContextEngine(root=tmp_path, session="s1", decision=good, auto_anchor=False)
    bury_config(e)
    e.reanchor(wait=True)
    kept = e.anchors

    e.decision = Fixed(boom=True)
    run = e.reanchor(wait=True)
    assert run.fallback and "model is down" in run.fallback
    assert e.anchors == kept
    assert e.messages("chat")  # the turn still assembles
    e.close()


def test_a_slow_model_never_blocks_the_turn(tmp_path):
    slow = Fixed({"res:file:config.py": 0.9}, delay=0.3)
    e = ContextEngine(root=tmp_path, session="s1", decision=slow, auto_anchor=False)
    bury_config(e)

    t0 = time.perf_counter()
    run = e.reanchor(wait=True, timeout=0.02)
    assert time.perf_counter() - t0 < 0.2
    assert run.fallback == "timeout"
    assert e.anchors == []

    e._inflight.result(timeout=2)  # the late answer still lands
    assert [a.node_id for a in e.anchors] == ["res:file:config.py"]
    e.close()


def test_anchors_and_every_score_are_kept_in_the_log(tmp_path):
    e = ContextEngine(
        root=tmp_path, session="s1",
        decision=Fixed({"res:file:config.py": 0.9}), auto_anchor=False,
    )
    bury_config(e)
    e.reanchor(wait=True)
    e.close()

    e2 = ContextEngine(root=tmp_path, session="s1")
    assert [a.node_id for a in e2.anchors] == ["res:file:config.py"]
    (run,) = e2.store.anchor_runs()
    assert run["scores"]["res:file:config.py"] == 0.9
    assert len(run["scores"]) == run["candidates"] > 1
    e2.close()


def test_ranking_runs_in_the_background_and_never_delays_a_step(tmp_path):
    slow = Fixed({"res:file:a.py": 0.9}, delay=0.2)
    e = ContextEngine(root=tmp_path, session="s1", decision=slow)
    t0 = time.perf_counter()
    for i in range(5):
        e.tool("read_file", {"path": "a.py"}, "x")
        e.tool("read_file", {"path": f"b{i}.py"}, "y")
    assert time.perf_counter() - t0 < 0.15  # recording never waited on the model
    assert slow.calls == 1  # one run at a time; triggers during it are skipped, not queued

    e.settle(timeout=5)  # an eval's view: bring the ranking up to date
    assert slow.calls == 2
    assert e.anchors[0].node_id == "res:file:a.py"
    e.close()


def test_names_reach_what_links_cannot(eng):
    """An old result that shares a rare name with the agent's plan becomes a
    candidate even though no file, task or derivation links them."""
    eng.set_goal("make the retry test pass")
    eng.tool("run", {"cmd": "cat settings.ini"}, "RETRY_LIMIT = 3\nTIMEOUT_MS = 50")
    for i in range(20):
        eng.tool("read_file", {"path": f"src/mod_{i}.py"}, filler(i))
    from ctxkernel.ir import Text
    eng.record_model([Text("The test gives up early; RETRY_LIMIT is probably too low.")])
    events = eng.store.events()
    state, cands = gather(eng.graph, events, decide(events))
    assert "RETRY_LIMIT" in state.plan
    old = next(c for c in cands if "settings.ini" in c.label)
    assert old.hops is None  # reached by name, not by link
    assert "RETRY_LIMIT" in old.links[0]


def test_proximity_scores_are_the_prior(eng):
    bury_config(eng)
    events = eng.store.events()
    state, cands = gather(eng.graph, events, decide(events))
    scores = ProximityDecision().score(state, cands)
    assert scores == {c.node_id: c.prior for c in cands}
    assert [c.prior for c in cands] == sorted((c.prior for c in cands), reverse=True)


# -- Jev adapter -----------------------------------------------------------------


def test_jev_asks_one_yes_no_question_per_candidate(eng, monkeypatch):
    sent = {}
    judged = []

    class Noul:
        def __init__(self, instructions):
            self.instructions = instructions

    class Client:
        def system_one(self, *, state, questions):
            if "failed" in questions:  # the judge's request, not the ranking's
                judged.append(questions["failed"].instructions)
                return types.SimpleNamespace(nouls={"failed": types.SimpleNamespace(noul=0.1)})
            sent.update(state=state, questions=questions)
            nouls = {k: types.SimpleNamespace(noul=0.8 if "config.py" in str(q.instructions) else 0.1)
                     for k, q in questions.items()}
            return types.SimpleNamespace(nouls=nouls)

    monkeypatch.setitem(sys.modules, "typesafe_sdk", types.SimpleNamespace(Noul=Noul))
    from ctxkernel.adapters.decision import JevDecision

    bury_config(eng)
    eng.decision = JevDecision(client=Client())
    run = eng.reanchor(wait=True)

    assert run.fallback is None
    assert len(sent["questions"]) == run.candidates
    assert sent["state"]["goal"] == "make the retry test pass"
    assert "res:file:config.py" in [a.node_id for a in eng.anchors]
    # Each item goes over whole: content and connections, not just a name.
    items = [q.instructions["item"] for q in sent["questions"].values()]
    config = next(i for i in items if i["label"].startswith("file:config.py"))
    assert config["content"] == CONFIG
    assert any(l.startswith("used by: run") for l in config["connected_to"])
    # The latest step was judged once, as its own request, carrying the step.
    assert len(judged) == 1 and "gave up after 3 retries" in judged[0]["step"]["content"]


def test_a_candidate_carries_its_content_and_connections(eng):
    bury_config(eng)
    events = eng.store.events()
    _, cands = gather(eng.graph, events, decide(events))
    config = next(c for c in cands if c.node_id == "res:file:config.py")
    assert config.content == CONFIG  # a resource speaks through its latest read
    assert any("gave up after 3 retries" in l for l in config.links)


def test_the_model_knows_the_task_and_the_latest_user_input(eng):
    eng.set_goal("ship the release")
    eng.record_user("actually, skip the changelog for now")
    with eng.task("fix the build"):
        with eng.task("check the lockfile"):
            eng.tool("read_file", {"path": "a.py"}, "x")
            events = eng.store.events()
            state, _ = gather(eng.graph, events, decide(events),
                              task=" > ".join(f.intent for f in eng._task_stack))
    assert state.task == "fix the build > check the lockfile"
    assert state.user_input == "actually, skip the changelog for now"


def test_jev_without_its_sdk_is_a_fallback_not_a_failure(eng, monkeypatch):
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)  # import raises
    from ctxkernel.adapters.decision import JevDecision

    bury_config(eng)
    eng.decision = JevDecision()
    run = eng.reanchor(wait=True)
    assert run.fallback and "Error" in run.fallback
    assert eng.messages("chat")


# -- failure judge ----------------------------------------------------------------


def test_a_failed_step_is_recorded_as_a_failure_note(tmp_path):
    e = ContextEngine(root=tmp_path, session="s1", decision=ProximityDecision())
    e.set_goal("ship it")
    e.tool("run", {"cmd": "make build"}, "compiling...\nerror: linker cannot find -lssl\nmake: *** [all] Error 1")
    e.settle(timeout=5)
    notes = [ev for ev in e.store.events() if ev.kind.value == "failure"]
    assert len(notes) == 1
    text = notes[0].blocks[0].text
    assert "make build" in text and "linker cannot find -lssl" in text  # quoted, not written
    assert notes[0].meta["auto"] is True
    # ...and, being a failure note, it is pinned into every context from now on.
    assert "linker cannot find -lssl" in e.messages("text")
    e.close()


def test_a_successful_step_records_nothing(tmp_path):
    e = ContextEngine(root=tmp_path, session="s1", decision=ProximityDecision())
    e.tool("run", {"cmd": "make build"}, "build ok in 3.2s")
    e.settle(timeout=5)
    assert not [ev for ev in e.store.events() if ev.kind.value == "failure"]
    e.close()


def test_each_step_is_judged_once(tmp_path):
    e = ContextEngine(root=tmp_path, session="s1", decision=ProximityDecision())
    e.tool("run", {"cmd": "pytest"}, "FAILED tests/test_a.py::test_x")
    e.settle(timeout=5)
    e.settle(timeout=5)
    e.assemble()
    assert len([ev for ev in e.store.events() if ev.kind.value == "failure"]) == 1
    e.close()


def test_a_broken_judge_records_nothing_and_breaks_nothing(tmp_path):
    class BrokenJudge(ProximityDecision):
        def judge(self, state, step, questions):
            raise RuntimeError("judge is down")

    e = ContextEngine(root=tmp_path, session="s1", decision=BrokenJudge())
    e.tool("run", {"cmd": "make"}, "error: boom")
    e.settle(timeout=5)
    assert not [ev for ev in e.store.events() if ev.kind.value == "failure"]
    assert e.messages("chat")
    e.close()


def test_the_default_engine_judges_nothing(eng):
    eng.tool("run", {"cmd": "make"}, "error: boom")
    eng.settle()
    assert not [ev for ev in eng.store.events() if ev.kind.value == "failure"]


def test_a_call_is_not_judged_before_its_result_arrives(tmp_path):
    e = ContextEngine(root=tmp_path, session="s1", decision=ProximityDecision())
    e.record_tool_call("t1", "run", {"cmd": "make"})
    e.settle(timeout=5)  # a refresh between the call and its result
    e.record_tool_result("t1", "error: linker cannot find -lssl")
    e.settle(timeout=5)
    notes = [ev for ev in e.store.events() if ev.kind.value == "failure"]
    assert len(notes) == 1 and "-lssl" in notes[0].blocks[0].text
    e.close()


def test_a_decision_model_brings_ranked_defaults(tmp_path):
    e = ContextEngine(root=tmp_path, session="a", decision=ProximityDecision())
    # It swaps, so recency keeps the whole target and ranking trades within it.
    assert e.assembler.budget.tail_frac == 1.0 and e.anchor_threshold == 0.0
    e.close()
    e = ContextEngine(root=tmp_path, session="b", decision=ProximityDecision(),
                      budget=budgets.small(32_000), anchor_threshold=0.4)
    assert e.assembler.budget == budgets.small(32_000) and e.anchor_threshold == 0.4
    e.close()
    e = ContextEngine(root=tmp_path, session="c")
    assert e.assembler.budget.tail_frac == 0.75  # no model: nothing changes
    e.close()


# -- swap: recency is the floor ----------------------------------------------------


class Swapping(Fixed):
    swap_threshold = 0.7


def twin(tmp_path, model, name):
    from dataclasses import replace

    budget = replace(budgets.small(32_000), tail_frac=1.0)
    e = ContextEngine(root=tmp_path / name, session="s", budget=budget, decision=model, auto_anchor=False)
    bury_config(e)
    return e


def test_without_a_confident_score_swap_is_exactly_recency(tmp_path):
    base = twin(tmp_path, NullDecision(), "base")
    unsure = twin(tmp_path, Swapping({"res:file:config.py": 0.6}), "unsure")
    unsure.reanchor(wait=True)
    assert unsure.anchors  # it did rank...
    assert unsure.messages("text") == base.messages("text")  # ...and changed nothing
    base.close(), unsure.close()


def test_a_confident_score_swaps_in_without_touching_the_newest(tmp_path):
    base = twin(tmp_path, NullDecision(), "base")
    sure = twin(tmp_path, Swapping({"res:file:config.py": 0.9}), "sure")
    sure.reanchor(wait=True)
    a, b = sure.assemble(), base.assemble()
    assert CONFIG in text_of(a.items) and CONFIG not in text_of(b.items)
    tail = next(z for z in a.zones if z.name == "tail")
    newest = [blk for _, blk in b.zones[-1].items][-2 * sure.assembler.budget.protect_recent:]
    assert all(blk in [x for _, x in tail.items] for blk in newest)
    budget = sure.assembler.budget.tail_budget
    assert sum(z.tokens for z in a.zones if z.name in ("tail", "retrieved")) <= budget * 1.05
    base.close(), sure.close()


# -- full content or summary -------------------------------------------------------

BIG = "\n".join(f"def function_{i}(a, b):\n    return a + b" for i in range(400)) + "\nSECRET_SAUCE = 42"


def test_full_content_replaces_the_summary_when_the_work_needs_it(tmp_path):
    from ctxkernel.ir import Text

    e = ContextEngine(root=tmp_path, session="s", decision=ProximityDecision())
    e.tool("read_file", {"path": "big.py"}, BIG)
    e.settle(timeout=5)
    assert "SECRET_SAUCE = 42" not in e.messages("text")  # summary only

    e.record_model([Text("I need the value of SECRET_SAUCE from big.py.")])
    e.tool("read_file", {"path": "notes.txt"}, "nothing here")
    e.settle(timeout=5)
    out = e.messages("text")
    assert "SECRET_SAUCE = 42" in out  # read in full, because the plan named it
    e.close()


def test_jev_decides_full_or_summary_in_one_request(tmp_path, monkeypatch):
    from ctxkernel.ir import Text

    asked = []

    class Noul:
        def __init__(self, instructions):
            self.instructions = instructions

    class Client:
        def system_one(self, *, state, questions):
            full = [k for k in questions if re.fullmatch(r"f\d+", k)]  # not "failed"
            if full:
                asked.append(questions)
            return types.SimpleNamespace(nouls={
                k: types.SimpleNamespace(noul=0.9 if k in full else 0.1) for k in questions
            })

    monkeypatch.setitem(sys.modules, "typesafe_sdk", types.SimpleNamespace(Noul=Noul))
    from ctxkernel.adapters.decision import JevDecision

    e = ContextEngine(root=tmp_path, session="s", decision=JevDecision(client=Client()))
    e.tool("read_file", {"path": "big.py"}, BIG)
    e.record_model([Text("check SECRET_SAUCE")])
    e.tool("read_file", {"path": "x.txt"}, "x")
    e.settle(timeout=5)
    item = asked[-1]["f0"].instructions["item"]
    assert "SECRET_SAUCE" in item["only_in_full_content"]
    assert "SECRET_SAUCE = 42" in e.messages("text")
    e.close()

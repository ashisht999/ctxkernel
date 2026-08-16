from ctxkernel.ir import Event, EventKind, ResourceId, Text
from ctxkernel.predicates import Reason, decide

F = ResourceId("file", "src/auth.py")
G = ResourceId("file", "src/db.py")


def ev(seq, kind=EventKind.TOOL_RESULT, **kw):
    kw.setdefault("blocks", [Text(f"e{seq}")])
    return Event(seq=seq, kind=kind, **kw)


def reasons(events, **kw):
    return {d.seq: d.reason for d in decide(events, **kw)}


def test_superseded_by_later_read_of_same_resource():
    r = reasons([ev(0, resource=F), ev(1, resource=G), ev(2, resource=F)])
    assert r[0] is Reason.SUPERSEDED
    assert r[1] is Reason.KEEP
    assert r[2] is Reason.KEEP  # newest read survives


def test_duplicate_keeps_the_later_copy():
    r = reasons([ev(0, digest="abc"), ev(1, digest="xyz"), ev(2, digest="abc")])
    assert r[0] is Reason.DUPLICATE
    assert r[2] is Reason.KEEP


def test_invalidated_by_a_later_write_to_a_source():
    # A test result derived from db.py, then db.py is written.
    events = [
        ev(0, resource=ResourceId("shell", "pytest"), derived_from=(G,)),
        ev(1, resource=G, is_write=True),
    ]
    r = reasons(events)
    assert r[0] is Reason.INVALIDATED


def test_invalidation_outranks_supersession():
    events = [
        ev(0, resource=F, derived_from=(G,)),
        ev(1, resource=G, is_write=True),
        ev(2, resource=F),
    ]
    # Both predicates fire on e0; the stronger claim wins for reporting.
    assert reasons(events)[0] is Reason.INVALIDATED


def test_writes_are_history_and_are_not_superseded():
    events = [ev(0, resource=F, is_write=True), ev(1, resource=F)]
    r = reasons(events)
    assert r[0] is Reason.KEEP
    assert r[1] is Reason.KEEP


def test_pinned_kinds_are_never_candidates():
    events = [
        ev(0, kind=EventKind.GOAL, digest="same"),
        ev(1, kind=EventKind.FAILURE, digest="same"),
        ev(2, kind=EventKind.INVARIANT, digest="same"),
        ev(3, kind=EventKind.USER_MESSAGE, digest="same"),
        ev(4, digest="same"),
    ]
    r = reasons(events)
    assert all(r[i] is Reason.PINNED for i in range(4))
    assert r[4] is Reason.KEEP  # last occurrence of the digest


def test_closed_scope_drops_interior_but_keeps_the_outcome():
    events = [
        ev(0, kind=EventKind.TASK_BEGIN, task_id="t1"),
        ev(1, task_id="t1"),
        ev(2, kind=EventKind.TASK_END, task_id="t1"),
        ev(3),
    ]
    r = reasons(events, closed_tasks={"t1"})
    assert r[0] is Reason.SCOPE_CLOSED
    assert r[1] is Reason.SCOPE_CLOSED
    assert r[2] is Reason.KEEP  # the outcome subsumes the trace
    assert r[3] is Reason.KEEP


def test_open_scope_is_untouched():
    events = [ev(0, task_id="t1"), ev(1, task_id="t1")]
    assert all(d.keep for d in decide(events, closed_tasks=set()))


def test_events_without_identity_still_get_digest_dedup():
    # The floor we promise when no extractor is registered for a custom tool.
    r = reasons([ev(0, digest="d"), ev(1, digest="d")])
    assert r[0] is Reason.DUPLICATE
    assert r[1] is Reason.KEEP

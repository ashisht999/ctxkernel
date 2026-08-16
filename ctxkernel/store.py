"""Durable layer: the append-only event log and the content-addressed blobs.

Two rules this module exists to enforce:

1. Nothing is ever deleted. Compaction excludes material from a *view*; it does
   not destroy it. That is what makes a wrong exclusion survivable.
2. A store is opened *scoped* to a tenant and session. There is deliberately no
   API that takes a session id as an argument, so cross-session reads are
   unrepresentable rather than merely discouraged.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from .ir import (
    Block,
    Elision,
    Event,
    EventKind,
    Handle,
    HandleRef,
    ResourceId,
    Text,
    Thinking,
    ToolResult,
    ToolUse,
    digest_of,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq          INTEGER PRIMARY KEY,
    ts           REAL    NOT NULL,
    kind         TEXT    NOT NULL,
    resource     TEXT,
    digest       TEXT,
    derived_from TEXT    NOT NULL DEFAULT '[]',
    is_write     INTEGER NOT NULL DEFAULT 0,
    task_id      TEXT,
    blocks       TEXT    NOT NULL,
    meta         TEXT    NOT NULL DEFAULT '{}'
);

-- The three indexes that make Tier A a lookup instead of a scan.
CREATE INDEX IF NOT EXISTS ix_events_resource ON events(resource, seq);
CREATE INDEX IF NOT EXISTS ix_events_digest   ON events(digest);
CREATE INDEX IF NOT EXISTS ix_events_task     ON events(task_id);

CREATE TABLE IF NOT EXISTS handles (
    id          TEXT PRIMARY KEY,
    digest      TEXT NOT NULL,
    media_type  TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL,
    extract     TEXT NOT NULL,
    resource    TEXT,
    path        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id        TEXT PRIMARY KEY,
    intent    TEXT NOT NULL,
    parent    TEXT,
    status    TEXT NOT NULL,
    outcome   TEXT,
    begin_seq INTEGER NOT NULL,
    end_seq   INTEGER
);
"""


# --------------------------------------------------------------------------
# Block (de)serialization
# --------------------------------------------------------------------------

_BLOCK_TYPES: dict[str, type[Block]] = {
    "text": Text,
    "thinking": Thinking,
    "tool_use": ToolUse,
    "tool_result": ToolResult,
    "handle_ref": HandleRef,
    "elision": Elision,
}
_BLOCK_TAGS = {v: k for k, v in _BLOCK_TYPES.items()}


def _dump_blocks(blocks: list[Block]) -> str:
    return json.dumps([{"_t": _BLOCK_TAGS[type(b)], **asdict(b)} for b in blocks])


def _load_blocks(raw: str) -> list[Block]:
    out: list[Block] = []
    for d in json.loads(raw):
        d = dict(d)
        cls = _BLOCK_TYPES[d.pop("_t")]
        out.append(cls(**d))
    return out


# --------------------------------------------------------------------------
# Blob store
# --------------------------------------------------------------------------


class BlobStore:
    """Content-addressed files on disk. Write-once, never mutated."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        return self.root / digest[:2] / digest[2:]

    def put(self, content: str) -> str:
        d = digest_of(content)
        p = self._path(d)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(p)  # atomic; concurrent writers converge on same bytes
        return d

    def get(self, digest: str) -> str:
        return self._path(digest).read_text(encoding="utf-8")

    def exists(self, digest: str) -> bool:
        return self._path(digest).exists()


# --------------------------------------------------------------------------
# Session store
# --------------------------------------------------------------------------


class SessionStore:
    """Event log + handle registry for exactly one (tenant, session).

    Open via :meth:`for_session`. The scope is bound at construction and there
    is no method that accepts another session's id.
    """

    def __init__(self, tenant: str, session: str, db_path: Path, blobs: BlobStore) -> None:
        self.tenant = tenant
        self.session = session
        self._blobs = blobs
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    @classmethod
    def for_session(
        cls, root: str | Path, tenant: str, session: str
    ) -> SessionStore:
        root = Path(root)
        # Tenant is a hard key prefix, not a filter: separate directories mean a
        # query cannot cross the boundary even if the code is wrong.
        sdir = root / _safe(tenant) / _safe(session)
        sdir.mkdir(parents=True, exist_ok=True)
        return cls(tenant, session, sdir / "log.db", BlobStore(root / _safe(tenant) / "blobs"))

    # -- events ------------------------------------------------------------

    def append(self, event: Event) -> Event:
        with self._lock:
            if event.seq < 0:
                cur = self._db.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM events")
                event.seq = int(cur.fetchone()[0])
            self._db.execute(
                "INSERT INTO events (seq, ts, kind, resource, digest, derived_from,"
                " is_write, task_id, blocks, meta) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    event.seq,
                    event.ts,
                    event.kind.value,
                    str(event.resource) if event.resource else None,
                    event.digest,
                    json.dumps([str(r) for r in event.derived_from]),
                    int(event.is_write),
                    event.task_id,
                    _dump_blocks(event.blocks),
                    json.dumps(event.meta),
                ),
            )
            self._db.commit()
        return event

    def next_seq(self) -> int:
        cur = self._db.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM events")
        return int(cur.fetchone()[0])

    def _row_to_event(self, r: sqlite3.Row) -> Event:
        return Event(
            seq=r["seq"],
            kind=EventKind(r["kind"]),
            blocks=_load_blocks(r["blocks"]),
            ts=r["ts"],
            resource=ResourceId.parse(r["resource"]) if r["resource"] else None,
            digest=r["digest"],
            derived_from=tuple(ResourceId.parse(s) for s in json.loads(r["derived_from"])),
            is_write=bool(r["is_write"]),
            task_id=r["task_id"],
            meta=json.loads(r["meta"]),
        )

    def events(self, since: int = 0) -> list[Event]:
        cur = self._db.execute("SELECT * FROM events WHERE seq >= ? ORDER BY seq", (since,))
        return [self._row_to_event(r) for r in cur]

    def iter_events(self, since: int = 0) -> Iterator[Event]:
        cur = self._db.execute("SELECT * FROM events WHERE seq >= ? ORDER BY seq", (since,))
        for r in cur:
            yield self._row_to_event(r)

    def count(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    # -- index lookups (the hot path; O(1)-ish, never a scan) ---------------

    def latest_seq_for_resource(self, resource: ResourceId) -> int | None:
        row = self._db.execute(
            "SELECT MAX(seq) FROM events WHERE resource = ?", (str(resource),)
        ).fetchone()
        return row[0]

    def write_seqs_for_resources(self, resources: list[ResourceId]) -> dict[str, int]:
        """Latest write seq per resource, for invalidation checks."""
        if not resources:
            return {}
        qs = ",".join("?" * len(resources))
        cur = self._db.execute(
            f"SELECT resource, MAX(seq) AS s FROM events"
            f" WHERE is_write = 1 AND resource IN ({qs}) GROUP BY resource",
            [str(r) for r in resources],
        )
        return {r["resource"]: r["s"] for r in cur}

    def first_seq_for_digest(self, digest: str) -> int | None:
        row = self._db.execute(
            "SELECT MIN(seq) FROM events WHERE digest = ?", (digest,)
        ).fetchone()
        return row[0]

    # -- handles -----------------------------------------------------------

    def put_handle(
        self,
        content: str,
        *,
        media_type: str,
        extract: str,
        resource: ResourceId | None = None,
    ) -> Handle:
        digest = self._blobs.put(content)
        hid = Handle.make_id(self.tenant, self.session, digest, resource)
        h = Handle(
            id=hid,
            digest=digest,
            media_type=media_type,
            size_bytes=len(content.encode("utf-8")),
            extract=extract,
            resource=resource,
        )
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO handles (id, digest, media_type, size_bytes,"
                " extract, resource, path) VALUES (?,?,?,?,?,?,?)",
                (h.id, h.digest, h.media_type, h.size_bytes, h.extract,
                 str(h.resource) if h.resource else None, digest),
            )
            self._db.commit()
        return h

    def get_handle(self, handle_id: str) -> Handle | None:
        r = self._db.execute("SELECT * FROM handles WHERE id = ?", (handle_id,)).fetchone()
        if r is None:
            return None  # scoped table: another session's id simply is not here
        return Handle(
            id=r["id"],
            digest=r["digest"],
            media_type=r["media_type"],
            size_bytes=r["size_bytes"],
            extract=r["extract"],
            resource=ResourceId.parse(r["resource"]) if r["resource"] else None,
        )

    def read_handle(self, handle_id: str) -> str | None:
        h = self.get_handle(handle_id)
        return self._blobs.get(h.digest) if h else None

    # -- tasks -------------------------------------------------------------

    def open_task(self, task_id: str, intent: str, parent: str | None, begin_seq: int) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO tasks (id, intent, parent, status, begin_seq)"
                " VALUES (?,?,?,'active',?)",
                (task_id, intent, parent, begin_seq),
            )
            self._db.commit()

    def close_task(self, task_id: str, outcome: str, end_seq: int, status: str = "done") -> None:
        with self._lock:
            self._db.execute(
                "UPDATE tasks SET status = ?, outcome = ?, end_seq = ? WHERE id = ?",
                (status, outcome, end_seq, task_id),
            )
            self._db.commit()

    def closed_task_ids(self) -> set[str]:
        cur = self._db.execute("SELECT id FROM tasks WHERE status IN ('done','abandoned')")
        return {r["id"] for r in cur}

    def task_outcomes(self) -> dict[str, dict[str, Any]]:
        cur = self._db.execute(
            "SELECT * FROM tasks WHERE status IN ('done','abandoned') ORDER BY begin_seq"
        )
        return {r["id"]: dict(r) for r in cur}

    def close(self) -> None:
        self._db.close()


def _safe(name: str) -> str:
    """Filesystem-safe key component; keeps tenant separation from being
    subverted by a crafted id containing path separators."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name) or "_"

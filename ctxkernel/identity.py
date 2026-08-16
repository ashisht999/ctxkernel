"""Deriving resource identity from tool calls.

The host is not asked to adopt a log format. We already see every tool call at
the wrap point, so identity is *derived* from the call's name and arguments.
A custom tool only needs a registered extractor when its identity is not
inferable from its arguments -- and without one, digest dedup still works, so
nothing breaks, one predicate just gets weaker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from .ir import ResourceId

Extractor = Callable[[str, dict[str, Any]], "ToolSemantics | None"]


@dataclass(frozen=True, slots=True)
class ToolSemantics:
    resource: ResourceId | None
    is_write: bool = False
    derived_from: tuple[ResourceId, ...] = ()


# --------------------------------------------------------------------------
# Argument sniffing
# --------------------------------------------------------------------------

_PATH_KEYS = ("file_path", "path", "filename", "file", "filepath", "target_file", "notebook_path")
_URL_KEYS = ("url", "uri", "endpoint", "href")
_CMD_KEYS = ("command", "cmd", "script", "shell_command")
_QUERY_KEYS = ("query", "pattern", "regex", "search", "q")


def _first(args: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


# --------------------------------------------------------------------------
# Name classification
# --------------------------------------------------------------------------

_READ_NAMES = re.compile(
    r"read|cat|open|view|get_file|show|fetch_file|load|inspect", re.I
)
_WRITE_NAMES = re.compile(
    r"write|create|edit|patch|replace|append|save|update|delete|remove|mkdir|move|rename",
    re.I,
)
_LIST_NAMES = re.compile(r"^(ls|list|glob|find|tree|dir)|list_(dir|files)|glob", re.I)
_SEARCH_NAMES = re.compile(r"grep|search|ripgrep|find_in", re.I)
_SHELL_NAMES = re.compile(r"^(bash|sh|shell|exec|run|terminal|command)", re.I)
_HTTP_NAMES = re.compile(r"http|fetch|curl|request|api_call|web", re.I)

#: Shell fragments that clearly mutate the filesystem. Best-effort by design:
#: a missed write costs a stale observation, which the model can still detect
#: and re-read, whereas treating every command as a write would invalidate the
#: whole context on each test run.
_SHELL_WRITE = re.compile(
    r"(^|[;&|]\s*)(rm|mv|cp|touch|mkdir|tee|patch|install)\b"
    r"|>>?\s*\S"
    r"|sed\s+-i"
    r"|git\s+(checkout|reset|apply|commit|merge|rebase|pull)\b"
    r"|(npm|yarn|pnpm|pip|poetry|cargo)\s+(install|add|remove|update)\b",
    re.I,
)
#: Redirect / in-place targets we can attribute a shell write to a real file.
_SHELL_TARGET = re.compile(r">>?\s*([\w./~-]+)|sed\s+-i(?:\.\w+)?\s+\S+\s+([\w./~-]+)")


def _norm_path(p: str) -> str:
    p = p.strip().strip("'\"")
    if p.startswith("./"):
        p = p[2:]
    return p.rstrip("/") or "/"


def _norm_cmd(c: str) -> str:
    return re.sub(r"\s+", " ", c.strip())[:200]


def builtin_extractor(name: str, args: dict[str, Any]) -> ToolSemantics | None:
    """Cover the tool shapes that appear in essentially every agent harness."""
    path = _first(args, _PATH_KEYS)
    url = _first(args, _URL_KEYS)
    cmd = _first(args, _CMD_KEYS)
    query = _first(args, _QUERY_KEYS)

    if _SHELL_NAMES.search(name) and cmd:
        is_write = bool(_SHELL_WRITE.search(cmd))
        target: ResourceId | None = None
        if is_write:
            m = _SHELL_TARGET.search(cmd)
            if m:
                target = ResourceId("file", _norm_path(m.group(1) or m.group(2)))
        if target is not None:
            return ToolSemantics(target, is_write=True)
        return ToolSemantics(ResourceId("shell", _norm_cmd(cmd)), is_write=is_write)

    if _HTTP_NAMES.search(name) and url:
        method = str(args.get("method", "GET")).upper()
        return ToolSemantics(ResourceId("http", f"{method} {url}"))

    if path:
        if _WRITE_NAMES.search(name):
            return ToolSemantics(ResourceId("file", _norm_path(path)), is_write=True)
        if _LIST_NAMES.search(name):
            return ToolSemantics(ResourceId("dir", _norm_path(path)))
        if _READ_NAMES.search(name) or True:  # default for a path-bearing tool
            return ToolSemantics(ResourceId("file", _norm_path(path)))

    if _SEARCH_NAMES.search(name) and query:
        scope = _first(args, ("path", "dir", "directory")) or "."
        return ToolSemantics(ResourceId("search", f"{query}@{_norm_path(scope)}"))

    if _LIST_NAMES.search(name):
        return ToolSemantics(ResourceId("dir", _norm_path(path or ".")))

    return None


class IdentityRegistry:
    """Resolves a tool call to a resource, with user overrides taking priority."""

    def __init__(self) -> None:
        self._exact: dict[str, Extractor] = {}

    def register(self, tool_name: str, extractor: Extractor) -> None:
        """Register an extractor for a custom tool.

        >>> reg.register("db_query", lambda n, a: ToolSemantics(
        ...     ResourceId("db", a["table"])))
        """
        self._exact[tool_name] = extractor

    def resolve(self, tool_name: str, args: dict[str, Any]) -> ToolSemantics:
        fn = self._exact.get(tool_name)
        if fn is not None:
            sem = fn(tool_name, args)
            if sem is not None:
                return sem
        sem = builtin_extractor(tool_name, args)
        if sem is not None:
            return sem
        # Unknown shape: no identity, so supersession is unavailable. Digest
        # dedup still applies, which is the floor we promised.
        return ToolSemantics(None)

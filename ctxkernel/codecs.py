"""Computed extracts.

Every extract in here is *parsed*, never model-written. That is the point: a
symbol outline is exact, instant, free, and cannot hallucinate, whereas "this
file defines several auth-related functions" is a guess that costs a model call.

You do not summarize something you can simply re-fetch. The extract exists to
tell the model the *shape* of the content and give it a handle, so it can decide
whether to pull the real thing back.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from .ir import ResourceId

#: Target ceiling for an extract, in characters (~250 tokens).
MAX_EXTRACT_CHARS = 1000

#: Ceiling for the names index appended to every extract (~150 tokens).
MAX_NAMES_CHARS = 600


class Codec(Protocol):
    media_type: str
    #: Optional (default True): whether the registry appends a names index.
    #: Set False on a codec whose cuts are deliberate rather than for size.

    def matches(self, content: str, resource: ResourceId | None, tool_name: str) -> bool: ...
    def extract(self, content: str, resource: ResourceId | None) -> str: ...


def _clip(s: str, limit: int = MAX_EXTRACT_CHARS) -> str:
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n / 1:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


# --------------------------------------------------------------------------
# Source files
# --------------------------------------------------------------------------

_LANGS: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript", ".go": "go",
    ".rs": "rust", ".java": "java", ".rb": "ruby", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".cs": "csharp", ".php": "php", ".swift": "swift",
    ".kt": "kotlin", ".scala": "scala", ".sh": "shell", ".sql": "sql",
}

_SYMBOLS: dict[str, list[re.Pattern[str]]] = {
    "python": [
        re.compile(r"^\s*class\s+(\w+)", re.M),
        re.compile(r"^\s*(?:async\s+)?def\s+(\w+)", re.M),
    ],
    "javascript": [
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+(\w+)", re.M),
        re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)", re.M),
        re.compile(r"^\s*(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?\(", re.M),
    ],
    "go": [
        re.compile(r"^type\s+(\w+)", re.M),
        re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+)", re.M),
    ],
    "rust": [
        re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)", re.M),
        re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", re.M),
    ],
    "java": [
        re.compile(r"^\s*(?:public|private|protected).*?\bclass\s+(\w+)", re.M),
        re.compile(r"^\s*(?:public|private|protected)[\w<>\[\], ]+\s(\w+)\s*\(", re.M),
    ],
    "ruby": [
        re.compile(r"^\s*class\s+(\w+)", re.M),
        re.compile(r"^\s*def\s+(\w+)", re.M),
    ],
}
_SYMBOLS["typescript"] = _SYMBOLS["javascript"]

_IMPORTS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.M),
    "javascript": re.compile(r"""^\s*(?:import\b[^'"]*from\s*|(?:const|let)\s+.*=\s*require\s*\()\s*['"]([^'"]+)""", re.M),
    "go": re.compile(r'^\s*(?:import\s+)?"([\w./-]+)"', re.M),
    "rust": re.compile(r"^\s*use\s+([\w:]+)", re.M),
}
_IMPORTS["typescript"] = _IMPORTS["javascript"]


def _lang_for(resource: ResourceId | None) -> str | None:
    if resource is None or resource.kind != "file":
        return None
    for ext, lang in _LANGS.items():
        if resource.key.endswith(ext):
            return lang
    return None


class SourceFileCodec:
    """Symbol outline for a recognized source file."""

    media_type = "text/x-source"

    def matches(self, content: str, resource: ResourceId | None, tool_name: str) -> bool:
        return _lang_for(resource) is not None

    def extract(self, content: str, resource: ResourceId | None) -> str:
        lang = _lang_for(resource) or "text"
        lines = content.count("\n") + 1
        head = [f"{resource.key} — {lines} lines, {lang}"]

        imp_re = _IMPORTS.get(lang)
        if imp_re:
            mods: list[str] = []
            for m in imp_re.finditer(content):
                mod = next((g for g in m.groups() if g), None)
                if mod and mod not in mods:
                    mods.append(mod.split(".")[0].split("/")[0])
                if len(mods) >= 8:
                    break
            if mods:
                head.append("imports: " + ", ".join(mods))

        syms: list[str] = []
        for pat in _SYMBOLS.get(lang, []):
            for m in pat.finditer(content):
                line = content.count("\n", 0, m.start()) + 1
                syms.append((line, m.group(1)))
        syms.sort()
        shown = syms[:24]
        head += [f"  L{ln:<5} {name}" for ln, name in shown]
        if len(syms) > len(shown):
            head.append(f"  … {len(syms) - len(shown)} more symbols")
        if not syms:
            head.append("  (no top-level symbols matched)")
        return _clip("\n".join(head))


# --------------------------------------------------------------------------
# Diffs
# --------------------------------------------------------------------------


class DiffCodec:
    media_type = "text/x-diff"

    _FILE = re.compile(r"^\+\+\+ [ab]/(.+)$", re.M)

    def matches(self, content: str, resource: ResourceId | None, tool_name: str) -> bool:
        return content.lstrip().startswith(("diff --git", "--- ", "Index: ")) and "@@" in content

    def extract(self, content: str, resource: ResourceId | None) -> str:
        files: dict[str, list[int]] = {}
        current: str | None = None
        for line in content.splitlines():
            m = self._FILE.match(line)
            if m:
                current = m.group(1)
                files.setdefault(current, [0, 0])
            elif current:
                if line.startswith("+") and not line.startswith("+++"):
                    files[current][0] += 1
                elif line.startswith("-") and not line.startswith("---"):
                    files[current][1] += 1
        adds = sum(v[0] for v in files.values())
        dels = sum(v[1] for v in files.values())
        out = [f"diff — {len(files)} file(s), +{adds} −{dels}"]
        for f, (a, d) in list(files.items())[:12]:
            out.append(f"  {f}  +{a} −{d}")
        if len(files) > 12:
            out.append(f"  … {len(files) - 12} more files")
        return _clip("\n".join(out))


# --------------------------------------------------------------------------
# Test output
# --------------------------------------------------------------------------


class TestOutputCodec:
    """Keeps failing test names and counts; drops the passing noise."""

    media_type = "text/x-test-output"
    #: What this codec drops, it drops on purpose: an index of the names it cut
    #: would put 200 passing tests back in front of the one that failed.
    index_names = False

    _PYTEST_SUM = re.compile(r"=+\s*(.*?(?:passed|failed|error).*?)\s*=+\s*$", re.M | re.I)
    _PYTEST_FAIL = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.M)
    _JEST_SUM = re.compile(r"^Tests:\s+(.+)$", re.M)
    _JEST_FAIL = re.compile(r"^\s*[✕×]\s+(.+?)(?:\s+\(\d+\s*ms\))?$", re.M)
    _GO_FAIL = re.compile(r"^\s*--- FAIL:\s+(\S+)", re.M)

    def matches(self, content: str, resource: ResourceId | None, tool_name: str) -> bool:
        return bool(
            self._PYTEST_SUM.search(content)
            or self._JEST_SUM.search(content)
            or self._GO_FAIL.search(content)
            or re.search(r"^(?:FAILED|ok\s+\S+\s+[\d.]+s)", content, re.M)
        )

    def extract(self, content: str, resource: ResourceId | None) -> str:
        summary = None
        for pat in (self._PYTEST_SUM, self._JEST_SUM):
            m = pat.search(content)
            if m:
                summary = m.group(1).strip()
                break

        fails: list[str] = []
        for pat in (self._PYTEST_FAIL, self._GO_FAIL, self._JEST_FAIL):
            fails += [m.group(1).strip() for m in pat.finditer(content)]
            if fails:
                break

        out = [f"test run — {summary}" if summary else "test run"]
        for f in fails[:15]:
            out.append(f"  FAIL {f}")
        if len(fails) > 15:
            out.append(f"  … {len(fails) - 15} more failures")
        if not fails and summary and "fail" not in summary.lower():
            out.append("  all passing")
        return _clip("\n".join(out))


# --------------------------------------------------------------------------
# Tracebacks
# --------------------------------------------------------------------------


class TracebackCodec:
    media_type = "text/x-traceback"

    _FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+), in (\S+)', re.M)
    _EXC = re.compile(r"^(\w+(?:Error|Exception|Warning))(?::\s*(.*))?$", re.M)

    def matches(self, content: str, resource: ResourceId | None, tool_name: str) -> bool:
        return "Traceback (most recent call last)" in content

    def extract(self, content: str, resource: ResourceId | None) -> str:
        frames = self._FRAME.findall(content)
        excs = self._EXC.findall(content)
        out = ["traceback"]
        if excs:
            typ, msg = excs[-1]
            out.append(f"  {typ}: {msg}".rstrip())
        # Innermost frames are the informative ones.
        for path, line, fn in frames[-3:]:
            out.append(f"  at {path}:{line} in {fn}")
        if len(frames) > 3:
            out.append(f"  ({len(frames)} frames total)")
        return _clip("\n".join(out))


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------


class JSONCodec:
    media_type = "application/json"

    def matches(self, content: str, resource: ResourceId | None, tool_name: str) -> bool:
        s = content.lstrip()[:1]
        if s not in ("{", "["):
            return False
        try:
            json.loads(content)
            return True
        except (ValueError, RecursionError):
            return False

    @staticmethod
    def _sketch(v: Any) -> str:
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, int):
            return "int"
        if isinstance(v, float):
            return "float"
        if isinstance(v, str):
            return "str"
        if v is None:
            return "null"
        if isinstance(v, list):
            return f"array[{len(v)}]"
        if isinstance(v, dict):
            return "object"
        return type(v).__name__

    def extract(self, content: str, resource: ResourceId | None) -> str:
        data = json.loads(content)
        out: list[str] = []
        if isinstance(data, list):
            out.append(f"json — array[{len(data)}]")
            first = next((x for x in data if isinstance(x, dict)), None)
            if first is not None:
                keys = ", ".join(f"{k}:{self._sketch(v)}" for k, v in list(first.items())[:12])
                out.append(f"  of object — {keys}")
                out.append("  sample: " + _clip(json.dumps(first)[:220], 220))
            elif data:
                out.append(f"  of {self._sketch(data[0])}")
        elif isinstance(data, dict):
            out.append(f"json — object, {len(data)} keys")
            for k, v in list(data.items())[:14]:
                out.append(f"  {k}: {self._sketch(v)}")
            if len(data) > 14:
                out.append(f"  … {len(data) - 14} more keys")
        return _clip("\n".join(out))


# --------------------------------------------------------------------------
# Fallback
# --------------------------------------------------------------------------


class DefaultCodec:
    """Head/tail slice. Always matches; registered last."""

    media_type = "text/plain"

    def matches(self, content: str, resource: ResourceId | None, tool_name: str) -> bool:
        return True

    def extract(self, content: str, resource: ResourceId | None) -> str:
        lines = content.splitlines()
        label = str(resource) if resource else "output"
        out = [f"{label} — {len(lines)} lines, {_human_bytes(len(content.encode()))}"]
        head, tail = lines[:8], lines[-4:] if len(lines) > 12 else []
        out += [f"  {ln[:120]}" for ln in head]
        if tail:
            out.append(f"  … {len(lines) - 12} lines elided …")
            out += [f"  {ln[:120]}" for ln in tail]
        return _clip("\n".join(out))


# --------------------------------------------------------------------------
# Names index
# --------------------------------------------------------------------------

_NAME = re.compile(r"[A-Za-z0-9_./\-]{3,}")
_FILEISH = re.compile(r"\.[A-Za-z]\w{0,5}$")


def name_ranks(text: str) -> dict[str, int]:
    """Name-shaped tokens in ``text``, in order of first appearance, each with
    its kind: 0 a path, 1 a file name, 2 an identifier.

    A name is what one piece of history uses to point at another -- a path, a
    symbol, a setting. Plain words are left out: "error" or "python" appear
    everywhere and point at nothing. Shared by the names index here and by the
    graph's mention links, so both agree on what a name is.
    """
    seen: dict[str, int] = {}
    for raw in _NAME.findall(text):
        t = raw.strip("./-")
        if len(t) < 3 or t in seen or not any(ch.isalpha() for ch in t):
            continue
        if "/" in t:
            seen[t] = 0
        elif _FILEISH.search(t):
            seen[t] = 1
        elif "_" in t or "." in t or re.search(r"[a-z][A-Z]", t):
            seen[t] = 2
    return seen


def names_index(content: str, extract: str, limit: int = MAX_NAMES_CHARS) -> str:
    """The names in ``content`` that ``extract`` does not already show.

    An extract keeps the shape of an output and cuts the middle, and the middle
    is where a listing keeps its paths: measured on real Claude Code sessions,
    over half of what an agent went on to use and could no longer see was a
    name that sat in a handle's blob, most of it long shell output. Without the
    name the agent cannot even ask for the content, so every extract carries
    the names it cut. Paths, file names and identifiers take turns, each in
    order of first appearance, so one long listing of paths cannot crowd out
    the lone ``README.md`` beside it. When they do not all fit, the index says
    how many were cut, so the agent knows to ``expand()``. Parsed, like the
    extract itself.
    """
    seen = {t: r for t, r in name_ranks(content).items() if t not in extract}
    if not seen:
        return ""
    groups = [[t for t, r in seen.items() if r == rank] for rank in (0, 1, 2)]
    out: list[str] = []
    used = len("names: ")
    full = False
    for i in range(max(len(g) for g in groups)):
        for g in groups:
            if i >= len(g):
                continue
            if used + len(g[i]) + 2 > limit:
                full = True
                break
            out.append(g[i])
            used += len(g[i]) + 2
        if full:
            break
    more = len(seen) - len(out)
    return "names: " + ", ".join(out) + (f"  (+{more} more)" if more else "")


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


class CodecRegistry:
    """Ordered most-specific-first; the first match wins."""

    def __init__(self) -> None:
        self._codecs: list[Codec] = [
            TracebackCodec(),
            DiffCodec(),
            TestOutputCodec(),
            JSONCodec(),
            SourceFileCodec(),
            DefaultCodec(),
        ]

    def register(self, codec: Codec, *, first: bool = True) -> None:
        """Add a codec for a domain-specific tool. Inserted ahead of the
        built-ins by default, since a custom codec is more specific."""
        self._codecs.insert(0 if first else len(self._codecs) - 1, codec)

    def encode(
        self, content: str, resource: ResourceId | None, tool_name: str = ""
    ) -> tuple[str, str]:
        """Return ``(extract, media_type)``."""
        for c in self._codecs:
            try:
                if c.matches(content, resource, tool_name):
                    extract = c.extract(content, resource)
                    if getattr(c, "index_names", True):
                        extract = _with_names(extract, content)
                    return extract, c.media_type
            except Exception:
                continue  # a broken codec must never break the turn
        return _with_names(DefaultCodec().extract(content, resource), content), "text/plain"


def _with_names(extract: str, content: str) -> str:
    try:
        names = names_index(content, extract)
    except Exception:
        return extract  # the index is a bonus; losing it must not lose the extract
    return f"{extract}\n{names}" if names else extract

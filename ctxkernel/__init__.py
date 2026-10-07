"""ctxkernel — a bounded view over an append-only event log.

The context window is not storage. It is a rendering of the log, the way a page
is a rendering of a database, and that is what makes discarding it safe.

    from ctxkernel import ContextEngine, budgets

    engine = ContextEngine(budget=budgets.window(200_000))
    engine.set_goal("make the flaky auth test pass", ["pytest tests/auth.py"])
    engine.record_tool_call("t1", "read_file", {"path": "src/auth.py"})
    engine.record_tool_result("t1", huge_file_contents)

    engine.assemble().report()
"""

from . import budgets
from .assembler import Assembler, Assembly, Budget, Zone
from .codecs import Codec, CodecRegistry
from .decision import DecisionModel, NullDecision, ProximityDecision
from .engine import ContextEngine, TaskFrame
from .identity import IdentityRegistry, ToolSemantics
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
from .predicates import Decision, Reason, decide
from .store import BlobStore, SessionStore
from .tokenizer import ApproxTokenizer, Tokenizer, default_tokenizer

__version__ = "0.1.0"

__all__ = [
    "ContextEngine",
    "TaskFrame",
    "budgets",
    "Budget",
    "Assembler",
    "Assembly",
    "Zone",
    "Event",
    "EventKind",
    "ResourceId",
    "Handle",
    "Block",
    "Text",
    "Thinking",
    "ToolUse",
    "ToolResult",
    "HandleRef",
    "Elision",
    "digest_of",
    "Decision",
    "Reason",
    "decide",
    "Codec",
    "CodecRegistry",
    "DecisionModel",
    "NullDecision",
    "ProximityDecision",
    "IdentityRegistry",
    "ToolSemantics",
    "SessionStore",
    "BlobStore",
    "Tokenizer",
    "ApproxTokenizer",
    "default_tokenizer",
    "__version__",
]

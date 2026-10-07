"""TypeSafe Jev as a ``DecisionModel``.

Jev answers typed questions about a state with calibrated probabilities rather
than generated text, which is the right shape for this job: the output is a
number per candidate, so there is nothing to parse and nothing to hallucinate.

Each candidate becomes one ``Noul`` (yes/no) question, all answered in a
single parallel request. Two properties of the model shape how the question is
asked:

* It reads questions literally and is weak at multi-hop reasoning, so every
  question carries its candidate whole -- label, a preview of its content, and
  its connections spelled out as sentences -- instead of asking the model to
  rank a list or follow an id.
* Its window is 32k tokens, so content is a bounded preview per item and the
  state is trimmed oldest-step-first; the goal is never cut.

The SDK is imported lazily: installing ctxkernel must not require it, and a
missing SDK surfaces as a fallback in ``AnchorRun``, not as a failed turn.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ...decision import FULL_QUESTION, Candidate, DecisionState

_QUESTION = (
    "Given the goal, the current task, the latest user input and the recent "
    "steps, will the agent need this item from its history to decide its next step?"
)


class JevDecision:
    name = "jev"
    #: Jev's probabilities are calibrated, so a low one means "not needed" and
    #: the item stays out -- accuracy and a smaller context together. Measured
    #: live: relevant items 0.56-0.85, filler 0.20-0.29.
    threshold = 0.3
    #: Displacing a recent item needs a confident call; recency is the floor.
    swap_threshold = 0.7
    full_threshold = 0.7

    def __init__(
        self,
        client: Any | None = None,
        *,
        max_state_chars: int = 20_000,
        max_item_chars: int = 600,
    ) -> None:
        """``client`` is a ``typesafe_sdk.TypeSafeClient``; one is created from
        ``TYPESAFE_API_KEY`` on first use if omitted."""
        self._client = client
        self.max_state_chars = max_state_chars
        self.max_item_chars = max_item_chars

    def score(self, state: DecisionState, candidates: Sequence[Candidate]) -> dict[str, float]:
        if not candidates:
            return {}
        from typesafe_sdk import Noul

        questions = {
            f"q{i}": Noul(instructions={"question": _QUESTION, "item": self._item(c)})
            for i, c in enumerate(candidates)
        }
        response = self._get_client().system_one(state=self._state(state), questions=questions)
        return {c.node_id: float(response.nouls[f"q{i}"].noul) for i, c in enumerate(candidates)}

    def fidelity(self, state: DecisionState, items: Sequence[Candidate]) -> dict[str, float]:
        """Summary or full content, one yes/no per large output, in one request."""
        if not items:
            return {}
        from typesafe_sdk import Noul

        asked = {
            f"f{i}": Noul(instructions={"question": FULL_QUESTION, "item": self._item(c)})
            for i, c in enumerate(items)
        }
        response = self._get_client().system_one(state=self._state(state), questions=asked)
        return {c.node_id: float(response.nouls[f"f{i}"].noul) for i, c in enumerate(items)}

    def judge(
        self, state: DecisionState, step: Candidate, questions: Mapping[str, str]
    ) -> dict[str, float]:
        """Yes/no judgments about one step, in the same single request shape
        as the ranking: each question carries the step whole."""
        if not questions:
            return {}
        from typesafe_sdk import Noul

        item = self._item(step)
        asked = {key: Noul(instructions={"question": q, "step": item}) for key, q in questions.items()}
        response = self._get_client().system_one(state=self._state(state), questions=asked)
        return {key: float(response.nouls[key].noul) for key in questions}

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def _get_client(self) -> Any:
        if self._client is None:
            from typesafe_sdk import TypeSafeClient

            self._client = TypeSafeClient()
        return self._client

    def _item(self, c: Candidate) -> dict[str, Any]:
        d = c.as_dict()
        if len(d.get("content", "")) > self.max_item_chars:
            d["content"] = d["content"][: self.max_item_chars - 1] + "…"
        return d

    def _state(self, state: DecisionState) -> dict[str, Any]:
        d = state.as_dict()
        # Trim oldest steps first; the goal is never cut (it is recorded
        # verbatim for a reason) and failures are the densest signal there is.
        while len(str(d)) > self.max_state_chars and d["recent_steps"]:
            d["recent_steps"] = d["recent_steps"][1:]
        return d

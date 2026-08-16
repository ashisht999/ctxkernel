"""Budget construction.

Three numbers matter, not one:

* **pinned floor** (~1%) — goal, invariants, failures. Must always fit, so the
  agent can never be wedged into a state where no prompt can be assembled.
* **target** (~15%) — what a healthy turn looks like. Empty space is the
  absence of noise, not waste.
* **pressure trigger** (~45%) — compaction fires, and it is logged as degraded,
  because a boundary should have caught it first.

They are fractions so a larger window inherits the policy instead of tempting
you to fill it and pay for it on every one of four hundred turns.
"""

from __future__ import annotations

from .assembler import Budget, window

__all__ = ["Budget", "window", "claude", "small"]


def claude() -> Budget:
    """200k window with room for extended thinking plus output."""
    return Budget(window=200_000, generation_reserve=32_000)


def small(n: int = 32_000) -> Budget:
    """For smaller windows the reserve is proportionally larger, since output
    length does not shrink with the context limit."""
    return Budget(window=n, generation_reserve=max(4_000, n // 5))

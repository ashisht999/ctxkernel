"""Decision-model adapters.

The only place a vendor's decision model is touched. Core code sees the
``DecisionModel`` protocol from ``ctxkernel.decision``; each module here
implements it for one vendor, importing that vendor's SDK lazily so the
package still installs and runs with none of them present.

* ``jev`` — TypeSafe Jev (``pip install ctxkernel[jev]``)
"""

from ...decision import DecisionModel, NullDecision, ProximityDecision
from .jev import JevDecision

__all__ = ["DecisionModel", "NullDecision", "ProximityDecision", "JevDecision"]

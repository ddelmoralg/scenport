"""Portfolio allocators.

Importing the package registers all of them. Group them by what they need so the
experiment scripts can talk about families rather than listing names.
"""

from . import convex, evolutionary, exact, heuristic  # noqa: F401  (import registers)
from .base import Allocation, Allocator, available, get, register
from .objective import Problem, decode, is_feasible, report, score

HEURISTIC = ("equal", "mcap", "random", "invvol", "single")
CONVEX = ("minvar", "maxsharpe", "riskparity", "hrp", "cvar_lp")
EVOLUTIONARY = ("ga", "de", "pso", "cmaes")
EXACT = ("milp",)

#: Allocators whose answer depends on the seed and therefore need repeated runs.
STOCHASTIC = ("random", *EVOLUTIONARY)

__all__ = [
    "CONVEX",
    "EVOLUTIONARY",
    "EXACT",
    "HEURISTIC",
    "STOCHASTIC",
    "Allocation",
    "Allocator",
    "Problem",
    "available",
    "decode",
    "get",
    "is_feasible",
    "register",
    "report",
    "score",
]

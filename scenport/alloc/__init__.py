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

#: Not allocators. They answer a relaxed problem, so they will hold more names than
#: the cardinality allows and beat the exact optimum for that reason alone. They
#: belong in the gap study as upper bounds and nowhere near a backtest.
BOUNDS = ("cvar_lp_bound",)

#: Benchmarks that sit outside the constraint set on purpose. A single-asset
#: reference cannot hold eight names capped at twenty per cent and is not supposed
#: to: the point of it is to say what doing nothing clever would have earned. The
#: paper has to say so where these are introduced, and any check on feasibility has
#: to leave them alone.
UNCONSTRAINED = ("single",)

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

"""Thin abstraction over the Benders backends.

Every backend solves the same two-stage stochastic CFLP and returns a
:class:`BendersResult`. The cut-separation routine
(:mod:`sflp.model.subproblem`) is shared; only the search strategy differs:

- :class:`~sflp.benders.classic.ClassicBackend` - iterative master/subproblem
  loop on HiGHS (open, no license).
- :class:`~sflp.benders.scip_backend.ScipBackend` - single-tree
  branch-and-Benders-cut via a SCIP constraint handler (open, Apache-2.0).
- :class:`~sflp.benders.gurobi_backend.GurobiBackend` - single-tree lazy cuts
  via a Gurobi callback (commercial fast path).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sflp.config import ModelConfig, SolverConfig
from sflp.data.instance import FloatArray, Instance, ScenarioSet


@dataclass(frozen=True)
class BendersResult:
    """Outcome of a Benders solve: objective, bounds, and convergence stats."""

    objective: float  # best upper bound = true cost of the incumbent
    y: FloatArray  # (I,) incumbent first stage
    open_facilities: list[int]
    lower_bound: float
    upper_bound: float
    gap: float
    iterations: int  # cut-loop iterations (classic) or separation rounds (single-tree)
    n_cuts: int
    converged: bool
    history: list[tuple[float, float]] = field(default_factory=list)  # (lb, ub) per iteration


class BendersBackend(ABC):
    """Solve the stochastic CFLP by Benders decomposition."""

    @abstractmethod
    def solve(
        self,
        instance: Instance,
        scenarios: ScenarioSet,
        model_cfg: ModelConfig,
        solver_cfg: SolverConfig,
    ) -> BendersResult:
        """Return the incumbent first stage, its cost, and bound information."""

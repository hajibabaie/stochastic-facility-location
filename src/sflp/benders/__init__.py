"""Benders / L-shaped decomposition: cuts, classic loop, and solver backends."""

from __future__ import annotations

from sflp.benders.backend import BendersBackend, BendersResult
from sflp.benders.classic import ClassicBackend, solve_classic_benders
from sflp.benders.cuts import BendersCut
from sflp.benders.scip_backend import ScipBackend
from sflp.config import ModelConfig, SolverConfig
from sflp.data.instance import Instance, ScenarioSet

__all__ = [
    "BendersBackend",
    "BendersCut",
    "BendersResult",
    "ClassicBackend",
    "ScipBackend",
    "solve_benders",
    "solve_classic_benders",
]


def get_backend(solver_cfg: SolverConfig) -> BendersBackend:
    """Select a Benders backend from the solver configuration.

    - ``classic`` -> HiGHS iterative loop.
    - ``branch_and_cut`` -> SCIP single-tree (or Gurobi if ``mip_solver`` is gurobi).
    - ``gurobi`` -> Gurobi single-tree lazy cuts.
    """
    if solver_cfg.backend == "classic":
        return ClassicBackend()
    if solver_cfg.backend == "branch_and_cut":
        if solver_cfg.mip_solver == "gurobi":
            return _gurobi_backend()
        return ScipBackend()
    if solver_cfg.backend == "gurobi":
        return _gurobi_backend()
    raise ValueError(f"Unknown solver.backend {solver_cfg.backend!r}.")


def _gurobi_backend() -> BendersBackend:
    """Import the optional Gurobi backend lazily (gurobipy is an extra)."""
    from sflp.benders.gurobi_backend import GurobiBackend

    return GurobiBackend()


def solve_benders(
    instance: Instance,
    scenarios: ScenarioSet,
    model_cfg: ModelConfig,
    solver_cfg: SolverConfig,
) -> BendersResult:
    """Solve the stochastic CFLP with the backend chosen by ``solver_cfg``."""
    return get_backend(solver_cfg).solve(instance, scenarios, model_cfg, solver_cfg)

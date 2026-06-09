"""Solver factory, backend-combination validation, and high-level solve calls.

Pyomo drives the monolith and the classic Benders loop on **HiGHS** (open) or
**Gurobi** (commercial). The single-tree ``branch_and_cut`` backend is *not* a
Pyomo solver: it is built natively on SCIP/Gurobi where lazy callbacks exist
(see :mod:`sflp.benders`). HiGHS cannot add lazy constraints, so the combination
``branch_and_cut + highs`` is rejected here with a clear message.
"""

from __future__ import annotations

from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers import Highs

from sflp.config import ModelConfig, SolverConfig
from sflp.data.instance import Instance, ScenarioSet
from sflp.model.facility_location import (
    CflpSolution,
    build_deterministic_cflp,
    extract_cflp_solution,
)
from sflp.model.saa_monolith import (
    SaaSolution,
    build_saa_monolith,
    extract_saa_solution,
)

#: Solvers usable through Pyomo for the monolith and classic Benders.
PYOMO_SOLVERS = frozenset({"highs", "gurobi"})
#: Backends that need a single search tree with lazy cuts (no HiGHS).
SINGLE_TREE_SOLVERS = frozenset({"scip", "gurobi"})


def validate_solver_config(cfg: SolverConfig) -> None:
    """Reject backend/solver combinations that cannot work, with guidance."""
    if cfg.backend == "classic":
        if cfg.mip_solver not in PYOMO_SOLVERS:
            raise ValueError(
                f"classic backend uses Pyomo; mip_solver must be one of "
                f"{sorted(PYOMO_SOLVERS)}, got {cfg.mip_solver!r}."
            )
    elif cfg.backend == "branch_and_cut":
        if cfg.mip_solver == "highs":
            raise ValueError(
                "branch_and_cut needs lazy-constraint callbacks, which HiGHS does "
                "not support. Use mip_solver 'scip' or 'gurobi', or switch to the "
                "'classic' backend on HiGHS."
            )
        if cfg.mip_solver not in SINGLE_TREE_SOLVERS:
            raise ValueError(
                f"branch_and_cut requires one of {sorted(SINGLE_TREE_SOLVERS)}, "
                f"got {cfg.mip_solver!r}."
            )
    elif cfg.backend == "gurobi":
        if cfg.mip_solver != "gurobi":
            raise ValueError("gurobi backend requires mip_solver 'gurobi'.")
    else:
        raise ValueError(f"Unknown solver.backend {cfg.backend!r}.")


def make_pyomo_solver(name: str, cfg: SolverConfig | None = None) -> Highs:
    """Create a configured Pyomo (APPSI) solver object.

    Only HiGHS is wired here; Gurobi is added with its backend. APPSI gives a
    persistent solver we can re-solve incrementally during Benders.
    """
    if name == "highs":
        solver = Highs()
        # Defer loading the primal so an infeasible solve returns a status
        # instead of raising; we load explicitly once optimality is confirmed.
        solver.config.load_solution = False
        options: dict[str, object] = {}
        if cfg is not None:
            options["mip_rel_gap"] = cfg.gap_tolerance
            if cfg.time_limit is not None:
                options["time_limit"] = cfg.time_limit
        solver.highs_options = options
        return solver
    raise ValueError(f"Pyomo solver {name!r} is not available yet (have: 'highs').")


def _solve_to_optimality(solver: Highs, model: object, context: str) -> None:
    """Solve, require an optimal status, then load the solution into the model."""
    results = solver.solve(model)
    cond = results.termination_condition
    if cond != TerminationCondition.optimal:
        raise RuntimeError(f"{context}: solver did not reach optimality ({cond}).")
    solver.load_vars()


def solve_deterministic_cflp(instance: Instance, cfg: SolverConfig) -> CflpSolution:
    """Build and solve the deterministic CFLP; return the structured solution."""
    if cfg.mip_solver not in PYOMO_SOLVERS:
        raise ValueError(
            f"solve_deterministic_cflp needs a Pyomo solver "
            f"{sorted(PYOMO_SOLVERS)}, got {cfg.mip_solver!r}."
        )
    model = build_deterministic_cflp(instance)
    solver = make_pyomo_solver(cfg.mip_solver, cfg)
    _solve_to_optimality(solver, model, f"CFLP {instance.name}")
    return extract_cflp_solution(model, instance)


def solve_saa_monolith(
    instance: Instance,
    scenarios: ScenarioSet,
    model_cfg: ModelConfig,
    solver_cfg: SolverConfig,
) -> SaaSolution:
    """Build and solve the SAA deterministic-equivalent monolith."""
    if solver_cfg.mip_solver not in PYOMO_SOLVERS:
        raise ValueError(
            f"solve_saa_monolith needs a Pyomo solver "
            f"{sorted(PYOMO_SOLVERS)}, got {solver_cfg.mip_solver!r}."
        )
    model = build_saa_monolith(instance, scenarios, model_cfg)
    solver = make_pyomo_solver(solver_cfg.mip_solver, solver_cfg)
    _solve_to_optimality(solver, model, f"SAA {instance.name}")
    return extract_saa_solution(model, instance, scenarios)

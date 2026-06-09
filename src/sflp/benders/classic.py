"""Classic (iterative) L-shaped Benders loop on an open Pyomo master.

Each iteration: solve the master MILP for a lower bound and a trial first stage
``(y, z)``; solve every scenario recourse LP for the true cost (an upper bound)
and a fresh optimality cut; add the violated cuts and repeat until the bound gap
closes. With relatively complete recourse there are no feasibility cuts.

This backend runs on HiGHS and needs no commercial license. The same cut routine
(:mod:`sflp.model.subproblem`) feeds the single-tree SCIP/Gurobi backends.
"""

from __future__ import annotations

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition

from sflp.benders.backend import BendersBackend, BendersResult
from sflp.benders.cuts import BendersCut, add_optimality_cut
from sflp.config import ModelConfig, SolverConfig
from sflp.data.instance import FloatArray, Instance, ScenarioSet
from sflp.model.master import build_master
from sflp.model.subproblem import build_subproblems
from sflp.solve import make_pyomo_solver


class ClassicBackend(BendersBackend):
    """Iterative master/subproblem L-shaped loop on HiGHS (open, no license)."""

    def solve(
        self,
        instance: Instance,
        scenarios: ScenarioSet,
        model_cfg: ModelConfig,
        solver_cfg: SolverConfig,
    ) -> BendersResult:
        return solve_classic_benders(instance, scenarios, model_cfg, solver_cfg)


def solve_classic_benders(
    instance: Instance,
    scenarios: ScenarioSet,
    model_cfg: ModelConfig,
    solver_cfg: SolverConfig,
) -> BendersResult:
    """Run the classic Benders loop; return the incumbent and bound history.

    With ``solver_cfg.pareto_cuts`` the optimality cut for each needy scenario is
    generated at an interior **core point** (Papadakos 2008 independent-point
    Magnanti-Wong), which on degenerate facility-location recourse yields stronger
    cuts and fewer iterations. The core point starts at ``0.5`` and is pulled
    halfway toward the master solution each iteration so it stays interior.
    """
    master = build_master(instance, scenarios, model_cfg)
    subs = build_subproblems(instance, scenarios.demand, model_cfg.chance_constraint)
    n_i, n_s = instance.n_facilities, scenarios.n_scenarios
    f, p = instance.fixed_cost, scenarios.probability
    has_chance = model_cfg.chance_constraint
    pareto = solver_cfg.pareto_cuts

    core_y = np.full(n_i, 0.5)
    core_z = np.full(n_s, model_cfg.gamma / 2.0 if has_chance else 0.0)

    # One persistent solver, reused across iterations with incremental cuts.
    solver = make_pyomo_solver(solver_cfg.mip_solver, solver_cfg)

    best_ub = float("inf")
    best_y = np.zeros(n_i)
    lower_bound = float("-inf")
    history: list[tuple[float, float]] = []
    n_cuts = 0
    converged = False
    iteration = 0

    while iteration < solver_cfg.max_iterations:
        iteration += 1
        results = solver.solve(master)
        if results.termination_condition != TerminationCondition.optimal:
            raise RuntimeError(f"Benders master not optimal ({results.termination_condition}).")
        solver.load_vars()

        y_hat = np.array([round(pyo.value(master.y[i])) for i in range(n_i)], dtype=np.float64)
        z_hat = _read_z(master, n_s, has_chance)
        theta_hat = np.array([pyo.value(master.theta[s]) for s in range(n_s)])
        lower_bound = float(pyo.value(master.obj))

        # Recourse at the master point: gives the true cost (UB) and tells us
        # which scenarios the master currently underestimates.
        recourse_total = 0.0
        needy: list[int] = []
        at_y_hat = []
        for s in range(n_s):
            r = subs[s].solve(y_hat, float(z_hat[s]))
            at_y_hat.append(r)
            recourse_total += p[s] * r.objective
            if r.objective - theta_hat[s] > 1e-6 * (1.0 + abs(r.objective)):
                needy.append(s)

        upper_bound = float(f @ y_hat) + recourse_total
        if upper_bound < best_ub:
            best_ub, best_y = upper_bound, y_hat
        history.append((lower_bound, best_ub))

        gap = (best_ub - lower_bound) / max(1.0, abs(best_ub))
        if not needy or gap <= solver_cfg.gap_tolerance:
            converged = True
            break

        for s in needy:
            # Pareto cuts come from an interior core point; standard cuts reuse
            # the recourse solve already done at the master point.
            result = subs[s].solve(core_y, float(core_z[s])) if pareto else at_y_hat[s]
            add_optimality_cut(master, BendersCut.from_recourse(s, result))
            n_cuts += 1

        if pareto:
            core_y = 0.5 * (core_y + y_hat)
            core_z = 0.5 * (core_z + z_hat)

    final_gap = (best_ub - lower_bound) / max(1.0, abs(best_ub))
    return BendersResult(
        objective=best_ub,
        y=best_y,
        open_facilities=[i for i in range(n_i) if best_y[i] > 0.5],
        lower_bound=lower_bound,
        upper_bound=best_ub,
        gap=final_gap,
        iterations=iteration,
        n_cuts=n_cuts,
        converged=converged,
        history=history,
    )


def _read_z(master: pyo.ConcreteModel, n_s: int, has_chance: bool) -> FloatArray:
    if not has_chance:
        return np.zeros(n_s)
    return np.array([round(pyo.value(master.z[s])) for s in range(n_s)], dtype=np.float64)

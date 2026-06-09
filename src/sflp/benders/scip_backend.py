"""Single-tree branch-and-Benders-cut on SCIP via a PySCIPOpt constraint handler.

HiGHS cannot add lazy constraints, so the only **open** single-tree option is
SCIP (>= 8, Apache-2.0). The master holds ``y``, the recourse epigraph ``theta``,
and (under the chance constraint) ``z``. A constraint handler separates Benders
optimality cuts whenever the search reaches a candidate solution whose ``theta``
underestimates the true recourse cost; the cut routine is shared with every
other backend (:mod:`sflp.model.subproblem`).

**Lazy-constraint correctness.** SCIP's dual reductions assume the constraint set
is complete. Because our cuts arrive lazily, those reductions can fix first-stage
variables and cut off the true optimum. We disable them
(``misc/allowstrongdualreds`` / ``misc/allowweakdualreds``); this is required for
the single-tree backend to match the monolith.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from pyscipopt import SCIP_RESULT, Conshdlr, Model, quicksum

from sflp.benders.backend import BendersBackend, BendersResult
from sflp.config import ModelConfig, SolverConfig
from sflp.data.instance import FloatArray, Instance, ScenarioSet
from sflp.model.subproblem import RecourseResult, RecourseSubproblem, build_subproblems

MIN_SCIP_MAJOR = 8


class _BendersConshdlr(Conshdlr):
    """Separates Benders optimality cuts for integer-/LP-feasible candidates."""

    def __init__(
        self,
        y_vars: dict[int, Any],
        theta_vars: dict[int, Any],
        z_vars: dict[int, Any] | None,
        subproblems: list[RecourseSubproblem],
        has_chance: bool,
        capacity: FloatArray,
        demand_total: FloatArray,
    ) -> None:
        super().__init__()
        self._y = y_vars
        self._theta = theta_vars
        self._z = z_vars
        self._subs = subproblems
        self._has_chance = has_chance
        self._capacity = capacity
        self._demand_total = demand_total
        self.n_cuts = 0
        self.n_rounds = 0

    def _read(self, sol: Any) -> tuple[FloatArray, FloatArray, FloatArray]:
        # The handler enforces integer-feasible candidates, so round y and z to
        # 0/1. This keeps the cut count down (one cut per integer point, not per
        # fractional LP iterate) and lets the subproblem cache hit repeatedly.
        n_i, n_s = len(self._y), len(self._subs)
        y = np.array(
            [round(self.model.getSolVal(sol, self._y[i])) for i in range(n_i)], dtype=float
        )
        theta = np.array([self.model.getSolVal(sol, self._theta[s]) for s in range(n_s)])
        if self._has_chance and self._z is not None:
            z = np.array(
                [round(self.model.getSolVal(sol, self._z[s])) for s in range(n_s)], dtype=float
            )
        else:
            z = np.zeros(n_s)
        return y, theta, z

    def _violations(self, sol: Any) -> Iterator[tuple[int, RecourseResult]]:
        y, theta, z = self._read(sol)
        total_capacity = float(self._capacity @ y)
        for s, sub in enumerate(self._subs):
            # When z_s = 0 the scenario must be fully served; that is feasible
            # only if open capacity covers its demand. If not, the master's
            # linear capacity constraint (not this handler) rejects the
            # candidate, so skip the (infeasible) recourse solve here.
            if self._has_chance and z[s] < 0.5 and total_capacity < self._demand_total[s] - 1e-6:
                continue
            r = sub.solve(y, float(z[s]))
            if r.objective - theta[s] > 1e-6 * (1.0 + abs(r.objective)):
                yield s, r

    def _enforce(self, sol: Any) -> dict[str, int]:
        self.n_rounds += 1
        added = False
        for s, r in self._violations(sol):
            rhs = r.intercept + quicksum(r.y_coef[i] * self._y[i] for i in self._y)
            if self._has_chance and self._z is not None and r.z_coef != 0.0:
                rhs = rhs + r.z_coef * self._z[s]
            self.model.addCons(self._theta[s] >= rhs, name=f"benders_{self.n_cuts}")
            self.n_cuts += 1
            added = True
        return {"result": SCIP_RESULT.CONSADDED if added else SCIP_RESULT.FEASIBLE}

    def consenfolp(
        self, constraints: Any, nusefulconss: int, solinfeasible: bool
    ) -> dict[str, int]:
        return self._enforce(None)

    def consenfops(
        self, constraints: Any, nusefulconss: int, solinfeasible: bool, objinfeasible: bool
    ) -> dict[str, int]:
        return self._enforce(None)

    def conscheck(
        self,
        constraints: Any,
        solution: Any,
        checkintegrality: bool,
        checklprows: bool,
        printreason: bool,
        completely: bool,
        **kwargs: Any,
    ) -> dict[str, int]:
        for _ in self._violations(solution):
            return {"result": SCIP_RESULT.INFEASIBLE}
        return {"result": SCIP_RESULT.FEASIBLE}

    def conslock(self, constraint: Any, locktype: Any, nlockspos: int, nlocksneg: int) -> None:
        pass


class ScipBackend(BendersBackend):
    """Single-tree branch-and-Benders-cut backend on SCIP."""

    def solve(
        self,
        instance: Instance,
        scenarios: ScenarioSet,
        model_cfg: ModelConfig,
        solver_cfg: SolverConfig,
    ) -> BendersResult:
        m = Model("sflp-scip-benders")
        version = m.version()
        if version < MIN_SCIP_MAJOR:
            raise RuntimeError(f"SCIP >= {MIN_SCIP_MAJOR} required (Apache-2.0), found {version}.")
        m.hideOutput()
        # Lazy cuts => dual reductions are unsound; disable them.
        m.setBoolParam("misc/allowstrongdualreds", False)
        m.setBoolParam("misc/allowweakdualreds", False)
        if solver_cfg.time_limit is not None:
            m.setRealParam("limits/time", solver_cfg.time_limit)
        m.setRealParam("limits/gap", solver_cfg.gap_tolerance)

        n_i, n_s = instance.n_facilities, scenarios.n_scenarios
        f, s_cap, p = instance.fixed_cost, instance.capacity, scenarios.probability
        has_chance = model_cfg.chance_constraint
        max_recourse = (scenarios.demand * instance.unmet_penalty[None, :]).sum(axis=1)
        demand_total = scenarios.demand.sum(axis=1)

        y = {i: m.addVar(vtype="B", name=f"y{i}") for i in range(n_i)}
        theta = {
            s: m.addVar(vtype="C", lb=0.0, ub=float(max_recourse[s]), name=f"theta{s}")
            for s in range(n_s)
        }
        z = None
        if has_chance:
            z = {s: m.addVar(vtype="B", name=f"z{s}") for s in range(n_s)}
            m.addCons(quicksum(p[s] * z[s] for s in range(n_s)) <= model_cfg.gamma)
            for s in range(n_s):
                m.addCons(
                    quicksum(s_cap[i] * y[i] for i in range(n_i)) >= demand_total[s] * (1 - z[s])
                )

        if model_cfg.budget is not None:
            m.addCons(quicksum(f[i] * y[i] for i in range(n_i)) <= model_cfg.budget)
        if model_cfg.cardinality is not None:
            m.addCons(quicksum(y[i] for i in range(n_i)) <= model_cfg.cardinality)

        m.setObjective(
            quicksum(f[i] * y[i] for i in range(n_i))
            + quicksum(p[s] * theta[s] for s in range(n_s)),
            "minimize",
        )

        subs = build_subproblems(instance, scenarios.demand, has_chance)

        # Warm start: seed initial cuts at several first-stage points (none open,
        # half open, all open) so the root relaxation already bounds theta well.
        # This shrinks the branch-and-bound tree dramatically versus relying on
        # lazy cuts alone.
        warm_points = [np.zeros(n_i), np.full(n_i, 0.5), np.ones(n_i)]
        for point in warm_points:
            for s in range(n_s):
                r = subs[s].solve(point, 1.0)
                rhs = r.intercept + quicksum(r.y_coef[i] * y[i] for i in range(n_i))
                if has_chance and z is not None and r.z_coef != 0.0:
                    rhs = rhs + r.z_coef * z[s]
                m.addCons(theta[s] >= rhs)

        handler = _BendersConshdlr(y, theta, z, subs, has_chance, s_cap, demand_total)
        m.includeConshdlr(
            handler,
            "sflpbenders",
            "Benders optimality cuts for the stochastic CFLP",
            enfopriority=-1,
            chckpriority=-1,
            needscons=True,
        )
        m.addPyCons(m.createCons(handler, "benders_handler"))

        m.optimize()
        status = m.getStatus()
        y_val = np.array([round(m.getVal(y[i])) for i in range(n_i)], dtype=np.float64)
        objective = float(m.getObjVal())
        lower_bound = float(m.getDualbound())
        gap = (objective - lower_bound) / max(1.0, abs(objective))
        return BendersResult(
            objective=objective,
            y=y_val,
            open_facilities=[i for i in range(n_i) if y_val[i] > 0.5],
            lower_bound=lower_bound,
            upper_bound=objective,
            gap=gap,
            iterations=handler.n_rounds,
            n_cuts=handler.n_cuts,
            converged=status == "optimal",
        )

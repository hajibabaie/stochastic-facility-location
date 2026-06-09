"""Single-tree branch-and-Benders-cut on Gurobi via lazy-constraint callbacks.

This is the commercial fast path used for the largest (150-node / 50-scenario)
headline instance. Gurobi keeps one branch-and-bound tree and calls back on every
integer-feasible solution; we solve the recourse subproblems there and inject the
violated optimality cuts with ``cbLazy``. The cut routine
(:mod:`sflp.model.subproblem`) is shared with the open backends, so the three
backends are interchangeable behind :class:`~sflp.benders.backend.BendersBackend`.

``gurobipy`` is an optional dependency (the ``[gurobi]`` extra); it is imported
lazily so the open install never needs it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sflp.benders.backend import BendersBackend, BendersResult
from sflp.config import ModelConfig, SolverConfig
from sflp.data.instance import Instance, ScenarioSet
from sflp.model.subproblem import build_subproblems


class GurobiBackend(BendersBackend):
    """Single-tree branch-and-Benders-cut backend on Gurobi (commercial)."""

    def solve(
        self,
        instance: Instance,
        scenarios: ScenarioSet,
        model_cfg: ModelConfig,
        solver_cfg: SolverConfig,
    ) -> BendersResult:
        import gurobipy as gp
        from gurobipy import GRB

        n_i, n_s = instance.n_facilities, scenarios.n_scenarios
        f, s_cap, p = instance.fixed_cost, instance.capacity, scenarios.probability
        has_chance = model_cfg.chance_constraint
        max_recourse = (scenarios.demand * instance.unmet_penalty[None, :]).sum(axis=1)
        demand_total = scenarios.demand.sum(axis=1)
        subs = build_subproblems(instance, scenarios.demand, has_chance)

        m = gp.Model("sflp-gurobi-benders")
        m.Params.OutputFlag = 0
        m.Params.LazyConstraints = 1
        m.Params.MIPGap = solver_cfg.gap_tolerance
        if solver_cfg.time_limit is not None:
            m.Params.TimeLimit = solver_cfg.time_limit

        y = m.addVars(range(n_i), vtype=GRB.BINARY, name="y")
        theta = m.addVars(range(n_s), lb=0.0, ub=[float(v) for v in max_recourse], name="theta")
        z = None
        if has_chance:
            z = m.addVars(range(n_s), vtype=GRB.BINARY, name="z")
            m.addConstr(gp.quicksum(p[s] * z[s] for s in range(n_s)) <= model_cfg.gamma)
            for s in range(n_s):
                m.addConstr(
                    gp.quicksum(s_cap[i] * y[i] for i in range(n_i)) >= demand_total[s] * (1 - z[s])
                )
        if model_cfg.budget is not None:
            m.addConstr(gp.quicksum(f[i] * y[i] for i in range(n_i)) <= model_cfg.budget)
        if model_cfg.cardinality is not None:
            m.addConstr(gp.quicksum(y[i] for i in range(n_i)) <= model_cfg.cardinality)

        m.setObjective(
            gp.quicksum(f[i] * y[i] for i in range(n_i))
            + gp.quicksum(p[s] * theta[s] for s in range(n_s)),
            GRB.MINIMIZE,
        )

        # Warm start: seed initial cuts at several first-stage points so the root
        # relaxation already bounds theta well and the tree stays small.
        for point in (np.zeros(n_i), np.full(n_i, 0.5), np.ones(n_i)):
            for s in range(n_s):
                r = subs[s].solve(point, 1.0)
                expr = r.intercept + gp.quicksum(r.y_coef[i] * y[i] for i in range(n_i))
                if has_chance and z is not None and r.z_coef != 0.0:
                    expr = expr + r.z_coef * z[s]
                m.addConstr(theta[s] >= expr)

        stats = {"rounds": 0, "cuts": 0}

        def callback(model: Any, where: int) -> None:
            if where != GRB.Callback.MIPSOL:
                return
            stats["rounds"] += 1
            y_val = np.maximum(np.array(model.cbGetSolution([y[i] for i in range(n_i)])), 0.0)
            theta_val = np.array(model.cbGetSolution([theta[s] for s in range(n_s)]))
            z_val = (
                np.array(model.cbGetSolution([z[s] for s in range(n_s)]))
                if has_chance and z is not None
                else np.zeros(n_s)
            )
            total_capacity = float(s_cap @ y_val)
            for s in range(n_s):
                if has_chance and z_val[s] < 0.5 and total_capacity < demand_total[s] - 1e-6:
                    continue
                r = subs[s].solve(y_val, float(z_val[s]))
                if r.objective - theta_val[s] > 1e-6 * (1.0 + abs(r.objective)):
                    expr = r.intercept + gp.quicksum(r.y_coef[i] * y[i] for i in range(n_i))
                    if has_chance and z is not None and r.z_coef != 0.0:
                        expr = expr + r.z_coef * z[s]
                    model.cbLazy(theta[s] >= expr)
                    stats["cuts"] += 1

        m.optimize(callback)

        y_sol = np.array([round(y[i].X) for i in range(n_i)], dtype=np.float64)
        objective = float(m.ObjVal)
        lower_bound = float(m.ObjBound)
        gap = (objective - lower_bound) / max(1.0, abs(objective))
        return BendersResult(
            objective=objective,
            y=y_sol,
            open_facilities=[i for i in range(n_i) if y_sol[i] > 0.5],
            lower_bound=lower_bound,
            upper_bound=objective,
            gap=gap,
            iterations=stats["rounds"],
            n_cuts=stats["cuts"],
            converged=m.Status == GRB.OPTIMAL,
        )

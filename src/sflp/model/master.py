"""Benders master problem (Pyomo) for the classic L-shaped loop.

The master keeps the first-stage decisions and a recourse epigraph variable per
scenario (multi-cut):

.. math::

    \\min \\sum_i f_i y_i + \\sum_s p_s \\theta_s

with ``theta_s >= 0`` and optimality cuts added lazily as the loop proceeds.
When the chance constraint is on it also carries the binary ``z_s``, the service
budget ``sum_s p_s z_s <= gamma``, and the capacity inequality

.. math::

    \\sum_i s_i y_i \\ge \\Big(\\sum_j d_{js}\\Big)(1 - z_s) \\quad\\forall s,

which guarantees that every ``z_s = 0`` scenario can be fully served (the
bipartite serve graph is complete, so total capacity >= total demand suffices).
That keeps every recourse subproblem feasible, so the loop needs optimality cuts
only.
"""

from __future__ import annotations

import pyomo.environ as pyo

from sflp.config import ModelConfig
from sflp.data.instance import Instance, ScenarioSet


def build_master(instance: Instance, scenarios: ScenarioSet, cfg: ModelConfig) -> pyo.ConcreteModel:
    """Build the Benders master MILP (no cuts yet)."""
    m = pyo.ConcreteModel(name=f"master-{instance.name}")
    m.I = pyo.RangeSet(0, instance.n_facilities - 1)
    m.S = pyo.RangeSet(0, scenarios.n_scenarios - 1)

    f, s_cap = instance.fixed_cost, instance.capacity
    p = scenarios.probability
    scenario_demand_total = scenarios.demand.sum(axis=1)  # (S,)

    m.y = pyo.Var(m.I, domain=pyo.Binary)
    m.theta = pyo.Var(m.S, domain=pyo.NonNegativeReals)
    m.cuts = pyo.ConstraintList()

    m.obj = pyo.Objective(
        expr=sum(f[i] * m.y[i] for i in m.I) + sum(p[s] * m.theta[s] for s in m.S),
        sense=pyo.minimize,
    )

    if cfg.chance_constraint:
        m.z = pyo.Var(m.S, domain=pyo.Binary)
        m.service_budget = pyo.Constraint(expr=sum(p[s] * m.z[s] for s in m.S) <= cfg.gamma)
        m.feasible_capacity = pyo.Constraint(
            m.S,
            rule=lambda mm, s: (
                sum(s_cap[i] * mm.y[i] for i in mm.I) >= scenario_demand_total[s] * (1 - mm.z[s])
            ),
        )

    if cfg.budget is not None:
        m.budget = pyo.Constraint(expr=sum(f[i] * m.y[i] for i in m.I) <= cfg.budget)
    if cfg.cardinality is not None:
        m.cardinality = pyo.Constraint(expr=sum(m.y[i] for i in m.I) <= cfg.cardinality)

    return m

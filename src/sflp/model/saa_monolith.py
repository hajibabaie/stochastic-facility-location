"""Sample-average-approximation (SAA) deterministic equivalent ("monolith").

This is the two-stage stochastic CFLP written as a single MILP over all sampled
scenarios at once. Facilities ``y`` are chosen once (first stage); flows ``x``
and unmet demand ``u`` are chosen per scenario (recourse):

.. math::

    \\min \\; \\sum_i f_i y_i
        + \\sum_s p_s \\Big( \\sum_{ij} c_{ij} x_{ijs} + \\sum_j q_j u_{js} \\Big)

subject to demand balance ``sum_i x_ijs + u_js = d_js``, capacity
``sum_j x_ijs <= s_i y_i``, and the strong link ``x_ijs <= d_js y_i``.

The unmet variable ``u`` gives **relatively complete recourse**: every scenario
is feasible for any ``y`` (worst case: serve nobody, pay the penalty). That is
what lets the Benders decomposition use optimality cuts only. With one scenario,
no uncertainty, and a large penalty, this reduces to the deterministic CFLP, so
it is the oracle the Benders backends must match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyomo.environ as pyo

from sflp.config import ModelConfig
from sflp.data.instance import FloatArray, Instance, ScenarioSet
from sflp.model.chance import add_chance_constraint


@dataclass(frozen=True)
class SaaSolution:
    """A solved SAA monolith with its first-stage / recourse cost split."""

    objective: float
    y: FloatArray  # (I,) rounded 0/1
    open_facilities: list[int]
    first_stage_cost: float  # sum_i f_i y_i
    expected_recourse_cost: float  # objective - first_stage_cost
    expected_unmet: float  # sum_s p_s sum_j u_js
    violation_probability: float  # sum_s p_s z_s (0 if no chance constraint)


def build_saa_monolith(
    instance: Instance, scenarios: ScenarioSet, cfg: ModelConfig
) -> pyo.ConcreteModel:
    """Build the SAA deterministic-equivalent MILP.

    ``cfg`` carries optional structural constraints (budget, cardinality). The
    service-level chance constraint is added by :mod:`sflp.model.chance` when
    ``cfg.chance_constraint`` is set; this base builder leaves it off.
    """
    if instance.n_customers != scenarios.n_customers:
        raise ValueError("instance and scenarios disagree on the number of customers.")

    m = pyo.ConcreteModel(name=f"saa-{instance.name}-S{scenarios.n_scenarios}")
    m.I = pyo.RangeSet(0, instance.n_facilities - 1)
    m.J = pyo.RangeSet(0, instance.n_customers - 1)
    m.S = pyo.RangeSet(0, scenarios.n_scenarios - 1)

    f, s_cap, c, q = (
        instance.fixed_cost,
        instance.capacity,
        instance.unit_cost,
        instance.unmet_penalty,
    )
    d = scenarios.demand  # (S, J)
    p = scenarios.probability  # (S,)

    m.y = pyo.Var(m.I, domain=pyo.Binary)
    m.x = pyo.Var(m.I, m.J, m.S, domain=pyo.NonNegativeReals)
    m.u = pyo.Var(m.J, m.S, domain=pyo.NonNegativeReals)

    m.first_stage = sum(f[i] * m.y[i] for i in m.I)
    m.recourse = sum(
        p[s]
        * (
            sum(c[i, j] * m.x[i, j, s] for i in m.I for j in m.J)
            + sum(q[j] * m.u[j, s] for j in m.J)
        )
        for s in m.S
    )
    m.obj = pyo.Objective(expr=m.first_stage + m.recourse, sense=pyo.minimize)

    m.demand_met = pyo.Constraint(
        m.J, m.S, rule=lambda mm, j, s: sum(mm.x[i, j, s] for i in mm.I) + mm.u[j, s] == d[s, j]
    )
    m.capacity = pyo.Constraint(
        m.I, m.S, rule=lambda mm, i, s: sum(mm.x[i, j, s] for j in mm.J) <= s_cap[i] * mm.y[i]
    )
    m.link = pyo.Constraint(
        m.I, m.J, m.S, rule=lambda mm, i, j, s: mm.x[i, j, s] <= d[s, j] * mm.y[i]
    )

    if cfg.chance_constraint:
        add_chance_constraint(m, instance, scenarios, cfg)
    _add_structural_constraints(m, instance, cfg)
    return m


def _add_structural_constraints(m: pyo.ConcreteModel, instance: Instance, cfg: ModelConfig) -> None:
    """Attach optional budget (C5) and cardinality (C6) constraints."""
    if cfg.budget is not None:
        f = instance.fixed_cost
        m.budget = pyo.Constraint(expr=sum(f[i] * m.y[i] for i in m.I) <= cfg.budget)
    if cfg.cardinality is not None:
        m.cardinality = pyo.Constraint(expr=sum(m.y[i] for i in m.I) <= cfg.cardinality)


def extract_saa_solution(
    model: pyo.ConcreteModel, instance: Instance, scenarios: ScenarioSet
) -> SaaSolution:
    """Read a solved SAA monolith into a :class:`SaaSolution`."""
    n_i = instance.n_facilities
    y = np.array([round(pyo.value(model.y[i])) for i in range(n_i)], dtype=np.float64)
    first_stage = float(instance.fixed_cost @ y)
    objective = float(pyo.value(model.obj))
    n_j = instance.n_customers
    p = scenarios.probability
    expected_unmet = float(
        sum(
            p[s] * sum(pyo.value(model.u[j, s]) for j in range(n_j))
            for s in range(scenarios.n_scenarios)
        )
    )
    violation_probability = 0.0
    if hasattr(model, "z"):
        violation_probability = float(
            sum(p[s] * round(pyo.value(model.z[s])) for s in range(scenarios.n_scenarios))
        )
    return SaaSolution(
        objective=objective,
        y=y,
        open_facilities=[i for i in range(n_i) if y[i] > 0.5],
        first_stage_cost=first_stage,
        expected_recourse_cost=objective - first_stage,
        expected_unmet=expected_unmet,
        violation_probability=violation_probability,
    )

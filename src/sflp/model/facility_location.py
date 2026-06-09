"""Deterministic capacitated facility location (CFLP) as a Pyomo model.

The model opens facilities (binary ``y``) and ships demand from open facilities
to customers (continuous ``x``), minimizing opening cost plus per-unit transport
cost:

.. math::

    \\min \\; \\sum_i f_i y_i + \\sum_{ij} c_{ij} x_{ij}
    \\quad\\text{s.t.}\\quad
    \\sum_i x_{ij} = d_j,\\;
    \\sum_j x_{ij} \\le s_i y_i,\\;
    x_{ij} \\le d_j y_i,\\;
    y \\in \\{0,1\\},\\; x \\ge 0.

The **disaggregated** link ``x_ij <= d_j y_i`` is redundant given the capacity
constraint but tightens the LP relaxation a lot. It is the standard strong CFLP
formulation; it is also what lets HiGHS' presolve solve these instances
correctly (the weak aggregated model triggers a presolve reduction that returns
a suboptimal incumbent as "optimal").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyomo.environ as pyo

from sflp.data.instance import FloatArray, Instance


@dataclass(frozen=True)
class CflpSolution:
    """A solved deterministic CFLP: objective split into its cost components."""

    objective: float
    open_facilities: list[int]
    y: FloatArray  # (I,) rounded 0/1
    flow: FloatArray  # (I, J) shipped amounts
    fixed_cost: float
    transport_cost: float


def build_deterministic_cflp(instance: Instance) -> pyo.ConcreteModel:
    """Build the strong-formulation CFLP model for ``instance``."""
    m = pyo.ConcreteModel(name=f"cflp-{instance.name}")
    m.I = pyo.RangeSet(0, instance.n_facilities - 1)
    m.J = pyo.RangeSet(0, instance.n_customers - 1)

    f = instance.fixed_cost
    s = instance.capacity
    d = instance.demand
    c = instance.unit_cost

    m.y = pyo.Var(m.I, domain=pyo.Binary)
    m.x = pyo.Var(m.I, m.J, domain=pyo.NonNegativeReals)

    m.obj = pyo.Objective(
        expr=sum(f[i] * m.y[i] for i in m.I) + sum(c[i, j] * m.x[i, j] for i in m.I for j in m.J),
        sense=pyo.minimize,
    )
    m.demand_met = pyo.Constraint(m.J, rule=lambda mm, j: sum(mm.x[i, j] for i in mm.I) == d[j])
    m.capacity = pyo.Constraint(
        m.I, rule=lambda mm, i: sum(mm.x[i, j] for j in mm.J) <= s[i] * mm.y[i]
    )
    m.link = pyo.Constraint(m.I, m.J, rule=lambda mm, i, j: mm.x[i, j] <= d[j] * mm.y[i])
    return m


def extract_cflp_solution(model: pyo.ConcreteModel, instance: Instance) -> CflpSolution:
    """Read a solved CFLP model into a :class:`CflpSolution`."""
    n_i, n_j = instance.n_facilities, instance.n_customers
    y = np.array([round(pyo.value(model.y[i])) for i in range(n_i)], dtype=np.float64)
    flow = np.array(
        [[pyo.value(model.x[i, j]) for j in range(n_j)] for i in range(n_i)],
        dtype=np.float64,
    )
    fixed = float(instance.fixed_cost @ y)
    transport = float((instance.unit_cost * flow).sum())
    return CflpSolution(
        objective=fixed + transport,
        open_facilities=[i for i in range(n_i) if y[i] > 0.5],
        y=y,
        flow=flow,
        fixed_cost=fixed,
        transport_cost=transport,
    )

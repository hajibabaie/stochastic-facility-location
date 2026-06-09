"""Service-level chance constraint for the SAA model.

We require that the chosen facilities **fully serve** demand in all but a small
fraction of scenarios. Following Luedtke & Ahmed (2008) and Pagnoncelli et al.
(2009), a binary ``z_s`` marks a scenario allowed to violate full service:

.. math::

    u_{js} \\le d_{js}\\, z_s \\quad\\forall j,s, \\qquad
    \\sum_s p_s z_s \\le \\gamma .

If ``z_s = 0`` the big-M link forces ``u_{js} = 0`` (scenario ``s`` is fully
served). The budget caps the **probability mass** of violating scenarios at the
SAA risk level ``gamma``. For equal-weight scenarios (``p_s = 1/N``) this is the
classic count form ``sum_s z_s <= floor(gamma N)``; with reduced (unequal-weight)
scenarios the probability-weighted form is the correct generalization.

The big-M is tight: ``u_{js} <= d_{js}`` always holds, so ``d_{js}`` is the
smallest valid coefficient.

**gamma is not epsilon.** ``gamma`` is the *in-sample* SAA risk level; the *true*
target is ``epsilon`` (each scenario served with probability >= 1 - epsilon). For
the SAA solution to be safe for the true constraint, set ``gamma <= epsilon``
(``gamma = 0`` is a guaranteed conservative inner approximation). The config
keeps them as separate fields and enforces ``gamma <= epsilon``.
"""

from __future__ import annotations

import pyomo.environ as pyo

from sflp.config import ModelConfig
from sflp.data.instance import Instance, ScenarioSet


def add_chance_constraint(
    m: pyo.ConcreteModel,
    instance: Instance,
    scenarios: ScenarioSet,
    cfg: ModelConfig,
) -> None:
    """Attach z_s, the big-M unmet link (C3), and the service budget (C4) to ``m``.

    The model ``m`` must already have the recourse variables ``u[j, s]`` and the
    scenario set ``S`` (i.e. it is a built SAA monolith).
    """
    d = scenarios.demand  # (S, J)
    p = scenarios.probability  # (S,)

    m.z = pyo.Var(m.S, domain=pyo.Binary)
    m.bigm_unmet = pyo.Constraint(m.J, m.S, rule=lambda mm, j, s: mm.u[j, s] <= d[s, j] * mm.z[s])
    m.service_budget = pyo.Constraint(expr=sum(p[s] * m.z[s] for s in m.S) <= cfg.gamma)

"""Benders optimality cut: a linear lower bound on one scenario's recourse cost.

A cut for scenario ``s`` reads ``theta_s >= intercept + y_coef . y + z_coef * z_s``
and is produced by :class:`~sflp.model.subproblem.RecourseSubproblem`. The same
cut object is consumed by every backend (the Pyomo master, the SCIP constraint
handler, the Gurobi callback).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyomo.environ as pyo

from sflp.data.instance import FloatArray
from sflp.model.subproblem import RecourseResult


@dataclass(frozen=True)
class BendersCut:
    """An optimality cut tied to a scenario index."""

    scenario: int
    intercept: float
    y_coef: FloatArray  # (I,)
    z_coef: float

    @classmethod
    def from_recourse(cls, scenario: int, result: RecourseResult) -> BendersCut:
        return cls(
            scenario=scenario,
            intercept=result.intercept,
            y_coef=result.y_coef,
            z_coef=result.z_coef,
        )

    def value_at(self, y: FloatArray, z_s: float) -> float:
        """The cut's right-hand side at a given first stage (for diagnostics)."""
        return float(self.intercept + self.y_coef @ y + self.z_coef * z_s)


def add_optimality_cut(master: pyo.ConcreteModel, cut: BendersCut) -> Any:
    """Append ``theta_s >= intercept + y_coef . y + z_coef z_s`` to the master.

    Returns the new constraint object so a persistent solver can register it
    incrementally.
    """
    s = cut.scenario
    rhs = cut.intercept + sum(cut.y_coef[i] * master.y[i] for i in master.I)
    if hasattr(master, "z") and cut.z_coef != 0.0:
        rhs = rhs + cut.z_coef * master.z[s]
    return master.cuts.add(master.theta[s] >= rhs)

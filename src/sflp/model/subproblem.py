"""Benders recourse subproblem and optimality-cut generation (plain Python).

For a fixed first stage ``(y, z)`` the second stage splits into one independent
LP per scenario. Each scenario LP serves demand from open facilities, paying
transport ``c`` and unmet penalty ``q``:

.. math::

    Q_s(y, z_s) = \\min \\sum_{ij} c_{ij} x_{ij} + \\sum_j q_j u_j
    \\;\\text{s.t.}\\;
    \\sum_i x_{ij} + u_j = d_j,\\;
    \\sum_j x_{ij} \\le s_i y_i,\\;
    u_j \\le d_j z_s,\\; x, u \\ge 0 .

By LP duality the optimum equals ``b^T \\pi`` for the optimal dual ``\\pi``, and
because the dual feasible region does **not** depend on ``(y, z)``, one dual
solution yields a cut valid for *all* ``(y, z)``:

.. math::

    Q_s(y, z_s) \\ge \\underbrace{\\sum_j \\pi_j d_j}_{\\text{intercept}}
        + \\sum_i (\\alpha_i s_i)\\, y_i + \\Big(\\sum_j \\delta_j d_j\\Big) z_s ,

where ``\\pi`` (demand, free), ``\\alpha \\le 0`` (capacity) and ``\\delta \\le 0``
(unmet link) are the row duals. This is an **optimality cut**; relatively
complete recourse (the ``u`` escape, kept feasible by the master's capacity
inequality) means no feasibility cuts are ever needed.

The LP is built once with HiGHS (highspy); only the right-hand side changes
across solves, so the per-iteration cost is a warm-startable re-solve, not a
model rebuild. The routine is solver-agnostic plain Python, so the classic loop
and the single-tree SCIP/Gurobi callbacks all share one cut routine.
"""

from __future__ import annotations

from dataclasses import dataclass

import highspy
import numpy as np
from scipy.sparse import coo_matrix

from sflp.data.instance import FloatArray, Instance

_INF = highspy.kHighsInf


@dataclass(frozen=True)
class RecourseResult:
    """Optimal recourse cost for one scenario and the optimality cut it induces.

    The cut is ``theta_s >= intercept + y_coef . y + z_coef * z_s`` and is tight
    at the queried ``(y, z_s)``: ``objective == intercept + y_coef . y_hat
    + z_coef * z_hat``.
    """

    objective: float
    intercept: float
    y_coef: FloatArray  # (I,)
    z_coef: float


class RecourseSubproblem:
    """Reusable per-scenario recourse LP; only the RHS changes across solves.

    Row layout: ``J`` demand equalities, then ``I`` capacity rows, then ``J``
    unmet-link rows. Columns: the ``I*J`` flows ``x`` then the ``J`` unmet ``u``.
    """

    def __init__(self, instance: Instance, demand_scenario: FloatArray, has_chance: bool) -> None:
        self._n_i = instance.n_facilities
        self._n_j = instance.n_customers
        self._capacity = instance.capacity
        self._demand = np.asarray(demand_scenario, dtype=np.float64)
        self._has_chance = has_chance

        n_i, n_j = self._n_i, self._n_j
        n_x = n_i * n_j
        n_var = n_x + n_j
        n_row = n_j + n_i + n_j

        cost = np.empty(n_var, dtype=np.float64)
        cost[:n_x] = instance.unit_cost.reshape(-1)
        cost[n_x:] = instance.unmet_penalty

        rows: list[int] = []
        cols: list[int] = []
        for j in range(n_j):  # demand row j: sum_i x_ij + u_j
            for i in range(n_i):
                rows.append(j)
                cols.append(i * n_j + j)
            rows.append(j)
            cols.append(n_x + j)
        for i in range(n_i):  # capacity row J+i: sum_j x_ij
            for j in range(n_j):
                rows.append(n_j + i)
                cols.append(i * n_j + j)
        for j in range(n_j):  # link row J+I+j: u_j
            rows.append(n_j + n_i + j)
            cols.append(n_x + j)
        csc = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_row, n_var)).tocsc()

        lp = highspy.HighsLp()
        lp.num_col_ = n_var
        lp.num_row_ = n_row
        lp.col_cost_ = cost
        lp.col_lower_ = np.zeros(n_var)
        lp.col_upper_ = np.full(n_var, _INF)
        lp.row_lower_ = np.concatenate([self._demand, np.full(n_i + n_j, -_INF)])
        lp.row_upper_ = np.concatenate([self._demand, np.zeros(n_i + n_j)])
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = csc.indptr
        lp.a_matrix_.index_ = csc.indices
        lp.a_matrix_.value_ = csc.data

        self._highs = highspy.Highs()
        self._highs.setOptionValue("output_flag", False)
        self._highs.setOptionValue("presolve", "off")  # keep vertex duals stable
        self._highs.passModel(lp)
        # Rows whose upper bound changes each solve: capacity then link.
        self._rhs_rows = np.arange(n_j, n_row, dtype=np.int32)
        # Single-tree backends query the same (y, z) candidate many times (LP
        # nodes, heuristics, the feasibility check); cache the last solves.
        self._cache: dict[bytes, RecourseResult] = {}

    def solve(self, y_hat: FloatArray, z_hat: float) -> RecourseResult:
        """Solve the recourse LP at ``(y_hat, z_hat)`` and build its optimality cut."""
        key = np.round(y_hat, 9).tobytes() + np.float64(round(z_hat, 9)).tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._solve(y_hat, z_hat)
        if len(self._cache) < 100_000:
            self._cache[key] = result
        return result

    def _solve(self, y_hat: FloatArray, z_hat: float) -> RecourseResult:
        n_i, n_j = self._n_i, self._n_j
        s, d = self._capacity, self._demand
        z_eff = z_hat if self._has_chance else 1.0

        upper = np.concatenate([s * y_hat, d * z_eff])
        lower = np.full(upper.size, -_INF)
        self._highs.changeRowsBounds(self._rhs_rows.size, self._rhs_rows, lower, upper)
        self._highs.run()

        status = self._highs.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"recourse subproblem not optimal (status {status}).")

        duals = np.array(self._highs.getSolution().row_dual)
        pi = duals[:n_j]
        alpha = duals[n_j : n_j + n_i]
        delta = duals[n_j + n_i :]

        intercept = float(pi @ d)
        y_coef = alpha * s
        link_term = float(delta @ d)
        if self._has_chance:
            z_coef = link_term
        else:
            intercept += link_term  # link RHS is the constant d_j (z fixed to 1)
            z_coef = 0.0

        return RecourseResult(
            objective=float(self._highs.getObjectiveValue()),
            intercept=intercept,
            y_coef=y_coef,
            z_coef=z_coef,
        )


def build_subproblems(
    instance: Instance, demand: FloatArray, has_chance: bool
) -> list[RecourseSubproblem]:
    """One :class:`RecourseSubproblem` per scenario row of ``demand`` (S, J)."""
    return [RecourseSubproblem(instance, demand[s], has_chance) for s in range(demand.shape[0])]

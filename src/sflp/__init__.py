"""Stochastic capacitated facility location with Benders decomposition.

This package builds and solves a two-stage stochastic capacitated facility
location problem (open facilities now, assign customers under uncertain demand).
It provides:

- A sample average approximation (SAA) deterministic-equivalent MILP, with an
  optional service-level chance constraint.
- A Benders / L-shaped decomposition with optional Magnanti-Wong Pareto-optimal
  cuts, runnable on open solvers (HiGHS, SCIP) and an optional Gurobi fast path.
- Stochastic-quality measures (out-of-sample gap, VSS, EVPI).
"""

__version__ = "0.1.0"

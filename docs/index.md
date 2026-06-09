# Stochastic Facility Location & Network Design

A clean, reproducible implementation of a **two-stage stochastic capacitated
facility location problem** (CFLP) with a service-level **chance constraint**,
solved by **sample average approximation (SAA)** and a **Benders / L-shaped
decomposition with Pareto-optimal cuts**.

We open facilities now (first stage), then assign customers to facilities under
**uncertain demand** (recourse), minimizing facility opening cost plus expected
transport and unmet-demand penalty cost, while guaranteeing that a target
fraction of demand scenarios is fully served.

## What this project demonstrates

- A two-stage stochastic MILP in [Pyomo](https://www.pyomo.org/), solved as an
  SAA deterministic equivalent.
- A joint service-level chance constraint (Luedtke–Ahmed; Pagnoncelli et al.).
- A Benders decomposition with **optimality cuts only** (relatively complete
  recourse), plus **Magnanti–Wong / Papadakos** Pareto-optimal cuts.
- Three interchangeable solver backends: HiGHS (classic loop), SCIP
  (single-tree branch-and-Benders-cut), and Gurobi (single-tree lazy cuts).
- Stochastic-quality reporting: out-of-sample SAA gap, VSS, and EVPI.
- Real city data (GeoNames) and validation against OR-Library published optima.

See [Algorithm & model](algorithm.md) for the math and
[Reproducing results](reproducing-results.md) to run it yourself.

## Solver backends

| Backend | Solver | License | Strategy |
| --- | --- | --- | --- |
| `classic` | HiGHS | MIT | Iterative master → S subproblems → cuts loop |
| `branch_and_cut` | SCIP | Apache-2.0 | Single tree, constraint-handler lazy cuts |
| `gurobi` | Gurobi | Commercial | Single tree, `cbLazy` callback cuts |

The SCIP backend is the default for the open instances. The cut routine is plain
Python (one recourse LP per scenario per candidate), which bounds practical
scale; the bundled configs use instances on the order of a dozen facilities,
verified against the monolith. See
[Reproducing results](reproducing-results.md#scale-and-the-large-instance-template).

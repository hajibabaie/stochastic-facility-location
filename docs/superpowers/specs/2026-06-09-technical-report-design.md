# Technical report — design

## Goal

An arXiv-style preprint (LaTeX `article`, single column, ~12 pages) presenting the
stochastic facility location solver in this repository and a computational study
of its methods. Author: Mohammad S. Hajibabaie.

Working title: *A Two-Stage Stochastic Capacitated Facility Location Solver with a
Service-Level Chance Constraint and Benders Decomposition: Implementation and
Computational Study.*

## Structure

1. Introduction — problem, motivation, contributions.
2. Problem formulation — SAA model, disaggregated link, chance constraint, gamma vs epsilon.
3. Solution methods — extensive form; Benders master/subproblem, optimality cuts;
   Pareto-optimal (Papadakos) cuts; the three backends.
4. Stochastic-quality measures — VSS, EVPI, SAA optimality-gap estimation.
5. Data and implementation — GeoNames, OR-Library, software stack, reproducibility.
6. Computational results:
   - Validation vs OR-Library optima; backend agreement with the monolith.
   - Pareto cuts: iteration counts, standard vs Pareto, on a degenerate instance.
   - Backend comparison (classic / SCIP / Gurobi) timing.
   - Scaling with Gurobi: optimal to ~20 facilities; bounded gap to ~50 in a time limit.
   - Stochastic value: VSS and EVPI; a sigma sweep.
   - Chance constraint: a gamma sweep (cost vs service).
7. Discussion and limitations — the plain-Python cut routine bounds scale.
8. Conclusion and future work — batched/compiled subproblems for larger instances.

## Results pipeline

`scripts/collect_results.py` runs every experiment once and writes a machine-readable
results file (`results/report/results.json`) plus figures, so every number in the
report is reproducible from the repository. The LaTeX reads the numbers from there
(transcribed into tables, with the JSON committed under `paper/data/`).

## Layout

`paper/` holds the LaTeX source (`main.tex`, `references.bib`, a build note) and a
copy of the results data and figures it uses. Builds with `latexmk`/`pdflatex`.

## Honesty

All numbers come from real runs on this machine. The scale limitation is stated
plainly (results to ~20 facilities optimal, ~50 with a reported gap). No fabricated
large-instance results.

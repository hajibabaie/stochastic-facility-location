# Stochastic Facility Location & Network Design

[![CI](https://github.com/hajibabaie/stochastic-facility-location/actions/workflows/ci.yml/badge.svg)](https://github.com/hajibabaie/stochastic-facility-location/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

A solver for the two-stage stochastic capacitated facility location problem
(CFLP) with a service-level chance constraint. Facilities are opened in the first
stage; customer demand is uncertain and is served in the recourse stage. The
problem is formulated as a sample-average-approximation (SAA) mixed-integer
program and solved either as a monolithic extensive form or by Benders
decomposition with three interchangeable solver backends.

## Problem

Let `I` be candidate facilities, `J` customers, and `S` a finite set of demand
scenarios with probabilities `p_s`. The first stage chooses which facilities to
open (`y_i ∈ {0,1}`). The recourse stage chooses flows `x_ijs ≥ 0` and unmet
demand `u_js ≥ 0` per scenario:

```
min  Σ_i f_i y_i + Σ_s p_s ( Σ_ij c_ij x_ijs + Σ_j q_j u_js )
s.t. Σ_i x_ijs + u_js = d_js          ∀ j, s      (demand balance)
     Σ_j x_ijs ≤ s_i y_i              ∀ i, s      (capacity)
     x_ijs ≤ d_js y_i                 ∀ i, j, s   (disaggregated link)
     u_js ≤ d_js z_s                  ∀ j, s      (chance: big-M)
     Σ_s p_s z_s ≤ γ                              (chance: service budget)
```

The unmet-demand variable `u` provides relatively complete recourse, so the
decomposition uses optimality cuts only. The binary `z_s` and the service budget
implement a joint chance constraint: at most a `γ` probability mass of scenarios
may fail to be fully served (Luedtke & Ahmed, 2008; Pagnoncelli et al., 2009).
The disaggregated link strengthens the LP relaxation; it is also required for the
HiGHS presolver to return the correct optimum on the validation instances.

The SAA risk level `γ` is distinct from the true target `ε`: `γ ≤ ε` yields a
conservative inner approximation. The two are separate configuration fields, and
the configuration enforces `γ ≤ ε`. See [docs/algorithm.md](docs/algorithm.md).

## Solver backends

All backends share a single cut-generation routine and differ only in search
strategy. They are selected through configuration and are interchangeable behind
one interface.

| Backend | Solver | License | Strategy |
| --- | --- | --- | --- |
| `classic` | [HiGHS](https://highs.dev/) | MIT | Iterative loop: solve master, solve `S` recourse LPs, add cuts, repeat |
| `branch_and_cut` | [SCIP](https://www.scipopt.org/) ≥ 8 | Apache-2.0 | Single search tree; cuts separated by a constraint handler |
| `gurobi` | [Gurobi](https://www.gurobi.com/) | Commercial | Single search tree; lazy cuts via callback |

Performance and licensing notes:

- The `branch_and_cut` (SCIP) backend is the default for the open instances. It
  keeps a single branch-and-bound tree and is substantially faster than the
  iterative loop at the sizes used here.
- The `classic` (HiGHS) backend is a dependency-free reference implementation and
  the only fully MIT-licensed path; it is slower because each iteration re-solves
  the master and every scenario subproblem.
- The cut-separation routine is plain Python: it solves one recourse LP per
  scenario at every candidate first stage. This keeps the code solver-agnostic
  and easy to follow, but it bounds practical scale. The bundled configurations
  use instances on the order of a dozen facilities, which solve in seconds and
  are verified against the monolith. Larger instances grow quickly in solve time;
  `configs/experiments/headline_150_50.yaml` is provided as a configuration
  template for the Gurobi backend rather than a quick reproducible result.

## Validation

- The deterministic CFLP solver reproduces the published optimal objective values
  of the OR-Library `cap*` instances (`cap71`, `cap101`, `cap131`).
- Each Benders backend is checked against the SAA monolith on shared instances;
  the objectives agree to solver tolerance.
- Scenario generation is verified to be deterministic under a fixed seed and
  mean-preserving.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management; `uv.lock` is committed for reproducible installs.

```bash
# open install (HiGHS + SCIP) plus development tools
uv sync --locked --extra dev

# add the Gurobi backend (requires a Gurobi license)
uv sync --locked --extra dev --extra gurobi
```

## Usage

```bash
# moderate instance on the open SCIP backend
uv run sflp run --config configs/default.yaml

# reproduce every configured experiment
bash scripts/run_all_experiments.sh
```

A run writes a JSON summary to `results/logs/` and figures to `results/figures/`.
Each summary records the seed, the resolved package and solver versions, and the
git commit. See [docs/usage.md](docs/usage.md) for configuration details.

## Technical report

A technical report describing the model, the methods, and a computational study
is in [`paper/`](paper/) (`paper/main.pdf`, built from `paper/main.tex`). Every
number and figure in it is produced by `scripts/collect_results.py`.

## Repository layout

```
src/sflp/
  config.py               # typed YAML configuration with a single seed
  data/                   # download (SHA-256 verified), parse, scenario generation
  model/                  # CFLP, SAA monolith, master, recourse subproblem
  benders/                # cuts, classic loop, SCIP and Gurobi backends
  saa.py                  # VSS, EVPI, and SAA optimality-gap estimation
  solve.py                # solver factory and backend-combination validation
  experiment.py  cli.py  plotting.py
configs/                  # default.yaml and configs/experiments/
tests/                    # unit and integration tests (known-optimum oracles)
docs/                     # MkDocs site
```

## Data sources

Raw third-party data is not redistributed; download and parse scripts fetch it on
demand and verify checksums.

- [GeoNames `cities5000`](https://download.geonames.org/export/dump/) — city
  coordinates and population, CC BY 4.0 (attribution required). The dataset is
  updated daily, so its checksum is not pinned.
- [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/capinfo.html)
  `cap*` instances and published optima (J. E. Beasley); checksums are pinned.

## References

- Santoso, Ahmed, Goetschalckx, Shapiro (2005), *European Journal of Operational
  Research* — stochastic supply-chain network design via SAA and Benders.
- Birge & Louveaux (2011), *Introduction to Stochastic Programming*.
- Kleywegt, Shapiro, Homem-de-Mello (2002) — the sample-average-approximation
  method.
- Luedtke & Ahmed (2008); Pagnoncelli, Ahmed, Shapiro (2009) —
  chance-constrained SAA.
- Magnanti & Wong (1981); Papadakos (2008) — Pareto-optimal Benders cuts.
- Rahmaniani, Crainic, Gendreau, Rei (2017) — Benders decomposition: a review.

## License

[MIT](LICENSE) © 2026 Mohammad S. Hajibabaie.

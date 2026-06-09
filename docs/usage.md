# Usage

## Command line

Solve an experiment described by a YAML config:

```bash
uv run sflp run --config configs/default.yaml
```

Options:

| Flag | Effect |
| --- | --- |
| `--config PATH` | config file to run (required) |
| `--output DIR` | output directory (default `results`) |
| `--name NAME` | run name (default: config file stem) |
| `--no-measures` | skip the VSS/EVPI computation (faster) |
| `--no-plots` | skip figure generation |

A run writes a JSON summary to `results/logs/<name>.json` and figures to
`results/figures/`. The summary records the seed, the resolved package/solver
versions, and the git commit, so any result can be traced to what produced it.

## Configuration

A config is plain YAML mapped onto typed dataclasses (`sflp.config.Config`).
Unknown keys are rejected, so typos fail fast. Key fields:

```yaml
seed: 20231015            # one seed drives all randomness
data:
  source: geonames        # geonames | or_library
  country: DE             # ISO code for the GeoNames instance
  n_facilities: 50        # top-N cities by population
scenarios:
  n_scenarios: 20         # working scenarios after reduction
  n_sample: 1000          # raw Monte-Carlo sample
  sigma: 0.2              # lognormal demand volatility
  correlation_length: null  # km; null = i.i.d. demand
  reduction: kmeans       # kmeans | fast_forward | none
model:
  chance_constraint: true
  epsilon: 0.10           # true target: served w.p. >= 1 - epsilon
  gamma: 0.05             # SAA risk level (keep gamma <= epsilon)
solver:
  backend: classic        # classic | branch_and_cut | gurobi
  mip_solver: highs       # highs | scip | gurobi
  pareto_cuts: true
  gap_tolerance: 1.0e-6
```

## Library

```python
from sflp.config import load_config
from sflp.experiment import run_experiment

result = run_experiment(load_config("configs/default.yaml"))
print(result.objective, result.open_facilities)
print(result.measures.vss, result.measures.evpi)
```

The lower-level pieces are also usable directly: `sflp.solve.solve_saa_monolith`,
`sflp.benders.solve_benders`, and `sflp.saa.compute_stochastic_measures`.

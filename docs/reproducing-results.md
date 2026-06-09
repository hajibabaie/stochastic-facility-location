# Reproducing results

Every run is seeded and records its provenance, so results are reproducible from
a clean checkout.

## Quality gate (what CI runs)

```bash
uv sync --locked --extra dev
uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run pytest -m "not slow and not gurobi"   # unit + integration
```

The integration suite asserts the deterministic CFLP solver reproduces the
**OR-Library** published optima (`cap71`, `cap101`, `cap131`) and that the
Benders backends equal the SAA monolith.

## Experiments

Configs live in `configs/`. Run one with the CLI:

```bash
# tiny smoke test (seconds)
uv run sflp run --config configs/experiments/smoke.yaml

# moderate open-solver default
uv run sflp run --config configs/default.yaml

# open single-tree path on SCIP
uv run sflp run --config configs/experiments/scip_single_tree.yaml

# demand-volatility sweep
for s in 0.10 0.20 0.30; do
  uv run sflp run --config configs/experiments/sigma_${s}.yaml --name sigma_${s}
done
```

Run everything at once:

```bash
bash scripts/run_all_experiments.sh
```

## Scale and the large-instance template

The cut-separation routine is plain Python and solves one recourse LP per
scenario at every candidate first stage. This keeps it solver-agnostic and easy
to read, but it bounds practical scale: the bundled configs use instances on the
order of a dozen facilities, which solve in seconds and are verified against the
monolith.

`configs/experiments/headline_150_50.yaml` is a configuration template for a
large chance-constrained instance on the Gurobi backend. It is not a quick
reproducible result — an instance of that size takes substantial time with this
cut routine, and the config sets a time limit so the run returns the best
solution found within it.

```bash
uv sync --locked --extra dev --extra gurobi
uv run sflp run --config configs/experiments/headline_150_50.yaml
```

## Data and provenance

Raw third-party data is not committed; the download scripts fetch it on demand
and verify SHA-256 checksums for the OR-Library files. GeoNames `cities5000` is a
living dataset (updated daily), so its checksum is not pinned; the population
snapshot used is recorded with each run. Attribution: GeoNames (CC BY 4.0),
OR-Library (Beasley).

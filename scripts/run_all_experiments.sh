#!/usr/bin/env bash
# Reproduce every figure and table. Open-solver experiments run by default; the
# Gurobi headline runs only if the [gurobi] extra and a license are present.
set -euo pipefail

cd "$(dirname "$0")/.."

run() {
  echo "=== $1 ==="
  uv run sflp run --config "$1" --name "$2"
}

# Moderate open-solver default (HiGHS classic Benders).
run configs/default.yaml default

# Open single-tree path on SCIP.
run configs/experiments/scip_single_tree.yaml scip_single_tree

# Demand-volatility sweep: VSS/EVPI vs sigma.
for s in 0.10 0.20 0.30; do
  run "configs/experiments/sigma_${s}.yaml" "sigma_${s}"
done

# Headline (150 nodes / 50 scenarios) on Gurobi, if available.
if uv run python -c "import gurobipy" >/dev/null 2>&1; then
  run configs/experiments/headline_150_50.yaml headline_150_50
else
  echo "=== headline skipped: gurobipy not installed (uv sync --extra gurobi) ==="
fi

echo "Done. Results in results/logs and results/figures."

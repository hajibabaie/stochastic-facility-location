# Installation

This project uses [**uv**](https://docs.astral.sh/uv/) for environment and
dependency management. The lockfile (`uv.lock`) is committed, so installs are
reproducible.

## Open install (no commercial license)

```bash
uv sync --locked --extra dev
```

This installs the core dependencies and the open solvers **HiGHS** (`highspy`)
and **SCIP** (`pyscipopt`), plus the dev tools (ruff, mypy, pytest, pre-commit).

## With the Gurobi fast path

```bash
uv sync --locked --extra dev --extra gurobi
```

`gurobipy` needs a valid Gurobi license. It is only required for the
`gurobi`/`branch_and_cut`-on-Gurobi backends (the largest headline instance).

## Python versions

The project targets Python 3.11–3.13; `.python-version` pins 3.12 for local work.
`uv` installs the right interpreter automatically.

## Verifying the install

```bash
uv run pytest -m "not slow and not gurobi and not network"
uv run ruff check . && uv run mypy src
```

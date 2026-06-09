"""End-to-end experiment runner: data -> scenarios -> solve -> measures -> record.

Ties the pieces into one reproducible run driven by a :class:`~sflp.config.Config`.
Every run records the seed, the resolved package/solver versions, and the git
commit, so a result can be traced back to exactly what produced it.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from sflp.benders import solve_benders
from sflp.benders.backend import BendersResult
from sflp.config import Config
from sflp.data.download import download_source
from sflp.data.generate import build_geonames_instance, generate_scenarios
from sflp.data.instance import Instance, ScenarioSet
from sflp.data.parse import read_geonames, read_or_library_cap
from sflp.saa import StochasticMeasures, compute_stochastic_measures
from sflp.solve import validate_solver_config

GEONAMES_MIN_POPULATION = 5000


@dataclass(frozen=True)
class RunResult:
    """Everything a single experiment produced, plus how to reproduce it."""

    instance_name: str
    n_facilities: int
    n_customers: int
    n_scenarios: int
    objective: float
    open_facilities: list[int]
    lower_bound: float
    gap: float
    iterations: int
    n_cuts: int
    runtime_seconds: float
    measures: StochasticMeasures | None
    benders: BendersResult
    metadata: dict[str, object]


def build_instance(cfg: Config) -> Instance:
    """Construct the deterministic instance from the configured data source."""
    if cfg.data.source == "geonames":
        path = download_source("geonames_cities5000")
        cities = read_geonames(path, GEONAMES_MIN_POPULATION, country=cfg.data.country)
        if cities.population.size < cfg.data.n_facilities:
            raise ValueError(
                f"GeoNames country {cfg.data.country!r} has only {cities.population.size} "
                f"cities >= {GEONAMES_MIN_POPULATION} pop; need {cfg.data.n_facilities}."
            )
        cities = cities.top_by_population(cfg.data.n_facilities)
        return build_geonames_instance(
            cities.names, cities.coordinates, cities.population, cfg.data
        )
    if cfg.data.source == "or_library":
        path = download_source(f"or_{cfg.data.instance}")
        return read_or_library_cap(path)
    raise ValueError(f"Unknown data.source {cfg.data.source!r}.")


def build_scenarios(instance: Instance, cfg: Config) -> ScenarioSet:
    """Generate the seeded scenario set for an instance."""
    rng = np.random.default_rng(cfg.seed)
    return generate_scenarios(instance, cfg.scenarios, rng)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in ("stochastic-facility-location", "numpy", "scipy", "pyomo", "highspy", "pyscipopt"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "n/a"
    return out


def run_experiment(cfg: Config, *, compute_measures: bool = True) -> RunResult:
    """Run one experiment end to end and return its result and provenance."""
    validate_solver_config(cfg.solver)
    instance = build_instance(cfg)
    scenarios = build_scenarios(instance, cfg)

    start = time.perf_counter()
    benders = solve_benders(instance, scenarios, cfg.model, cfg.solver)
    measures = (
        compute_stochastic_measures(instance, scenarios, cfg.solver) if compute_measures else None
    )
    runtime = time.perf_counter() - start

    metadata: dict[str, object] = {
        "seed": cfg.seed,
        "backend": cfg.solver.backend,
        "mip_solver": cfg.solver.mip_solver,
        "pareto_cuts": cfg.solver.pareto_cuts,
        "chance_constraint": cfg.model.chance_constraint,
        "gamma": cfg.model.gamma,
        "git_commit": _git_commit(),
        "versions": _versions(),
    }
    return RunResult(
        instance_name=instance.name,
        n_facilities=instance.n_facilities,
        n_customers=instance.n_customers,
        n_scenarios=scenarios.n_scenarios,
        objective=benders.objective,
        open_facilities=benders.open_facilities,
        lower_bound=benders.lower_bound,
        gap=benders.gap,
        iterations=benders.iterations,
        n_cuts=benders.n_cuts,
        runtime_seconds=runtime,
        measures=measures,
        benders=benders,
        metadata=metadata,
    )


def save_result(result: RunResult, output_dir: str | Path, name: str) -> Path:
    """Write a JSON summary of a run (without the heavy arrays) to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "instance_name": result.instance_name,
        "n_facilities": result.n_facilities,
        "n_customers": result.n_customers,
        "n_scenarios": result.n_scenarios,
        "objective": result.objective,
        "open_facilities": result.open_facilities,
        "lower_bound": result.lower_bound,
        "gap": result.gap,
        "iterations": result.iterations,
        "n_cuts": result.n_cuts,
        "runtime_seconds": result.runtime_seconds,
        "measures": asdict(result.measures) if result.measures else None,
        "metadata": result.metadata,
    }
    path = output_dir / f"{name}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path

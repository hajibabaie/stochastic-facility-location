"""Typed, YAML-backed configuration for an SFLP run.

A single :class:`Config` object drives a whole experiment: which data to use,
how to generate scenarios, the model's risk levels, and the solver choice. One
``seed`` field makes every run reproducible.

The config is intentionally flat-ish nested dataclasses (no magic numbers in
code). Solver-combination validation lives in :mod:`sflp.solve`; here we only
check values that must hold for any backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class DataConfig:
    """Where demand/cost data comes from and how model-set numbers are derived.

    Coordinates and population are real (from the source). Capacity and fixed
    cost are *model-supplied* by the documented rules below; only these are
    invented, never the geography.
    """

    source: str = "geonames"
    """One of ``geonames``, ``or_library``, ``sslp``."""

    country: str = "DE"
    """ISO country code for the GeoNames headline instance."""

    n_facilities: int = 50
    """Top-N cities by population; each is a customer and a candidate facility."""

    instance: str = "cap71"
    """OR-Library instance name (used when ``source == 'or_library'``)."""

    capacity_rule_k: float = 3.0
    """Capacity ``s_i = k * mean(demand)`` (GeoNames only)."""

    fixed_cost_base: float = 1.0
    """Dimensionless fixed-cost knob; ``f_i = base * (mean_demand * typical_distance)
    * (pop_i/mean_pop)**scale`` so opening a facility trades off against transport."""

    fixed_cost_pop_scale: float = 0.5
    """Population sensitivity of fixed cost, ``f_i ~ (pop/mean_pop)**scale``."""

    unmet_penalty_scale: float = 3.0
    """Unmet-demand penalty ``q_j = scale * max_i c_ij`` (must exceed serve cost)."""


@dataclass(frozen=True)
class ScenarioConfig:
    """Stochastic demand: sample many, optionally reduce to a working set."""

    n_scenarios: int = 20
    """Number of working scenarios S (after reduction)."""

    n_sample: int = 1000
    """Raw Monte-Carlo sample size before reduction."""

    sigma: float = 0.2
    """Lognormal multiplicative volatility; sweepable."""

    correlation_length: float | None = None
    """Distance decay L (km) for spatially correlated demand; ``None`` = i.i.d."""

    reduction: str = "kmeans"
    """Scenario reduction: ``kmeans``, ``fast_forward``, or ``none``."""


@dataclass(frozen=True)
class ModelConfig:
    """Risk levels and optional structural constraints.

    ``gamma`` (the SAA risk level used in C4) is NOT ``epsilon`` (the true
    shortfall target). Set ``gamma < epsilon`` for a conservative inner
    approximation; see ``docs/algorithm``.
    """

    chance_constraint: bool = True
    """Enable the service-level chance constraint (z_s, big-M, C3-C4)."""

    epsilon: float = 0.10
    """True target: each scenario fully served with probability >= 1 - epsilon."""

    gamma: float = 0.05
    """SAA risk level: at most floor(gamma * N) scenarios may violate service."""

    budget: float | None = None
    """Optional opening-cost budget B (constraint C5); ``None`` disables it."""

    cardinality: int | None = None
    """Optional max number of open facilities p (constraint C6)."""


@dataclass(frozen=True)
class SolverConfig:
    """Backend choice and decomposition controls."""

    backend: str = "classic"
    """One of ``classic`` (HiGHS loop), ``branch_and_cut`` (SCIP), ``gurobi``."""

    mip_solver: str = "highs"
    """Solver for the master MIP / monolith."""

    lp_solver: str = "highs"
    """Solver for the Benders subproblem LPs."""

    pareto_cuts: bool = True
    """Use Magnanti-Wong / Papadakos Pareto-optimal cuts."""

    max_iterations: int = 200
    """Cut-loop iteration cap (classic backend)."""

    gap_tolerance: float = 1.0e-6
    """Relative optimality gap (UB - LB) / |UB| stopping tolerance."""

    time_limit: float | None = None
    """Wall-clock limit in seconds; ``None`` = no limit."""


@dataclass(frozen=True)
class SaaConfig:
    """Out-of-sample statistical validation of the SAA solution."""

    replications: int = 20
    """Number of independent SAA replications M for the optimality-gap CI."""

    reference_sample: int = 5000
    """Large out-of-sample reference for the upper bound and VSS/EVPI."""

    confidence: float = 0.95
    """Confidence level for the gap interval."""


@dataclass(frozen=True)
class Config:
    """Top-level run configuration."""

    seed: int = 20231015
    data: DataConfig = field(default_factory=DataConfig)
    scenarios: ScenarioConfig = field(default_factory=ScenarioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    saa: SaaConfig = field(default_factory=SaaConfig)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.data.n_facilities < 2:
            raise ValueError("data.n_facilities must be at least 2.")
        if self.scenarios.n_scenarios < 1:
            raise ValueError("scenarios.n_scenarios must be at least 1.")
        if self.scenarios.n_sample < self.scenarios.n_scenarios:
            raise ValueError("scenarios.n_sample must be >= n_scenarios.")
        if not 0.0 <= self.model.gamma <= 1.0:
            raise ValueError("model.gamma must be in [0, 1].")
        if not 0.0 < self.model.epsilon <= 1.0:
            raise ValueError("model.epsilon must be in (0, 1].")
        if self.model.gamma > self.model.epsilon:
            raise ValueError(
                "model.gamma must not exceed model.epsilon; a conservative SAA "
                "inner approximation needs gamma <= epsilon (gamma=0 is safest)."
            )
        if self.solver.gap_tolerance <= 0:
            raise ValueError("solver.gap_tolerance must be positive.")


def _build(cls: type[T], data: Any) -> T:
    """Recursively build a (possibly nested) dataclass from plain dicts.

    Unknown keys raise, so typos in YAML fail fast instead of being ignored.
    """
    if not is_dataclass(cls):
        return data
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping for {cls.__name__}, got {type(data).__name__}.")

    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}.")

    # Resolve string annotations (PEP 563) to real types so nested dataclasses
    # are detected and built recursively.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        ftype = hints.get(name)
        if is_dataclass(ftype) and isinstance(ftype, type):
            kwargs[name] = _build(ftype, data[name])
        else:
            kwargs[name] = data[name]
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load and validate a :class:`Config` from a YAML file."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _build(Config, raw)

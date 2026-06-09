"""Parse real third-party data files into instances and reference values.

Supported formats:

- **OR-Library ``cap*``** capacitated facility location instances. The classic
  Beasley layout: ``m n`` (facilities, customers); then ``s_i f_i`` per facility;
  then per customer a demand ``d_j`` followed by ``m`` costs ``c_ij`` giving the
  cost of serving **all** of customer ``j`` from facility ``i``. We store the
  per-unit cost ``c_ij / d_j`` so the model's flow term is ``unit_cost * flow``.
- **OR-Library ``capopt``** the table of published optimal objective values,
  one ``name value`` pair per line.
- **GeoNames ``cities*``** tab-separated dumps; we keep name, latitude,
  longitude, and population.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sflp.data.instance import FloatArray, Instance


@dataclass(frozen=True)
class GeoCities:
    """Parsed GeoNames rows: aligned name / lat / lon / population arrays."""

    names: list[str]
    latitude: FloatArray
    longitude: FloatArray
    population: FloatArray

    @property
    def coordinates(self) -> FloatArray:
        return np.column_stack([self.latitude, self.longitude])

    def top_by_population(self, n: int) -> GeoCities:
        """Keep the ``n`` most populous cities (ties broken by original order)."""
        order = np.argsort(-self.population, kind="stable")[:n]
        return GeoCities(
            names=[self.names[i] for i in order],
            latitude=self.latitude[order],
            longitude=self.longitude[order],
            population=self.population[order],
        )


def parse_or_library_cap(text: str, name: str = "cap") -> Instance:
    """Parse an OR-Library ``cap*`` instance from its raw text."""
    tokens = text.split()
    pos = 0

    def take() -> str:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        return tok

    m = int(take())
    n = int(take())

    capacity = np.empty(m, dtype=np.float64)
    fixed_cost = np.empty(m, dtype=np.float64)
    for i in range(m):
        capacity[i] = float(take())
        fixed_cost[i] = float(take())

    demand = np.empty(n, dtype=np.float64)
    total_cost = np.empty((m, n), dtype=np.float64)  # cost to fully serve j from i
    for j in range(n):
        demand[j] = float(take())
        for i in range(m):
            total_cost[i, j] = float(take())

    # Convert "serve all of j" cost to per-unit cost so flow * unit_cost is right.
    unit_cost = total_cost / demand[None, :]
    # A penalty above the largest unit cost keeps unmet demand strictly dominated;
    # the deterministic CFLP never uses it (demand is always met), but the
    # stochastic model does.
    unmet_penalty = np.full(n, 10.0 * float(unit_cost.max()), dtype=np.float64)

    return Instance(
        facility_ids=[f"f{i}" for i in range(m)],
        customer_ids=[f"c{j}" for j in range(n)],
        fixed_cost=fixed_cost,
        capacity=capacity,
        demand=demand,
        unit_cost=unit_cost,
        unmet_penalty=unmet_penalty,
        name=name,
    )


def parse_or_library_optima(text: str) -> dict[str, float]:
    """Parse the OR-Library ``capopt`` published-optima table."""
    optima: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0].lower().removesuffix(".txt")
        try:
            optima[key] = float(parts[1])
        except ValueError:
            continue
    return optima


def parse_geonames(text: str, min_population: int = 1, country: str | None = None) -> GeoCities:
    """Parse a GeoNames dump (tab-separated, 19 columns per the schema).

    Columns used: 1 name, 4 latitude, 5 longitude, 8 country code, 14 population.
    Rows with non-positive population are dropped (population is the demand). If
    ``country`` (an ISO code) is given, only that country's rows are kept.
    """
    names: list[str] = []
    lat: list[float] = []
    lon: list[float] = []
    pop: list[float] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 15:
            continue
        if country is not None and cols[8] != country:
            continue
        try:
            population = float(cols[14])
            latitude = float(cols[4])
            longitude = float(cols[5])
        except ValueError:
            continue
        if population < min_population:
            continue
        names.append(cols[1])
        lat.append(latitude)
        lon.append(longitude)
        pop.append(population)
    return GeoCities(
        names=names,
        latitude=np.array(lat, dtype=np.float64),
        longitude=np.array(lon, dtype=np.float64),
        population=np.array(pop, dtype=np.float64),
    )


def read_or_library_cap(path: str | Path) -> Instance:
    """Read and parse an OR-Library ``cap*`` file from disk."""
    path = Path(path)
    return parse_or_library_cap(path.read_text(encoding="utf-8"), name=path.stem)


def read_geonames(
    path: str | Path, min_population: int = 1, country: str | None = None
) -> GeoCities:
    """Read and parse a GeoNames dump file from disk."""
    return parse_geonames(Path(path).read_text(encoding="utf-8"), min_population, country)

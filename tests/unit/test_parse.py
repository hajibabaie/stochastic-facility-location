"""Tests for parsing OR-Library and GeoNames data."""

import numpy as np

from sflp.data.parse import (
    parse_geonames,
    parse_or_library_cap,
    parse_or_library_optima,
)

# 2 facilities, 3 customers. Per facility: capacity fixed_cost.
# Per customer: demand, then one cost per facility (cost to fully serve j from i).
TINY_CAP = """
2 3
100 10
120 20
30
5 8
40
6 9
20
7 4
"""


def test_parse_or_library_cap_shapes_and_values() -> None:
    inst = parse_or_library_cap(TINY_CAP, name="tiny")
    assert inst.n_facilities == 2
    assert inst.n_customers == 3
    np.testing.assert_allclose(inst.capacity, [100, 120])
    np.testing.assert_allclose(inst.fixed_cost, [10, 20])
    np.testing.assert_allclose(inst.demand, [30, 40, 20])
    # unit_cost[i, j] = total_cost[i, j] / demand[j]
    np.testing.assert_allclose(inst.unit_cost[0], [5 / 30, 6 / 40, 7 / 20])
    np.testing.assert_allclose(inst.unit_cost[1], [8 / 30, 9 / 40, 4 / 20])
    # penalty must dominate the largest unit cost
    assert inst.unmet_penalty.min() > inst.unit_cost.max()


def test_parse_or_library_optima() -> None:
    text = "cap71 932615.75\ncap72.txt   977799.40\n# header line\n"
    optima = parse_or_library_optima(text)
    assert optima["cap71"] == 932615.75
    assert optima["cap72"] == 977799.40
    assert "# header line" not in optima


def _geonames_row(name: str, lat: float, lon: float, pop: int) -> str:
    cols = [""] * 19
    cols[0] = "1"
    cols[1] = name
    cols[4] = str(lat)
    cols[5] = str(lon)
    cols[14] = str(pop)
    return "\t".join(cols)


def test_parse_geonames_filters_and_keeps_fields() -> None:
    text = "\n".join(
        [
            _geonames_row("Berlin", 52.524, 13.411, 3426354),
            _geonames_row("Munich", 48.137, 11.575, 1260391),
            _geonames_row("NoPop", 50.0, 8.0, 0),  # dropped: population 0
            "short\trow",  # dropped: too few columns
        ]
    )
    cities = parse_geonames(text, min_population=1)
    assert cities.names == ["Berlin", "Munich"]
    np.testing.assert_allclose(cities.population, [3426354, 1260391])
    assert cities.coordinates.shape == (2, 2)


def test_top_by_population() -> None:
    text = "\n".join(
        [
            _geonames_row("Small", 1.0, 1.0, 10),
            _geonames_row("Big", 2.0, 2.0, 1000),
            _geonames_row("Mid", 3.0, 3.0, 100),
        ]
    )
    cities = parse_geonames(text).top_by_population(2)
    assert cities.names == ["Big", "Mid"]

"""Validate the deterministic CFLP solver against OR-Library published optima.

These tests download real instances (``cap71``, ``cap101``, ``cap131``) and the
published-optima table, solve each with the open HiGHS backend, and assert the
objective matches Beasley's published value. They skip cleanly when the data is
unreachable (offline / server down) so CI stays green without the network.
"""

import urllib.error

import pytest

from sflp.config import SolverConfig
from sflp.data.download import download_source
from sflp.data.parse import parse_or_library_optima, read_or_library_cap
from sflp.solve import solve_deterministic_cflp

CASES = ["or_cap71", "or_cap101", "or_cap131"]


@pytest.fixture(scope="module")
def optima() -> dict[str, float]:
    try:
        path = download_source("or_capopt")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        pytest.skip(f"OR-Library unreachable: {exc}")
    return parse_or_library_optima(path.read_text(encoding="utf-8"))


@pytest.mark.network
@pytest.mark.parametrize("key", CASES)
def test_cflp_matches_published_optimum(key: str, optima: dict[str, float]) -> None:
    try:
        path = download_source(key)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        pytest.skip(f"OR-Library unreachable: {exc}")

    instance = read_or_library_cap(path)
    published = optima[instance.name]
    solution = solve_deterministic_cflp(instance, SolverConfig(mip_solver="highs"))
    # Published values carry rounding; match to a tight relative tolerance.
    assert solution.objective == pytest.approx(published, rel=1e-5)

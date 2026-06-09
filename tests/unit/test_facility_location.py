"""Unit tests for the deterministic CFLP model and solver wiring."""

import numpy as np
import pytest

from sflp.config import SolverConfig
from sflp.data.instance import Instance
from sflp.solve import solve_deterministic_cflp, validate_solver_config


def _toy_instance() -> Instance:
    # Facility 0: expensive to open (100) but cheap transport (1/unit).
    # Facility 1: cheap to open (5) but expensive transport (10/unit).
    # Each facility alone can cover all demand (cap 25 >= 20).
    # Optimum: open only facility 0 -> 100 + (10+10)*1 = 120.
    return Instance(
        facility_ids=["f0", "f1"],
        customer_ids=["c0", "c1"],
        fixed_cost=np.array([100.0, 5.0]),
        capacity=np.array([25.0, 25.0]),
        demand=np.array([10.0, 10.0]),
        unit_cost=np.array([[1.0, 1.0], [10.0, 10.0]]),
        unmet_penalty=np.array([1000.0, 1000.0]),
        name="toy",
    )


def test_solve_toy_cflp_known_optimum() -> None:
    sol = solve_deterministic_cflp(_toy_instance(), SolverConfig(mip_solver="highs"))
    assert sol.objective == pytest.approx(120.0, abs=1e-6)
    assert sol.open_facilities == [0]
    assert sol.fixed_cost == pytest.approx(100.0)
    assert sol.transport_cost == pytest.approx(20.0)
    # all demand is shipped from the open facility
    np.testing.assert_allclose(sol.flow.sum(axis=0), [10.0, 10.0])
    assert sol.flow[1].sum() == pytest.approx(0.0)


def test_validate_solver_config_rejects_branch_and_cut_on_highs() -> None:
    with pytest.raises(ValueError, match="HiGHS does not support"):
        validate_solver_config(SolverConfig(backend="branch_and_cut", mip_solver="highs"))


def test_validate_solver_config_accepts_classic_highs() -> None:
    validate_solver_config(SolverConfig(backend="classic", mip_solver="highs"))


def test_validate_solver_config_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown solver"):
        validate_solver_config(SolverConfig(backend="nonsense"))

"""Tests for the service-level chance constraint behavior."""

import numpy as np
import pytest

from sflp.config import ModelConfig, SolverConfig
from sflp.data.instance import Instance, ScenarioSet
from sflp.solve import solve_saa_monolith

SOLVER = SolverConfig(mip_solver="highs")


def _two_facility_instance() -> Instance:
    # Facility 0: cheap to open (10), small capacity (50).
    # Facility 1: expensive to open (1000), large capacity (500).
    return Instance(
        facility_ids=["small", "big"],
        customer_ids=["c0"],
        fixed_cost=np.array([10.0, 1000.0]),
        capacity=np.array([50.0, 500.0]),
        demand=np.array([50.0]),
        unit_cost=np.array([[1.0], [1.0]]),
        unmet_penalty=np.array([5.0]),  # low penalty: violating is attractive
        name="chance",
    )


def _two_scenarios() -> ScenarioSet:
    # Scenario 1 (demand 400) can only be fully served by the big facility.
    return ScenarioSet(demand=np.array([[50.0], [400.0]]), probability=np.array([0.5, 0.5]))


def test_gamma_zero_forces_full_service() -> None:
    inst, scen = _two_facility_instance(), _two_scenarios()
    cfg = ModelConfig(chance_constraint=True, gamma=0.0)
    sol = solve_saa_monolith(inst, scen, cfg, SOLVER)
    assert sol.expected_unmet == pytest.approx(0.0, abs=1e-6)
    assert sol.violation_probability == pytest.approx(0.0, abs=1e-9)
    assert 1 in sol.open_facilities  # the big facility must be open to serve scenario 1


def test_larger_gamma_relaxes_and_lowers_cost() -> None:
    inst, scen = _two_facility_instance(), _two_scenarios()
    strict = solve_saa_monolith(inst, scen, ModelConfig(chance_constraint=True, gamma=0.0), SOLVER)
    relaxed = solve_saa_monolith(inst, scen, ModelConfig(chance_constraint=True, gamma=0.5), SOLVER)
    assert relaxed.objective < strict.objective
    assert relaxed.expected_unmet > 0.0  # the costly scenario is now left unmet
    assert relaxed.violation_probability <= 0.5 + 1e-9


def test_gamma_zero_infeasible_when_scenario_exceeds_capacity() -> None:
    inst = Instance(
        facility_ids=["f0", "f1"],
        customer_ids=["c0", "c1"],
        fixed_cost=np.array([10.0, 10.0]),
        capacity=np.array([15.0, 15.0]),  # total 30
        demand=np.array([10.0, 10.0]),
        unit_cost=np.array([[1.0, 1.0], [1.0, 1.0]]),
        unmet_penalty=np.array([50.0, 50.0]),
        name="infeasible-at-gamma0",
    )
    scen = ScenarioSet(demand=np.array([[30.0, 30.0]]), probability=np.array([1.0]))  # 60 > 30
    with pytest.raises(RuntimeError, match="optimality"):
        solve_saa_monolith(inst, scen, ModelConfig(chance_constraint=True, gamma=0.0), SOLVER)

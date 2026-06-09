"""Tests for the SAA deterministic-equivalent monolith."""

import numpy as np
import pytest

from sflp.config import ModelConfig, SolverConfig
from sflp.data.instance import Instance, ScenarioSet
from sflp.solve import solve_deterministic_cflp, solve_saa_monolith

SOLVER = SolverConfig(mip_solver="highs")
NO_CHANCE = ModelConfig(chance_constraint=False)


def _toy_instance() -> Instance:
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


def test_single_scenario_matches_deterministic_cflp() -> None:
    """With one scenario at nominal demand, the SAA monolith is the CFLP oracle."""
    inst = _toy_instance()
    scen = ScenarioSet(demand=inst.demand[None, :].copy(), probability=np.array([1.0]))
    saa = solve_saa_monolith(inst, scen, NO_CHANCE, SOLVER)
    det = solve_deterministic_cflp(inst, SOLVER)
    assert saa.objective == pytest.approx(det.objective, abs=1e-6)
    assert saa.open_facilities == det.open_facilities == [0]
    assert saa.expected_unmet == pytest.approx(0.0, abs=1e-9)


def test_recourse_cost_split_is_consistent() -> None:
    inst = _toy_instance()
    demand = np.array([[10.0, 10.0], [12.0, 8.0]])
    scen = ScenarioSet(demand=demand, probability=np.array([0.5, 0.5]))
    saa = solve_saa_monolith(inst, scen, NO_CHANCE, SOLVER)
    assert saa.first_stage_cost + saa.expected_recourse_cost == pytest.approx(saa.objective)
    assert saa.first_stage_cost == pytest.approx(100.0)  # facility 0 open


def test_capacity_shortage_forces_unmet() -> None:
    """A scenario whose demand exceeds total capacity must leave demand unmet."""
    inst = Instance(
        facility_ids=["f0", "f1"],
        customer_ids=["c0", "c1"],
        fixed_cost=np.array([10.0, 10.0]),
        capacity=np.array([15.0, 15.0]),  # total 30
        demand=np.array([10.0, 10.0]),
        unit_cost=np.array([[1.0, 2.0], [2.0, 1.0]]),
        unmet_penalty=np.array([50.0, 50.0]),
        name="tight",
    )
    scen = ScenarioSet(demand=np.array([[30.0, 30.0]]), probability=np.array([1.0]))  # 60 > 30
    saa = solve_saa_monolith(inst, scen, NO_CHANCE, SOLVER)
    assert saa.expected_unmet == pytest.approx(30.0, abs=1e-6)  # 60 demand - 30 capacity


def test_cardinality_constraint_limits_open_facilities() -> None:
    inst = _toy_instance()
    scen = ScenarioSet(demand=inst.demand[None, :].copy(), probability=np.array([1.0]))
    cfg = ModelConfig(chance_constraint=False, cardinality=1)
    saa = solve_saa_monolith(inst, scen, cfg, SOLVER)
    assert len(saa.open_facilities) <= 1

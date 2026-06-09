"""Tests for SAA statistics: VSS, EVPI, and the optimality-gap estimate."""

import numpy as np
import pytest

from sflp.config import DataConfig, ScenarioConfig, SolverConfig
from sflp.data.generate import build_geonames_instance, generate_scenarios
from sflp.data.instance import Instance, ScenarioSet
from sflp.saa import (
    compute_stochastic_measures,
    estimate_optimality_gap,
    evaluate_first_stage,
)
from sflp.solve import solve_saa_monolith

SOLVER = SolverConfig(mip_solver="highs")


def _geo_instance() -> Instance:
    rng = np.random.default_rng(2024)
    names = [f"c{i}" for i in range(6)]
    coords = rng.uniform(0, 5, size=(6, 2))
    population = rng.uniform(50, 200, size=6)
    return build_geonames_instance(names, coords, population, DataConfig(capacity_rule_k=2.0))


def _scenarios(instance: Instance, seed: int = 1) -> ScenarioSet:
    cfg = ScenarioConfig(n_scenarios=8, n_sample=300, sigma=0.35, reduction="kmeans")
    return generate_scenarios(instance, cfg, np.random.default_rng(seed))


def test_stochastic_measure_inequalities() -> None:
    inst = _geo_instance()
    scen = _scenarios(inst)
    m = compute_stochastic_measures(inst, scen, SOLVER)
    # WS <= RP <= EEV always holds for a minimization stochastic program.
    assert m.ws <= m.rp + 1e-6
    assert m.rp <= m.eev + 1e-6
    assert m.evpi >= -1e-6
    assert m.vss >= -1e-6
    assert m.evpi == pytest.approx(m.rp - m.ws)
    assert m.vss == pytest.approx(m.eev - m.rp)


def test_evaluate_first_stage_matches_monolith_objective() -> None:
    inst = _geo_instance()
    scen = _scenarios(inst)
    from sflp.config import ModelConfig

    rp = solve_saa_monolith(inst, scen, ModelConfig(chance_constraint=False), SOLVER)
    assert evaluate_first_stage(inst, scen, rp.y) == pytest.approx(rp.objective, rel=1e-6)


def test_vss_and_evpi_positive_with_capacity_hedging() -> None:
    # One customer, two equal facilities; each facility holds 60 units. The
    # mean demand (55) fits one facility, so the expected-value solution opens
    # just one. But the high scenario (70) overflows it, so the stochastic
    # solution opens both to hedge -> VSS > 0; perfect foresight helps -> EVPI > 0.
    inst = Instance(
        facility_ids=["f0", "f1"],
        customer_ids=["c0"],
        fixed_cost=np.array([10.0, 10.0]),
        capacity=np.array([60.0, 60.0]),
        demand=np.array([55.0]),
        unit_cost=np.array([[1.0], [1.0]]),
        unmet_penalty=np.array([10.0]),
        name="hedge",
    )
    scen = ScenarioSet(demand=np.array([[40.0], [70.0]]), probability=np.array([0.5, 0.5]))
    m = compute_stochastic_measures(inst, scen, SOLVER)
    assert m.vss > 0.0  # the stochastic solution beats the mean-value one
    assert m.evpi > 0.0  # perfect foresight strictly helps
    assert m.ws <= m.rp <= m.eev


@pytest.mark.slow
def test_optimality_gap_estimate() -> None:
    inst = _geo_instance()
    cfg = ScenarioConfig(n_scenarios=10, n_sample=300, sigma=0.3, reduction="kmeans")
    gap = estimate_optimality_gap(inst, cfg, SOLVER, seed=7, replications=6, reference_size=400)
    assert gap.replications == 6
    assert gap.reference_size == 400
    assert gap.lower_bound > 0
    assert gap.upper_bound > 0
    assert gap.gap_ci_high >= gap.gap
    # the SAA lower bound should not exceed the candidate's true cost by much
    assert gap.gap >= -3.0 * (gap.lower_stderr + gap.upper_stderr)

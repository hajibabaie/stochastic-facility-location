"""Tests for instance building and seeded scenario generation."""

import numpy as np
import pytest

from sflp.config import DataConfig, ScenarioConfig
from sflp.data.generate import (
    build_geonames_instance,
    generate_scenarios,
    haversine_matrix,
    reduce_scenarios,
    sample_demand,
)


def test_haversine_self_zero_and_symmetric() -> None:
    coords = np.array([[51.5074, -0.1278], [48.8566, 2.3522]])  # London, Paris
    d = haversine_matrix(coords)
    assert d[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert d[0, 1] == pytest.approx(d[1, 0])
    # London-Paris great-circle distance is ~344 km.
    assert d[0, 1] == pytest.approx(344.0, abs=10.0)


def _small_instance() -> "tuple[list[str], np.ndarray, np.ndarray]":
    names = ["A", "B", "C"]
    coords = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    population = np.array([100.0, 200.0, 50.0])
    return names, coords, population


def test_build_geonames_instance_rules() -> None:
    from sflp.data.generate import GEONAMES_DEMAND_MEAN

    names, coords, pop = _small_instance()
    cfg = DataConfig(capacity_rule_k=3.0, fixed_cost_base=1.0, fixed_cost_pop_scale=0.5)
    inst = build_geonames_instance(names, coords, pop, cfg)

    # demand is population rescaled to a fixed mean (relative sizes preserved)
    assert inst.demand.mean() == pytest.approx(GEONAMES_DEMAND_MEAN)
    np.testing.assert_allclose(inst.demand / inst.demand.sum(), pop / pop.sum())
    assert np.allclose(inst.capacity, 3.0 * inst.demand.mean())  # s_i = k * mean demand
    # fixed cost rises with population
    assert inst.fixed_cost[1] > inst.fixed_cost[0] > inst.fixed_cost[2]
    # penalty dominates the worst serve cost per customer
    assert np.all(inst.unmet_penalty >= inst.unit_cost.max(axis=0))
    assert np.allclose(np.diag(inst.unit_cost), 0.0)  # zero distance to self


def test_sample_demand_is_deterministic() -> None:
    base = np.array([100.0, 200.0, 50.0])
    a = sample_demand(base, 0.2, 100, np.random.default_rng(7))
    b = sample_demand(base, 0.2, 100, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (100, 3)
    assert np.all(a > 0)  # lognormal => strictly positive


def test_sample_demand_is_mean_preserving() -> None:
    base = np.array([100.0, 200.0, 50.0])
    samples = sample_demand(base, 0.3, 50_000, np.random.default_rng(0))
    np.testing.assert_allclose(samples.mean(axis=0), base, rtol=0.03)


def test_correlated_sampling_mean_preserving() -> None:
    base = np.array([100.0, 120.0, 80.0])
    coords = np.array([[0.0, 0.0], [0.0, 0.5], [10.0, 10.0]])
    samples = sample_demand(
        base, 0.3, 50_000, np.random.default_rng(1), coords=coords, correlation_length=100.0
    )
    np.testing.assert_allclose(samples.mean(axis=0), base, rtol=0.03)
    # nearby cities (0, 1) should correlate more than distant ones (0, 2)
    corr = np.corrcoef(samples.T)
    assert corr[0, 1] > corr[0, 2]


def test_reduce_scenarios_probabilities_sum_to_one() -> None:
    rng = np.random.default_rng(3)
    samples = sample_demand(np.array([100.0, 50.0]), 0.25, 500, rng)
    scen = reduce_scenarios(samples, 10, "kmeans", rng)
    assert scen.n_scenarios <= 10
    assert scen.demand.shape[1] == 2
    assert scen.probability.sum() == pytest.approx(1.0)
    assert np.all(scen.probability > 0)


def test_generate_scenarios_end_to_end_deterministic() -> None:
    names, coords, pop = _small_instance()
    inst = build_geonames_instance(names, coords, pop, DataConfig())
    scfg = ScenarioConfig(n_scenarios=8, n_sample=400, sigma=0.2, reduction="kmeans")
    s1 = generate_scenarios(inst, scfg, np.random.default_rng(42))
    s2 = generate_scenarios(inst, scfg, np.random.default_rng(42))
    np.testing.assert_array_equal(s1.demand, s2.demand)
    np.testing.assert_array_equal(s1.probability, s2.probability)
    assert s1.demand.shape[1] == inst.n_customers


def test_reduction_none_uses_exact_count() -> None:
    names, coords, pop = _small_instance()
    inst = build_geonames_instance(names, coords, pop, DataConfig())
    scfg = ScenarioConfig(n_scenarios=5, n_sample=100, sigma=0.15, reduction="none")
    scen = generate_scenarios(inst, scfg, np.random.default_rng(11))
    assert scen.n_scenarios == 5
    np.testing.assert_allclose(scen.probability, 0.2)

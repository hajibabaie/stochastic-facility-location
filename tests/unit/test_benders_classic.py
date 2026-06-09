"""Classic Benders must reproduce the SAA monolith optimum (the oracle)."""

import numpy as np
import pytest

from sflp.benders.classic import solve_classic_benders
from sflp.config import DataConfig, ModelConfig, ScenarioConfig, SolverConfig
from sflp.data.generate import build_geonames_instance, generate_scenarios
from sflp.data.instance import Instance, ScenarioSet
from sflp.solve import solve_saa_monolith

SOLVER = SolverConfig(mip_solver="highs")


def _toy_instance() -> Instance:
    return Instance(
        facility_ids=["f0", "f1", "f2"],
        customer_ids=["c0", "c1", "c2"],
        fixed_cost=np.array([100.0, 120.0, 80.0]),
        capacity=np.array([60.0, 60.0, 60.0]),
        demand=np.array([20.0, 20.0, 20.0]),
        unit_cost=np.array([[1.0, 4.0, 3.0], [4.0, 1.0, 2.0], [3.0, 2.0, 1.0]]),
        unmet_penalty=np.array([40.0, 40.0, 40.0]),
        name="toy3",
    )


def _scenarios() -> ScenarioSet:
    demand = np.array([[20.0, 20.0, 20.0], [30.0, 10.0, 25.0], [15.0, 25.0, 18.0]])
    return ScenarioSet(demand=demand, probability=np.array([0.4, 0.3, 0.3]))


def test_benders_matches_monolith_no_chance() -> None:
    inst, scen = _toy_instance(), _scenarios()
    cfg = ModelConfig(chance_constraint=False)
    monolith = solve_saa_monolith(inst, scen, cfg, SOLVER)
    benders = solve_classic_benders(inst, scen, cfg, SOLVER)
    assert benders.converged
    assert benders.objective == pytest.approx(monolith.objective, rel=1e-6)
    assert sorted(benders.open_facilities) == sorted(monolith.open_facilities)
    assert benders.lower_bound <= benders.upper_bound + 1e-6


def test_benders_matches_monolith_with_chance() -> None:
    inst, scen = _toy_instance(), _scenarios()
    cfg = ModelConfig(chance_constraint=True, gamma=0.3)
    monolith = solve_saa_monolith(inst, scen, cfg, SOLVER)
    benders = solve_classic_benders(inst, scen, cfg, SOLVER)
    assert benders.converged
    assert benders.objective == pytest.approx(monolith.objective, rel=1e-6)


def _degenerate_ring(n: int, n_scen: int, seed: int) -> tuple[Instance, ScenarioSet]:
    """Facilities on a symmetric ring: tied recourse costs => degenerate duals."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords = np.column_stack([np.cos(angles), np.sin(angles)]) * 10.0
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    inst = Instance(
        facility_ids=[f"f{i}" for i in range(n)],
        customer_ids=[f"c{j}" for j in range(n)],
        fixed_cost=np.full(n, 100.0),
        capacity=np.full(n, 120.0),
        demand=np.full(n, 30.0),
        unit_cost=dist,
        unmet_penalty=np.full(n, 40.0),
        name="ring",
    )
    rng = np.random.default_rng(seed)
    demand = np.clip(30.0 + rng.normal(0, 6, size=(n_scen, n)), 1.0, None)
    return inst, ScenarioSet(demand=demand, probability=np.full(n_scen, 1.0 / n_scen))


@pytest.mark.slow
def test_pareto_cuts_reduce_iterations_on_degenerate_instance() -> None:
    """Pareto (Papadakos) cuts cut iterations vs standard cuts; same optimum."""
    inst, scen = _degenerate_ring(n=6, n_scen=5, seed=1)
    cfg = ModelConfig(chance_constraint=False)
    monolith = solve_saa_monolith(inst, scen, cfg, SOLVER)

    standard = solve_classic_benders(
        inst, scen, cfg, SolverConfig(mip_solver="highs", pareto_cuts=False)
    )
    pareto = solve_classic_benders(
        inst, scen, cfg, SolverConfig(mip_solver="highs", pareto_cuts=True)
    )

    assert standard.objective == pytest.approx(monolith.objective, rel=1e-6)
    assert pareto.objective == pytest.approx(monolith.objective, rel=1e-6)
    assert pareto.iterations < standard.iterations
    assert pareto.n_cuts < standard.n_cuts


def test_benders_matches_monolith_on_generated_instance() -> None:
    rng = np.random.default_rng(2024)
    names = [f"c{i}" for i in range(8)]
    coords = rng.uniform(0, 5, size=(8, 2))
    population = rng.uniform(50, 200, size=8)
    inst = build_geonames_instance(names, coords, population, DataConfig())
    scen = generate_scenarios(
        inst, ScenarioConfig(n_scenarios=6, n_sample=200, sigma=0.25, reduction="kmeans"), rng
    )
    cfg = ModelConfig(chance_constraint=False)
    monolith = solve_saa_monolith(inst, scen, cfg, SOLVER)
    benders = solve_classic_benders(inst, scen, cfg, SOLVER)
    assert benders.converged
    assert benders.objective == pytest.approx(monolith.objective, rel=1e-5)

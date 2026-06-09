"""Gurobi single-tree backend must match the monolith (needs a Gurobi license)."""

import numpy as np
import pytest

from sflp.config import ModelConfig, SolverConfig
from sflp.data.instance import Instance, ScenarioSet
from sflp.solve import solve_saa_monolith

gp = pytest.importorskip("gurobipy")

SOLVER = SolverConfig(mip_solver="highs")


def _toy() -> tuple[Instance, ScenarioSet]:
    inst = Instance(
        facility_ids=["f0", "f1", "f2"],
        customer_ids=["c0", "c1", "c2"],
        fixed_cost=np.array([100.0, 120.0, 80.0]),
        capacity=np.array([60.0, 60.0, 60.0]),
        demand=np.array([20.0, 20.0, 20.0]),
        unit_cost=np.array([[1.0, 4.0, 3.0], [4.0, 1.0, 2.0], [3.0, 2.0, 1.0]]),
        unmet_penalty=np.array([40.0, 40.0, 40.0]),
        name="toy3",
    )
    demand = np.array([[20.0, 20.0, 20.0], [30.0, 10.0, 25.0], [15.0, 25.0, 18.0]])
    return inst, ScenarioSet(demand=demand, probability=np.array([0.4, 0.3, 0.3]))


@pytest.fixture(scope="module")
def gurobi_available() -> bool:
    try:
        env = gp.Env()
        env.dispose()
        return True
    except gp.GurobiError as exc:  # no license
        pytest.skip(f"Gurobi license unavailable: {exc}")


@pytest.mark.gurobi
def test_gurobi_backend_matches_monolith(gurobi_available: bool) -> None:
    from sflp.benders.gurobi_backend import GurobiBackend

    inst, scen = _toy()
    cfg = ModelConfig(chance_constraint=False)
    monolith = solve_saa_monolith(inst, scen, cfg, SOLVER)
    result = GurobiBackend().solve(
        inst, scen, cfg, SolverConfig(backend="gurobi", mip_solver="gurobi")
    )
    assert result.converged
    assert result.objective == pytest.approx(monolith.objective, rel=1e-5)
    assert result.n_cuts > 0


@pytest.mark.gurobi
def test_gurobi_backend_with_chance(gurobi_available: bool) -> None:
    from sflp.benders.gurobi_backend import GurobiBackend

    inst, scen = _toy()
    cfg = ModelConfig(chance_constraint=True, gamma=0.3)
    monolith = solve_saa_monolith(inst, scen, cfg, SOLVER)
    result = GurobiBackend().solve(
        inst, scen, cfg, SolverConfig(backend="gurobi", mip_solver="gurobi")
    )
    assert result.converged
    assert result.objective == pytest.approx(monolith.objective, rel=1e-5)

"""SCIP single-tree branch-and-Benders-cut must equal classic and monolith."""

import numpy as np
import pytest

from sflp.benders.classic import solve_classic_benders
from sflp.benders.scip_backend import ScipBackend
from sflp.config import ModelConfig, SolverConfig
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


def test_scip_backend_matches_monolith_and_classic_no_chance() -> None:
    inst, scen = _toy_instance(), _scenarios()
    cfg = ModelConfig(chance_constraint=False)
    monolith = solve_saa_monolith(inst, scen, cfg, SOLVER)
    classic = solve_classic_benders(inst, scen, cfg, SOLVER)
    scip_cfg = SolverConfig(backend="branch_and_cut", mip_solver="scip")
    scip = ScipBackend().solve(inst, scen, cfg, scip_cfg)

    assert scip.converged
    assert scip.objective == pytest.approx(monolith.objective, rel=1e-5)
    assert scip.objective == pytest.approx(classic.objective, rel=1e-5)
    assert sorted(scip.open_facilities) == sorted(monolith.open_facilities)
    assert scip.n_cuts > 0


def test_scip_backend_matches_monolith_with_chance() -> None:
    inst, scen = _toy_instance(), _scenarios()
    cfg = ModelConfig(chance_constraint=True, gamma=0.3)
    monolith = solve_saa_monolith(inst, scen, cfg, SOLVER)
    scip_cfg = SolverConfig(backend="branch_and_cut", mip_solver="scip")
    scip = ScipBackend().solve(inst, scen, cfg, scip_cfg)

    assert scip.converged
    assert scip.objective == pytest.approx(monolith.objective, rel=1e-5)

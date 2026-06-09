"""Sample-average-approximation statistics: optimality gap, VSS, and EVPI.

These measures answer "is the stochastic model worth it, and is the sample big
enough?" using the standard stochastic-programming quantities (Birge & Louveaux;
Kleywegt-Shapiro-Homem-de-Mello 2002; Mak-Morton-Wood 1999):

- **RP** - the recourse (stochastic) problem optimum.
- **WS** - wait-and-see: average over scenarios of the per-scenario optimum
  (perfect foresight). ``WS <= RP``.
- **EEV** - expected result of the expected-value solution: the cost of the
  mean-value first stage, evaluated over the real scenarios. ``EEV >= RP``.
- **EVPI = RP - WS** - value of perfect information.
- **VSS = EEV - RP** - value of the stochastic solution over the mean-value one.
- **SAA optimality gap** - a statistical bound on how far an SAA first stage is
  from the true optimum: a lower bound from the mean of several SAA replications
  and an upper bound from evaluating one candidate on a large reference sample.

All measures use the base two-stage recourse (no chance constraint), which is the
textbook setting for VSS/EVPI.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import stats

from sflp.config import ModelConfig, ScenarioConfig, SolverConfig
from sflp.data.generate import generate_scenarios
from sflp.data.instance import FloatArray, Instance, ScenarioSet
from sflp.model.subproblem import build_subproblems
from sflp.solve import solve_saa_monolith

_BASE = ModelConfig(chance_constraint=False)


def _monolith_cfg(solver_cfg: SolverConfig) -> SolverConfig:
    """A Pyomo (HiGHS) solver config for the small single-scenario monoliths.

    WS and the expected-value problem are single-scenario MILPs solved as a
    monolith, which needs a Pyomo solver; the experiment's backend (e.g. SCIP)
    may not be one, so force HiGHS here while keeping its tolerances.
    """
    return SolverConfig(
        backend="classic",
        mip_solver="highs",
        gap_tolerance=solver_cfg.gap_tolerance,
        time_limit=solver_cfg.time_limit,
    )


@dataclass(frozen=True)
class StochasticMeasures:
    """The classic stochastic-programming value measures for one instance."""

    rp: float  # recourse (stochastic) problem optimum
    ws: float  # wait-and-see (perfect foresight)
    eev: float  # expected result of the expected-value solution
    evpi: float  # rp - ws >= 0
    vss: float  # eev - rp >= 0


@dataclass(frozen=True)
class GapEstimate:
    """A statistical confidence interval on an SAA first stage's optimality gap."""

    lower_bound: float  # mean of replication optima (estimates a lower bound on z*)
    lower_stderr: float
    upper_bound: float  # true expected cost of the candidate (estimates >= z*)
    upper_stderr: float
    gap: float  # upper_bound - lower_bound
    gap_ci_high: float  # one-sided (1 - alpha) upper bound on the gap
    replications: int
    reference_size: int


def evaluate_first_stage(instance: Instance, scenarios: ScenarioSet, y: FloatArray) -> float:
    """Expected total cost of a fixed first stage ``y`` over ``scenarios``.

    Uses the base recourse (transport + unmet penalty, no chance constraint).
    """
    subs = build_subproblems(instance, scenarios.demand, has_chance=False)
    recourse = sum(
        scenarios.probability[s] * subs[s].solve(y, 0.0).objective
        for s in range(scenarios.n_scenarios)
    )
    return float(instance.fixed_cost @ y + recourse)


def _single_scenario(demand_row: FloatArray) -> ScenarioSet:
    return ScenarioSet(demand=demand_row[None, :].copy(), probability=np.array([1.0]))


def wait_and_see(instance: Instance, scenarios: ScenarioSet, solver_cfg: SolverConfig) -> float:
    """Probability-weighted average of the per-scenario perfect-foresight optima."""
    cfg = _monolith_cfg(solver_cfg)
    total = 0.0
    for s in range(scenarios.n_scenarios):
        sol = solve_saa_monolith(instance, _single_scenario(scenarios.demand[s]), _BASE, cfg)
        total += float(scenarios.probability[s]) * sol.objective
    return total


def expected_value_solution(
    instance: Instance, scenarios: ScenarioSet, solver_cfg: SolverConfig
) -> FloatArray:
    """First stage that is optimal for the mean-demand (expected-value) problem."""
    mean_problem = _single_scenario(scenarios.mean_demand())
    return solve_saa_monolith(instance, mean_problem, _BASE, _monolith_cfg(solver_cfg)).y


def compute_stochastic_measures(
    instance: Instance, scenarios: ScenarioSet, solver_cfg: SolverConfig
) -> StochasticMeasures:
    """Compute RP, WS, EEV, EVPI, and VSS for an instance and scenario set.

    RP (the full-scenario problem) is solved with the configured Benders backend,
    so it scales like the main experiment rather than the large monolith. WS and
    the expected-value solution are single-scenario problems, solved directly.
    """
    from sflp.benders import solve_benders

    rp = solve_benders(instance, scenarios, _BASE, solver_cfg).objective
    ws = wait_and_see(instance, scenarios, solver_cfg)
    y_ev = expected_value_solution(instance, scenarios, solver_cfg)
    eev = evaluate_first_stage(instance, scenarios, y_ev)
    return StochasticMeasures(rp=rp, ws=ws, eev=eev, evpi=rp - ws, vss=eev - rp)


def estimate_optimality_gap(
    instance: Instance,
    scenario_cfg: ScenarioConfig,
    solver_cfg: SolverConfig,
    seed: int,
    replications: int = 20,
    reference_size: int = 2000,
    confidence: float = 0.95,
) -> GapEstimate:
    """Estimate the SAA optimality gap of a candidate first stage.

    Solves ``replications`` independent SAA problems (each on a fresh scenario
    sample) for a statistical lower bound on the true optimum, then evaluates the
    first candidate's first stage on a large reference sample for an upper bound.
    The gap and a one-sided ``confidence`` upper bound on it are returned.
    """
    rng = np.random.default_rng(seed)
    monolith_cfg = _monolith_cfg(solver_cfg)
    replication_optima = np.empty(replications)
    candidate_y: FloatArray | None = None
    for m in range(replications):
        scen = generate_scenarios(instance, scenario_cfg, rng)
        sol = solve_saa_monolith(instance, scen, _BASE, monolith_cfg)
        replication_optima[m] = sol.objective
        if candidate_y is None:
            candidate_y = sol.y
    assert candidate_y is not None

    lower_mean = float(replication_optima.mean())
    lower_stderr = float(replication_optima.std(ddof=1) / np.sqrt(replications))

    ref_cfg = replace(scenario_cfg, n_scenarios=reference_size, reduction="none")
    ref = generate_scenarios(instance, ref_cfg, rng)
    subs = build_subproblems(instance, ref.demand, has_chance=False)
    fixed = float(instance.fixed_cost @ candidate_y)
    per_scenario = np.array(
        [fixed + subs[s].solve(candidate_y, 0.0).objective for s in range(reference_size)]
    )
    upper_mean = float(per_scenario.mean())
    upper_stderr = float(per_scenario.std(ddof=1) / np.sqrt(reference_size))

    gap = upper_mean - lower_mean
    z = float(stats.norm.ppf(confidence))
    gap_ci_high = gap + z * np.sqrt(lower_stderr**2 + upper_stderr**2)
    return GapEstimate(
        lower_bound=lower_mean,
        lower_stderr=lower_stderr,
        upper_bound=upper_mean,
        upper_stderr=upper_stderr,
        gap=gap,
        gap_ci_high=gap_ci_high,
        replications=replications,
        reference_size=reference_size,
    )

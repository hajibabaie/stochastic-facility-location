"""Run every experiment for the technical report and write reproducible results.

Produces ``results/report/results.json`` and figures under
``results/report/figures``. Numbers in the report are transcribed from this file.
Run with: ``uv run python scripts/collect_results.py`` (Gurobi parts need the
[gurobi] extra and a license; they are skipped gracefully otherwise).
"""

from __future__ import annotations

import json
import time
import urllib.error
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sflp import plotting
from sflp.benders import solve_benders
from sflp.benders.classic import solve_classic_benders
from sflp.config import (
    DataConfig,
    ModelConfig,
    ScenarioConfig,
    SolverConfig,
)
from sflp.data.download import download_source
from sflp.data.generate import build_geonames_instance, generate_scenarios
from sflp.data.instance import Instance, ScenarioSet
from sflp.data.parse import parse_or_library_optima, read_or_library_cap
from sflp.saa import compute_stochastic_measures
from sflp.solve import solve_deterministic_cflp, solve_saa_monolith

OUT = Path("results/report")
FIG = OUT / "figures"
SEED = 20231015
HIGHS = SolverConfig(mip_solver="highs")
# Fast backend for the stochastic measures' full-scenario RP solve.
SCIP = SolverConfig(backend="branch_and_cut", mip_solver="scip", pareto_cuts=False)


def log(msg: str) -> None:
    print(msg, flush=True)


def read_or_geonames(country: str, n: int) -> Instance:
    path = download_source("geonames_cities5000")
    from sflp.data.parse import read_geonames

    cities = read_geonames(path, 5000, country=country).top_by_population(n)
    return build_geonames_instance(
        cities.names, cities.coordinates, cities.population, DataConfig()
    )


def scenarios_for(inst: Instance, n_scen: int, sigma: float, seed: int = SEED) -> ScenarioSet:
    cfg = ScenarioConfig(n_scenarios=n_scen, n_sample=600, sigma=sigma, reduction="kmeans")
    return generate_scenarios(inst, cfg, np.random.default_rng(seed))


def or_library_validation() -> list[dict[str, object]]:
    log("== OR-Library validation ==")
    try:
        optima = parse_or_library_optima(download_source("or_capopt").read_text("utf-8"))
    except (urllib.error.URLError, OSError) as exc:
        log(f"  skipped (offline): {exc}")
        return []
    rows = []
    for key in ("or_cap71", "or_cap101", "or_cap131"):
        inst = read_or_library_cap(download_source(key))
        sol = solve_deterministic_cflp(inst, HIGHS)
        published = optima[inst.name]
        rows.append(
            {
                "instance": inst.name,
                "solver_objective": sol.objective,
                "published_optimum": published,
                "relative_error": abs(sol.objective - published) / abs(published),
            }
        )
        log(f"  {inst.name}: solver={sol.objective:.3f} published={published:.3f}")
    return rows


def degenerate_ring(n: int, n_scen: int, seed: int) -> tuple[Instance, ScenarioSet]:
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


def pareto_comparison() -> dict[str, object]:
    log("== Pareto cuts (degenerate ring 6/5) ==")
    inst, scen = degenerate_ring(6, 5, seed=1)
    cfg = ModelConfig(chance_constraint=False)
    mono = solve_saa_monolith(inst, scen, cfg, HIGHS)
    std = solve_classic_benders(
        inst, scen, cfg, SolverConfig(mip_solver="highs", pareto_cuts=False)
    )
    par = solve_classic_benders(inst, scen, cfg, SolverConfig(mip_solver="highs", pareto_cuts=True))
    log(f"  standard iters={std.iterations} cuts={std.n_cuts}")
    log(f"  pareto   iters={par.iterations} cuts={par.n_cuts}")
    return {
        "monolith_objective": mono.objective,
        "standard": {
            "iterations": std.iterations,
            "cuts": std.n_cuts,
            "objective": std.objective,
        },
        "pareto": {"iterations": par.iterations, "cuts": par.n_cuts, "objective": par.objective},
    }


def backend_comparison(inst: Instance, scen: ScenarioSet) -> dict[str, object]:
    log("== Backend comparison + agreement ==")
    cfg = ModelConfig(chance_constraint=False)
    out: dict[str, object] = {"monolith": solve_saa_monolith(inst, scen, cfg, HIGHS).objective}
    backends = [
        ("classic", SolverConfig(backend="classic", mip_solver="highs", pareto_cuts=False)),
        ("scip", SolverConfig(backend="branch_and_cut", mip_solver="scip", pareto_cuts=False)),
    ]
    try:
        import gurobipy  # noqa: F401

        backends.append(
            ("gurobi", SolverConfig(backend="gurobi", mip_solver="gurobi", pareto_cuts=False))
        )
    except ImportError:
        log("  gurobi not available; skipping that backend")
    for name, sc in backends:
        t = time.perf_counter()
        r = solve_benders(inst, scen, cfg, sc)
        out[name] = {
            "objective": r.objective,
            "iterations": r.iterations,
            "n_cuts": r.n_cuts,
            "time_seconds": time.perf_counter() - t,
            "converged": r.converged,
        }
        log(f"  {name}: obj={r.objective:.2f} time={out[name]['time_seconds']:.1f}s")
    return out


def gurobi_scaling() -> list[dict[str, object]]:
    log("== Gurobi scaling ==")
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        log("  gurobi not available; skipping")
        return []
    rows = []
    for nf, ns in [(12, 8), (20, 15), (30, 20), (50, 30)]:
        inst = read_or_geonames("DE", nf)
        scen = scenarios_for(inst, ns, 0.2)
        sc = SolverConfig(backend="gurobi", mip_solver="gurobi", pareto_cuts=False, time_limit=120)
        t = time.perf_counter()
        r = solve_benders(inst, scen, ModelConfig(chance_constraint=False), sc)
        rows.append(
            {
                "n_facilities": nf,
                "n_scenarios": ns,
                "objective": r.objective,
                "gap": r.gap,
                "time_seconds": time.perf_counter() - t,
                "converged": r.converged,
            }
        )
        log(f"  {nf}/{ns}: obj={r.objective:.1f} gap={r.gap:.2e} converged={r.converged}")
    return rows


def sigma_sweep(inst: Instance) -> list[dict[str, object]]:
    log("== Sigma sweep (VSS/EVPI) ==")
    rows = []
    for sigma in (0.10, 0.20, 0.30):
        scen = scenarios_for(inst, 12, sigma)
        m = compute_stochastic_measures(inst, scen, SCIP)
        rows.append(
            {"sigma": sigma, "rp": m.rp, "ws": m.ws, "eev": m.eev, "evpi": m.evpi, "vss": m.vss}
        )
        log(f"  sigma={sigma}: EVPI={m.evpi:.1f} VSS={m.vss:.1f}")
    return rows


def gamma_sweep(inst: Instance) -> list[dict[str, object]]:
    log("== Gamma sweep (chance constraint) ==")
    scen = scenarios_for(inst, 10, 0.3)
    rows = []
    for gamma in (0.0, 0.1, 0.2, 0.3):
        cfg = ModelConfig(chance_constraint=True, gamma=gamma, epsilon=max(gamma, 0.3))
        sol = solve_saa_monolith(inst, scen, cfg, HIGHS)
        rows.append(
            {
                "gamma": gamma,
                "objective": sol.objective,
                "expected_unmet": sol.expected_unmet,
                "violation_probability": sol.violation_probability,
            }
        )
        log(f"  gamma={gamma}: obj={sol.objective:.1f} unmet={sol.expected_unmet:.2f}")
    return rows


def make_figures(inst: Instance, scen: ScenarioSet, sigma_rows: list[dict[str, object]]) -> None:
    log("== Figures ==")
    cfg = ModelConfig(chance_constraint=False)
    classic = solve_classic_benders(
        inst, scen, cfg, SolverConfig(mip_solver="highs", pareto_cuts=False)
    )
    plotting.plot_facility_map(inst, classic, FIG / "facility_map.png")
    plotting.plot_convergence(classic, FIG / "convergence.png")
    measures = compute_stochastic_measures(inst, scen, SCIP)
    plotting.plot_stochastic_measures(measures, FIG / "measures.png")
    # sigma sweep curve
    sigmas = [r["sigma"] for r in sigma_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sigmas, [r["evpi"] for r in sigma_rows], "-o", label="EVPI")
    ax.plot(sigmas, [r["vss"] for r in sigma_rows], "-s", label="VSS")
    ax.set_xlabel("demand volatility sigma")
    ax.set_ylabel("value")
    ax.set_title("EVPI and VSS vs demand volatility")
    ax.legend()
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "sigma_sweep.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    base = read_or_geonames("DE", 12)
    base_scen = scenarios_for(base, 12, 0.2)

    results: dict[str, object] = {
        "seed": SEED,
        "or_library_validation": or_library_validation(),
        "pareto": pareto_comparison(),
        "backend_comparison": backend_comparison(base, base_scen),
        "gurobi_scaling": gurobi_scaling(),
        "sigma_sweep": sigma_sweep(base),
        "gamma_sweep": gamma_sweep(base),
    }
    # Persist results before the (slower, optional) figures so a plotting issue
    # never loses the numbers.
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"Wrote {OUT / 'results.json'}")
    try:
        make_figures(base, base_scen, results["sigma_sweep"])  # type: ignore[arg-type]
        log(f"Wrote figures in {FIG}")
    except Exception as exc:
        # Report and continue; the numeric results are already saved.
        log(f"Figure generation failed: {exc!r}")


if __name__ == "__main__":
    main()

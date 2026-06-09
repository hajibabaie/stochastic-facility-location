"""Command-line interface: ``sflp run --config <file>``."""

from __future__ import annotations

import argparse
from pathlib import Path

from sflp import plotting
from sflp.config import Config, load_config
from sflp.experiment import RunResult, build_instance, run_experiment, save_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sflp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="solve an experiment from a YAML config")
    run_p.add_argument("--config", required=True, help="path to a config YAML file")
    run_p.add_argument("--output", default="results", help="output directory (default: results)")
    run_p.add_argument("--name", default=None, help="run name (default: config file stem)")
    run_p.add_argument("--no-measures", action="store_true", help="skip VSS/EVPI computation")
    run_p.add_argument("--no-plots", action="store_true", help="skip figure generation")

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


def _run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    name = args.name or Path(args.config).stem
    output = Path(args.output)

    print(f"[sflp] running '{name}' (backend={cfg.solver.backend}, solver={cfg.solver.mip_solver})")
    result = run_experiment(cfg, compute_measures=not args.no_measures)

    record = save_result(result, output / "logs", name)
    _print_summary(result)
    print(f"[sflp] wrote {record}")

    if not args.no_plots:
        _make_plots(result, cfg, output / "figures", name)
    return 0


def _print_summary(result: RunResult) -> None:
    print(
        f"[sflp] {result.instance_name}: I={result.n_facilities} J={result.n_customers} "
        f"S={result.n_scenarios}"
    )
    print(
        f"[sflp] objective={result.objective:.4f} gap={result.gap:.2e} "
        f"open={len(result.open_facilities)} iters={result.iterations} cuts={result.n_cuts} "
        f"time={result.runtime_seconds:.2f}s"
    )
    if result.measures is not None:
        m = result.measures
        print(
            f"[sflp] RP={m.rp:.2f} WS={m.ws:.2f} EEV={m.eev:.2f} EVPI={m.evpi:.2f} VSS={m.vss:.2f}"
        )


def _make_plots(result: RunResult, cfg: Config, figures_dir: Path, name: str) -> None:
    if result.benders.history:
        plotting.plot_convergence(result.benders, figures_dir / f"{name}_convergence.png")
    if result.measures is not None:
        plotting.plot_stochastic_measures(result.measures, figures_dir / f"{name}_measures.png")
    # Rebuild the instance (data is cached) only to draw the geographic map.
    instance = build_instance(cfg)
    if instance.coordinates is not None:
        plotting.plot_facility_map(instance, result.benders, figures_dir / f"{name}_map.png")
    print(f"[sflp] wrote figures to {figures_dir}")


if __name__ == "__main__":
    raise SystemExit(main())

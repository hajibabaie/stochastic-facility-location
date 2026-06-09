"""Result figures: facility map, Benders bound convergence, and value measures.

Uses a non-interactive backend so figures render without a display (CI, servers).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sflp.benders.backend import BendersResult
from sflp.data.instance import Instance
from sflp.saa import StochasticMeasures


def plot_facility_map(instance: Instance, result: BendersResult, path: str | Path) -> Path:
    """Scatter customers (sized by demand) and highlight the open facilities."""
    if instance.coordinates is None:
        raise ValueError("instance has no coordinates to plot.")
    coords = instance.coordinates
    fig, ax = plt.subplots(figsize=(7, 6))
    sizes = 20 + 180 * instance.demand / instance.demand.max()
    ax.scatter(
        coords[:, 1],
        coords[:, 0],
        s=sizes,
        c="#9bb3d4",
        edgecolors="none",
        label="customer (size = demand)",
        zorder=1,
    )
    opened = result.open_facilities
    ax.scatter(
        coords[opened, 1],
        coords[opened, 0],
        marker="*",
        s=320,
        c="#d1495b",
        edgecolors="black",
        linewidths=0.5,
        label="open facility",
        zorder=3,
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"{instance.name}: {len(opened)} facilities open")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    return _save(fig, path)


def plot_convergence(result: BendersResult, path: str | Path) -> Path:
    """Plot the Benders lower and upper bounds per iteration (classic backend)."""
    if not result.history:
        raise ValueError("no bound history to plot (single-tree backends omit it).")
    lbs = [lb for lb, _ in result.history]
    ubs = [ub for _, ub in result.history]
    iterations = range(1, len(result.history) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(iterations, ubs, "-o", ms=3, label="upper bound (incumbent)", color="#d1495b")
    ax.plot(iterations, lbs, "-o", ms=3, label="lower bound (master)", color="#30638e")
    ax.set_xlabel("iteration")
    ax.set_ylabel("objective")
    ax.set_title("Benders bound convergence")
    ax.legend(loc="best")
    fig.tight_layout()
    return _save(fig, path)


def plot_stochastic_measures(measures: StochasticMeasures, path: str | Path) -> Path:
    """Bar chart of WS <= RP <= EEV, annotating EVPI and VSS."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = ["WS", "RP", "EEV"]
    values = [measures.ws, measures.rp, measures.eev]
    ax.bar(labels, values, color=["#6a994e", "#30638e", "#bc4749"])
    ax.set_ylabel("expected cost")
    ax.set_title(f"Stochastic value measures (EVPI={measures.evpi:.1f}, VSS={measures.vss:.1f})")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    return _save(fig, path)


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

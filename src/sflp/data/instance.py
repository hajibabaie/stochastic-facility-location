"""Core data structures for a (stochastic) capacitated facility location instance.

An :class:`Instance` holds the deterministic data shared by every scenario:
candidate facilities, customers, opening cost, capacity, the per-unit serve-cost
matrix, and the unmet-demand penalty. The uncertain part (demand realizations) is
a :class:`ScenarioSet`.

Conventions
-----------
- ``I`` indexes facilities, ``J`` indexes customers.
- ``unit_cost[i, j]`` is the cost to ship **one unit** of demand from facility
  ``i`` to customer ``j`` (so the serve-cost term is ``unit_cost * flow``).
- ``demand`` on the instance is the **mean / nominal** demand ``d_j^0``; the
  per-scenario demand lives in a :class:`ScenarioSet`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Instance:
    """Deterministic data for a capacitated facility location problem."""

    facility_ids: list[str]
    customer_ids: list[str]
    fixed_cost: FloatArray  # (I,)  f_i
    capacity: FloatArray  # (I,)  s_i
    demand: FloatArray  # (J,)  nominal d_j^0
    unit_cost: FloatArray  # (I, J)  c_ij per unit
    unmet_penalty: FloatArray  # (J,)  q_j
    coordinates: FloatArray | None = None  # (J, 2) lat/lon, if geographic
    name: str = "instance"

    def __post_init__(self) -> None:
        if self.unit_cost.shape != (self.n_facilities, self.n_customers):
            raise ValueError(
                f"unit_cost must be ({self.n_facilities}, {self.n_customers}), "
                f"got {self.unit_cost.shape}."
            )
        for arr, n, label in (
            (self.fixed_cost, self.n_facilities, "fixed_cost"),
            (self.capacity, self.n_facilities, "capacity"),
            (self.demand, self.n_customers, "demand"),
            (self.unmet_penalty, self.n_customers, "unmet_penalty"),
        ):
            if arr.shape != (n,):
                raise ValueError(f"{label} must have shape ({n},), got {arr.shape}.")
        if float(self.capacity.sum()) < float(self.demand.sum()):
            raise ValueError(
                "Total capacity is below total nominal demand; the instance can "
                "never serve all customers even with every facility open."
            )

    @property
    def n_facilities(self) -> int:
        return len(self.facility_ids)

    @property
    def n_customers(self) -> int:
        return len(self.customer_ids)


@dataclass(frozen=True)
class ScenarioSet:
    """A finite set of demand realizations with probabilities.

    ``demand[s, j]`` is the demand of customer ``j`` under scenario ``s``;
    ``probability[s]`` is its weight (the weights sum to one).
    """

    demand: FloatArray  # (S, J)
    probability: FloatArray  # (S,)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.demand.ndim != 2:
            raise ValueError(f"demand must be 2-D (S, J), got shape {self.demand.shape}.")
        if self.probability.shape != (self.n_scenarios,):
            raise ValueError(
                f"probability must have shape ({self.n_scenarios},), got {self.probability.shape}."
            )
        total = float(self.probability.sum())
        if not np.isclose(total, 1.0, atol=1e-9):
            raise ValueError(f"probability must sum to 1, got {total}.")
        if np.any(self.probability < 0):
            raise ValueError("probability has negative entries.")

    @property
    def n_scenarios(self) -> int:
        return self.demand.shape[0]

    @property
    def n_customers(self) -> int:
        return self.demand.shape[1]

    def mean_demand(self) -> FloatArray:
        """Probability-weighted mean demand vector, shape (J,)."""
        return self.probability @ self.demand

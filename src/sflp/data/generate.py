"""Build instances from real geography and generate seeded demand scenarios.

The geography (coordinates, population) is real. Only capacity and fixed cost are
*model-supplied*, by the documented rules in :class:`~sflp.config.DataConfig`:

- ``s_i = capacity_rule_k * mean(demand)`` for every candidate facility ``i``.
- ``f_i = fixed_cost_base * (pop_i / mean_pop) ** fixed_cost_pop_scale``.
- ``q_j = unmet_penalty_scale * max_i c_ij`` (penalty must exceed any serve cost).

Demand uncertainty is a mean-preserving multiplicative lognormal shock,
``d_js = d_j^0 * xi_js`` with ``E[xi] = 1``. With a finite ``correlation_length``
the log-shocks are spatially correlated so nearby cities move together.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sflp.config import DataConfig, ScenarioConfig
from sflp.data.instance import FloatArray, Instance, ScenarioSet

IntArray = NDArray[np.intp]
EARTH_RADIUS_KM = 6371.0088
#: GeoNames populations span 10^3-10^6; we rescale demand to this mean so the
#: objective stays at a numerically clean magnitude (raw values give costs ~10^9
#: that corrupt LP duals and the Benders cuts). Rescaling is uniform, so it does
#: not change the optimal solution, only the units of cost.
GEONAMES_DEMAND_MEAN = 100.0


def haversine_matrix(coords: FloatArray) -> FloatArray:
    """Pairwise great-circle distance (km) for an ``(n, 2)`` lat/lon array."""
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    cos_lat = np.cos(lat)
    a = np.sin(dlat / 2.0) ** 2 + cos_lat[:, None] * cos_lat[None, :] * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def build_geonames_instance(
    names: list[str],
    coordinates: FloatArray,
    population: FloatArray,
    cfg: DataConfig,
) -> Instance:
    """Assemble a CFLP instance where every city is a candidate facility.

    Population is the nominal demand; transport cost is the great-circle
    distance. Capacity and fixed cost follow the documented model rules.
    """
    population = np.asarray(population, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if population.ndim != 1 or coordinates.shape != (population.size, 2):
        raise ValueError("coordinates must be (n, 2) and align with population (n,).")

    mean_pop = float(population.mean())
    # Rescale population to a clean demand magnitude (relative sizes preserved).
    demand = population * (GEONAMES_DEMAND_MEAN / mean_pop)
    mean_demand = float(demand.mean())

    # Distance is symmetric (n x n); facilities and customers are the same cities.
    distance = haversine_matrix(coordinates)
    unit_cost = distance  # cost per unit shipped == km

    capacity = np.full(population.size, cfg.capacity_rule_k * mean_demand, dtype=np.float64)
    # Fixed cost must be comparable to transport savings, else opening every
    # facility is trivially optimal. Scale it by the cost of serving one mean
    # customer over a typical distance; fixed_cost_base is then a dimensionless
    # knob (~1), and the cost rises with city population.
    off_diagonal = distance[~np.eye(population.size, dtype=bool)]
    typical_distance = float(np.median(off_diagonal)) if off_diagonal.size else 1.0
    cost_scale = mean_demand * typical_distance
    fixed_cost = (
        cfg.fixed_cost_base * cost_scale * (population / mean_pop) ** cfg.fixed_cost_pop_scale
    )
    # Penalty per unit must dominate the worst serve cost, else unmet is "free".
    max_cost_per_customer = unit_cost.max(axis=0)
    unmet_penalty = cfg.unmet_penalty_scale * max_cost_per_customer

    return Instance(
        facility_ids=list(names),
        customer_ids=list(names),
        fixed_cost=fixed_cost,
        capacity=capacity,
        demand=demand,
        unit_cost=unit_cost,
        unmet_penalty=unmet_penalty,
        coordinates=coordinates,
        name=f"geonames-{cfg.country}-{population.size}",
    )


def _log_covariance(
    n: int, sigma: float, coords: FloatArray | None, correlation_length: float | None
) -> FloatArray:
    """Covariance of the log-shocks: diagonal sigma^2, or distance-decayed."""
    if correlation_length is None or coords is None:
        return np.eye(n) * sigma**2
    distance = haversine_matrix(coords)
    return sigma**2 * np.exp(-distance / correlation_length)


def sample_demand(
    base_demand: FloatArray,
    sigma: float,
    n_sample: int,
    rng: np.random.Generator,
    coords: FloatArray | None = None,
    correlation_length: float | None = None,
) -> FloatArray:
    """Draw ``n_sample`` mean-preserving lognormal demand vectors, shape (n_sample, J).

    Log-shocks are Gaussian with the covariance from :func:`_log_covariance`; the
    mean is shifted by ``-0.5 * diag(cov)`` so that ``E[d_js] = base_demand[j]``.
    """
    j = base_demand.size
    cov = _log_covariance(j, sigma, coords, correlation_length)
    mean = -0.5 * np.diag(cov)
    log_shock = rng.multivariate_normal(mean, cov, size=n_sample)
    return base_demand[None, :] * np.exp(log_shock)


def reduce_scenarios(
    samples: FloatArray, n_scenarios: int, method: str, rng: np.random.Generator
) -> ScenarioSet:
    """Reduce a large demand sample to ``n_scenarios`` weighted representatives."""
    n_sample = samples.shape[0]
    if n_scenarios > n_sample:
        raise ValueError("n_scenarios cannot exceed the number of samples.")

    if method == "none":
        idx = rng.choice(n_sample, size=n_scenarios, replace=False)
        chosen = samples[idx]
        prob = np.full(n_scenarios, 1.0 / n_scenarios)
        return ScenarioSet(demand=chosen, probability=prob, metadata={"reduction": "none"})

    if method in ("kmeans", "fast_forward"):
        centroids, labels = _kmeans(samples, n_scenarios, rng)
        counts = np.bincount(labels, minlength=n_scenarios).astype(np.float64)
        # Drop empty clusters so every kept scenario has positive probability.
        keep = counts > 0
        centroids, counts = centroids[keep], counts[keep]
        prob = counts / counts.sum()
        return ScenarioSet(demand=centroids, probability=prob, metadata={"reduction": method})

    raise ValueError(f"Unknown scenario reduction method: {method!r}.")


def _kmeans(
    samples: FloatArray, k: int, rng: np.random.Generator, max_iter: int = 100
) -> tuple[FloatArray, IntArray]:
    """Lloyd's k-means with k-means++ seeding (no external ML dependency)."""
    n = samples.shape[0]
    centroids = _kmeans_plusplus(samples, k, rng)
    labels = np.zeros(n, dtype=np.intp)
    for _ in range(max_iter):
        dist = np.linalg.norm(samples[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            members = samples[labels == c]
            if members.size:
                centroids[c] = members.mean(axis=0)
    return centroids, labels


def _kmeans_plusplus(samples: FloatArray, k: int, rng: np.random.Generator) -> FloatArray:
    """k-means++ initial centroid selection."""
    n = samples.shape[0]
    first = int(rng.integers(n))
    centroids = [samples[first]]
    closest_sq = np.linalg.norm(samples - centroids[0], axis=1) ** 2
    for _ in range(1, k):
        probs = closest_sq / closest_sq.sum() if closest_sq.sum() > 0 else None
        nxt = int(rng.choice(n, p=probs))
        centroids.append(samples[nxt])
        new_sq = np.linalg.norm(samples - samples[nxt], axis=1) ** 2
        closest_sq = np.minimum(closest_sq, new_sq)
    return np.array(centroids, dtype=np.float64)


def generate_scenarios(
    instance: Instance, cfg: ScenarioConfig, rng: np.random.Generator
) -> ScenarioSet:
    """Sample and reduce demand scenarios for an instance."""
    if cfg.reduction == "none":
        samples = sample_demand(
            instance.demand,
            cfg.sigma,
            cfg.n_scenarios,
            rng,
            instance.coordinates,
            cfg.correlation_length,
        )
        prob = np.full(cfg.n_scenarios, 1.0 / cfg.n_scenarios)
        return ScenarioSet(demand=samples, probability=prob, metadata={"reduction": "none"})

    samples = sample_demand(
        instance.demand,
        cfg.sigma,
        cfg.n_sample,
        rng,
        instance.coordinates,
        cfg.correlation_length,
    )
    return reduce_scenarios(samples, cfg.n_scenarios, cfg.reduction, rng)

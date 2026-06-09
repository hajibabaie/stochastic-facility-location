"""Tests for loading and validating run configuration."""

import textwrap
from pathlib import Path

import pytest

from sflp.config import Config, load_config


def test_default_config_is_valid() -> None:
    cfg = Config()
    assert cfg.seed == 20231015
    assert cfg.solver.backend == "classic"
    assert cfg.model.gamma <= cfg.model.epsilon


def test_load_default_yaml() -> None:
    cfg = load_config(Path(__file__).parents[2] / "configs" / "default.yaml")
    assert cfg.data.source == "geonames"
    assert cfg.scenarios.n_scenarios >= 1
    assert cfg.solver.backend == "branch_and_cut"
    assert cfg.model.gamma <= cfg.model.epsilon


def test_nested_overrides_round_trip(tmp_path: Path) -> None:
    text = textwrap.dedent(
        """
        seed: 7
        data:
          country: US
          n_facilities: 150
        solver:
          backend: gurobi
          pareto_cuts: false
        """
    )
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    cfg = load_config(path)
    assert cfg.seed == 7
    assert cfg.data.country == "US"
    assert cfg.data.n_facilities == 150
    assert cfg.solver.backend == "gurobi"
    assert cfg.solver.pareto_cuts is False
    # untouched fields keep their defaults
    assert cfg.scenarios.sigma == 0.2


def test_unknown_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("data:\n  bogus_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown DataConfig keys"):
        load_config(path)


def test_gamma_above_epsilon_rejected() -> None:
    from sflp.config import ModelConfig

    with pytest.raises(ValueError, match="gamma"):
        Config(model=ModelConfig(epsilon=0.05, gamma=0.10))


def test_too_few_facilities_rejected() -> None:
    from sflp.config import DataConfig

    with pytest.raises(ValueError, match="n_facilities"):
        Config(data=DataConfig(n_facilities=1))

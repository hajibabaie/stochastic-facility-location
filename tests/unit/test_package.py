"""Smoke tests for the package skeleton."""

import sflp


def test_version_is_exposed() -> None:
    assert isinstance(sflp.__version__, str)
    assert sflp.__version__.count(".") == 2

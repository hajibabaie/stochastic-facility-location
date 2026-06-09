"""Tests for checksum verification (no network access required)."""

from pathlib import Path

import pytest

from sflp.data.download import SOURCES, sha256_of, verify_checksum


def test_sha256_of_known_content(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello")
    # sha256("hello")
    assert sha256_of(f) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_verify_checksum_passes_and_returns_digest(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello")
    digest = verify_checksum(f, None)  # None skips verification
    assert verify_checksum(f, digest) == digest


def test_verify_checksum_mismatch_raises(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_checksum(f, "0" * 64)


def test_sources_registry_has_expected_keys() -> None:
    assert "geonames_cities5000" in SOURCES
    assert SOURCES["geonames_cities5000"].member == "cities5000.txt"
    assert "or_cap71" in SOURCES

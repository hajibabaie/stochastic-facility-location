"""Fetch real third-party datasets with SHA-256 verification.

We do **not** redistribute raw third-party files. Instead we ship this script
plus checksums, so anyone can reproduce the exact data we used. Downloaded files
land in ``data/raw/`` (gitignored).

Attribution / licensing (see README):
- GeoNames ``cities*`` — CC BY 4.0 (attribution required).
- OR-Library ``cap*`` / optima — J. E. Beasley.
- SIPLIB ``SSLP`` — Ntaimo & Sen.
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

GEONAMES_BASE = "https://download.geonames.org/export/dump/"
OR_LIBRARY_BASE = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"
SIPLIB_BASE = "https://www2.isye.gatech.edu/~sahmed/siplib/sslp/"

_USER_AGENT = "sflp-data-fetcher/0.1 (+https://github.com/hajibabaie/stochastic-facility-location)"


@dataclass(frozen=True)
class DataSource:
    """A downloadable file. ``sha256`` is the shipped checksum (``None`` = unpinned).

    For zip archives, ``member`` names the file to extract from the archive; the
    checksum then refers to the **extracted** member, not the zip.
    """

    name: str
    url: str
    sha256: str | None = None
    member: str | None = None


# Registry of the exact files used. Hashes are pinned after a verified download;
# an unpinned (None) hash means "compute and report" rather than "verify".
SOURCES: dict[str, DataSource] = {
    "geonames_cities5000": DataSource(
        name="cities5000",
        url=GEONAMES_BASE + "cities5000.zip",
        member="cities5000.txt",
    ),
    "or_cap71": DataSource(
        name="cap71",
        url=OR_LIBRARY_BASE + "cap71.txt",
        sha256="dda7a533a674bd5c82766f58aa3ccb995f9cf7f776210f9aa1a7a49406ded8dc",
    ),
    "or_cap101": DataSource(
        name="cap101",
        url=OR_LIBRARY_BASE + "cap101.txt",
        sha256="7f1d0e32808b8c1bd9e3f49ee42dd758abb5a157da63ed52182f22bd14919ebe",
    ),
    "or_cap131": DataSource(
        name="cap131",
        url=OR_LIBRARY_BASE + "cap131.txt",
        sha256="3ea70c64d029f21a2065b588c4ffa7b0d60cab6ee82fd7775360e0f7e007b36a",
    ),
    "or_capopt": DataSource(
        name="capopt",
        url=OR_LIBRARY_BASE + "capopt.txt",
        sha256="62a1ebca1a102f3fe5a43ebef3262a5350534c7dcd4d0e60578cc5016f3193ca",
    ),
}


def sha256_of(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: str | Path, expected: str | None) -> str:
    """Check a file's digest against ``expected``; return the actual digest.

    A ``None`` expectation skips verification (used to discover/pin a hash). A
    mismatch raises so corrupted or swapped downloads fail loudly.
    """
    actual = sha256_of(path)
    if expected is not None and actual.lower() != expected.lower():
        raise ValueError(
            f"Checksum mismatch for {path}:\n  expected {expected}\n  actual   {actual}"
        )
    return actual


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request) as response:
        data: bytes = response.read()
    return data


def download_source(key: str, raw_dir: str | Path = "data/raw") -> Path:
    """Download (and, for zips, extract) one registered source; verify its hash.

    Returns the path to the usable file. Re-downloads only if the target is
    missing or fails verification.
    """
    if key not in SOURCES:
        raise KeyError(f"Unknown data source {key!r}; known: {sorted(SOURCES)}.")
    source = SOURCES[key]
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    target_name = source.member if source.member else Path(source.url).name
    target = raw_dir / target_name
    if target.exists():
        try:
            verify_checksum(target, source.sha256)
            return target
        except ValueError:
            target.unlink()  # corrupt cache; fall through to re-download

    payload = _http_get(source.url)
    if source.member:
        archive = raw_dir / Path(source.url).name
        archive.write_bytes(payload)
        with zipfile.ZipFile(archive) as zf:
            zf.extract(source.member, raw_dir)
        archive.unlink()
    else:
        target.write_bytes(payload)

    verify_checksum(target, source.sha256)
    return target

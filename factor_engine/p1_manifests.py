"""Content-addressed build manifests for the P1 factor library (WS3).

A factor's output parquet is UP TO DATE iff its manifest key is unchanged.
The key covers the full closure of things that can change its value:

    1. tree structure     — the factor's Merkle root from p1_factor_dag
    2. input data         — fingerprint of every dataset file the factor's
                            evaluation reads (tree var leaves + audited opaque
                            kernel footprints + link tables + the template
                            source monthlyCRSP), plus direct artifacts read
                            outside the DatasetStore (daily buckets directory,
                            cached series like ps_innov_pit)
    3. operator code      — hand-bumped integer versions per op (KERNEL_VERSIONS).
                            Deliberately dumb: hashing bytecode misses changed
                            constants (co_consts), so a human bumps the number
                            when an operator's semantics change
    4. engine plumbing    — ENGINE_SCHEMA_VERSION covers template construction,
                            alignment, downcast and write conventions

File fingerprints are (size, mtime_ns): cheap, and safe because these files
are only replaced wholesale by data refreshes. Over-invalidation (recompute)
is always safe; the closure errs toward including more files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from Factor_Construction.p1_factor_engine import INTERMEDIATE_DIR, ROOT
except ModuleNotFoundError:
    from p1_factor_engine import INTERMEDIATE_DIR, ROOT

ENGINE_SCHEMA_VERSION = "p1-v2.0"
MANIFEST_FILENAME = "_build_manifest.json"

# Bump an op's entry when its implementation changes semantics. Missing = 1.
KERNEL_VERSIONS: dict[str, int] = {
    "cash_rdq_signal": 2,   # 2026-08-17 deterministic (gvkey, rdq) tie-break
    "price_delay": 2,       # 2026-08-17 fixed 12-month PIT forward-fill horizon
    "coskew_signal": 2,     # 2026-08-17 12-month PIT bound on anchored windows
}

# Artifacts read OUTSIDE the DatasetStore by the daily-track kernels.
# Any factor whose tree contains a daily kernel op gets this whole closure
# folded into its key (over-inclusion is safe; see module docstring).
# Data/tailrisk_series.parquet is deliberately ABSENT: it is a read-through
# cache the betatailrisk op itself writes (derived from the buckets, which
# ARE fingerprinted) — keying on it would leave BetaTailRisk never current.
DAILY_ARTIFACT_PATHS = [
    ROOT / "Data" / "daily_buckets",
    ROOT / "Data" / "ps_innov_pit.parquet",
    INTERMEDIATE_DIR / "dailyFF.parquet",
    INTERMEDIATE_DIR / "monthlyFF.parquet",
    INTERMEDIATE_DIR / "monthlyMarket.parquet",
]

# Same rule for per-op audited direct files: never fingerprint self-written caches.
EXCLUDED_DIRECT_FILES = {"Data/tailrisk_series.parquet"}


def kernel_version(op: str) -> int:
    return KERNEL_VERSIONS.get(op, 1)


def file_fingerprint(path: Path) -> str:
    """(size, mtime_ns) fingerprint; directories roll up their parquet members.

    A missing file raises — a factor whose input is absent must fail the
    plan loudly, not silently key on 'missing'.
    """

    path = Path(path)
    if path.is_dir():
        h = hashlib.blake2b(digest_size=16)
        members = sorted(p for p in path.rglob("*.parquet"))
        if not members:
            raise FileNotFoundError(f"fingerprint: directory has no parquet members: {path}")
        for p in members:
            st = p.stat()
            h.update(f"{p.relative_to(path).as_posix()}|{st.st_size}|{st.st_mtime_ns}".encode())
        return f"dir:{h.hexdigest()}"
    st = path.stat()  # raises FileNotFoundError if absent
    return f"{st.st_size}:{st.st_mtime_ns}"


def factor_manifest_key(
    factor: str,
    merkle_root: str,
    tree_ops: list[str],
    dataset_files: list[Path],
    direct_files: list[Path],
) -> str:
    """Deterministic content key for one factor build."""

    h = hashlib.blake2b(digest_size=16)
    h.update(f"schema|{ENGINE_SCHEMA_VERSION}|".encode())
    h.update(f"tree|{factor}|{merkle_root}|".encode())
    for op in sorted(set(tree_ops)):
        h.update(f"op|{op}|{kernel_version(op)}|".encode())
    for p in sorted(set(map(Path, dataset_files)) | set(map(Path, direct_files))):
        h.update(f"file|{p.name}|{file_fingerprint(p)}|".encode())
    return h.hexdigest()


def load_manifest(output_dir: Path) -> dict[str, dict[str, Any]]:
    path = Path(output_dir) / MANIFEST_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(output_dir: Path, manifest: dict[str, dict[str, Any]]) -> None:
    path = Path(output_dir) / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")


def factor_is_current(
    factor: str,
    key: str,
    output_dir: Path,
    manifest: dict[str, dict[str, Any]],
) -> bool:
    entry = manifest.get(factor)
    if entry is None or entry.get("key") != key:
        return False
    return (Path(output_dir) / f"{factor}.parquet").exists()

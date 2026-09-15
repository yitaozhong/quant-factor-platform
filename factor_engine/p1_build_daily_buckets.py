"""
p1_build_daily_buckets.py — (re)build Data/daily_buckets/ from dailyCRSP.parquet.

The daily kernels (p1_daily_kernels.py) stream Data/daily_buckets/bucket=N/*.parquet:
dailyCRSP reorganized into N_BUCKETS permno-hash buckets so each stock's daily
history sits in one bucket. iter_buckets() sorts and asserts contiguity itself,
so this builder only needs to guarantee (a) every row lands in exactly one bucket
and (b) all DAILY_COLUMNS are present.

Rebuild whenever dailyCRSP.parquet changes (the manifest fingerprints the bucket
directory, so factors depending on it are invalidated automatically).

Run: python p1_build_daily_buckets.py
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from p1_daily_kernels import BUCKET_DIR, DAILY_COLUMNS, N_BUCKETS
from p1_factor_engine import INTERMEDIATE_DIR

SOURCE = INTERMEDIATE_DIR / "dailyCRSP.parquet"


def build(source: Path = SOURCE, out_dir: Path = BUCKET_DIR, n_buckets: int = N_BUCKETS) -> None:
    t0 = time.time()
    if not source.exists():
        raise FileNotFoundError(source)
    src_rows = pq.ParquetFile(source).metadata.num_rows
    print(f"source {source.name}: {src_rows:,} rows")

    tmp = out_dir.with_name(out_dir.name + "_building")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    cols = ", ".join(DAILY_COLUMNS)
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(
        f"""
        COPY (
            SELECT {cols}, (hash(permno) % {n_buckets})::INTEGER AS bucket
            FROM read_parquet('{source.as_posix()}')
            ORDER BY bucket, permno, time_d
        )
        TO '{tmp.as_posix()}' (FORMAT PARQUET, PARTITION_BY (bucket), OVERWRITE_OR_IGNORE)
        """
    )
    con.close()

    # conservation check: every source row is in exactly one bucket
    out_rows = 0
    for b in range(n_buckets):
        part = tmp / f"bucket={b}"
        files = sorted(part.glob("*.parquet"))
        if not files:
            raise RuntimeError(f"bucket {b} empty after build — hash distribution broken?")
        out_rows += sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    if out_rows != src_rows:
        raise RuntimeError(f"row conservation failed: source {src_rows:,} vs buckets {out_rows:,}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    print(f"built {n_buckets} buckets, {out_rows:,} rows, in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    build()

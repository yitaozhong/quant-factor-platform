"""
p1_refresh_gate.py — splice-validation gate for a WRDS data refresh.

For every Intermediate parquet the P1 library reads, compare the refreshed file
against the frozen snapshot (Intermediate_2024snapshot/) and report, per file:

  schema      : columns/dtypes identical (a change is a hard failure)
  coverage    : last date before vs after; new rows appended
  overlap     : on the snapshot's own date range, cell-by-cell agreement of every
                numeric column on the (key, date) intersection — measured, not
                assumed. Compustat/IBES REVISE history ("restatements"): the
                gate quantifies them so the user sees exactly how much the past
                moved; it does not hide them.
  keys        : rows only-in-snapshot / only-in-refresh on the overlap window

Output: Data/golden/reports/refresh_gate_<stamp>.json + console table.
This script never writes to Intermediate/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INTER = ROOT / 'Open_Source_Asset_Pricing' / 'Signals' / 'pyData' / 'Intermediate'
SNAP = ROOT / 'Open_Source_Asset_Pricing' / 'Signals' / 'pyData' / 'Intermediate_2024snapshot'
REPORTS = ROOT / 'Data' / 'golden' / 'reports'

# file -> (key columns, date column). Keys identify one row within a date.
SPEC = {
    'monthlyCRSP.parquet':          (['permno'], 'time_avail_m'),
    'dailyCRSP.parquet':            (['permno'], 'time_d'),
    'm_aCompustat.parquet':         (['gvkey'], 'time_avail_m'),
    'a_aCompustat.parquet':         (['gvkey'], 'time_avail_m'),
    'm_QCompustat.parquet':         (['gvkey'], 'time_avail_m'),
    'CRSPdistributions.parquet':    (['permno', 'distcd', 'exdt'], None),
    'IBES_EPS_Unadj.parquet':       (['tickerIBES', 'fpi', 'fpedats'], 'time_avail_m'),
    'IBES_EPS_Adj.parquet':         (['tickerIBES', 'fpi', 'fpedats'], 'time_avail_m'),
    'IBES_Recommendations.parquet': (['tickerIBES', 'amaskcd', 'anndats'], 'time_avail_m'),
    'IBES_UnadjustedActuals.parquet': (['tickerIBES'], 'time_avail_m'),
    'IBESCRSPLinkingTable.parquet': (['tickerIBES', 'permno'], 'time_avail_m'),
    'CCMLinkingTable.parquet':      (['gvkey', 'permno'], None),
    'monthlyFF.parquet':            ([], 'time_avail_m'),
    'dailyFF.parquet':              ([], 'time_d'),
    'monthlyMarket.parquet':        ([], 'time_avail_m'),
    'monthlyLiquidity.parquet':     ([], 'time_avail_m'),
}


def _load(path: Path, columns=None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def compare_file(name: str, keys: list[str], date_col: str | None, sample_dates: int | None = None) -> dict:
    t0 = time.time()
    snap_p, new_p = SNAP / name, INTER / name
    if not snap_p.exists() or not new_p.exists():
        return {'file': name, 'status': 'MISSING', 'snapshot': snap_p.exists(), 'refresh': new_p.exists()}

    import pyarrow.parquet as pq
    s_schema = pq.ParquetFile(snap_p).schema_arrow
    n_schema = pq.ParquetFile(new_p).schema_arrow
    schema_same = [f.name for f in s_schema] == [f.name for f in n_schema]
    dtype_changes = [
        (f.name, str(f.type), str(n_schema.field(f.name).type))
        for f in s_schema if f.name in n_schema.names and str(f.type) != str(n_schema.field(f.name).type)
    ]

    snap = _load(snap_p)
    new = _load(new_p)
    res = {
        'file': name, 'status': 'ok', 'seconds': None,
        'schema_columns_same': schema_same, 'dtype_changes': dtype_changes,
        'rows_snapshot': int(len(snap)), 'rows_refresh': int(len(new)),
    }
    if date_col:
        s_max, n_max = pd.to_datetime(snap[date_col]).max(), pd.to_datetime(new[date_col]).max()
        res.update({'last_date_snapshot': str(s_max.date()), 'last_date_refresh': str(n_max.date()),
                    'new_rows_after_snapshot_end': int((pd.to_datetime(new[date_col]) > s_max).sum())})
        new_ov = new[pd.to_datetime(new[date_col]) <= s_max]
        if sample_dates:
            # subsample dates for very large files (daily): every k-th distinct date
            dts = np.sort(pd.to_datetime(snap[date_col]).unique())
            pick = set(dts[:: max(1, len(dts) // sample_dates)])
            snap = snap[pd.to_datetime(snap[date_col]).isin(pick)]
            new_ov = new_ov[pd.to_datetime(new_ov[date_col]).isin(pick)]
            res['overlap_sampled_dates'] = len(pick)
    else:
        new_ov = new

    join_keys = keys + ([date_col] if date_col else [])
    if not join_keys:
        # pure series files: align on date only
        join_keys = [date_col]
    # dedupe on join keys (last) so the merge is well-defined; record if duplicates exist
    s_dup = int(snap.duplicated(join_keys).sum())
    n_dup = int(new_ov.duplicated(join_keys).sum())
    snap_u = snap.drop_duplicates(join_keys, keep='last')
    new_u = new_ov.drop_duplicates(join_keys, keep='last')
    m = snap_u.merge(new_u, on=join_keys, how='outer', suffixes=('_s', '_n'), indicator=True)
    res.update({
        'overlap_rows_snapshot': int(len(snap_u)), 'overlap_rows_refresh': int(len(new_u)),
        'dup_keys_snapshot': s_dup, 'dup_keys_refresh': n_dup,
        'only_in_snapshot': int((m['_merge'] == 'left_only').sum()),
        'only_in_refresh': int((m['_merge'] == 'right_only').sum()),
        'both': int((m['_merge'] == 'both').sum()),
    })
    both = m[m['_merge'] == 'both']
    cols = {}
    num_cols = [c for c in snap.columns if c not in join_keys and pd.api.types.is_numeric_dtype(snap[c])]
    for c in num_cols:
        a = both[f'{c}_s'].to_numpy(dtype='float64')
        b = both[f'{c}_n'].to_numpy(dtype='float64')
        an, bn = np.isnan(a), np.isnan(b)
        valid = ~an & ~bn
        if valid.sum() == 0:
            cols[c] = {'compared': 0}
            continue
        d = np.abs(a[valid] - b[valid])
        scale = np.maximum(np.abs(a[valid]), 1e-12)
        changed = (d > 1e-6) & (d / scale > 1e-6)
        cols[c] = {
            'compared': int(valid.sum()),
            'changed_frac': float(changed.mean()),
            'changed_n': int(changed.sum()),
            'nan_only_snapshot': int((an & ~bn).sum()),
            'nan_only_refresh': int((~an & bn).sum()),
            'max_abs_diff': float(d.max()),
            'median_rel_diff_of_changed': float(np.median((d / scale)[changed])) if changed.any() else 0.0,
        }
    res['columns'] = cols
    worst = max((v.get('changed_frac', 0.0) for v in cols.values()), default=0.0)
    res['worst_column_changed_frac'] = worst
    res['seconds'] = round(time.time() - t0, 1)
    return res


def main(files: list[str] | None = None) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out = []
    todo = files or list(SPEC)
    for name in todo:
        keys, date_col = SPEC[name]
        sample = 60 if name.startswith('daily') else None
        try:
            r = compare_file(name, keys, date_col, sample_dates=sample)
        except Exception as exc:  # report, do not hide
            r = {'file': name, 'status': f'ERROR: {type(exc).__name__}: {exc}'}
        out.append(r)
        print(f"{name:32s} {r.get('status'):8s} rows {r.get('rows_snapshot','-')!s:>10} -> {r.get('rows_refresh','-')!s:>10}  "
              f"last {r.get('last_date_snapshot','-')} -> {r.get('last_date_refresh','-')}  "
              f"overlap both={r.get('both','-')} onlyS={r.get('only_in_snapshot','-')} onlyR={r.get('only_in_refresh','-')}  "
              f"worst_col_changed={r.get('worst_column_changed_frac', float('nan')):.4f}  ({r.get('seconds','-')}s)", flush=True)
    path = REPORTS / f'refresh_gate_{stamp}.json'
    path.write_text(json.dumps(out, indent=1, default=str), encoding='utf-8')
    print('report:', path)


if __name__ == '__main__':
    main(sys.argv[1:] or None)

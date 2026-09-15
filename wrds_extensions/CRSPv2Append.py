# ABOUTME: Appends post-2024 CRSP months from the CIZ-format tables (msf_v2/dsf_v2) onto the
# ABOUTME: legacy-format monthlyCRSP/dailyCRSP parquets, mapping v2 codes to legacy shrcd/exchcd.
"""
Background
----------
CRSP retired the legacy stock-file format (crsp.msf / crsp.dsf) at 2024-12-31 and
continues only the CIZ format (crsp.msf_v2 / crsp.dsf_v2), which differs in:
  * column names (mthret/mthprc/... vs ret/prc/...)
  * classification: sharetype/securitytype/usincflg/issuertype/primaryexch
    instead of numeric shrcd/exchcd
  * delisting returns are folded into mthret/dlyret natively (legacy kept a
    separate msedelist table and the OSAP CRSPMonthly.py applies Shumway-style
    imputation, -35%/-55%, when dlret is missing)
  * a revised return methodology: on the 2024 overlap ~6% of common-stock monthly
    returns differ by ~0.03pp (prices identical) — a CRSP methodology change

Design (measured 2026-08-17, see p1_deviation_register.md "CRSP CIZ append"):
  * Legacy files remain the source of truth through LEGACY_END (2024-12); this
    script never touches those rows, so every validated golden stays reproducible.
  * Rows strictly after LEGACY_END are appended from *_v2 with the code mappings
    below, which were verified one-to-one on 2024-06 (exchcd<->primaryexch exact;
    shrcd digits reconstructed from sharetype + securitytype/usincflg/issuertype).
  * Legacy delisting imputation is NOT re-applied to v2 rows: v2 mthret already
    carries the delisting return, and the imputation branch fired only 2 times in
    2020-2024 legacy data.

Inputs : ../pyData/Intermediate/monthlyCRSP.parquet, dailyCRSP.parquet, dailyCRSPprc.parquet
         (legacy-format, produced by CRSPMonthly.py / CRSPDaily.py)
Outputs: same files, extended through the latest v2 month
Run    : python3 DataDownloads/CRSPv2Append.py   (from pyCode/, .env present)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / '.env')
from wrds_connection import create_wrds_engine  # noqa: E402

INTER = Path(__file__).resolve().parents[2] / 'pyData' / 'Intermediate'
LEGACY_END = pd.Timestamp('2024-12-31')

EXCH_MAP = {'N': 1, 'A': 2, 'Q': 3, 'R': 4, 'B': 5, 'X': 0}
FIRST_DIGIT = {'NS': 1, 'AD': 3, 'SB': 4, 'UG': 7}


def shrcd_from_v2(sharetype, securitytype, usincflg, issuertype):
    """Reconstruct legacy shrcd = 10*first + second from v2 classification.

    second digit: 1 ordinary US common, 2 foreign-incorporated, 4 closed-end
    fund, 8 REIT (issuertype REIT), 5 foreign fund. Verified against the
    2024-06 legacy/v2 crosstab (11 <-> EQTY/NS/Y/CORP 3929 rows, 12 <->
    EQTY/NS/N/CORP, 18 <-> REIT, 31 <-> AD, 44/73 <-> FUND/SB|NS ...).
    Unmappable combinations get NaN — never a guessed code.
    """
    first = FIRST_DIGIT.get(sharetype)
    if first is None:
        return np.nan
    if issuertype == 'REIT':
        second = 8
    elif securitytype == 'FUND':
        second = 4 if usincflg == 'Y' else 5
        if sharetype == 'NS':
            # legacy: 73 for open-end/ETF units (ACOR issuer), 14 for others
            first, second = (7, 3) if issuertype == 'ACOR' and first == 1 else (first, second)
    elif usincflg == 'Y':
        second = 1
    elif usincflg == 'N':
        second = 2
    else:
        return np.nan
    return 10 * first + second


def fetch(engine, sql):
    with engine.connect() as con:
        return pd.read_sql_query(text(sql), con)


def append_monthly(engine):
    path = INTER / 'monthlyCRSP.parquet'
    legacy = pd.read_parquet(path)
    legacy = legacy[legacy['time_avail_m'] <= LEGACY_END]
    print(f'legacy monthlyCRSP: {len(legacy):,} rows through {legacy["time_avail_m"].max().date()}')

    v2 = fetch(engine, f"""
        SELECT permno, permco, mthcaldt, mthret, mthretx, mthvol, shrout, mthprc, mthcumfacshr,
               siccd, ticker, sharetype, securitytype, usincflg, issuertype, primaryexch
        FROM crsp.msf_v2
        WHERE mthcaldt > '{LEGACY_END.date()}'
    """)
    print(f'v2 rows after {LEGACY_END.date()}: {len(v2):,} through {pd.to_datetime(v2["mthcaldt"]).max().date()}')
    if v2.empty:
        raise RuntimeError('no post-legacy v2 monthly rows — nothing to append')

    out = pd.DataFrame({
        'permno': v2['permno'].astype('Int64'),
        'permco': v2['permco'].astype('Int64'),
        'ret': v2['mthret'].astype('float64'),
        'retx': v2['mthretx'].astype('float64'),
        # legacy crsp.msf.vol is in HUNDREDS of shares and CRSPMonthly.py divides by
        # 10,000 -> millions of shares. crsp.msf_v2.mthvol is in SHARES (verified
        # AAPL 2024-06: msf vol 16,973,587 vs msf_v2 mthvol 1,697,358,745), so the
        # equivalent scaling is /1e6. (dsf and dsf_v2 daily vol are both in shares.)
        'vol': v2['mthvol'].astype('float64') / 1_000_000.0,
        'shrout': v2['shrout'].astype('float64') / 1000.0,    # both in thousands -> legacy /1000
        'prc': v2['mthprc'].astype('float64'),
        'cfacshr': v2['mthcumfacshr'].astype('float64'),
        'bidlo': np.nan, 'askhi': np.nan,
        'shrcd': [shrcd_from_v2(a, b, c, d) for a, b, c, d in
                  zip(v2['sharetype'], v2['securitytype'], v2['usincflg'], v2['issuertype'])],
        'exchcd': v2['primaryexch'].map(EXCH_MAP).astype('float64'),
        'ticker': v2['ticker'].fillna(''),
        'shrcls': '',
        'sicCRSP': pd.to_numeric(v2['siccd'], errors='coerce'),
        'time_avail_m': pd.to_datetime(v2['mthcaldt']).dt.to_period('M').dt.to_timestamp(),
    })
    out['sic2D'] = pd.to_numeric(out['sicCRSP'].astype('Int64').astype(str).str[:2], errors='coerce')
    out['ret_b4_dl'] = out['ret']       # v2 ret already includes delisting return
    out['mve_c'] = out['shrout'] * out['prc'].abs()
    mve_permco = (out.dropna(subset=['permco'])
                  .groupby(['permco', 'time_avail_m'], as_index=False)['mve_c']
                  .sum(min_count=1).rename(columns={'mve_c': 'mve_permco'}))
    out = out.merge(mve_permco, on=['permco', 'time_avail_m'], how='left')

    unmapped = out['shrcd'].isna().mean()
    print(f'v2->legacy shrcd unmapped share: {unmapped:.4%}; exchcd unmapped: {out["exchcd"].isna().mean():.4%}')
    if unmapped > 0.02:
        raise RuntimeError(f'shrcd mapping left {unmapped:.2%} rows unmapped — extend shrcd_from_v2 before appending')

    missing_cols = set(legacy.columns) - set(out.columns)
    extra_cols = set(out.columns) - set(legacy.columns)
    if missing_cols or extra_cols:
        raise RuntimeError(f'schema mismatch — missing {sorted(missing_cols)}, extra {sorted(extra_cols)}')
    out = out[legacy.columns].astype({c: legacy[c].dtype for c in legacy.columns if str(legacy[c].dtype) != 'object'}, errors='ignore')

    combined = pd.concat([legacy, out], ignore_index=True)
    combined.to_parquet(path, index=False)
    print(f'monthlyCRSP.parquet -> {len(combined):,} rows through {combined["time_avail_m"].max().date()}')
    return combined['time_avail_m'].max()


def append_daily(engine):
    path_full = INTER / 'dailyCRSP.parquet'
    path_prc = INTER / 'dailyCRSPprc.parquet'
    legacy = pd.read_parquet(path_full)
    legacy = legacy[legacy['time_d'] <= LEGACY_END]
    print(f'legacy dailyCRSP: {len(legacy):,} rows through {legacy["time_d"].max().date()}')

    v2 = fetch(engine, f"""
        SELECT permno, dlycaldt, dlyret, dlyvol, dlyprc, dlycumfacpr, shrout
        FROM crsp.dsf_v2
        WHERE dlycaldt > '{LEGACY_END.date()}'
    """)
    print(f'v2 daily rows after {LEGACY_END.date()}: {len(v2):,} through {pd.to_datetime(v2["dlycaldt"]).max().date()}')
    if v2.empty:
        raise RuntimeError('no post-legacy v2 daily rows — nothing to append')

    out = pd.DataFrame({
        'permno': v2['permno'].astype('int32'),
        'time_d': pd.to_datetime(v2['dlycaldt']).dt.floor('D'),
        'ret': v2['dlyret'].astype('float64'),
        'vol': v2['dlyvol'].astype('float64'),
        'prc': v2['dlyprc'].astype('float64'),
        'cfacpr': v2['dlycumfacpr'].astype('float64'),
        'shrout': v2['shrout'].astype('float64'),
    })
    # legacy dsf: vol in shares (NOT scaled), shrout in thousands (NOT scaled) — verified from
    # CRSPDaily.py, which stores raw columns. dsf_v2 shrout is also in thousands.
    out = out[legacy.columns]
    combined = pd.concat([legacy, out], ignore_index=True)
    combined.to_parquet(path_full, index=False)
    combined[['permno', 'time_d', 'prc', 'cfacpr', 'shrout']].to_parquet(path_prc, index=False)
    print(f'dailyCRSP.parquet -> {len(combined):,} rows through {combined["time_d"].max().date()}')


if __name__ == '__main__':
    engine = create_wrds_engine()
    try:
        append_monthly(engine)
        append_daily(engine)
    finally:
        engine.dispose()
    print('CRSPv2Append completed')

# ABOUTME: Appends post-2024 rows to monthlyMarket.parquet, reconstructing CRSP's vwretd/ewretd/usdval
# ABOUTME: from stock-level crsp.msf_v2 because the legacy market index table (crsp.msi) is frozen at 2024-12.
"""
Validation (2026-08-17, on 2022-01..2024-12 where both exist): reconstructing the index from
ALL securities in the legacy stock file (value weights = prior month-end |prc|*shrout) matches
the published crsp.msi with corr 0.99985 (vw) / 0.99985 (ew), max |diff| 0.0026 (vw) /
0.0044 (ew); usdval reconstruction / published median ratio 0.958. The residual is CRSP's
internal weighting detail (delisting-month treatment); the appended 2025 values therefore
carry a documented ~0.3% return-level uncertainty. Only three factors read this file
(Beta via ewretd/vwretd regressors, and the market-volatility factors); Beta's regressor is
a 60-month rolling estimate, insensitive at this magnitude. Registered in
p1_deviation_register.md section D.

Legacy rows through LEGACY_END are never touched.
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
LEGACY_END = pd.Timestamp('2024-12-01')


def main() -> None:
    path = INTER / 'monthlyMarket.parquet'
    legacy = pd.read_parquet(path)
    legacy = legacy[legacy['time_avail_m'] <= LEGACY_END]
    print(f'legacy monthlyMarket: {len(legacy)} rows through {legacy["time_avail_m"].max().date()}')

    engine = create_wrds_engine()
    try:
        with engine.connect() as con:
            # need Dec-2024 caps for Jan-2025 weights: pull from 2024-11 onward
            v2 = pd.read_sql_query(text("""
                SELECT permno, mthcaldt, mthret, mthprc, shrout
                FROM crsp.msf_v2 WHERE mthcaldt >= '2024-11-01'
            """), con)
    finally:
        engine.dispose()

    v2['ym'] = pd.to_datetime(v2['mthcaldt']).dt.to_period('M')
    v2['cap'] = v2['mthprc'].abs() * v2['shrout']          # $ thousands (shrout in 000s)
    v2 = v2.sort_values(['permno', 'ym'])
    v2['cap_lag'] = v2.groupby('permno')['cap'].shift(1)
    valid = v2.dropna(subset=['mthret', 'cap_lag'])
    vw = valid.groupby('ym').apply(lambda g: np.average(g['mthret'], weights=g['cap_lag']), include_groups=False)
    ew = valid.groupby('ym')['mthret'].mean()
    tot = v2.groupby('ym')['cap'].sum()
    out = pd.DataFrame({'vwretd': vw, 'ewretd': ew, 'usdval': tot}).reset_index()
    out['time_avail_m'] = out['ym'].dt.to_timestamp()
    out = out[out['time_avail_m'] > LEGACY_END][['vwretd', 'ewretd', 'usdval', 'time_avail_m']]
    if out.empty:
        raise RuntimeError('no post-legacy months reconstructed')
    print(f'appending {len(out)} months: {out["time_avail_m"].min().date()} .. {out["time_avail_m"].max().date()}')
    print(out.round(4).to_string(index=False))

    combined = pd.concat([legacy, out[legacy.columns]], ignore_index=True)
    combined.to_parquet(path, index=False)
    print(f'monthlyMarket.parquet -> {len(combined)} rows through {combined["time_avail_m"].max().date()}')


if __name__ == '__main__':
    main()

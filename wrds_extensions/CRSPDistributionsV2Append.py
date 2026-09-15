# ABOUTME: Appends post-2024 CRSP distributions from the CIZ table crsp.stkdistributions onto the
# ABOUTME: legacy CRSPdistributions.parquet, reconstructing the 4-digit distcd from v2 classification fields.
"""
crsp.msedist (legacy) is frozen at exdt 2024-12-31; crsp.stkdistributions continues (2025-12-31).
The P1 dividend factors (DivInit, DivOmit, DivSeason, DivYieldST) read divamt, exdt, distcd and
its digits cd1..cd3. Crosswalk measured on 2023-01..2024-12 (45,099 legacy/v2 rows joined on
permno+exdt+amount):
    cd1 (type):       CD->1 cash dividend, SD->1 (stock-dividend rows carry cd1=1 in legacy, e.g.
                      1272/1274), ROC->1, CG->2 (capital gains, 22xx), CP->2/3, SP->3/4 (spinoff/
                      security dist.), other -> reconstructed conservatively (see MAP)
    cd2 (payment):    USD ordinary->2, FX->3, OS (other security)->7, SS->5
    cd3 (frequency):  U->1, M->2, Q->3, S->4, A->5, Y->6, E (special/extra)->7 (crosstab-verified)
    cd4 (tax status): NOT reconstructible one-to-one (9/39 codes ambiguous); no P1 factor reads
                      cd4, so it is set to the modal legacy value per (cd1,cd2,cd3) tuple.
Legacy rows are untouched. Registered in p1_deviation_register.md section D.
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

CD1 = {'CD': 1, 'SD': 1, 'ROC': 1, 'CG': 2, 'CP': 2, 'SP': 3, 'LQ': 3, 'RI': 4, 'RD': 4,
       'FRS': 5, 'N/A': 4}                       # FRS forward split -> 55xx; N/A/X -> 4999
CD2 = {'USD': 2, 'FX': 3, 'OS': 7, 'SS': 5, 'N/A': 2, 'OP': 8, 'X': 9}
# cd3 (legacy frequency digit) from disfreqtype — crosstab on 2022-24 regular cash rows:
#   U->1 (924), M->2 (29,335), Q->3 (29,718), S->4 (811), A->5 (832), Y->6 (849), E->7 (996)
# (an earlier draft had M/Q swapped, collapsing DivSeason 2025 coverage to 16% — caught by the
#  refresh factor gate.)
CD3 = {'U': 1, 'M': 2, 'Q': 3, 'S': 4, 'A': 5, 'Y': 6, 'E': 7, 'N/A': 2, 'N': 9}
# measured on 2020-24 overlap: FRS/SS -> 5523 (1917 rows), SP/OP -> 3888 (79), N/A/X -> 4999,
# CD/USD/N -> 1292 (12). None of these are cash-dividend rows read by DivInit/DivOmit/DivSeason/
# DivYieldST except CD/USD/N (cd1=1,cd2=2, cd3=9), which is mapped exactly.
# per-(cd1,cd2,cd3) modal cd4 from the legacy overlap; default 2 (taxable) if unseen
CD4_DEFAULT = 2


def main() -> None:
    path = INTER / 'CRSPdistributions.parquet'
    legacy = pd.read_parquet(path)
    legacy = legacy[legacy['exdt'] <= LEGACY_END]
    print(f'legacy CRSPdistributions: {len(legacy):,} rows, max exdt {legacy["exdt"].max().date()}')

    # modal cd4 per (cd1,cd2,cd3) tuple from recent legacy rows
    recent = legacy[legacy['exdt'] >= '2015-01-01']
    cd4_mode = recent.groupby(['cd1', 'cd2', 'cd3'])['cd4'].agg(lambda s: int(s.mode().iloc[0]))

    engine = create_wrds_engine()
    try:
        with engine.connect() as con:
            v2 = pd.read_sql_query(text(f"""
                SELECT permno, disexdt, distype, disfreqtype, dispaymenttype, disdetailtype,
                       disdivamt, disfacshr, disrecorddt, dispaydt
                FROM crsp.stkdistributions WHERE disexdt > '{LEGACY_END.date()}'
            """), con)
    finally:
        engine.dispose()
    print(f'v2 rows after {LEGACY_END.date()}: {len(v2):,}, max disexdt {pd.to_datetime(v2["disexdt"]).max().date()}')

    cd1 = v2['distype'].map(CD1)
    cd2 = v2['dispaymenttype'].map(CD2)
    cd3 = v2['disfreqtype'].map(CD3)
    unmapped = cd1.isna() | cd2.isna() | cd3.isna()
    if unmapped.mean() > 0.01:
        raise RuntimeError(f'{unmapped.mean():.2%} v2 distribution rows unmapped — extend crosswalk: '
                           f'{v2.loc[unmapped, ["distype", "dispaymenttype", "disfreqtype"]].drop_duplicates().head(10).to_dict("records")}')
    print(f'unmapped v2 rows dropped: {int(unmapped.sum())} ({unmapped.mean():.3%})')
    v2 = v2[~unmapped].copy()
    cd1, cd2, cd3 = cd1[~unmapped].astype(int), cd2[~unmapped].astype(int), cd3[~unmapped].astype(int)
    keys = list(zip(cd1, cd2, cd3))
    cd4 = np.array([cd4_mode.get(k, CD4_DEFAULT) for k in keys], dtype=int)

    out = pd.DataFrame({
        'permno': v2['permno'].astype(legacy['permno'].dtype),
        'divamt': v2['disdivamt'].astype('float64'),
        'distcd': (cd1.to_numpy() * 1000 + cd2.to_numpy() * 100 + cd3.to_numpy() * 10 + cd4).astype(legacy['distcd'].dtype),
        'facshr': v2['disfacshr'].astype('float64'),
        'rcrddt': pd.to_datetime(v2['disrecorddt']),
        'exdt': pd.to_datetime(v2['disexdt']),
        'paydt': pd.to_datetime(v2['dispaydt']),
        'cd1': cd1.to_numpy(), 'cd2': cd2.to_numpy(), 'cd3': cd3.to_numpy(), 'cd4': cd4,
    })
    for c in ('cd1', 'cd2', 'cd3', 'cd4'):
        out[c] = out[c].astype(legacy[c].dtype)
    id_cols = ['permno', 'rcrddt', 'exdt', 'paydt', 'distcd']
    out = out.sort_values(id_cols).drop_duplicates(id_cols, keep='first')
    missing = set(legacy.columns) - set(out.columns)
    extra = set(out.columns) - set(legacy.columns)
    if missing or extra:
        raise RuntimeError(f'schema mismatch: missing {sorted(missing)} extra {sorted(extra)}')

    print('appended distcd distribution:', out['distcd'].value_counts().head(8).to_dict())
    combined = pd.concat([legacy, out[legacy.columns]], ignore_index=True)
    combined.to_parquet(path)
    print(f'CRSPdistributions.parquet -> {len(combined):,} rows, max exdt {combined["exdt"].max().date()}')


if __name__ == '__main__':
    main()

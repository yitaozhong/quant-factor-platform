# WRDS download extensions (CRSP CIZ-format adapters)

CRSP retired its legacy stock-file format at 2024-12-31: the WRDS tables
`crsp.msf`, `crsp.dsf`, `crsp.msi`, and `crsp.msedist` are frozen at that date,
and coverage continues only in the CIZ-format tables (`crsp.msf_v2`,
`crsp.dsf_v2`, `crsp.stkdistributions`), which use different column names,
different classification codes, and a revised return methodology.

The three scripts here extend the [Open Source Asset Pricing](https://www.openassetpricing.com/)
download pipeline (`Signals/pyCode/DataDownloads/`) past that freeze. Each one
keeps the legacy download as the untouched source of truth through 2024-12 and
appends later months from the CIZ tables, with every mapping **measured on the
overlap period** rather than assumed:

| Script | Extends | Key mapping (verified on 2022–24 overlap) |
| --- | --- | --- |
| `CRSPv2Append.py` | monthly/daily stock files | `exchcd`↔`primaryexch` one-to-one; `shrcd` reconstructed from `sharetype`×`issuertype`×`usincflg` (0.0000% unmapped); volume unit fix — `mthvol` is in shares, legacy `vol` in hundreds |
| `MarketReturnsV2Append.py` | market index (vwretd/ewretd/usdval) | reconstructed from stock level; corr 0.99985 vs the published index, max diff 26 bp |
| `CRSPDistributionsV2Append.py` | distributions (dividends/splits) | 4-digit `distcd` rebuilt from `distype`/`dispaymenttype`/`disfreqtype` crosstabs (e.g. freq M→2, Q→3); 0 of 37k rows unmapped |

Run order matters (the IBES link expands ranges against the CRSP calendar):

```
CRSPMonthly.py → CRSPDaily.py → CRSPv2Append.py → MarketReturnsV2Append.py
→ CRSPDistributionsV2Append.py → IBESCRSPLink.py → SignalMasterTable.py → factor build
```

All scripts read WRDS credentials from a `.env` file (`WRDS_USERNAME`,
`WRDS_PASSWORD`) that is never committed. They are drop-ins for the OSAP
`DataDownloads/` folder; the OSAP repository itself is not included here.

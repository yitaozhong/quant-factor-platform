"""
p1_refresh_factor_gate.py — compare the rebuilt factor library against the pre-refresh
snapshot on their common (month, permno) grid.

After a data refresh the two libraries have different shapes (new months, new
permnos), so the standard full-grid harness cannot be used directly. This gate:

  1. restricts both panels to the intersection of months and permnos
     (the snapshot's own 1925-12..2024-12 x 38,843 grid),
  2. runs the same NaN-aware cell diff (p1_validation.diff_panels) at zero tolerance,
  3. classifies each factor:
        IDENTICAL       — 0 changed cells (data it reads was not restated)
        RESTATED        — changed cells, but concentrated in the last N years of the
                          overlap (Compustat/IBES restatements only ever touch recent
                          fiscal years) or of tiny magnitude
        SUSPECT         — changes deep in history (before restatement horizon) or
                          large mass changes: must be explained before acceptance
  4. reports new-month coverage (2025) per factor so an empty 2025 is caught.

Output: Data/golden/reports/refresh_factor_gate.csv (+ console summary).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from p1_validation import diff_panels

ROOT = Path(__file__).resolve().parent
NEW = ROOT / "Data" / "P1_Parquet"
OLD = ROOT / "Data" / "P1_Parquet_2024snapshot"
OUT = ROOT / "Data" / "golden" / "reports" / "refresh_factor_gate.csv"

RESTATEMENT_HORIZON = pd.Timestamp("2019-01-01")   # changes before this are suspect
NEW_MONTHS_START = pd.Timestamp("2025-01-01")


def gate(factors: list[str] | None = None) -> pd.DataFrame:
    names = sorted(factors or [p.stem for p in OLD.glob("*.parquet") if not p.name.startswith("_")])
    rows = []
    for name in names:
        old = pd.read_parquet(OLD / f"{name}.parquet")
        new_path = NEW / f"{name}.parquet"
        if not new_path.exists():
            rows.append({"factor": name, "status": "MISSING_NEW"})
            continue
        new = pd.read_parquet(new_path)

        common_idx = old.index.intersection(new.index)
        common_col = old.columns.intersection(new.columns)
        o = old.loc[common_idx, common_col]
        n = new.loc[common_idx, common_col]
        rep = diff_panels(n, o, factor=name, atol=0.0, rtol=0.0)

        # where do the changed cells live in time?
        ov = o.to_numpy(dtype="float64"); nv = n.to_numpy(dtype="float64")
        on, nn = np.isnan(ov), np.isnan(nv)
        both = ~on & ~nn
        changed = np.zeros_like(both)
        changed[both] = ov[both] != nv[both]
        mask_changed = changed | (on ^ nn)
        per_month = mask_changed.sum(axis=1)
        months = common_idx
        n_changed = int(mask_changed.sum())
        early = int(per_month[months < RESTATEMENT_HORIZON].sum())
        first_change = str(months[per_month > 0].min().date()) if n_changed else ""
        # magnitude among jointly-valued changed cells
        d = np.abs(ov[changed] - nv[changed]) if changed.any() else np.array([])
        med_rel = float(np.median(d / np.maximum(np.abs(ov[changed]), 1e-12))) if changed.any() else 0.0

        # new-month coverage
        new_months = new.index[new.index >= NEW_MONTHS_START]
        new_cells = int(new.loc[new_months].notna().sum().sum()) if len(new_months) else 0
        old_last_year_cells = int(old.loc[old.index >= "2024-01-01"].notna().sum().sum())

        total_valued = int(both.sum())
        frac = n_changed / max(total_valued, 1)
        if n_changed == 0:
            status = "IDENTICAL"
        elif early == 0 and (frac < 0.05):
            status = "RESTATED"
        elif early > 0 and early / max(n_changed, 1) < 0.02 and frac < 0.05:
            status = "RESTATED_minor_early"
        else:
            status = "SUSPECT"

        rows.append({
            "factor": name, "status": status,
            "overlap_valued_cells": total_valued, "changed_cells": n_changed,
            "changed_frac": round(frac, 6), "changed_before_2019": early,
            "first_changed_month": first_change, "median_rel_change": round(med_rel, 6),
            "nan_only_new": rep.nan_only_in_candidate, "nan_only_old": rep.nan_only_in_golden,
            "cells_2024_old": old_last_year_cells, "cells_2025_new": new_cells,
        })
        print(f"{name:26s} {status:22s} changed {n_changed:>9,} ({frac:.4%})  early {early:>7,}  "
              f"first {first_change:10s}  2025 cells {new_cells:>9,}", flush=True)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("\nstatus counts:", df["status"].value_counts().to_dict())
    print("report:", OUT)
    return df


if __name__ == "__main__":
    gate(sys.argv[1:] or None)

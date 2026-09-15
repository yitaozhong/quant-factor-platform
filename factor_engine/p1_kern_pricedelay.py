"""Numba kernels for PriceDelaySlope / PriceDelayRsq / PriceDelayTstat.

Reference: Open_Source_Asset_Pricing/Signals/pyCode/Predictors/
ZZ2_PriceDelaySlope_PriceDelayRsq_PriceDelayTstat.py (Hou-Moskowitz 2005 price
delay, OSAP implementation).

Spec-vs-script verification notes (the task brief said "weekly returns" — that
is WRONG for this reference; the OSAP script is DAILY, and the spec entries in
p1_daily_kernel_specs.json agree with the script):

* Week construction: NONE. The script regresses DAILY excess returns inside
  calendar July(y-1)-June(y) annual buckets, one regression per
  (permno, bucket). No resampling, no compounding — verified L97-106.
* Market series: dailyFF mktrf used as-is; mktLag1..4 = mktrf.shift(n) built on
  the FULL FF trading calendar BEFORE the join (L52-57), so lags are market-
  calendar lags independent of the stock's missing days. Verified dailyFF has
  zero null/NaN mktrf/rf; the only null lags are the first 4 FF days
  (1926-07-01..07-07). Stock excess return y = ret - rf (L72). The
  CRSP-to-FF join is INNER on time_d (L69): stock days absent from the FF
  calendar do not exist for any count or endpoint check.
* Window/restriction: per (permno, bucket) require raw rows >= 26 AND
  non-null ret >= 26 AND non-null mktrf >= 26 AND var(ret)>0 AND var(mktrf)>0
  (ddof=1 over non-null; polars var matches exact two-pass — verified
  empirically, constant series give exactly 0.0), AND the group's last joined
  trading day falls in June (L130-138, L248-257).
* R2 definitions: polars_ols mode="statistics" r2 is the standard CENTERED
  R^2 = 1 - SSE/SST(y - ybar) — verified empirically to 1e-16 against numpy.
  Restricted model: y ~ const + mktrf. Unrestricted: y ~ const + mktrf +
  mktLag1..4. PriceDelayRsq = 1 - R2_restricted / R2_unrestricted.
* Tstat formula: t_j = b_j / se_j, se_j = sqrt(sigma2 * diag((X'X)^-1)),
  sigma2 = SSE / (n - 6) — verified empirically (dof = n minus ALL columns
  including intercept). t order [mktrf, mktLag1..4, const].
  PriceDelayTstat = (1*t_1 + 2*t_2 + 3*t_3 + 4*t_4) / (t_0 + t_1 + .. + t_4).
  PriceDelaySlope is the same ratio over the b's.
* Output stamping: bucket keyed June y is stamped time_avail_m = July y
  (June + 1mo, L301-303), then forward-filled on a monthly grid per permno from
  the FIRST July stamp to the LAST July stamp: each annual value covers its
  July through the month before the next stamp (missing years fill straight
  across); the last annual value covers only its own July.

BIT-CONSTANT-GROUP GATE (root cause of the one full-panel divergence found and
fixed, 2026-08-01): a stale/non-trading stock can have ret == 0.0 on every
joined day of a July-June bucket while daily rf is bit-constant across the
window, making y = ret - rf bit-identical on ALL valid rows (22 such groups
across 21 permnos, e.g. permno 48980 June-1977, 12233 June-1990, all with
ret == 0.0 all year). In exact arithmetic var(y) = 0, and polars' grouped var
returns EXACTLY 0.0 for bit-identical input (verified at every tested frame
offset), so the reference dropped every one of these groups — the oracle's
cells over those windows are forward-fills from each permno's previous valid
July stamp, never fresh regression values. A naive exact two-pass ss > 0 gate
does NOT reproduce that drop: on bit-constant input the naive-summation mean
is off by ~1ulp, giving ss ~1e-39 > 0, so the group wrongly survives and the
kernel publishes a regression on constant y — pure linear-algebra noise
(R^2 negative, betas ~1e-18, slope/tstat = ratios of dust that differ across
implementations and even across frame layouts of the same implementation).
Hence the gate below also requires min < max over the valid values (variance
exactly 0 <=> all values bit-equal): bit-constant groups are dropped exactly
like the reference, the previous July's value fills across, and the full-panel
diff is clean without publishing noise.

Degenerate-branch pinning (empirical, this polars_ols build): mode="statistics"
computes (X'X)^-1 via CHOLESKY regardless of solve_method="svd" and PANICS the
whole script on: n_valid <= k (dof <= 0), singular X'X, all rows dropped, or
NaN (non-null) inputs. The reference run completed, therefore NO surviving
group in the real data hits those branches — this kernel raises loudly if one
ever appears instead of silently emitting anything (per the surface-bugs
convention). Perfect fits do NOT panic: SSE is fp dust > 0, t's are huge finite
numbers; their weighted ratio is scale-stable (sigma cancels), so fidelity
survives. inf/NaN produced by the final ratios (zero denominators, 0/0) are
KEPT, exactly as the reference keeps them through save_predictor.
"""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd

from p1_daily_kernels import _month_codes, _stock_offsets, iter_buckets, load_daily_ff

NLAG = 4
MIN_OBS = 26  # raw rows AND non-null ret AND non-null mktrf, all >= 26 (script L130-138)

ORACLE_DIR = "Data/golden/oracle"


@numba.njit(cache=True)
def _june_key(month_code: np.int64) -> np.int64:
    """Month code -> month code of the June that keys its July-June bucket.

    Script L97-106: time_avail_m = June of year(month + 6mo). Floor division
    handles pre-1970 (negative) codes exactly like polars' calendar arithmetic.
    """

    return ((month_code + 6) // 12) * 12 + 5


@numba.njit(cache=True, error_model="numpy")
def _price_delay_stock(
    y: np.ndarray,
    mkt: np.ndarray,
    l1: np.ndarray,
    l2: np.ndarray,
    l3: np.ndarray,
    l4: np.ndarray,
    months: np.ndarray,
    min_obs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One stock's joined daily history -> (July stamp code, slope, rsq, tstat).

    Inputs are the post-inner-join rows (days absent from the FF calendar were
    dropped by the caller), sorted by time_d. NaN in ``y`` == null ret
    (dailyCRSP has zero float-NaN rets — verified). ``mkt`` is never NaN;
    l1..l4 are NaN only for the first 4 FF days ever.

    error_model="numpy": the factor ratios must yield IEEE inf/NaN on zero
    denominators (the reference keeps them), not raise ZeroDivisionError.
    """

    n = y.shape[0]
    out_stamp = np.empty(n, dtype=np.int64)
    out_slope = np.empty(n, dtype=np.float64)
    out_rsq = np.empty(n, dtype=np.float64)
    out_tstat = np.empty(n, dtype=np.float64)
    n_out = 0

    i = 0
    while i < n:
        jk = _june_key(months[i])
        j = i + 1
        while j < n and _june_key(months[j]) == jk:
            j += 1
        # group rows [i, j): one July-June bucket.
        # Quality gates (script L130-138) + June endpoint (L248-257). The
        # script filters the endpoint AFTER regressing, but both are pure row
        # filters so the surviving set is identical either way.
        if j - i >= min_obs and months[j - 1] % 12 == 5:
            cnt_y = 0
            sum_y = 0.0
            min_y = np.inf
            max_y = -np.inf
            cnt_m = 0
            sum_m = 0.0
            min_m = np.inf
            max_m = -np.inf
            for k in range(i, j):
                if not np.isnan(y[k]):
                    cnt_y += 1
                    sum_y += y[k]
                    if y[k] < min_y:
                        min_y = y[k]
                    if y[k] > max_y:
                        max_y = y[k]
                if not np.isnan(mkt[k]):
                    cnt_m += 1
                    sum_m += mkt[k]
                    if mkt[k] < min_m:
                        min_m = mkt[k]
                    if mkt[k] > max_m:
                        max_m = mkt[k]
            if cnt_y >= min_obs and cnt_m >= min_obs:
                # exact two-pass sums of squares; var>0 <=> ss>0 (n-1 > 0).
                # The var>0 gate is evaluated in EXACT arithmetic: a group whose
                # valid values are bit-identical has variance exactly 0 and is
                # dropped (min==max check). Naive two-pass "dust" (mean rounding
                # makes ss ~1e-39 > 0 on constant input) must NOT pass the gate:
                # the reference's own grouped-var kernel returns exactly 0.0 for
                # such groups when re-run today, and the handful of constant-ret
                # groups the ORACLE kept were execution-context noise in the
                # original run (see module docstring, registered deviation).
                mean_y = sum_y / cnt_y
                mean_m = sum_m / cnt_m
                ss_y = 0.0
                ss_m = 0.0
                for k in range(i, j):
                    if not np.isnan(y[k]):
                        d = y[k] - mean_y
                        ss_y += d * d
                    if not np.isnan(mkt[k]):
                        d = mkt[k] - mean_m
                        ss_m += d * d
                if ss_y > 0.0 and ss_m > 0.0 and min_y < max_y and min_m < max_m:
                    # ---- restricted OLS: y ~ const + mktrf, drop rows with
                    # null y or mktrf (null_policy="drop")
                    n_r = 0
                    for k in range(i, j):
                        if not (np.isnan(y[k]) or np.isnan(mkt[k])):
                            n_r += 1
                    if n_r <= 2:
                        raise ValueError(
                            "PriceDelay: restricted regression has dof <= 0; the "
                            "reference polars_ols run would have panicked here — "
                            "input data no longer matches the oracle's"
                        )
                    Xr = np.empty((n_r, 2), dtype=np.float64)
                    yr = np.empty(n_r, dtype=np.float64)
                    r = 0
                    for k in range(i, j):
                        if not (np.isnan(y[k]) or np.isnan(mkt[k])):
                            Xr[r, 0] = mkt[k]
                            Xr[r, 1] = 1.0
                            yr[r] = y[k]
                            r += 1
                    br, _res_r, rank_r, _sv_r = np.linalg.lstsq(Xr, yr)
                    if rank_r < 2:
                        raise ValueError(
                            "PriceDelay: singular restricted design; the reference "
                            "statistics-mode Cholesky would have panicked — "
                            "input data no longer matches the oracle's"
                        )
                    mean_yr = 0.0
                    for k in range(n_r):
                        mean_yr += yr[k]
                    mean_yr /= n_r
                    sse_r = 0.0
                    sst_r = 0.0
                    for k in range(n_r):
                        e = yr[k] - (br[0] * Xr[k, 0] + br[1])
                        sse_r += e * e
                        d = yr[k] - mean_yr
                        sst_r += d * d
                    r2_r = 1.0 - sse_r / sst_r

                    # ---- unrestricted OLS: y ~ const + mktrf + mktLag1..4,
                    # drop rows with null y, mktrf, or ANY lag
                    n_u = 0
                    for k in range(i, j):
                        if not (
                            np.isnan(y[k])
                            or np.isnan(mkt[k])
                            or np.isnan(l1[k])
                            or np.isnan(l2[k])
                            or np.isnan(l3[k])
                            or np.isnan(l4[k])
                        ):
                            n_u += 1
                    if n_u <= 6:
                        raise ValueError(
                            "PriceDelay: unrestricted regression has dof <= 0; the "
                            "reference polars_ols run would have panicked here — "
                            "input data no longer matches the oracle's"
                        )
                    Xu = np.empty((n_u, 6), dtype=np.float64)
                    yu = np.empty(n_u, dtype=np.float64)
                    r = 0
                    for k in range(i, j):
                        if not (
                            np.isnan(y[k])
                            or np.isnan(mkt[k])
                            or np.isnan(l1[k])
                            or np.isnan(l2[k])
                            or np.isnan(l3[k])
                            or np.isnan(l4[k])
                        ):
                            Xu[r, 0] = mkt[k]
                            Xu[r, 1] = l1[k]
                            Xu[r, 2] = l2[k]
                            Xu[r, 3] = l3[k]
                            Xu[r, 4] = l4[k]
                            Xu[r, 5] = 1.0
                            yu[r] = y[k]
                            r += 1
                    bu, _res_u, rank_u, _sv_u = np.linalg.lstsq(Xu, yu)
                    if rank_u < 6:
                        raise ValueError(
                            "PriceDelay: singular unrestricted design; the reference "
                            "statistics-mode Cholesky would have panicked — "
                            "input data no longer matches the oracle's"
                        )
                    mean_yu = 0.0
                    for k in range(n_u):
                        mean_yu += yu[k]
                    mean_yu /= n_u
                    sse_u = 0.0
                    sst_u = 0.0
                    for k in range(n_u):
                        p = bu[5]
                        for c in range(5):
                            p += bu[c] * Xu[k, c]
                        e = yu[k] - p
                        sse_u += e * e
                        d = yu[k] - mean_yu
                        sst_u += d * d
                    r2_u = 1.0 - sse_u / sst_u
                    sigma2 = sse_u / (n_u - 6)
                    xtx = Xu.T @ Xu
                    xtx_inv = np.linalg.inv(xtx)
                    t0 = bu[0] / np.sqrt(sigma2 * xtx_inv[0, 0])
                    t1 = bu[1] / np.sqrt(sigma2 * xtx_inv[1, 1])
                    t2 = bu[2] / np.sqrt(sigma2 * xtx_inv[2, 2])
                    t3 = bu[3] / np.sqrt(sigma2 * xtx_inv[3, 3])
                    t4 = bu[4] / np.sqrt(sigma2 * xtx_inv[4, 4])

                    # ---- factor ratios (script L264-296, weightscale=1);
                    # left-to-right addition mirrors sum_horizontal's fold
                    slope = (1.0 * bu[1] + 2.0 * bu[2] + 3.0 * bu[3] + 4.0 * bu[4]) / (
                        bu[0] + (bu[1] + bu[2] + bu[3] + bu[4])
                    )
                    tstat = (1.0 * t1 + 2.0 * t2 + 3.0 * t3 + 4.0 * t4) / (
                        t0 + (t1 + t2 + t3 + t4)
                    )
                    rsq = 1.0 - r2_r / r2_u

                    out_stamp[n_out] = jk + 1  # June key + 1mo -> July (L301-303)
                    out_slope[n_out] = slope
                    out_rsq[n_out] = rsq
                    out_tstat[n_out] = tstat
                    n_out += 1
        i = j
    return out_stamp[:n_out], out_slope[:n_out], out_rsq[:n_out], out_tstat[:n_out]


def _expand_monthly(
    permno: np.ndarray,
    stamp: np.ndarray,
    slope: np.ndarray,
    rsq: np.ndarray,
    tstat: np.ndarray,
) -> pd.DataFrame:
    """Annual July-stamped rows -> a point-in-time forward-filled monthly grid.

    Reference (script L308-383): each annual value fills forward until the
    permno's NEXT stamp, and the LAST stamp covers only its own July. That
    fill length depends on information not available at the stamp date — the
    2026-08-17 data refresh exposed it: extending history by one year turned
    44,343 previously-NaN 2010-2024 cells into values purely because a later
    stamp appeared (a vintage-dependent look-ahead; deviation registered
    section D). Point-in-time rule used here: every stamp fills a fixed
    12-month horizon (July..next June) regardless of later stamps. NaN
    annual values still propagate as NaN across their horizon.
    """

    order = np.lexsort((stamp, permno))
    p = permno[order]
    s = stamp[order]
    same_next = p[1:] == p[:-1]
    if not np.all(s[1:][same_next] > s[:-1][same_next]):
        raise AssertionError(
            "duplicate or non-increasing July stamps within a permno — "
            "annual rows are corrupt, investigate"
        )
    lengths = np.full(len(p), 12, dtype=np.int64)
    total = int(lengths.sum())
    row_idx = np.repeat(np.arange(len(p)), lengths)
    csum = np.cumsum(lengths)
    offs = np.arange(total) - np.repeat(csum - lengths, lengths)
    month_code = s[row_idx] + offs
    return pd.DataFrame(
        {
            "permno": p[row_idx],
            "time_avail_m": month_code.astype("datetime64[M]").astype("datetime64[ns]"),
            "slope": slope[order][row_idx],
            "rsq": rsq[order][row_idx],
            "tstat": tstat[order][row_idx],
        }
    )


def price_delay_panels(permnos=None) -> pd.DataFrame:
    """Long frame [permno, time_avail_m, slope, rsq, tstat] from all buckets.

    slope = PriceDelaySlope, rsq = PriceDelayRsq, tstat = PriceDelayTstat, at
    monthly frequency after the reference's July-stamp + forward-fill grid.
    All three factors share every regression, so they are computed in one
    pass. ``permnos`` (optional iterable) restricts output to those stocks —
    every bucket is still streamed, non-target stocks are skipped.
    """

    ff_dates, ff_fac = load_daily_ff()
    nff = len(ff_dates)
    mkt_ff = np.ascontiguousarray(ff_fac[:, 0])
    rf_ff = np.ascontiguousarray(ff_fac[:, 3])
    # market-calendar lags built BEFORE the join (script L52-57)
    lag_ff = np.full((NLAG, nff), np.nan, dtype=np.float64)
    for lag in range(1, NLAG + 1):
        lag_ff[lag - 1, lag:] = mkt_ff[:-lag]

    want = None if permnos is None else np.unique(np.asarray(list(permnos), dtype=np.int64))

    all_permno: list[np.ndarray] = []
    all_stamp: list[np.ndarray] = []
    all_slope: list[np.ndarray] = []
    all_rsq: list[np.ndarray] = []
    all_tstat: list[np.ndarray] = []

    for bucket in iter_buckets(["ret"]):
        permno = bucket["permno"].to_numpy(np.int64)
        keep = None
        if want is not None:
            keep = np.isin(permno, want)
            if not keep.any():
                continue
        dates = bucket["time_d"].to_numpy("datetime64[D]").astype(np.int64)
        # INNER join on the FF calendar: unmatched days cease to exist
        pos = np.minimum(np.searchsorted(ff_dates, dates), nff - 1)
        matched = ff_dates[pos] == dates
        keep = matched if keep is None else (matched & keep)
        if not keep.any():
            continue
        idx = np.flatnonzero(keep)
        permno_k = permno[idx]
        months_k = _month_codes(bucket["time_d"].to_numpy()[idx])
        posk = pos[idx]
        y = bucket["ret"].to_numpy(np.float64)[idx] - rf_ff[posk]
        mkt = mkt_ff[posk]
        l1 = lag_ff[0][posk]
        l2 = lag_ff[1][posk]
        l3 = lag_ff[2][posk]
        l4 = lag_ff[3][posk]

        starts, ids = _stock_offsets(permno_k)
        for s_idx in range(len(ids)):
            lo, hi = starts[s_idx], starts[s_idx + 1]
            stamp, slope, rsq, tstat = _price_delay_stock(
                y[lo:hi], mkt[lo:hi], l1[lo:hi], l2[lo:hi], l3[lo:hi], l4[lo:hi],
                months_k[lo:hi], MIN_OBS,
            )
            if len(stamp):
                all_permno.append(np.full(len(stamp), ids[s_idx], dtype=np.int64))
                all_stamp.append(stamp)
                all_slope.append(slope)
                all_rsq.append(rsq)
                all_tstat.append(tstat)

    if not all_stamp:
        raise RuntimeError(
            "price_delay_panels produced zero annual rows — the bucket artifact "
            "or FF file is broken, investigate"
        )
    return _expand_monthly(
        np.concatenate(all_permno),
        np.concatenate(all_stamp),
        np.concatenate(all_slope),
        np.concatenate(all_rsq),
        np.concatenate(all_tstat),
    )


# ---------------------------------------------------------------------------
# Self-test against the golden oracle
# ---------------------------------------------------------------------------

_ORACLE_FILES = {
    "slope": "PriceDelaySlope.parquet",
    "rsq": "PriceDelayRsq.parquet",
    "tstat": "PriceDelayTstat.parquet",
}

LIQUID_PERMNOS = (14593, 10107, 12490)  # AAPL, MSFT, IBM


def _oracle_columns(factor: str) -> list[str]:
    import pyarrow.parquet as pq
    from p1_daily_kernels import ROOT

    path = ROOT / ORACLE_DIR / _ORACLE_FILES[factor]
    return [n for n in pq.ParquetFile(path).schema_arrow.names if n != "month"]


def _read_oracle(factor: str, permnos: list[int]) -> pd.DataFrame:
    """Wide float32 oracle restricted to the requested permnos, month-indexed."""

    import pyarrow.parquet as pq
    from p1_daily_kernels import ROOT

    path = ROOT / ORACLE_DIR / _ORACLE_FILES[factor]
    cols = [str(p) for p in permnos] + ["month"]
    df = pq.read_table(path, columns=cols).to_pandas()
    if df.index.name != "month":
        df = df.set_index("month")
    # pandas metadata restores the permno column names as integers
    df.columns = df.columns.astype(str)
    return df


def self_test(n_random: int = 2, seed: int = 20260801, rtol: float = 1e-5, atol: float = 1e-8) -> dict:
    """Compare 3 liquid + ``n_random`` random stocks against the golden oracle.

    Returns a plain-data dict: per factor, per permno, cell counts for
    value_match / value_diff / missing_in_mine / extra_in_mine plus the worst
    relative difference and up to 3 example mismatches.
    """

    rng = np.random.default_rng(seed)
    candidates = np.array([int(c) for c in _oracle_columns("slope")], dtype=np.int64)
    pool = candidates[~np.isin(candidates, LIQUID_PERMNOS)]

    # draw random stocks that actually have oracle data
    randoms: list[int] = []
    draw = rng.permutation(pool)
    ptr = 0
    while len(randoms) < n_random and ptr < len(draw):
        batch = [int(p) for p in draw[ptr:ptr + 16]]
        ptr += 16
        wide = _read_oracle("slope", batch)
        for p in batch:
            if len(randoms) < n_random and wide[str(p)].notna().any():
                randoms.append(p)
    if len(randoms) < n_random:
        raise RuntimeError("could not find random permnos with oracle data")

    targets = list(LIQUID_PERMNOS) + randoms
    mine = price_delay_panels(permnos=targets)

    report: dict = {"targets": {"liquid": list(LIQUID_PERMNOS), "random": randoms}}
    for factor in ("slope", "rsq", "tstat"):
        oracle = _read_oracle(factor, targets)
        fac_rep: dict = {}
        for p in targets:
            o = oracle[str(p)].astype(np.float64)
            m_rows = mine[mine["permno"] == p].set_index("time_avail_m")[factor]
            m = m_rows.reindex(o.index).astype(np.float64)
            # compare at float32 precision: the oracle was stored as float32
            m32 = m.to_numpy().astype(np.float32).astype(np.float64)
            ov = o.to_numpy()
            has_o = ~np.isnan(ov)
            has_m = ~np.isnan(m32)
            both = has_o & has_m
            close = np.zeros(len(ov), dtype=bool)
            close[both] = np.isclose(m32[both], ov[both], rtol=rtol, atol=atol)
            # months mine has beyond the oracle's index would be silently lost
            # by reindex — surface them
            extra_months = m_rows.index.difference(o.index)
            if len(extra_months):
                raise AssertionError(
                    f"permno {p}: computed months outside the oracle grid: "
                    f"{list(extra_months[:3])}"
                )
            value_diff_idx = np.flatnonzero(both & ~close)
            missing_idx = np.flatnonzero(has_o & ~has_m)
            extra_idx = np.flatnonzero(has_m & ~has_o)
            rel = np.zeros(0, dtype=np.float64)
            if both.any():
                denom = np.maximum(np.abs(ov[both]), 1e-300)
                rel = np.abs(m32[both] - ov[both]) / denom
            examples = [
                {
                    "month": str(o.index[k].date()),
                    "mine": float(m32[k]),
                    "oracle": float(ov[k]),
                }
                for k in value_diff_idx[:3]
            ]
            fac_rep[p] = {
                "compared": int(both.sum()),
                "value_match": int(close.sum()),
                "value_diff": int(len(value_diff_idx)),
                "missing_in_mine": int(len(missing_idx)),
                "extra_in_mine": int(len(extra_idx)),
                "max_rel_diff": float(rel.max()) if len(rel) else 0.0,
                "examples": examples,
                "missing_months": [str(o.index[k].date()) for k in missing_idx[:3]],
                "extra_months": [str(o.index[k].date()) for k in extra_idx[:3]],
            }
        report[factor] = fac_rep
    report["pass"] = all(
        r["value_diff"] == 0 and r["missing_in_mine"] == 0 and r["extra_in_mine"] == 0
        for f in ("slope", "rsq", "tstat")
        for r in report[f].values()
    )
    return report


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), indent=2, default=str))

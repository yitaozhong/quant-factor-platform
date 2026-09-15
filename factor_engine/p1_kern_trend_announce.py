"""Bucket-streamed kernels for TrendFactor and AnnouncementReturn (WS8 follow-on).

Two factors, one function each, both returning LONG [permno, time_avail_m, value]:

* ``trendfactor_panel``      — Han–Zhou–Zhu (2016) trend factor. Stage 1 (numba,
  bucket-streamed): positional rolling means of the split-adjusted price
  P = |prc|/cfacpr at 11 horizons, collapsed to the last trading row of each
  stock-month and normalized by that row's P. Stage 2 (numpy/scipy, ~1.2k
  months): monthly cross-sectional OLS of next-month return on the 11 lagged
  MA signals over the reference's filtered universe, 12-month trailing mean of
  the coefficients, then the expected-return forecast Σ_L EBeta_L(t)·A_L,i(t).
* ``announcement_return_panel`` — CAR over the positional earnings-announcement
  window [ann-2, ann+1] (business rows), market-adjusted (ret − mktrf − rf),
  stamped at the month of the window's LAST surviving day, forward-filled up
  to 6 months.

Reference scripts (fidelity targets, validated against Data/golden/oracle/):
  Open_Source_Asset_Pricing/Signals/pyCode/Predictors/TrendFactor.py
  Open_Source_Asset_Pricing/Signals/pyCode/Predictors/ZZ2_AnnouncementReturn.py

LOOK-AHEAD AUDIT (TrendFactor, mandatory P3): the reference is point-in-time
correct — no deviation needed. b_L(s) is estimated from fRet(s) = ret(s+1)
regressed on A_L(s), so b_L(s) is knowable at the end of month s+1. The
smoothing is ``shift(1).rolling_mean(12)``: EBeta_L(t) averages b_L(t-12..t-1),
whose most recent member b_L(t-1) needs ret(t) — available when the signal is
stamped at end of month t to predict ret(t+1). Dropping the shift would leak
ret(t+1) into the month-t signal; the script has the shift, so this module
implements it as-is.

Null-vs-NaN note: the reference distinguishes polars null (missing) from NaN
(computed, e.g. inf/inf). Here both are numpy NaN. This is observationally
identical in the oracle's wide matrix: a reference NaN cell and a dropped row
are the same NaN cell. ±inf values are kept wherever the reference keeps them
(cfacpr==0 → P=+inf exists 34,595 times in dailyCRSP).
"""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.linalg import qr as _scipy_qr

from p1_daily_kernels import (
    ROOT,
    _month_codes,
    _stock_offsets,
    iter_buckets,
    load_daily_ff,
)

INTERMEDIATE = ROOT / "Open_Source_Asset_Pricing" / "Signals" / "pyData" / "Intermediate"

# ---------------------------------------------------------------------------
# TrendFactor — stage 1: daily rolling MAs, monthly collapse (numba)
# ---------------------------------------------------------------------------

TREND_LAGS = np.array([3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000], dtype=np.int64)


@numba.njit(cache=True)
def _trend_ma_bucket(
    P: np.ndarray,
    months: np.ndarray,
    starts: np.ndarray,
    lags: np.ndarray,
    out_sid: np.ndarray,
    out_m: np.ndarray,
    out_A: np.ndarray,
) -> int:
    """Rolling means of P at each lag, written at each stock's last row per month.

    Window = trailing ``lag`` POSITIONAL rows of the stock (the reference's
    asrol over the row counter time_temp — calendar gaps ignored, and null
    rows still occupy window positions). NaN P is missing (polars null:
    skipped, doesn't count toward the mean); ±inf P is a VALUE tracked in
    counters (polars SumWindow: window mean is inf exactly while an inf is
    inside, finite again once it exits; +inf and −inf together → NaN).

    The accumulation is BITWISE-identical to polars 1.40's SumWindow
    (polars-compute/src/rolling/sum.rs): per window step, leaving values are
    subtracted FIRST, then entering values added, each through Kahan
    compensation with SEPARATE err terms for the add and subtract streams.
    Bitwise fidelity matters: in 1926–1929 every stock has fewer rows than the
    long MA horizons, so with min_samples=1 the A_600/A_800/A_1000 columns are
    exactly duplicated and the monthly regression's QR pivot tie-break (which
    duplicate column gets the coefficient, the others 0.0) is decided by
    1-ulp differences in the column norms.

    Emitted value is A_L(last row of month) divided by that row's P (IEEE
    semantics: x/NaN→NaN, inf/inf→NaN, finite/inf→0.0, matching polars float
    division). Returns rows written; out arrays must be len(P).
    """

    K = lags.shape[0]
    n_out = 0
    for s in range(starts.shape[0] - 1):
        lo = starts[s]
        hi = starts[s + 1]
        sums = np.zeros(K, dtype=np.float64)
        err_add = np.zeros(K, dtype=np.float64)
        err_sub = np.zeros(K, dtype=np.float64)
        cnts = np.zeros(K, dtype=np.int64)
        pinf = np.zeros(K, dtype=np.int64)
        ninf = np.zeros(K, dtype=np.int64)
        for i in range(lo, hi):
            v = P[i]
            v_ok = not np.isnan(v)
            for k in range(K):
                # subtract the leaving value FIRST (polars update order)
                j = i - lags[k]
                if j >= lo:
                    u = P[j]
                    if not np.isnan(u):
                        if u == np.inf:
                            pinf[k] -= 1
                        elif u == -np.inf:
                            ninf[k] -= 1
                        else:
                            y = (-u) - err_sub[k]
                            t = sums[k] + y
                            err_sub[k] = (t - sums[k]) - y
                            sums[k] = t
                        cnts[k] -= 1
                # then add the entering value
                if v_ok:
                    if v == np.inf:
                        pinf[k] += 1
                    elif v == -np.inf:
                        ninf[k] += 1
                    else:
                        y = v - err_add[k]
                        t = sums[k] + y
                        err_add[k] = (t - sums[k]) - y
                        sums[k] = t
                    cnts[k] += 1
            if i == hi - 1 or months[i + 1] != months[i]:
                out_sid[n_out] = s
                out_m[n_out] = months[i]
                for k in range(K):
                    if cnts[k] == 0 or (pinf[k] > 0 and ninf[k] > 0):
                        a = np.nan
                    elif pinf[k] > 0:
                        a = np.inf
                    elif ninf[k] > 0:
                        a = -np.inf
                    else:
                        a = sums[k] / cnts[k]
                    out_A[n_out, k] = a / v
                n_out += 1
    return n_out


# ---------------------------------------------------------------------------
# TrendFactor — stage 2: universe filters, cross-sectional betas, forecast
# ---------------------------------------------------------------------------


def _stata_p10(a: np.ndarray) -> float:
    """Exact port of utils.stata_replication.stata_quantile(x, 0.10).

    NaNs dropped, stable sort; P = 0.10·n (through the qs*100/100 round trip
    the reference performs); the value is the first order statistic whose rank
    exceeds P, except when P is within 1e-12 of an integer k (1 ≤ k < n), in
    which case the midpoint (arr[k-1]+arr[k])/2 is used. Empty → NaN.
    """

    arr = a[~np.isnan(a)]
    arr = np.sort(arr, kind="mergesort")
    n = arr.size
    if n == 0:
        return np.nan
    q = 0.10 * 100.0  # reference maps fractional qs to percent, then back
    p = (q / 100.0) * n
    if p <= 0:
        return float(arr[0])
    if p >= n:
        return float(arr[-1])
    idx = int(np.searchsorted(np.arange(1, n + 1), p, side="right"))
    val = arr[idx]
    k = int(np.floor(p + 1e-12))
    if abs(p - k) < 1e-12 and 1 <= k < n:
        val = (arr[k - 1] + arr[k]) / 2
    return float(val)


def _xs_beta_month(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """One month's cross-sectional betas, replicating asreg_collinear.

    Estimation sample: rows finite in y AND all 11 regressors (np.isfinite —
    NaN and ±inf both excluded). Zero valid rows → all-NaN betas (the
    reference's exception path; the month still occupies a row in the beta
    table so the positional EBeta window sees it).

    Collinearity exactly as drop_collinear(method='qr', scale=True): constant
    columns (max==min on the sample) dropped first; the rest L2-normalized and
    rank-revealed by scipy QR with column pivoting, tol = eps·max(n,p)·max|diag R|;
    only piv[:rank] kept. Dropped columns get beta 0.0 (NOT NaN) — with n < 12
    valid rows this is how the reference produces exact-zero betas that drag
    EBeta toward 0. Kept columns + prepended intercept are fit by least
    squares (statsmodels' pinv and lstsq agree to fp noise; if the intercept
    is itself in the span of the kept columns both give the minimum-norm
    solution — degenerate-branch caveat, see module notes).
    """

    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if not mask.any():
        return np.full(X.shape[1], np.nan)
    ys = y[mask]
    Xs = X[mask]
    n = ys.size
    betas = np.zeros(X.shape[1], dtype=np.float64)
    ptp = Xs.max(axis=0) - Xs.min(axis=0)
    noncst = ptp != 0.0
    keep_idx = np.empty(0, dtype=np.int64)
    if noncst.any():
        A = Xs[:, noncst]
        p_pre = A.shape[1]
        norms = np.linalg.norm(A, axis=0)
        norms[norms == 0.0] = 1.0
        At = A / norms
        _q, R, piv = _scipy_qr(At, mode="economic", pivoting=True)
        diagR = np.abs(np.diag(R))
        tol = np.finfo(At.dtype).eps * max(n, p_pre) * diagR.max()
        rank = int((diagR > tol).sum())
        orig = np.flatnonzero(noncst)
        keep_idx = np.sort(orig[np.asarray(piv[:rank])])
    design = np.empty((n, keep_idx.size + 1), dtype=np.float64)
    design[:, 0] = 1.0
    design[:, 1:] = Xs[:, keep_idx]
    coef, _res, _rank, _sv = np.linalg.lstsq(design, ys, rcond=None)
    betas[keep_idx] = coef[1:]
    return betas


def _load_smt_filtered() -> pd.DataFrame:
    """SignalMasterTable rows surviving the reference's regression filters.

    Returns [permno, mcode, ret]. Filters (stata_ineq semantics — nulls fill
    to +inf on both sides of >=, so null prc and null mve_c PASS):
    exchcd ∈ {1,2,3}, shrcd ∈ {10,11}, |prc| ≥ 5, mve_c ≥ month's NYSE
    (exchcd==1) 10th percentile of mve_c (stata_quantile). Months with no
    NYSE rows have null qu10 (→ +inf): only null-mve rows pass there.
    """

    smt = pq.read_table(
        INTERMEDIATE / "SignalMasterTable.parquet",
        columns=["permno", "time_avail_m", "ret", "prc", "exchcd", "shrcd", "mve_c"],
    ).to_pandas()
    mcode = smt["time_avail_m"].to_numpy("datetime64[M]").astype(np.int64)
    exch = smt["exchcd"].to_numpy(np.float64)
    shr = smt["shrcd"].to_numpy(np.float64)
    prc = smt["prc"].to_numpy(np.float64)
    mve = smt["mve_c"].to_numpy(np.float64)

    nyse = exch == 1.0
    ny_m = mcode[nyse]
    ny_v = mve[nyse]
    order = np.argsort(ny_m, kind="stable")
    ny_m = ny_m[order]
    ny_v = ny_v[order]
    u_months, u_starts = np.unique(ny_m, return_index=True)
    bounds = np.append(u_starts, ny_m.size)
    qu10 = np.array(
        [_stata_p10(ny_v[bounds[i] : bounds[i + 1]]) for i in range(u_months.size)]
    )
    pos = np.searchsorted(u_months, mcode)
    pos_c = np.minimum(pos, max(u_months.size - 1, 0))
    row_qu10 = np.where(
        (u_months.size > 0) & (u_months[pos_c] == mcode), qu10[pos_c], np.nan
    )

    prc_f = np.where(np.isnan(prc), np.inf, np.abs(prc))
    mve_f = np.where(np.isnan(mve), np.inf, mve)
    qu_f = np.where(np.isnan(row_qu10), np.inf, row_qu10)
    keep = (
        ((exch == 1.0) | (exch == 2.0) | (exch == 3.0))
        & ((shr == 10.0) | (shr == 11.0))
        & (prc_f >= 5.0)
        & (mve_f >= qu_f)
    )
    return pd.DataFrame(
        {
            "permno": smt["permno"].to_numpy(np.int64)[keep],
            "mcode": mcode[keep],
            "ret": smt["ret"].to_numpy(np.float64)[keep],
        }
    )


def trendfactor_panel() -> pd.DataFrame:
    """LONG [permno, time_avail_m, value] for TrendFactor.

    Emits a row for every filtered stock-month whose 11 (EBeta_L, A_L) inputs
    are all non-missing (the reference's N_MA_used == 11 gate); the value may
    be NaN or ±inf where the reference computes and saves one (kept, not
    filtered — the oracle keeps them too).
    """

    a_cols = [f"A_{L}" for L in TREND_LAGS]
    smt_f = _load_smt_filtered()

    parts: list[pd.DataFrame] = []
    for bucket in iter_buckets(["prc", "cfacpr"]):
        permno = bucket["permno"].to_numpy(np.int64)
        months = _month_codes(bucket["time_d"].to_numpy())
        prc = bucket["prc"].to_numpy(np.float64)
        cf = bucket["cfacpr"].to_numpy(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.abs(prc) / cf  # cfacpr==0 → +inf, kept (polars division)
        starts, ids = _stock_offsets(permno)
        n = P.shape[0]
        out_sid = np.empty(n, dtype=np.int64)
        out_m = np.empty(n, dtype=np.int64)
        out_A = np.empty((n, TREND_LAGS.size), dtype=np.float64)
        n_out = _trend_ma_bucket(P, months, starts, TREND_LAGS, out_sid, out_m, out_A)
        part = pd.DataFrame(
            {"permno": ids[out_sid[:n_out]], "mcode": out_m[:n_out]}
            | {c: out_A[:n_out, k] for k, c in enumerate(a_cols)}
        )
        merged = smt_f.merge(part, on=["permno", "mcode"], how="inner")
        if len(merged):
            parts.append(merged)
    reg = pd.concat(parts, ignore_index=True)
    del parts

    # fRet(t) = ret(t+1), a self-lead of the filtered+joined table: fRet exists
    # only if the stock also survives the filters AND the MA join at t+1.
    lead = reg[["permno", "mcode", "ret"]].copy()
    lead["mcode"] = lead["mcode"] - 1
    lead = lead.rename(columns={"ret": "fRet"})
    reg = reg.merge(lead, on=["permno", "mcode"], how="left")
    # the reference sorts [time_avail_m, permno] before the regressions; the
    # ROW ORDER matters bitwise — column norms in the QR collinearity screen
    # are fp sums, and in rank-deficient months (duplicated MA columns,
    # 1926-1929) the pivot tie-break decides which column gets the coefficient
    reg = reg.sort_values(["mcode", "permno"], kind="stable", ignore_index=True)

    mcodes = reg["mcode"].to_numpy(np.int64)
    frets = reg["fRet"].to_numpy(np.float64)
    Amat = reg[a_cols].to_numpy(np.float64)
    months_u, m_starts = np.unique(mcodes, return_index=True)
    bounds = np.append(m_starts, mcodes.size)

    B = np.empty((months_u.size, TREND_LAGS.size), dtype=np.float64)
    for t in range(months_u.size):
        g = slice(bounds[t], bounds[t + 1])
        B[t] = _xs_beta_month(frets[g], Amat[g])

    # EBeta_L(t): mean of the non-NaN betas over the 12 preceding rows of the
    # beta table (shift(1).rolling_mean(12, min_samples=1) — positional over
    # months PRESENT, calendar gaps compress the window). NaN rows (failed
    # months, e.g. the final sample month whose fRet is entirely missing) are
    # skipped, not poisonous.
    EB = np.full_like(B, np.nan)
    for t in range(months_u.size):
        w = B[max(0, t - 12) : t]
        if w.shape[0]:
            cnt = (~np.isnan(w)).sum(axis=0)
            s = np.nansum(w, axis=0)
            EB[t] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)

    row_eb = EB[np.searchsorted(months_u, mcodes)]
    gate = ~np.isnan(row_eb).any(axis=1) & ~np.isnan(Amat).any(axis=1)
    value = (row_eb * Amat).sum(axis=1)

    out = pd.DataFrame(
        {
            "permno": reg["permno"].to_numpy(np.int64)[gate],
            "time_avail_m": mcodes[gate].astype("datetime64[M]").astype("datetime64[ns]"),
            "value": value[gate],
        }
    )
    return out


# ---------------------------------------------------------------------------
# AnnouncementReturn — positional event windows around rdq (numba)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _ann_stock(
    ar: np.ndarray,
    months: np.ndarray,
    ann: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One stock's announcement CARs, stamped and ≤6-month forward-filled.

    Inputs are the stock's SURVIVING rows (link-valid AND FF-covered), date
    ascending. Window assignment replicates the reference's sequential
    np.where overwrites — anndat, then f1, then f2, then l1 (LAST write wins,
    so a row shared by two adjacent event windows goes to the earlier
    announcement via l1, except an announcement row 1-2 rows before another
    announcement is stolen by the later one via f1/f2). Groups are keyed by
    the announcement's own row index t0; each group's CAR is the skipna sum of
    ar over its rows (all-NaN → 0.0, pandas .agg('sum')), stamped at the
    month of the group's LAST surviving row (max time_d — normally the day
    after the announcement, so windows can spill into the next month).
    Several groups stamping one month: the largest t0 wins (stable-sort +
    tail(1)). Forward fill: value(m) = original(m−j) for the smallest j in
    0..6 with an original value, only within [first, last] stamped months.

    Returns (month_codes, values), month ascending.
    """

    n = ar.shape[0]
    w = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if ann[i]:
            w[i] = i
    for i in range(n - 1):
        if ann[i + 1]:
            w[i] = i + 1
    for i in range(n - 2):
        if ann[i + 2]:
            w[i] = i + 2
    for i in range(1, n):
        if ann[i - 1]:
            w[i] = i - 1

    seen = np.zeros(n, dtype=np.bool_)
    gsum = np.zeros(n, dtype=np.float64)
    gmon = np.zeros(n, dtype=np.int64)
    for i in range(n):
        t = w[i]
        if t >= 0:
            seen[t] = True
            v = ar[i]
            if not np.isnan(v):
                gsum[t] += v
            gmon[t] = months[i]  # rows ascend in time: last write = max time_d

    k = 0
    gm = np.empty(n, dtype=np.int64)
    gv = np.empty(n, dtype=np.float64)
    for t in range(n):  # ascending t0 == ascending time_ann_d
        if seen[t]:
            gm[k] = gmon[t]
            gv[k] = gsum[t]
            k += 1
    if k == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    # per stamped month keep the LAST group in t0 order (stamped months are
    # not guaranteed monotone in t0 when windows collide)
    order = np.argsort(gm[:k], kind="mergesort")
    um = np.empty(k, dtype=np.int64)
    uv = np.empty(k, dtype=np.float64)
    ku = 0
    for a in range(k):
        g = order[a]
        if ku > 0 and um[ku - 1] == gm[g]:
            uv[ku - 1] = gv[g]
        else:
            um[ku] = gm[g]
            uv[ku] = gv[g]
            ku += 1

    out_m = np.empty(7 * ku, dtype=np.int64)
    out_v = np.empty(7 * ku, dtype=np.float64)
    n_out = 0
    for a in range(ku):
        if a == ku - 1:
            stop = um[a]  # the reindexed panel ends at the last stamped month
        else:
            stop = um[a] + 6
            if um[a + 1] - 1 < stop:
                stop = um[a + 1] - 1
        for m in range(um[a], stop + 1):
            out_m[n_out] = m
            out_v[n_out] = uv[a]
            n_out += 1
    return out_m[:n_out], out_v[:n_out]


def _load_links() -> dict[int, list[tuple[int, int, int]]]:
    """CCM crosswalk: permno -> [(gvkey, start_day, end_day)] (int64 days).

    Missing timeLinkEnd_d means the link is still active (+inf sentinel). The
    linking table has no NaT starts, no non-numeric and no overlapping-window
    gvkeys per permno (verified 2026-08-01); overlaps would duplicate daily
    rows in the reference — assert so a data regression surfaces loudly.
    """

    cw = pd.read_parquet(
        INTERMEDIATE / "CCMLinkingTable.parquet",
        columns=["gvkey", "permno", "timeLinkStart_d", "timeLinkEnd_d"],
    )
    g = pd.to_numeric(cw["gvkey"], errors="coerce")
    if g.isna().any() or (g % 1 != 0).any():
        raise ValueError("CCMLinkingTable gvkey no longer clean numeric — revisit link matching")
    if cw["timeLinkStart_d"].isna().any():
        raise ValueError("CCMLinkingTable has NaT timeLinkStart_d — reference drops such rows; revisit")
    gvkey = g.to_numpy(np.int64)
    permno = cw["permno"].to_numpy(np.int64)
    start = cw["timeLinkStart_d"].to_numpy("datetime64[D]").astype(np.int64)
    end_raw = cw["timeLinkEnd_d"]
    end = end_raw.to_numpy("datetime64[D]").astype(np.int64)
    end[end_raw.isna().to_numpy()] = np.iinfo(np.int64).max

    order = np.lexsort((start, permno))
    prev_end = np.empty(order.size, dtype=np.int64)
    prev_end[0] = np.iinfo(np.int64).min
    prev_end[1:] = end[order[:-1]]
    same = np.empty(order.size, dtype=np.bool_)
    same[0] = False
    same[1:] = permno[order[1:]] == permno[order[:-1]]
    if (same & (start[order] <= prev_end)).any():
        raise ValueError(
            "overlapping CCM link windows for one permno — the reference would "
            "duplicate daily rows; this kernel assumes disjoint links"
        )

    links: dict[int, list[tuple[int, int, int]]] = {}
    for i in range(permno.size):
        links.setdefault(int(permno[i]), []).append((int(gvkey[i]), int(start[i]), int(end[i])))
    return links


def _load_rdq_map() -> dict[int, np.ndarray]:
    """m_QCompustat: gvkey -> sorted unique announcement days (int64 D)."""

    qc = pq.read_table(
        INTERMEDIATE / "m_QCompustat.parquet", columns=["gvkey", "rdq"]
    ).to_pandas()
    qc = qc.dropna(subset=["rdq"]).drop_duplicates()
    gv = qc["gvkey"].to_numpy(np.int64)
    dt = qc["rdq"].to_numpy("datetime64[D]").astype(np.int64)
    order = np.argsort(gv, kind="stable")
    gv = gv[order]
    dt = dt[order]
    out: dict[int, np.ndarray] = {}
    u, s = np.unique(gv, return_index=True)
    b = np.append(s, gv.size)
    for i in range(u.size):
        out[int(u[i])] = np.unique(dt[b[i] : b[i + 1]])
    return out


def announcement_return_panel() -> pd.DataFrame:
    """LONG [permno, time_avail_m, value] for AnnouncementReturn.

    Surviving rows per stock = daily rows inside a CCM link window AND with a
    dailyFF date (the reference's link filter + FF inner join, both applied
    BEFORE the business-day counter, so window positions skip dropped days).
    An announcement is flagged only when rdq equals a surviving trading date
    exactly (weekend/holiday rdq silently produces nothing — reference
    behavior). All-NaN windows are saved as real 0.0 signals.
    """

    links = _load_links()
    rdq_map = _load_rdq_map()
    ff_dates, ff_fac = load_daily_ff()
    ff_mkt = ff_fac[:, 0]
    ff_rf = ff_fac[:, 3]

    p_parts: list[np.ndarray] = []
    m_parts: list[np.ndarray] = []
    v_parts: list[np.ndarray] = []
    for bucket in iter_buckets(["ret"]):
        permno = bucket["permno"].to_numpy(np.int64)
        dates = bucket["time_d"].to_numpy("datetime64[D]").astype(np.int64)
        months = _month_codes(bucket["time_d"].to_numpy())
        ret = bucket["ret"].to_numpy(np.float64)
        pos = np.searchsorted(ff_dates, dates)
        pos_c = np.minimum(pos, ff_dates.size - 1)
        matched = ff_dates[pos_c] == dates
        mkt = ff_mkt[pos_c]
        rf = ff_rf[pos_c]
        starts, ids = _stock_offsets(permno)
        for s_idx in range(ids.size):
            lo, hi = starts[s_idx], starts[s_idx + 1]
            entry = links.get(int(ids[s_idx]))
            if entry is None:
                continue  # no link rows: every day fails the validity filter
            d = dates[lo:hi]
            valid = np.zeros(d.size, dtype=np.bool_)
            ann = np.zeros(d.size, dtype=np.bool_)
            for gv, st, en in entry:
                within = (d >= st) & (d <= en)
                rd = rdq_map.get(gv)
                if rd is not None:
                    p2 = np.searchsorted(rd, d)
                    p2c = np.minimum(p2, rd.size - 1)
                    ann |= within & (rd[p2c] == d)
                valid |= within
            surv = valid & matched[lo:hi]
            if not surv.any():
                continue
            ann_s = ann[surv]
            if not ann_s.any():
                continue
            ar = ret[lo:hi][surv] - (mkt[lo:hi][surv] + rf[lo:hi][surv])
            m_out, v_out = _ann_stock(ar, months[lo:hi][surv], ann_s)
            if m_out.size:
                p_parts.append(np.full(m_out.size, ids[s_idx], dtype=np.int64))
                m_parts.append(m_out)
                v_parts.append(v_out)

    out = pd.DataFrame(
        {
            "permno": np.concatenate(p_parts),
            "time_avail_m": np.concatenate(m_parts).astype("datetime64[M]").astype("datetime64[ns]"),
            "value": np.concatenate(v_parts),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Self-test against Data/golden/oracle (run as a script)
# ---------------------------------------------------------------------------


def _oracle_long(name: str) -> pd.DataFrame:
    """Oracle wide (month × permno, float32) -> long [permno, mcode, value]."""

    wide = pq.read_table(ROOT / "Data" / "golden" / "oracle" / f"{name}.parquet").to_pandas()
    if "month" in wide.columns:
        wide = wide.set_index("month")
    mask = wide.notna().to_numpy()
    rows, cols = np.nonzero(mask)
    return pd.DataFrame(
        {
            "permno": np.asarray([int(c) for c in wide.columns], dtype=np.int64)[cols],
            "mcode": wide.index.to_numpy("datetime64[M]").astype(np.int64)[rows],
            "oracle": wide.to_numpy()[mask].astype(np.float64),
        }
    )


def _compare(name: str, mine: pd.DataFrame, permnos: list[int]) -> dict:
    orc = _oracle_long(name)
    got = mine.copy()
    got["mcode"] = got["time_avail_m"].to_numpy("datetime64[M]").astype(np.int64)
    got = got[~np.isnan(got["value"].to_numpy())]  # NaN value == missing cell in the oracle
    merged = orc.merge(got[["permno", "mcode", "value"]], on=["permno", "mcode"], how="outer")
    o = merged["oracle"].to_numpy()
    v = merged["value"].to_numpy()
    both = ~np.isnan(o) & ~np.isnan(v)
    # oracle is float32: compare at float32 resolution
    close = both & (
        (np.float32(0) + np.abs(v.astype(np.float32) - o.astype(np.float32)))
        <= 3e-6 * np.abs(o.astype(np.float32)) + 1e-9
    )
    inf_ok = both & np.isinf(o) & (o == v)
    report = {
        "oracle_cells": int((~np.isnan(o)).sum()),
        "mine_cells": int((~np.isnan(v)).sum()),
        "matched_cells": int(both.sum()),
        "value_agree": int((close | inf_ok).sum()),
        "value_disagree": int((both & ~(close | inf_ok)).sum()),
        "oracle_only": int((~np.isnan(o) & np.isnan(v)).sum()),
        "mine_only": int((np.isnan(o) & ~np.isnan(v)).sum()),
        "per_permno": {},
    }
    bad = merged[both & ~(close | inf_ok)]
    if len(bad):
        d = np.abs(bad["value"].to_numpy() - bad["oracle"].to_numpy())
        report["worst_abs_diff"] = float(np.nanmax(d))
    for p in permnos:
        sub = merged[merged["permno"] == p]
        so = sub["oracle"].to_numpy()
        sv = sub["value"].to_numpy()
        sb = ~np.isnan(so) & ~np.isnan(sv)
        diffs = np.abs(sv[sb].astype(np.float32) - so[sb].astype(np.float32))
        rel = diffs / np.maximum(np.abs(so[sb].astype(np.float32)), 1e-12)
        report["per_permno"][p] = {
            "oracle_months": int((~np.isnan(so)).sum()),
            "mine_months": int((~np.isnan(sv)).sum()),
            "common": int(sb.sum()),
            "max_rel_diff": float(rel.max()) if sb.any() else None,
            "oracle_only": int((~np.isnan(so) & np.isnan(sv)).sum()),
            "mine_only": int((np.isnan(so) & ~np.isnan(sv)).sum()),
        }
    return report


if __name__ == "__main__":
    import json

    liquid = [10107, 14593, 11850]  # MSFT, AAPL, XOM
    rng = np.random.default_rng(20260801)

    ann = announcement_return_panel()
    tf = trendfactor_panel()

    orc_cols = pq.ParquetFile(ROOT / "Data" / "golden" / "oracle" / "TrendFactor.parquet").schema_arrow.names
    candidates = [int(c) for c in orc_cols if c != "month" and int(c) not in liquid]
    random2 = [int(x) for x in rng.choice(candidates, size=2, replace=False)]
    permnos = liquid + random2
    print("test permnos:", permnos)

    print(json.dumps({"AnnouncementReturn": _compare("AnnouncementReturn", ann, permnos)}, indent=1))
    print(json.dumps({"TrendFactor": _compare("TrendFactor", tf, permnos)}, indent=1))

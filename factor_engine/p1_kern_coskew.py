"""Numba kernels for the two coskewness factors.

Coskewness  -- Harvey & Siddique (2000) systematic coskewness from MONTHLY
               simple excess returns over an anchored trailing 60-calendar-month
               window (reference: Predictors/Coskewness.py).
CoskewACX   -- Ang, Chen & Xing (2006, Table 8B) coskewness from DAILY log
               excess returns over an anchored trailing 12-calendar-month
               window (reference: Predictors/CoskewACX.py).

Both reference scripts build windows with the same K-batch modulo trick
(K = 60 months for Coskewness, K = 12 calendar months for CoskewACX): rows are
sorted permno-asc / time-desc, rows whose month code is congruent to the batch
index (mod K) are stamped with their own month code, the stamp is forward-filled
per permno (i.e. propagated to OLDER rows), and unstamped rows are dropped.
Net semantics, replicated exactly here per stock:

* Every permno-month that has at least one post-join row becomes an anchor t
  exactly once.
* The window of anchor t contains the stock's rows whose month code lies in
  (t_prev, t], where t_prev is the stock's most recent DATA-BEARING month code
  strictly before t with t_prev == t (mod K).  When the stock has data exactly
  K months earlier this is the trailing K calendar months [t-K+1, t]; when it
  does not, OLDER rows LEAK IN -- the window extends back to the previous
  existing same-class month, or to the start of the stock's history if there is
  none (the reference's gap-leak quirk, kept deliberately).
* The value is stamped at the anchor month t itself (window includes t; no lag).

Moment structure (identical in both factors; two-pass POPULATION moments,
every mean is sum/count-of-contributing-rows, no ddof anywhere):

    E_ret  = mean(ret) over rows with VALID ret only        (polars skips nulls)
    E_mkt  = mean(mkt) over ALL window rows                 (mkt never null)
    r~ = ret - E_ret (only defined on valid rows),  m~ = mkt - E_mkt
    E_ret_mkt2 = mean(r~ * m~^2)  over valid-ret rows
    E_ret2     = mean(r~^2)       over valid-ret rows
    E_mkt2     = mean(m~^2)       over ALL window rows (incl. null-ret rows)
    value = E_ret_mkt2 / ( sqrt(E_ret2) * E_mkt2 )          # denominator uses
                                                            # E_mkt2, NOT its sqrt

FIDELITY ASYMMETRY (the load-bearing trap): market moments average over ALL
window rows while stock moments average over valid-ret rows only.  Null-ret
rows are therefore window members that contribute to E_mkt/E_mkt2 but not to
E_ret/E_ret2/E_ret_mkt2 or to nobs.

Edge semantics (kept, not sanitized -- the references write these to CSV):
0/0 -> NaN (exact zeros from two-pass demeaning, no one-pass dust); x/0 -> +-inf;
CoskewACX log transform: ret == -1 -> -inf kept as a VALID observation whose
propagation turns the group's stock moments (and the value) into NaN; ret < -1
-> NaN kept likewise.  Neither occurs in the current dailyCRSP vintage
(verified: zero rows with ret <= -1), but the semantics are implemented.

Output contract: each public function returns a LONG DataFrame
[permno (int64), time_avail_m (datetime64[ns] month-start), value (float64)].
"""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from p1_daily_kernels import (
    ROOT,
    _month_codes,
    _stock_offsets,
    iter_buckets,
    load_daily_ff,
)

INTERMEDIATE = ROOT / "Open_Source_Asset_Pricing" / "Signals" / "pyData" / "Intermediate"

# dailyCRSP hard filter in CoskewACX.py L39: keep time_d >= 1962-07-02.
_ACX_START_D = np.datetime64("1962-07-02", "D").astype(np.int64)


@numba.njit(cache=True, error_model="numpy")
def _coskew_windows_stock(
    ret: np.ndarray,
    mkt: np.ndarray,
    valid: np.ndarray,
    codes: np.ndarray,
    cycle: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Anchored-window coskewness for ONE stock's contiguous rows.

    Inputs are one stock's rows sorted ascending by month code:
      ret   -- (possibly transformed) stock return; only read where valid=True,
               may be NaN/-inf on valid rows (CoskewACX log edge) and then
               propagates IEEE-style exactly like polars means over non-nulls.
      mkt   -- market return, never NaN.
      valid -- True where the reference's ret is non-null (nulls are excluded
               from stock moments and nobs but stay window members).
      codes -- int64 month codes (months since 1970-01, non-decreasing).  The
               reference uses months since 1960-01; the 120-month offset is
               congruent to 0 mod 60 and mod 12, so classes and windows agree.
      cycle -- 60 (Coskewness) or 12 (CoskewACX).

    Emits one row per DISTINCT month code t (every data month is an anchor):
    window = rows strictly after the stock's previous same-class (t mod cycle)
    data month's block, through the end of t's block -- gap-leak included.
    Returns (anchor_codes, values, nobs) with NO min-obs filtering; callers
    apply their factor's rule.  nobs counts valid-ret rows in the window.
    Groups with zero valid rets emit value = NaN (the reference emits null;
    both land as NaN in downstream float frames).

    CONSTANT-WINDOW EXACTNESS: when every valid ret in the window is bitwise
    identical (stale stocks: ret = 0.0 daily for months while daily rf is a
    constant, so the log excess return is one repeated constant), the mean is
    taken as that value EXACTLY instead of sum/count.  Polars returns the exact
    constant here, so the reference gets exact-zero demeaned stock moments and
    0/0 = NaN; a naive sequential sum/count mean is ~1 ulp off, and the ratio
    structure cancels the dust magnitude leaving sign(dust) * (n_all/n_valid
    scaled) ~= +-1.0 -- a fabricated value where the reference publishes NaN
    (observed on 57 CoskewACX cells before this branch existed).  If any valid
    ret is NaN, the equality chain breaks or the mean is NaN either way, and
    NaN propagates exactly like polars.

    error_model="numpy": divisions follow IEEE (0/0 -> NaN, x/0 -> +-inf),
    matching polars float division -- these are honest reference semantics,
    not error suppression.
    """

    n = ret.shape[0]
    out_code = np.empty(n, dtype=np.int64)
    out_val = np.empty(n, dtype=np.float64)
    out_nobs = np.empty(n, dtype=np.int64)
    # prev_end[c] = end offset (exclusive) of the block of this stock's most
    # recent data month with month-code class c; 0 until that class is seen,
    # which reproduces the reference's leak back to the start of history.
    prev_end = np.zeros(cycle, dtype=np.int64)
    n_out = 0
    i = 0
    while i < n:
        t = codes[i]
        j = i
        while j < n and codes[j] == t:
            j += 1
        c = (t % cycle + cycle) % cycle
        lo = prev_end[c]
        # Point-in-time bound: the anchored window must not reach rows older than
        # 12 calendar months before t. The reference (prev_end back to history
        # start) lets a stock relisted after a multi-year gap pull pre-gap rows
        # into its window; its inflated nobs then dominates the RELATIVE
        # min-obs rule (max_nobs - nobs <= 5) and silently drops every other
        # stock in that anchor month (2025-07..12 collapsed to 1 stock/month at
        # the 2026-08-17 refresh). Deviation registered section D.
        while lo < j and codes[lo] <= t - cycle:
            lo += 1
        n_all = j - lo
        # pass 1: means (population; market over ALL rows, stock over valid)
        sum_mkt = 0.0
        sum_ret = 0.0
        n_valid = 0
        first_ret = 0.0
        all_equal = True
        for k in range(lo, j):
            sum_mkt += mkt[k]
            if valid[k]:
                v = ret[k]
                if n_valid == 0:
                    first_ret = v
                elif v != first_ret:  # NaN anywhere -> False, NaN path below
                    all_equal = False
                n_valid += 1
                sum_ret += v
        e_mkt = sum_mkt / n_all
        if n_valid == 0:
            val = np.nan
        else:
            # constant window: the exact mean, so demeaned moments are exact
            # zeros (0/0 -> NaN) like polars, not 1-ulp dust (see docstring)
            e_ret = first_ret if all_equal else sum_ret / n_valid
            # pass 2: exact demeaned moments (no one-pass forms)
            s_mkt2_all = 0.0
            s_ret2 = 0.0
            s_ret_mkt2 = 0.0
            for k in range(lo, j):
                dm = mkt[k] - e_mkt
                dm2 = dm * dm
                s_mkt2_all += dm2
                if valid[k]:
                    dr = ret[k] - e_ret
                    s_ret2 += dr * dr
                    s_ret_mkt2 += dr * dm2
            e_mkt2 = s_mkt2_all / n_all
            e_ret2 = s_ret2 / n_valid
            e_ret_mkt2 = s_ret_mkt2 / n_valid
            val = e_ret_mkt2 / (np.sqrt(e_ret2) * e_mkt2)
        out_code[n_out] = t
        out_val[n_out] = val
        out_nobs[n_out] = n_valid
        n_out += 1
        prev_end[c] = j
        i = j
    return out_code[:n_out], out_val[:n_out], out_nobs[:n_out]


def _finalize(ids: np.ndarray, codes: np.ndarray, vals: np.ndarray) -> pd.DataFrame:
    """Assemble the LONG contract frame, sorted (permno, time_avail_m)."""

    order = np.lexsort((codes, ids))
    return pd.DataFrame(
        {
            "permno": ids[order],
            "time_avail_m": codes[order].astype("datetime64[M]").astype("datetime64[ns]"),
            "value": vals[order],
        }
    )


def coskewness_panel() -> pd.DataFrame:
    """Coskewness (Harvey-Siddique 2000) -- monthly, 60-month anchored window.

    Exact semantics implemented (mirrors Predictors/Coskewness.py):
      * monthlyCRSP [permno, time_avail_m, ret] INNER-joined to monthlyFF
        [time_avail_m, mktrf, rf], keeping only months with non-null mktrf and
        rf (months absent from FF drop their stock rows entirely -- they are
        never window members).  No share-code/exchange/date filters.
      * ret := ret - rf (simple excess; null ret stays null = invalid row),
        mkt := mktrf (already excess, never null).
      * Anchored calendar 60-month window per stock (see module docstring),
        gap-leak included; two-pass population moments with the market/stock
        row-membership asymmetry; value = E[r~ m~^2]/(sqrt(E[r~^2]) E[m~^2]).
      * Min-obs: keep anchors with nobs >= 12 valid monthly excess returns in
        the window.  No cross-sectional filter.
      * Stamped at the anchor month itself (window includes it; no lag).
        NaN (0/0 on zero-variance stock windows) and +-inf survive to the
        output exactly as the reference writes them to CSV.
    """

    crsp = pq.read_table(
        INTERMEDIATE / "monthlyCRSP.parquet", columns=["permno", "time_avail_m", "ret"]
    ).to_pandas()
    ff = (
        pq.read_table(INTERMEDIATE / "monthlyFF.parquet", columns=["time_avail_m", "mktrf", "rf"])
        .to_pandas()
        .sort_values("time_avail_m")
    )
    ff_codes = _month_codes(ff["time_avail_m"].to_numpy())
    mktrf = ff["mktrf"].to_numpy(np.float64)
    rf = ff["rf"].to_numpy(np.float64)
    ff_ok = ~(np.isnan(mktrf) | np.isnan(rf))  # reference L48 (no-op on current vintage)
    ff_codes, mktrf, rf = ff_codes[ff_ok], mktrf[ff_ok], rf[ff_ok]

    permno = crsp["permno"].to_numpy(np.int64)
    codes = _month_codes(crsp["time_avail_m"].to_numpy())
    ret_raw = crsp["ret"].to_numpy(np.float64)

    # inner join on month code (ff_codes unique + sorted)
    pos = np.searchsorted(ff_codes, codes)
    pos_c = np.minimum(pos, len(ff_codes) - 1)
    keep = ff_codes[pos_c] == codes
    permno, codes, ret_raw, idx = permno[keep], codes[keep], ret_raw[keep], pos_c[keep]

    order = np.lexsort((codes, permno))
    permno, codes, ret_raw, idx = permno[order], codes[order], ret_raw[order], idx[order]

    valid = ~np.isnan(ret_raw)  # monthlyCRSP ret has nulls but no float NaN (verified)
    ret_ex = ret_raw - rf[idx]
    mkt = mktrf[idx]

    starts, ids = _stock_offsets(permno)
    all_ids: list[np.ndarray] = []
    all_codes: list[np.ndarray] = []
    all_vals: list[np.ndarray] = []
    all_nobs: list[np.ndarray] = []
    for s in range(len(ids)):
        lo, hi = starts[s], starts[s + 1]
        c, v, nob = _coskew_windows_stock(
            ret_ex[lo:hi], mkt[lo:hi], valid[lo:hi], codes[lo:hi], 60
        )
        all_ids.append(np.full(c.shape[0], ids[s], dtype=np.int64))
        all_codes.append(c)
        all_vals.append(v)
        all_nobs.append(nob)

    ids_a = np.concatenate(all_ids)
    codes_a = np.concatenate(all_codes)
    vals_a = np.concatenate(all_vals)
    nobs_a = np.concatenate(all_nobs)
    keep_out = nobs_a >= 12  # reference L161
    return _finalize(ids_a[keep_out], codes_a[keep_out], vals_a[keep_out])


def coskew_acx_panel() -> pd.DataFrame:
    """CoskewACX (Ang-Chen-Xing 2006) -- daily, 12-calendar-month anchored window.

    Exact semantics implemented (mirrors Predictors/CoskewACX.py):
      * dailyCRSP [permno, time_d, ret] with HARD FILTER time_d >= 1962-07-02,
        INNER-joined to dailyFF [time_d, mktrf, rf] on the exact day, keeping
        only days with non-null mktrf and rf.  Days absent from dailyFF never
        enter any window (dropped, not NaN-masked).
      * Log excess returns, computed exactly as the script's ln(1+x) forms:
        mkt := ln(1 + mktrf + rf) - ln(1 + rf); ret := ln(1 + ret) - ln(1 + rf).
        Null ret stays null (invalid row, still a window member); ret == -1
        gives -inf and ret < -1 gives NaN, both on VALID rows that poison the
        group's stock moments to NaN exactly like polars (none in current data).
      * Anchored calendar 12-month window per stock over its daily rows (see
        module docstring; class = calendar month of year), gap-leak included;
        two-pass population moments with the market/stock row-membership
        asymmetry; value = E[r~ m~^2] / (sqrt(E[r~^2]) * E[m~^2]).
      * Min-obs is RELATIVE and CROSS-SECTIONAL only (reference L162-168):
        nobs = valid-ret days in the window; max_nobs = max of nobs across ALL
        permnos sharing anchor month t (computed here globally across buckets,
        equivalent because each anchor month lives in exactly one reference
        batch); keep iff max_nobs - nobs <= 5.  NO absolute minimum: groups
        with zero valid rets emit NaN (reference: null) and survive only when
        max_nobs <= 5.
      * Stamped at the anchor month itself (window includes it; no lag).
    """

    ff_dates, ff_fac = load_daily_ff()  # sorted; cols = mktrf smb hml rf
    ff_mktrf = ff_fac[:, 0]
    ff_rf = ff_fac[:, 3]
    ff_ok = ~(np.isnan(ff_mktrf) | np.isnan(ff_rf))  # reference L53 (no-op on current vintage)
    ff_dates, ff_mktrf, ff_rf = ff_dates[ff_ok], ff_mktrf[ff_ok], ff_rf[ff_ok]
    # per-FF-day transforms, script-literal ln(1+x) (NOT log1p)
    lrf = np.log(1.0 + ff_rf)
    lmkt = np.log(1.0 + ff_mktrf + ff_rf) - lrf

    all_ids: list[np.ndarray] = []
    all_codes: list[np.ndarray] = []
    all_vals: list[np.ndarray] = []
    all_nobs: list[np.ndarray] = []
    for bucket in iter_buckets(["ret"]):
        permno = bucket["permno"].to_numpy(np.int64)
        time_d = bucket["time_d"].to_numpy()
        dates = time_d.astype("datetime64[D]").astype(np.int64)
        months = _month_codes(time_d)
        ret_raw = bucket["ret"].to_numpy(np.float64)

        pos = np.searchsorted(ff_dates, dates)
        pos_c = np.minimum(pos, len(ff_dates) - 1)
        keep = (dates >= _ACX_START_D) & (ff_dates[pos_c] == dates)
        permno, months, ret_raw, idx = permno[keep], months[keep], ret_raw[keep], pos_c[keep]

        valid = ~np.isnan(ret_raw)  # dailyCRSP ret has nulls but no float NaN (verified)
        with np.errstate(divide="ignore", invalid="ignore"):
            # ln(0) = -inf / ln(<0) = NaN are kept per reference edge semantics
            ret_log = np.log(1.0 + ret_raw) - lrf[idx]
        mkt = lmkt[idx]

        starts, ids = _stock_offsets(permno)
        for s in range(len(ids)):
            lo, hi = starts[s], starts[s + 1]
            c, v, nob = _coskew_windows_stock(
                ret_log[lo:hi], mkt[lo:hi], valid[lo:hi], months[lo:hi], 12
            )
            all_ids.append(np.full(c.shape[0], ids[s], dtype=np.int64))
            all_codes.append(c)
            all_vals.append(v)
            all_nobs.append(nob)

    ids_a = np.concatenate(all_ids)
    codes_a = np.concatenate(all_codes)
    vals_a = np.concatenate(all_vals)
    nobs_a = np.concatenate(all_nobs)

    # relative cross-sectional filter: per anchor month, keep max_nobs - nobs <= 5
    base = codes_a.min()
    max_nobs = np.zeros(codes_a.max() - base + 1, dtype=np.int64)
    np.maximum.at(max_nobs, codes_a - base, nobs_a)
    keep_out = (max_nobs[codes_a - base] - nobs_a) <= 5
    return _finalize(ids_a[keep_out], codes_a[keep_out], vals_a[keep_out])

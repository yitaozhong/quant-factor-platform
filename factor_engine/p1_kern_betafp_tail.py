"""Bucket-streamed numba kernels for BetaFP and BetaTailRisk (WS8 continuation).

Two factors, one function each, contract ``panel() -> long [permno, time_avail_m, value]``:

* ``betafp_panel``       — Frazzini-Pedersen beta (ZZ2_BetaFP.py): |rolling corr of
  overlapping 3-day log returns| x (252-day stock vol / 252-day market vol),
  last finite daily value per calendar month.
* ``betatailrisk_panel`` — Kelly-Jiang tail-risk beta (BetaTailRisk.py): PART 1
  builds the market-wide monthly tail-risk series from the pooled daily
  cross-section (5th-percentile threshold + Hill-style mean of ln(ret/retp5));
  PART 2 runs per-stock rolling 120-month OLS of monthly raw returns on that
  series and keeps the slope.

Fidelity provenance (all semantics below verified against the reference scripts
and, where polars/polars_ols behavior was ambiguous, probed empirically on the
installed versions — polars 1.40.1, the same environment that froze the oracles
on 2026-07-23):

BetaFP (ZZ2_BetaFP.py):
  * dailyCRSP INNER-joined to dailyFF on time_d — stock days without FF coverage
    (149 pre-1926-07 days) are dropped BEFORE any window forms; windows are
    positional over the post-join rows.
  * LogRet = log1p(RAW ret) — NOT excess (the script's with_columns overwrites
    ``ret`` to ret-rf in the same batch, but every expression evaluates against
    the input frame and the excess column is never used again). LogMkt =
    log1p(mktrf), which IS excess-of-rf. The asymmetry is real script behavior.
  * 3-day overlapping returns: tempRi[t] = LogRet[t] + LogRet[t-1] + LogRet[t-2]
    (positional shifts inside the permno; first two rows null; a null ret
    poisons three consecutive rows). Same for tempRm from LogMkt.
  * Rolling windows are trailing POSITIONAL row windows (252 rows / min 120
    non-null for the two vols; 1260 rows / min 500 non-null for each rolling
    corr component). Probed: nulls occupy positions but do not count toward
    min_samples; aggregates skip them; each of the five rolling components
    aggregates over ITS OWN non-null rows (E[xy], E[x], E[y] can disagree on
    row sets when the stock has null-ret days — replicated here via separate
    per-series prefix counts).
  * corr_hat = (mean(xy) - mean(x)mean(y)) / (std(x, ddof=1) * std(y, ddof=1))
    — the script's mixed population-cov / sample-std construction, kept as-is.
  * BetaFP = sqrt(|corr_hat^2|) * sd252(LogRet)/sd252(LogMkt); the correlation
    SIGN is destroyed by construction (reference behavior).
  * Probed: polars rolling_std of a constant window is exactly 0.0 (not null) —
    negative-variance dust is clamped to 0 here to match; a zero std then flows
    to inf/NaN and is removed by the monthly is_finite gate exactly like the
    reference.
  * Monthly stamping: time_avail_m = calendar month of time_d, value = LAST
    FINITE daily BetaFP within the month (null/NaN/inf rows skipped), no lag.
  * dailyCRSP holds no stored-NaN returns and no ret <= -1 (verified), so the
    NaN-vs-null distinction that pandas erases cannot bite: NaN here == null
    in the reference.

BetaTailRisk (BetaTailRisk.py):
  * PART 1 (series): retp5(month) = polars quantile(0.05, interpolation='lower')
    of ALL daily returns pooled across stocks that month = the element at index
    trunc(0.05*(n-1)) of the ascending non-null sort — an actual data point.
    Tail rows are ret <= retp5 (inclusive; >= 1 row always qualifies), and the
    series value is the arithmetic mean of ln(ret/retp5) over them. Built here
    in three bucket streams: (A) per-month valid counts -> exact selection rank,
    (B) bounded max-heap per month selecting the (rank+1) smallest returns ->
    retp5 is exactly the reference's data point, (C) accumulate the tail mean
    with the exact threshold, so boundary TIES beyond the heap are counted
    exactly. The builder raises if any series value is non-finite: a NaN month
    would trigger polars_ols' state-poisoning (probed: one NaN regressor row
    nulls every subsequent output), which this kernel intentionally does not
    replicate — surfacing beats silently diverging.
  * PART 2 (regressions) uses monthlyCRSP (the script regresses MONTHLY raw
    returns on the series; the daily buckets play no role here). polars_ols
    rolling_ols(window_size=120, min_periods=72, add_intercept, null_policy=
    'drop') semantics, probed on the installed version:
      - the window is the trailing 120 VALID rows (ret and tailex both
        non-null) of the permno's joined series — positional over valid rows,
        calendar gaps stretch the span;
      - min_periods gates on the VALID-row count (first emission at the 72nd
        valid row), NOT on the raw row index (the raw-row-gate defect recorded
        for Beta in p1_daily_kernel_specs.json does not reproduce on the
        installed version, which postdates that note and froze the oracle);
      - once gated, a coefficient is emitted at EVERY subsequent row of the
        permno, including rows whose own pair is invalid (probed: the invalid
        row carries the trailing-valid-window coefficient).
    Slope = sum((x-xbar)(y-ybar))/sum((x-xbar)^2) over the window; intercept
    fitted but discarded; no dof adjustment anywhere.
  * Registered deviation: a singular window (zero tailex variance) makes
    polars_ols emit finite minimum-norm garbage (probed); this kernel emits
    nothing. Unreachable on real data — the series varies every month.
  * Output filter: slope non-null AND shrcd <= 11 (null shrcd drops — NaN <= 11
    is False here, same outcome). Stamped at the row's own time_avail_m, no lag.
  * monthlyCRSP holds no stored-NaN ret and no duplicate permno-months
    (verified), so NaN == null again.

Read-only imports from p1_daily_kernels: iter_buckets, _stock_offsets,
_month_codes, load_daily_ff. This module registers nothing — wiring into
p1_factor_specs/engine is a separate, deliberate step.
"""

from __future__ import annotations

from pathlib import Path

import numba
import numpy as np
import pandas as pd

from p1_daily_kernels import (
    ROOT,
    _month_codes,
    _stock_offsets,
    iter_buckets,
    load_daily_ff,
)

INTERMEDIATE_DIR = ROOT / "Open_Source_Asset_Pricing" / "Signals" / "pyData" / "Intermediate"
MONTHLY_CRSP = INTERMEDIATE_DIR / "monthlyCRSP.parquet"

# Month-code addressing for the tail-risk accumulators: codes are months since
# 1970-01 (datetime64[M] int64); shift by 1900-01 so 1900-01..2099-12 index a
# fixed 2400-slot array. Data outside that range is a data bug and raises.
_MONTH_BASE = int(np.datetime64("1900-01", "M").astype(np.int64))  # -840
_MONTH_SLOTS = 2400


def _codes_to_month_index(time_values: np.ndarray) -> np.ndarray:
    """datetime64[ns] month codes -> indices into the fixed slot arrays."""

    idx = _month_codes(time_values) - _MONTH_BASE
    if len(idx) and (idx.min() < 0 or idx.max() >= _MONTH_SLOTS):
        raise ValueError(
            f"month codes outside 1900-01..2099-12 (index range [{idx.min()}, {idx.max()}]) "
            f"— widen _MONTH_BASE/_MONTH_SLOTS or investigate the data"
        )
    return idx


# ---------------------------------------------------------------------------
# BetaFP — Frazzini-Pedersen beta (ZZ2_BetaFP.py)
# ---------------------------------------------------------------------------


@numba.njit(cache=True, parallel=True, error_model="numpy")
def _betafp_bucket_kernel(
    ret: np.ndarray,
    mkt: np.ndarray,
    months: np.ndarray,
    starts: np.ndarray,
    window_vol: int,
    min_vol: int,
    window_cor: int,
    min_cor: int,
    out_month: np.ndarray,
    out_val: np.ndarray,
    out_n: np.ndarray,
) -> None:
    """One bucket, all stocks in parallel; per stock emit the last finite daily
    BetaFP of each calendar month into out_month/out_val[starts[s]:...].

    Arrays are the stock's post-FF-inner-join rows (compacted by the caller).
    NaN ret == null ret (verified: the source holds no stored NaNs). Window
    statistics use per-window prefix-sum differences — exact enough at f64 for
    the 1e-6/1e-5 validation tolerance, and each rolling component counts and
    aggregates its OWN non-null rows exactly like the five independent polars
    rolling expressions in the script.
    """

    n_stocks = starts.shape[0] - 1
    for s in numba.prange(n_stocks):
        lo = starts[s]
        hi = starts[s + 1]
        n = hi - lo
        lr = np.empty(n, dtype=np.float64)
        lm = np.empty(n, dtype=np.float64)
        for i in range(n):
            lr[i] = np.log1p(ret[lo + i])
            lm[i] = np.log1p(mkt[lo + i])
        # overlapping 3-day log returns, positional within the stock
        x = np.empty(n, dtype=np.float64)
        y = np.empty(n, dtype=np.float64)
        for i in range(n):
            if i >= 2:
                x[i] = lr[i] + lr[i - 1] + lr[i - 2]
                y[i] = lm[i] + lm[i - 1] + lm[i - 2]
            else:
                x[i] = np.nan
                y[i] = np.nan
        # prefix count/sum/sumsq per series (NaN rows occupy positions, add 0)
        c_lr = np.zeros(n + 1, dtype=np.int64)
        s_lr = np.zeros(n + 1, dtype=np.float64)
        q_lr = np.zeros(n + 1, dtype=np.float64)
        c_lm = np.zeros(n + 1, dtype=np.int64)
        s_lm = np.zeros(n + 1, dtype=np.float64)
        q_lm = np.zeros(n + 1, dtype=np.float64)
        c_x = np.zeros(n + 1, dtype=np.int64)
        s_x = np.zeros(n + 1, dtype=np.float64)
        q_x = np.zeros(n + 1, dtype=np.float64)
        c_y = np.zeros(n + 1, dtype=np.int64)
        s_y = np.zeros(n + 1, dtype=np.float64)
        q_y = np.zeros(n + 1, dtype=np.float64)
        c_p = np.zeros(n + 1, dtype=np.int64)
        s_p = np.zeros(n + 1, dtype=np.float64)
        for i in range(n):
            v = lr[i]
            if np.isnan(v):
                c_lr[i + 1] = c_lr[i]
                s_lr[i + 1] = s_lr[i]
                q_lr[i + 1] = q_lr[i]
            else:
                c_lr[i + 1] = c_lr[i] + 1
                s_lr[i + 1] = s_lr[i] + v
                q_lr[i + 1] = q_lr[i] + v * v
            v = lm[i]
            if np.isnan(v):
                c_lm[i + 1] = c_lm[i]
                s_lm[i + 1] = s_lm[i]
                q_lm[i + 1] = q_lm[i]
            else:
                c_lm[i + 1] = c_lm[i] + 1
                s_lm[i + 1] = s_lm[i] + v
                q_lm[i + 1] = q_lm[i] + v * v
            v = x[i]
            if np.isnan(v):
                c_x[i + 1] = c_x[i]
                s_x[i + 1] = s_x[i]
                q_x[i + 1] = q_x[i]
            else:
                c_x[i + 1] = c_x[i] + 1
                s_x[i + 1] = s_x[i] + v
                q_x[i + 1] = q_x[i] + v * v
            v = y[i]
            if np.isnan(v):
                c_y[i + 1] = c_y[i]
                s_y[i + 1] = s_y[i]
                q_y[i + 1] = q_y[i]
            else:
                c_y[i + 1] = c_y[i] + 1
                s_y[i + 1] = s_y[i] + v
                q_y[i + 1] = q_y[i] + v * v
            v = x[i] * y[i]
            if np.isnan(v):
                c_p[i + 1] = c_p[i]
                s_p[i + 1] = s_p[i]
            else:
                c_p[i + 1] = c_p[i] + 1
                s_p[i + 1] = s_p[i] + v
        # walk rows; remember the last finite BetaFP per calendar month
        n_emit = 0
        cur_month = np.int64(-(2**62))
        have = False
        last_val = 0.0
        for i in range(n):
            m = months[lo + i]
            if m != cur_month:
                if have:
                    out_month[lo + n_emit] = cur_month
                    out_val[lo + n_emit] = last_val
                    n_emit += 1
                cur_month = m
                have = False
            b = np.nan
            l1 = i - (window_vol - 1) if i >= window_vol - 1 else 0
            cr = c_lr[i + 1] - c_lr[l1]
            cm = c_lm[i + 1] - c_lm[l1]
            if cr >= min_vol and cm >= min_vol and cr >= 2 and cm >= 2:
                su = s_lr[i + 1] - s_lr[l1]
                var_r = ((q_lr[i + 1] - q_lr[l1]) - su * su / cr) / (cr - 1)
                if var_r < 0.0:
                    var_r = 0.0  # polars yields exactly 0.0 on constant windows
                sd_r = np.sqrt(var_r)
                su = s_lm[i + 1] - s_lm[l1]
                var_m = ((q_lm[i + 1] - q_lm[l1]) - su * su / cm) / (cm - 1)
                if var_m < 0.0:
                    var_m = 0.0
                sd_m = np.sqrt(var_m)
                l2 = i - (window_cor - 1) if i >= window_cor - 1 else 0
                cx = c_x[i + 1] - c_x[l2]
                cy = c_y[i + 1] - c_y[l2]
                cp = c_p[i + 1] - c_p[l2]
                if cx >= min_cor and cy >= min_cor and cp >= min_cor:
                    mean_x = (s_x[i + 1] - s_x[l2]) / cx
                    var_x = ((q_x[i + 1] - q_x[l2]) - cx * mean_x * mean_x) / (cx - 1)
                    if var_x < 0.0:
                        var_x = 0.0
                    std_x = np.sqrt(var_x)
                    mean_y = (s_y[i + 1] - s_y[l2]) / cy
                    var_y = ((q_y[i + 1] - q_y[l2]) - cy * mean_y * mean_y) / (cy - 1)
                    if var_y < 0.0:
                        var_y = 0.0
                    std_y = np.sqrt(var_y)
                    mean_p = (s_p[i + 1] - s_p[l2]) / cp
                    cov = mean_p - mean_x * mean_y
                    corr = cov / (std_x * std_y)  # inf/NaN allowed; is_finite gate below
                    r2 = corr * corr
                    b = np.sqrt(np.abs(r2)) * (sd_r / sd_m)
            if np.isfinite(b):
                last_val = b
                have = True
        if have:
            out_month[lo + n_emit] = cur_month
            out_val[lo + n_emit] = last_val
            n_emit += 1
        out_n[s] = n_emit


def betafp_panel(permnos: np.ndarray | list[int] | None = None) -> pd.DataFrame:
    """Long frame [permno, time_avail_m, value] of BetaFP over all buckets.

    ``permnos`` restricts computation to the given stocks (self-test/spot-diff
    convenience; the per-stock computation is independent, so the restriction
    is exact). Production callers omit it.
    """

    ff_dates, ff_fac = load_daily_ff()
    mkt_all = ff_fac[:, 0]
    wanted = None if permnos is None else np.asarray(permnos, dtype=np.int64)
    parts: list[pd.DataFrame] = []
    for bucket in iter_buckets(["ret"]):
        permno = bucket["permno"].to_numpy(np.int64)
        if wanted is not None:
            sel = np.isin(permno, wanted)
            if not sel.any():
                continue
            bucket = bucket[sel]
            permno = permno[sel]
        dates = bucket["time_d"].to_numpy("datetime64[D]").astype(np.int64)
        pos = np.searchsorted(ff_dates, dates)
        pos_clipped = np.minimum(pos, len(ff_dates) - 1)
        keep = ff_dates[pos_clipped] == dates  # INNER join: unmatched days drop
        if not keep.any():
            continue
        permno_k = permno[keep]
        months = _month_codes(bucket["time_d"].to_numpy()[keep])
        ret = bucket["ret"].to_numpy(np.float64)[keep]
        mkt = mkt_all[pos_clipped[keep]]
        starts, ids = _stock_offsets(permno_k)
        n_rows = len(ret)
        out_month = np.empty(n_rows, dtype=np.int64)
        out_val = np.empty(n_rows, dtype=np.float64)
        out_n = np.zeros(len(ids), dtype=np.int64)
        _betafp_bucket_kernel(
            ret, mkt, months, starts, 252, 120, 1260, 500, out_month, out_val, out_n
        )
        mask = np.zeros(n_rows, dtype=np.bool_)
        for s in range(len(ids)):
            mask[starts[s] : starts[s] + out_n[s]] = True
        parts.append(
            pd.DataFrame(
                {
                    "permno": np.repeat(ids, out_n),
                    "month_code": out_month[mask],
                    "value": out_val[mask],
                }
            )
        )
    long = pd.concat(parts, ignore_index=True)
    long["time_avail_m"] = long.pop("month_code").astype("datetime64[M]").astype("datetime64[ns]")
    return long[["permno", "time_avail_m", "value"]]


# ---------------------------------------------------------------------------
# BetaTailRisk — Kelly-Jiang tail risk beta (BetaTailRisk.py)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _tail_select_kernel(
    ret: np.ndarray,
    midx: np.ndarray,
    heapbuf: np.ndarray,
    heap_off: np.ndarray,
    heap_size: np.ndarray,
    heap_cap: np.ndarray,
) -> None:
    """Stream rows into per-month bounded max-heaps of the k+1 smallest returns.

    After all buckets, each full heap's root is exactly the reference's
    quantile(0.05, 'lower') data point (ascending-sort element at rank k).
    """

    for i in range(ret.shape[0]):
        v = ret[i]
        if np.isnan(v):
            continue
        m = midx[i]
        cap = heap_cap[m]
        if cap == 0:
            continue
        off = heap_off[m]
        sz = heap_size[m]
        if sz < cap:
            heapbuf[off + sz] = v
            j = sz
            heap_size[m] = sz + 1
            while j > 0:
                p = (j - 1) >> 1
                if heapbuf[off + p] < heapbuf[off + j]:
                    tmp = heapbuf[off + p]
                    heapbuf[off + p] = heapbuf[off + j]
                    heapbuf[off + j] = tmp
                    j = p
                else:
                    break
        elif v < heapbuf[off]:
            heapbuf[off] = v
            j = 0
            while True:
                left = 2 * j + 1
                if left >= cap:
                    break
                big = left
                right = left + 1
                if right < cap and heapbuf[off + right] > heapbuf[off + left]:
                    big = right
                if heapbuf[off + big] > heapbuf[off + j]:
                    tmp = heapbuf[off + big]
                    heapbuf[off + big] = heapbuf[off + j]
                    heapbuf[off + j] = tmp
                    j = big
                else:
                    break


@numba.njit(cache=True, error_model="numpy")
def _tail_accum_kernel(
    ret: np.ndarray,
    midx: np.ndarray,
    retp5: np.ndarray,
    acc_sum: np.ndarray,
    acc_cnt: np.ndarray,
) -> None:
    """Accumulate sum/count of ln(ret/retp5) over tail rows (ret <= retp5).

    The exact <= threshold on the exact retp5 data point makes boundary ties
    count exactly as in the reference (each tie contributes ln(1) = 0 to the
    sum but inflates the denominator).
    """

    for i in range(ret.shape[0]):
        v = ret[i]
        if np.isnan(v):
            continue
        m = midx[i]
        p5 = retp5[m]
        if np.isnan(p5):
            continue
        if v <= p5:
            acc_sum[m] += np.log(v / p5)
            acc_cnt[m] += 1


def build_tailrisk_series(
    cache: str | Path | None = None, force: bool = False
) -> pd.DataFrame:
    """Monthly tail-risk series [time_avail_m, tailex] — script PART 1, exact.

    Three bucket streams: per-month valid counts (fixes the selection rank),
    bounded-heap selection of the rank+1 smallest pooled daily returns (retp5),
    then the tail mean with exact tie handling. ``cache`` (parquet path) skips
    the rebuild when present unless ``force``.

    Raises if any series value is non-finite: the reference's downstream
    polars_ols would then poison every later window (probed), a behavior this
    module refuses to replicate silently.
    """

    if cache is not None:
        cache = Path(cache)
        if cache.exists() and not force:
            return pd.read_parquet(cache)

    # pass A: per-month non-null counts
    counts = np.zeros(_MONTH_SLOTS, dtype=np.int64)
    for bucket in iter_buckets(["ret"]):
        midx = _codes_to_month_index(bucket["time_d"].to_numpy())
        ret = bucket["ret"].to_numpy(np.float64)
        counts += np.bincount(midx[~np.isnan(ret)], minlength=_MONTH_SLOTS)

    # selection rank: polars quantile(0.05, 'lower') index = trunc(0.05*(n-1)),
    # computed in f64 exactly as polars does
    nz = counts > 0
    cap = np.zeros(_MONTH_SLOTS, dtype=np.int64)
    cap[nz] = np.floor(0.05 * (counts[nz] - 1.0)).astype(np.int64) + 1
    heap_off = np.zeros(_MONTH_SLOTS + 1, dtype=np.int64)
    np.cumsum(cap, out=heap_off[1:])
    heapbuf = np.empty(int(heap_off[-1]), dtype=np.float64)
    heap_size = np.zeros(_MONTH_SLOTS, dtype=np.int64)

    # pass B: bounded-heap selection of the cap smallest returns per month
    for bucket in iter_buckets(["ret"]):
        midx = _codes_to_month_index(bucket["time_d"].to_numpy())
        ret = bucket["ret"].to_numpy(np.float64)
        _tail_select_kernel(ret, midx, heapbuf, heap_off[:-1], heap_size, cap)
    if not (heap_size == cap).all():
        raise AssertionError(
            "tail selection heaps did not fill to capacity — counts and stream disagree"
        )
    retp5 = np.full(_MONTH_SLOTS, np.nan)
    retp5[nz] = heapbuf[heap_off[:-1][nz]]

    # pass C: exact tail mean per month
    acc_sum = np.zeros(_MONTH_SLOTS, dtype=np.float64)
    acc_cnt = np.zeros(_MONTH_SLOTS, dtype=np.int64)
    for bucket in iter_buckets(["ret"]):
        midx = _codes_to_month_index(bucket["time_d"].to_numpy())
        ret = bucket["ret"].to_numpy(np.float64)
        _tail_accum_kernel(ret, midx, retp5, acc_sum, acc_cnt)
    if not (acc_cnt[nz] > 0).all():
        raise AssertionError(
            "a month with data produced no tail rows — retp5 must itself qualify"
        )
    tailex = acc_sum[nz] / acc_cnt[nz]
    if not np.isfinite(tailex).all():
        bad = np.flatnonzero(nz)[~np.isfinite(tailex)] + _MONTH_BASE
        raise ValueError(
            "non-finite tail-risk series values at months "
            f"{bad.astype('datetime64[M]')}: the reference's polars_ols would "
            "poison all subsequent windows with such a value — implement that "
            "explicitly before proceeding (refusing to diverge silently)"
        )
    series = pd.DataFrame(
        {
            "time_avail_m": (np.flatnonzero(nz) + _MONTH_BASE)
            .astype("datetime64[M]")
            .astype("datetime64[ns]"),
            "tailex": tailex,
        }
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        series.to_parquet(cache)
    return series


@numba.njit(cache=True, parallel=True)
def _tailbeta_kernel(
    y: np.ndarray,
    x: np.ndarray,
    starts: np.ndarray,
    window: int,
    min_obs: int,
    out_val: np.ndarray,
    out_has: np.ndarray,
) -> None:
    """Rolling OLS slope of y on x (+intercept) per stock, polars_ols
    rolling_ols(null_policy='drop') semantics as probed on the installed
    version: window = trailing `window` VALID pairs, gate = valid count >=
    min_obs, and once gated a slope is emitted at EVERY row (including rows
    whose own pair is invalid — they carry the trailing-valid-window value).

    Singular windows (sxx == 0) emit nothing — registered deviation from
    polars_ols' finite minimum-norm output; unreachable on the real series.
    """

    n_stocks = starts.shape[0] - 1
    for s in numba.prange(n_stocks):
        lo = starts[s]
        hi = starts[s + 1]
        n = hi - lo
        vt = np.empty(n, dtype=np.int64)
        c = 0
        for t in range(n):
            yv = y[lo + t]
            xv = x[lo + t]
            if not (np.isnan(yv) or np.isnan(xv)):
                vt[c] = lo + t
                c += 1
            if c < min_obs or c < 2:
                continue
            w = window if c >= window else c
            mx = 0.0
            my = 0.0
            for q in range(c - w, c):
                k = vt[q]
                mx += x[k]
                my += y[k]
            mx /= w
            my /= w
            sxx = 0.0
            sxy = 0.0
            for q in range(c - w, c):
                k = vt[q]
                dx = x[k] - mx
                sxx += dx * dx
                sxy += dx * (y[k] - my)
            if sxx > 0.0:
                out_val[lo + t] = sxy / sxx
                out_has[lo + t] = True


def betatailrisk_panel(
    tail_cache: str | Path | None = None,
    permnos: np.ndarray | list[int] | None = None,
) -> pd.DataFrame:
    """Long frame [permno, time_avail_m, value] of BetaTailRisk.

    PART 1 series via ``build_tailrisk_series`` (optionally cached at
    ``tail_cache``), PART 2 rolling regressions on monthlyCRSP. ``permnos``
    restricts PART 2 to the given stocks (exact — regressions are per-stock);
    the series is always built from the full cross-section.
    """

    series = build_tailrisk_series(cache=tail_cache)
    lookup = np.full(_MONTH_SLOTS, np.nan)
    scodes = series["time_avail_m"].to_numpy("datetime64[M]").astype(np.int64) - _MONTH_BASE
    lookup[scodes] = series["tailex"].to_numpy(np.float64)

    mc = pd.read_parquet(MONTHLY_CRSP, columns=["permno", "time_avail_m", "ret", "shrcd"])
    if permnos is not None:
        mc = mc[mc["permno"].isin(np.asarray(permnos, dtype=np.int64))]
    mc = mc.sort_values(["permno", "time_avail_m"], kind="stable", ignore_index=True)
    permno = mc["permno"].to_numpy(np.int64)
    codes = mc["time_avail_m"].to_numpy("datetime64[M]").astype(np.int64) - _MONTH_BASE
    if len(codes) and (codes.min() < 0 or codes.max() >= _MONTH_SLOTS):
        raise ValueError("monthlyCRSP months outside the fixed 1900-2099 slot range")
    x = lookup[codes]  # months absent from the series -> NaN -> dropped pair
    y = mc["ret"].to_numpy(np.float64)
    shrcd = mc["shrcd"].to_numpy(np.float64)

    starts, ids = _stock_offsets(permno)
    out_val = np.empty(len(y), dtype=np.float64)
    out_has = np.zeros(len(y), dtype=np.bool_)
    _tailbeta_kernel(y, x, starts, 120, 72, out_val, out_has)

    keep = out_has & (shrcd <= 11.0)  # NaN shrcd is False, matching null-drop
    return pd.DataFrame(
        {
            "permno": permno[keep],
            "time_avail_m": mc["time_avail_m"].to_numpy()[keep],
            "value": out_val[keep],
        }
    )

"""Bucket-streamed numba kernels for the daily/rolling factor family (WS8).

Data source: Data/daily_buckets/bucket=N/*.parquet — the one-time artifact holding
dailyCRSP reorganized into 64 permno-hash buckets, each sorted (permno, time_d),
so every stock's daily history is contiguous. A kernel streams one bucket
(~600 stocks, ~20 MB) at a time; peak memory is bucket-sized, never the
day×permno wide pivot (~8 GB/variable) that kept these factors script-backed.

Design guardrail (locked with the user): every kernel is BUCKET-SHAPED —
``kernel(bucket arrays) -> (permno, month, value) partials`` — so when the v2
executor arrives, the same functions slot into its parallel fold/merge/finalize
machinery and only this module's serial driver is replaced.

Fidelity: each factor built on these kernels is validated cell-by-cell against
its reference script's output (Data/golden/oracle/) through p1_validation.
Statistical definitions therefore matter to the bit: std is ddof=1; skewness is
the bias-adjusted Fisher–Pearson G1 (pandas/polars ``skew(bias=False)``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import numba
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
BUCKET_DIR = ROOT / "Data" / "daily_buckets"
N_BUCKETS = 64

DAILY_COLUMNS = ("permno", "time_d", "ret", "vol", "prc", "cfacpr", "shrout")


def iter_buckets(columns: list[str]) -> Iterator[pd.DataFrame]:
    """Yield each bucket as a DataFrame sorted (permno, time_d).

    Raises if the artifact is missing or a bucket is empty — a broken artifact
    must surface, not silently yield fewer stocks.
    """

    if not BUCKET_DIR.exists():
        raise FileNotFoundError(
            f"{BUCKET_DIR} missing — build it once with the daily bucket "
            f"repartition step (see p1-factor-engine-status / WS6)"
        )
    want = list(dict.fromkeys(["permno", "time_d", *columns]))
    for b in range(N_BUCKETS):
        part = BUCKET_DIR / f"bucket={b}"
        files = sorted(part.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"bucket {b} has no parquet files under {part}")
        frames = [pq.read_table(f, columns=want).to_pandas() for f in files]
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        # DuckDB's multithreaded partitioned COPY does NOT preserve ORDER BY inside
        # bucket files (measured: 595 contiguous blocks for 577 stocks — split
        # stocks silently became partial months). Sort here and assert, so a
        # future artifact regression fails loudly instead of corrupting stats.
        df = df.sort_values(["permno", "time_d"], kind="stable", ignore_index=True)
        p = df["permno"].to_numpy()
        n_blocks = int((np.diff(p) != 0).sum()) + 1
        assert n_blocks == df["permno"].nunique(), (
            f"bucket {b}: {n_blocks} blocks for {df['permno'].nunique()} stocks after sort — investigate"
        )
        yield df


def _stock_offsets(permno: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start offsets of each stock's contiguous block plus the stock ids."""

    change = np.flatnonzero(np.diff(permno)) + 1
    starts = np.concatenate(([0], change, [len(permno)]))
    ids = permno[starts[:-1]]
    return starts, ids


def _month_codes(time_d: np.ndarray) -> np.ndarray:
    """datetime64 -> integer month code (months since epoch)."""

    return time_d.astype("datetime64[M]").astype(np.int64)


# ---------------------------------------------------------------------------
# Batch 1: within-calendar-month grouped statistics
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _monthly_stats_stock(
    values: np.ndarray,
    months: np.ndarray,
    out_month: np.ndarray,
    out_count: np.ndarray,
    out_sum: np.ndarray,
    out_sumsq: np.ndarray,
    out_sumcube: np.ndarray,
    out_max: np.ndarray,
) -> int:
    """Single pass over one stock's (month-sorted) daily values.

    Accumulates per-month count/sum/sum²/sum³/max of the NON-NaN values into the
    out arrays; returns the number of months written. Moments are accumulated
    around zero — the statistics are derived later in float64 two-pass form
    where numerically required.
    """

    n_out = 0
    cur = -1
    for i in range(values.shape[0]):
        m = months[i]
        if m != cur:
            cur = m
            out_month[n_out] = m
            out_count[n_out] = 0
            out_sum[n_out] = 0.0
            out_sumsq[n_out] = 0.0
            out_sumcube[n_out] = 0.0
            out_max[n_out] = -np.inf
            n_out += 1
        v = values[i]
        if not np.isnan(v):
            j = n_out - 1
            out_count[j] += 1
            out_sum[j] += v
            out_sumsq[j] += v * v
            out_sumcube[j] += v * v * v
            if v > out_max[j]:
                out_max[j] = v
    return n_out


@numba.njit(cache=True)
def _finalize_monthly_stat(
    stat_code: int,
    min_obs: int,
    count_rows: bool,
    bias_adjusted: bool,
    values: np.ndarray,
    months: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one stock's monthly statistic with an exact two-pass formula.

    stat_code: 0=std(ddof=1), 1=skew, 2=max, 3=mean.
    count_rows: the min_obs gate counts ALL rows of the month (the reference
    ReturnSkew.py counts missing-return days toward its >=15 filter) instead of
    valid values.
    bias_adjusted: skew only — True = Fisher-Pearson G1 (pandas skew(bias=False)),
    False = population g1 (polars .skew() default), the reference's choice.
    Two-pass moments (mean first) — the one-pass sum-of-squares form suffers
    catastrophic cancellation for near-constant series, a bug this project has
    already paid for once (EarningsSurprise).
    """

    res_month = np.empty(values.shape[0], dtype=np.int64)
    res_value = np.empty(values.shape[0], dtype=np.float64)
    n_out = 0
    i = 0
    n = values.shape[0]
    while i < n:
        m = months[i]
        j = i
        while j < n and months[j] == m:
            j += 1
        rows = j - i
        cnt = 0
        s = 0.0
        mx = -np.inf
        for k in range(i, j):
            v = values[k]
            if not np.isnan(v):
                cnt += 1
                s += v
                if v > mx:
                    mx = v
        gate = rows if count_rows else cnt
        if stat_code == 4:
            # skipna SUM over the month's rows (pandas groupby.agg('sum'):
            # empty/all-NaN month -> 0.0). Emitted for every observed month.
            if gate >= min_obs:
                res_month[n_out] = m
                res_value[n_out] = s
                n_out += 1
        elif stat_code == 5:
            # LAST non-NaN value of the month (Stata 'lastnm')
            if gate >= min_obs and cnt > 0:
                last = np.nan
                for k in range(i, j):
                    v = values[k]
                    if not np.isnan(v):
                        last = v
                res_month[n_out] = m
                res_value[n_out] = last
                n_out += 1
        elif gate >= min_obs and cnt > 0:
            if stat_code == 2:
                res_month[n_out] = m
                res_value[n_out] = mx
                n_out += 1
            elif stat_code == 3:
                res_month[n_out] = m
                res_value[n_out] = s / cnt
                n_out += 1
            else:
                mean = s / cnt
                m2 = 0.0
                m3 = 0.0
                for k in range(i, j):
                    v = values[k]
                    if not np.isnan(v):
                        d = v - mean
                        m2 += d * d
                        m3 += d * d * d
                if stat_code == 0 and cnt >= 2:
                    res_month[n_out] = m
                    res_value[n_out] = np.sqrt(m2 / (cnt - 1))
                    n_out += 1
                elif stat_code == 1 and m2 > 0.0:
                    if bias_adjusted:
                        if cnt >= 3:
                            g1 = (m3 / cnt) / (m2 / cnt) ** 1.5
                            res_month[n_out] = m
                            res_value[n_out] = g1 * np.sqrt(cnt * (cnt - 1.0)) / (cnt - 2.0)
                            n_out += 1
                    else:
                        # population g1 — polars .skew(bias=True), the reference default
                        res_month[n_out] = m
                        res_value[n_out] = (m3 / cnt) / (m2 / cnt) ** 1.5
                        n_out += 1
        i = j
    return res_month[:n_out], res_value[:n_out]


_STAT_CODES = {"std": 0, "skew": 1, "max": 2, "mean": 3, "sum": 4, "last": 5}


def monthly_stat_panel(
    stat: str,
    *,
    min_obs: int,
    min_obs_mode: str = "valid",
    bias_adjusted: bool = False,
    value_expr: Callable[[pd.DataFrame], np.ndarray] | None = None,
    column: str = "ret",
) -> pd.DataFrame:
    """Stream all buckets and build the (month × permno) long result.

    ``value_expr`` derives the daily series from the bucket frame (default: the
    raw column) — e.g. Amihud's |ret|/(|prc|·vol) for Illiquidity.
    Returns a LONG frame [permno, month(datetime64), value]; the engine op
    aligns it to the output template.
    """

    if stat not in _STAT_CODES:
        raise KeyError(f"unknown stat {stat!r}; add it to _STAT_CODES/_finalize_monthly_stat explicitly")
    if min_obs_mode not in ("valid", "rows"):
        raise KeyError(f"min_obs_mode must be 'valid' or 'rows', got {min_obs_mode!r}")
    code = _STAT_CODES[stat]
    count_rows = min_obs_mode == "rows"
    need = [column] if value_expr is None else ["ret", "vol", "prc", "cfacpr", "shrout"]
    parts: list[pd.DataFrame] = []
    for bucket in iter_buckets(need):
        permno = bucket["permno"].to_numpy(np.int64)
        months = _month_codes(bucket["time_d"].to_numpy())
        values = (
            bucket[column].to_numpy(np.float64)
            if value_expr is None
            else np.asarray(value_expr(bucket), dtype=np.float64)
        )
        starts, ids = _stock_offsets(permno)
        for s_idx in range(len(ids)):
            lo, hi = starts[s_idx], starts[s_idx + 1]
            m_out, v_out = _finalize_monthly_stat(code, min_obs, count_rows, bias_adjusted, values[lo:hi], months[lo:hi])
            if len(m_out):
                parts.append(pd.DataFrame({"permno": ids[s_idx], "month_code": m_out, "value": v_out}))
    long = pd.concat(parts, ignore_index=True)
    long["time_avail_m"] = long.pop("month_code").astype("datetime64[M]").astype("datetime64[ns]")
    return long


# ---------------------------------------------------------------------------
# Batch 3: within-month FF3 regressions (IdioVol3F, ReturnSkew3F)
# ---------------------------------------------------------------------------


def load_daily_ff() -> tuple[np.ndarray, np.ndarray]:
    """(dates as datetime64[D] int64, factors[n,4] = mktrf smb hml rf), sorted."""

    ff = pq.read_table(
        ROOT / "Open_Source_Asset_Pricing" / "Signals" / "pyData" / "Intermediate" / "dailyFF.parquet",
        columns=["time_d", "mktrf", "smb", "hml", "rf"],
    ).to_pandas().sort_values("time_d")
    dates = ff["time_d"].to_numpy("datetime64[D]").astype(np.int64)
    fac = ff[["mktrf", "smb", "hml", "rf"]].to_numpy(np.float64)
    return dates, fac


@numba.njit(cache=True)
def _ff3_month_stock(
    exret: np.ndarray,
    mkt: np.ndarray,
    smb: np.ndarray,
    hml: np.ndarray,
    months: np.ndarray,
    min_obs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per calendar month: OLS of excess return on FF3 (+intercept) over valid
    rows; returns (month, std(resid) ddof=1, population skew(resid)).

    Valid row = all four series non-NaN (polars null_policy='drop' after the
    inner FF join). Groups with < min_obs valid rows, or a rank-deficient design
    (polars_ols singular -> null residuals -> group removed), emit nothing.
    """

    n = exret.shape[0]
    out_m = np.empty(n, dtype=np.int64)
    out_iv = np.empty(n, dtype=np.float64)
    out_sk = np.empty(n, dtype=np.float64)
    n_out = 0
    i = 0
    while i < n:
        m = months[i]
        j = i
        while j < n and months[j] == m:
            j += 1
        cnt = 0
        for k in range(i, j):
            if not (np.isnan(exret[k]) or np.isnan(mkt[k]) or np.isnan(smb[k]) or np.isnan(hml[k])):
                cnt += 1
        if cnt >= min_obs:
            X = np.empty((cnt, 4), dtype=np.float64)
            y = np.empty(cnt, dtype=np.float64)
            r = 0
            for k in range(i, j):
                if not (np.isnan(exret[k]) or np.isnan(mkt[k]) or np.isnan(smb[k]) or np.isnan(hml[k])):
                    X[r, 0] = 1.0
                    X[r, 1] = mkt[k]
                    X[r, 2] = smb[k]
                    X[r, 3] = hml[k]
                    y[r] = exret[k]
                    r += 1
            beta, _res, rank, _sv = np.linalg.lstsq(X, y)
            if rank == 4:
                resid = y - X @ beta
                mean = resid.mean()
                m2 = 0.0
                m3 = 0.0
                for k in range(cnt):
                    d = resid[k] - mean
                    m2 += d * d
                    m3 += d * d * d
                iv = np.sqrt(m2 / (cnt - 1))
                # perfect-fit months: residuals are float dust (~1e-20) and their
                # skew is noise, not statistics — undefined (NaN) per P4. The
                # reference publishes the noise (registered deviation).
                sk = (m3 / cnt) / (m2 / cnt) ** 1.5 if np.sqrt(m2 / cnt) > 1e-10 else np.nan
                out_m[n_out] = m
                out_iv[n_out] = iv
                out_sk[n_out] = sk
                n_out += 1
        i = j
    return out_m[:n_out], out_iv[:n_out], out_sk[:n_out]


def ff3_idio_panels(min_obs: int = 15) -> pd.DataFrame:
    """Long frame [permno, time_avail_m, idiovol, skew3f] from all buckets."""

    ff_dates, ff_fac = load_daily_ff()
    parts: list[pd.DataFrame] = []
    for bucket in iter_buckets(["ret"]):
        permno = bucket["permno"].to_numpy(np.int64)
        dates = bucket["time_d"].to_numpy("datetime64[D]").astype(np.int64)
        months = _month_codes(bucket["time_d"].to_numpy())
        ret = bucket["ret"].to_numpy(np.float64)
        # align FF factors by exact date; days without FF coverage become NaN
        pos = np.searchsorted(ff_dates, dates)
        pos_clipped = np.minimum(pos, len(ff_dates) - 1)
        matched = ff_dates[pos_clipped] == dates
        fac = np.where(matched[:, None], ff_fac[pos_clipped], np.nan)
        exret = ret - fac[:, 3]
        starts, ids = _stock_offsets(permno)
        for s_idx in range(len(ids)):
            lo, hi = starts[s_idx], starts[s_idx + 1]
            m_out, iv, sk = _ff3_month_stock(
                exret[lo:hi], fac[lo:hi, 0], fac[lo:hi, 1], fac[lo:hi, 2], months[lo:hi], min_obs
            )
            if len(m_out):
                parts.append(pd.DataFrame({"permno": ids[s_idx], "month_code": m_out, "idiovol": iv, "skew3f": sk}))
    long = pd.concat(parts, ignore_index=True)
    long["time_avail_m"] = long.pop("month_code").astype("datetime64[M]").astype("datetime64[ns]")
    return long


# ---------------------------------------------------------------------------
# Batch 4: rolling-observation market-model RMSE (IdioVolAHT)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _rolling_rmse_stock(
    y: np.ndarray,
    x: np.ndarray,
    months: np.ndarray,
    window: int,
    min_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per row: RMSE of OLS(y ~ 1 + x) over the trailing `window` VALID rows
    (arrays are pre-compacted to valid days, matching the reference's filter-
    then-window order); emit the LAST row's value per calendar month.

    Two-pass per window (means first, then centered moments) — exact, no
    prefix-sum cancellation. dof = n - 2; n < min_obs emits nothing.
    """

    n = y.shape[0]
    res_month = np.empty(n, dtype=np.int64)
    res_value = np.empty(n, dtype=np.float64)
    n_out = 0
    for i in range(n):
        lo = i - window + 1
        if lo < 0:
            lo = 0
        cnt = i - lo + 1
        is_last_of_month = i == n - 1 or months[i + 1] != months[i]
        if not is_last_of_month:
            continue
        if cnt < min_obs or cnt < 3:
            continue
        my = 0.0
        mx = 0.0
        for k in range(lo, i + 1):
            my += y[k]
            mx += x[k]
        my /= cnt
        mx /= cnt
        sxx = 0.0
        sxy = 0.0
        syy = 0.0
        for k in range(lo, i + 1):
            dx = x[k] - mx
            dy = y[k] - my
            sxx += dx * dx
            sxy += dx * dy
            syy += dy * dy
        if sxx <= 0.0:
            continue
        beta = sxy / sxx
        sse = syy - beta * sxy
        if sse < 0.0:
            sse = 0.0
        res_month[n_out] = months[i]
        res_value[n_out] = np.sqrt(sse / (cnt - 2))
        n_out += 1
    return res_month[:n_out], res_value[:n_out]


def rolling_market_rmse_panel(window: int = 252, min_obs: int = 100) -> pd.DataFrame:
    """Long frame [permno, time_avail_m, value] of IdioVolAHT."""

    ff_dates, ff_fac = load_daily_ff()
    parts: list[pd.DataFrame] = []
    for bucket in iter_buckets(["ret"]):
        permno = bucket["permno"].to_numpy(np.int64)
        dates = bucket["time_d"].to_numpy("datetime64[D]").astype(np.int64)
        months = _month_codes(bucket["time_d"].to_numpy())
        ret = bucket["ret"].to_numpy(np.float64)
        pos = np.searchsorted(ff_dates, dates)
        pos_clipped = np.minimum(pos, len(ff_dates) - 1)
        matched = ff_dates[pos_clipped] == dates
        mkt = np.where(matched, ff_fac[pos_clipped, 0], np.nan)
        rf = np.where(matched, ff_fac[pos_clipped, 3], np.nan)
        exret = ret - rf
        valid = ~(np.isnan(exret) | np.isnan(mkt))
        starts, ids = _stock_offsets(permno)
        for s_idx in range(len(ids)):
            lo, hi = starts[s_idx], starts[s_idx + 1]
            v = valid[lo:hi]
            if not v.any():
                continue
            m_out, r_out = _rolling_rmse_stock(
                exret[lo:hi][v], mkt[lo:hi][v], months[lo:hi][v], window, min_obs
            )
            if len(m_out):
                parts.append(pd.DataFrame({"permno": ids[s_idx], "month_code": m_out, "value": r_out}))
    long = pd.concat(parts, ignore_index=True)
    long["time_avail_m"] = long.pop("month_code").astype("datetime64[M]").astype("datetime64[ns]")
    return long


# ---------------------------------------------------------------------------
# Batch 5: monthly rolling regressions (Beta, BetaLiquidityPS)
# ---------------------------------------------------------------------------


@numba.njit(cache=True, parallel=True)
def rolling_beta_panel_kernel(
    Y: np.ndarray,
    x: np.ndarray,
    R: np.ndarray,
    window: int,
    min_obs: int,
) -> np.ndarray:
    """Rolling OLS slope of each column of Y on the common series x.

    Window = trailing `window` CALENDAR rows (the reference rolls over each
    stock's listed rows — positional; calendar is project policy, residual is
    the registered family). A pair is valid when both y and x are non-NaN.
    Two-pass per window; slope NaN when valid pairs < min_obs or x has no
    variance within the stock's valid pattern.
    """

    n_t, n_s = Y.shape
    out = np.full((n_t, n_s), np.nan)
    for s in numba.prange(n_s):
        y = Y[:, s]
        # polars_ols rolling null_policy='drop' semantics, PROVEN empirically
        # (Beta cell 10508 @1928-02: emission at the stock's 20th post-join row
        # with only 13 valid pairs; 25 emissions == rows 20..44 exactly):
        #   * the regression uses the trailing `window` VALID pairs (compacted);
        #   * min_periods gates on the stock's RAW post-join ROW COUNT (R), not
        #     on the number of valid pairs;
        #   * the coefficient is emitted at every row past the gate (>=2 pairs,
        #     nonsingular), including rows whose own pair is invalid.
        vt = np.empty(n_t, dtype=np.int64)  # times of valid pairs, in order
        n_valid = 0
        n_rows = 0
        for t in range(n_t):
            if R[t, s]:
                n_rows += 1
            if not (np.isnan(y[t]) or np.isnan(x[t])):
                vt[n_valid] = t
                n_valid += 1
            c = n_valid
            # gate on VALID pairs. The reference library has a degenerate branch:
            # stocks whose TOTAL valid count never reaches min_periods emit from
            # the min_periods-th RAW row (betas from as few as 13 pairs) — a
            # polars_ols edge-case defect we refuse to replicate (registered).
            if c < min_obs or c < 2:
                continue
            w = window if c >= window else c
            my = 0.0
            mx = 0.0
            for q in range(c - w, c):
                k = vt[q]
                my += y[k]
                mx += x[k]
            my /= w
            mx /= w
            sxx = 0.0
            sxy = 0.0
            for q in range(c - w, c):
                k = vt[q]
                dx = x[k] - mx
                sxx += dx * dx
                sxy += dx * (y[k] - my)
            if sxx > 0.0:
                out[t, s] = sxy / sxx
    return out


@numba.njit(cache=True, parallel=True)
def rolling_multibeta_panel_kernel(
    Y: np.ndarray,
    X: np.ndarray,
    keep: int,
    window: int,
    min_obs: int,
) -> np.ndarray:
    """Rolling OLS of each column of Y on common regressors X[n_t, K] (+ intercept);
    returns the coefficient of regressor `keep` (0-based, excluding intercept).

    Same proven polars_ols semantics as the single-regressor kernel: window =
    trailing `window` VALID rows (a row is valid when y and ALL regressors are
    non-NaN), gate on valid count, emission at every gated row. Rank-deficient
    windows emit nothing (reference: null -> dropped).
    """

    n_t, n_s = Y.shape
    K = X.shape[1]
    P = K + 1
    out = np.full((n_t, n_s), np.nan)
    x_ok = np.empty(n_t, dtype=np.bool_)
    for t in range(n_t):
        ok = True
        for j in range(K):
            if np.isnan(X[t, j]):
                ok = False
                break
        x_ok[t] = ok
    for s in numba.prange(n_s):
        y = Y[:, s]
        vt = np.empty(n_t, dtype=np.int64)
        n_valid = 0
        for t in range(n_t):
            if x_ok[t] and not np.isnan(y[t]):
                vt[n_valid] = t
                n_valid += 1
            c = n_valid
            if c < min_obs or c < P:
                continue
            w = window if c >= window else c
            xtx = np.zeros((P, P))
            xty = np.zeros(P)
            for q in range(c - w, c):
                k = vt[q]
                # design row: [1, X[k, 0..K-1]]
                xtx[0, 0] += 1.0
                xty[0] += y[k]
                for a in range(K):
                    xa = X[k, a]
                    xtx[0, a + 1] += xa
                    xty[a + 1] += xa * y[k]
                    for b in range(a, K):
                        xtx[a + 1, b + 1] += xa * X[k, b]
            for a in range(P):
                for b in range(a):
                    xtx[a, b] = xtx[b, a]
            beta, _res, rank, _sv = np.linalg.lstsq(xtx, xty)
            if rank == P:
                out[t, s] = beta[keep + 1]
    return out


@numba.njit(cache=True, parallel=True)
def rolling_ff3_residual_kernel(
    Y: np.ndarray,
    X: np.ndarray,
    window: int,
) -> np.ndarray:
    """Rolling FF3 residual of the LAST row of each trailing `window`-VALID-row
    regression (min = window exactly, per the reference's asreg-style call).
    Residual placed at its row's time; invalid rows stay NaN."""

    n_t, n_s = Y.shape
    K = X.shape[1]
    P = K + 1
    out = np.full((n_t, n_s), np.nan)
    x_ok = np.empty(n_t, dtype=np.bool_)
    for t in range(n_t):
        ok = True
        for j in range(K):
            if np.isnan(X[t, j]):
                ok = False
                break
        x_ok[t] = ok
    for s in numba.prange(n_s):
        y = Y[:, s]
        vt = np.empty(n_t, dtype=np.int64)
        n_valid = 0
        for t in range(n_t):
            if not (x_ok[t] and not np.isnan(y[t])):
                continue
            vt[n_valid] = t
            n_valid += 1
            if n_valid < window:
                continue
            xtx = np.zeros((P, P))
            xty = np.zeros(P)
            for q in range(n_valid - window, n_valid):
                k = vt[q]
                xtx[0, 0] += 1.0
                xty[0] += y[k]
                for a in range(K):
                    xa = X[k, a]
                    xtx[0, a + 1] += xa
                    xty[a + 1] += xa * y[k]
                    for b in range(a, K):
                        xtx[a + 1, b + 1] += xa * X[k, b]
            for a in range(P):
                for b in range(a):
                    xtx[a, b] = xtx[b, a]
            beta, _res, rank, _sv = np.linalg.lstsq(xtx, xty)
            if rank == P:
                pred = beta[0]
                for a in range(K):
                    pred += beta[a + 1] * X[t, a]
                out[t, s] = y[t] - pred
    return out


if __name__ == "__main__":
    # smoke: one stat over all buckets, timed
    import time

    t0 = time.time()
    out = monthly_stat_panel("max", min_obs=1)
    print(f"monthly max: {len(out):,} stock-months in {time.time()-t0:.1f}s")
    print(out.head())

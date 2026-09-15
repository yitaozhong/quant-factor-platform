"""Point-in-time Pastor-Stambaugh (2003) aggregate liquidity innovations.

Task #11 (user decision): the published WRDS series ``ff.liq_ps.ps_innov`` (mirrored
locally in Open_Source_Asset_Pricing/.../monthlyLiquidity.parquet) is the residual of
a regression estimated ONCE over the FULL sample — the value stamped at month t
therefore embeds coefficients fitted on data after t (look-ahead). This module
rebuilds the series so that the innovation stamped at month t is the residual of the
same regression fitted on months <= t only (expanding window), and each month's value
is FROZEN when first computed — never revised by later data.

Construction (Pastor & Stambaugh, JPE 2003, Section II) — replicated as closely as
the paper and our data allow; every deviation is documented below:

1. Per stock-month "gamma" (return-reversal-per-dollar-volume, the paper's eq. 1):
   within calendar month t, for each stock i, OLS of
       y   = r_{i,d+1} - mkt_{d+1}                    (next-day return in excess of market)
   on  X   = [ 1,  r_{i,d},  sign(r_{i,d} - mkt_d) * dollar_volume_{i,d} ]
   gamma_{i,t} = the coefficient on the signed-dollar-volume term. A (d, d+1) pair is
   valid only when d and d+1 are CONSECUTIVE trading days (per the Fama-French daily
   calendar), both inside month t, with finite ret on both days, finite price on day
   d, and STRICTLY POSITIVE volume on day d. The positive-volume screen (v_{i,d,t} >
   0) is NOT in the 2003 paper's text but IS part of the official construction —
   Pastor & Stambaugh, "Liquidity Risk After 20 Years" (Critical Finance Review 2019,
   Section 2.3) disclose that they always imposed it and that replications without it
   correlate only 27-39% with the published series (Li-Novy-Marx-Velikov 2017;
   Pontiff-Singla 2019). Rationale: on zero-volume days CRSP's price is a bid/ask
   midpoint, making the bid-ask-bounce control r_{i,d} misspecified. Dollar volume is
   |prc| * shares / 1e6 (millions of dollars, the paper's unit). Requirement:
   >= MIN_PAIRS (15) valid pairs (task spec; the 2003 paper says "more than 15"), and
   a full-rank (rank 3) design — rank-deficient months emit nothing rather than a
   garbage coefficient. One documented exception: 2001-09 requires only >= 11 pairs
   (SEPT_2001_MIN_PAIRS), matching the official series' accommodation of the 9/11
   exchange closure (15 trading days that month, so at most 14 pairs); without it the
   month is structurally empty while the published series has a 2001-09 value.

2. Universe and filters (paper section II.B), applied via monthlyCRSP at month t-1
   because dailyCRSP carries no exchange or share code:
     - NYSE/AMEX only: exchcd in {1, 2} on the month t-1 row (Nasdaq volume
       conventions differ; this is the paper's universe).
     - Ordinary common shares: shrcd in {10, 11}.
     - Price filter: 5 <= |prc at end of month t-1| <= 1000 (the paper excludes
       stocks priced below $5 or above $1000 at the end of the PREVIOUS month; we
       adopt end-of-t-1 = "month start" inclusively on both bounds).
   A stock-month enters only if the (permno, t-1) monthlyCRSP row exists and passes
   all three (plus finite positive mve_c, which the price filter already implies).

3. Aggregate level: gamma_hat_t = (1/N_t) * sum_i gamma_{i,t}; scaled level
   g_t = (m_t / m_1) * gamma_hat_t, where m_t = total dollar value (monthlyCRSP
   mve_c) at the end of month t-1 of the stocks included in month t, and m_1 is that
   of the first sample month (1962-08, matching the paper — only the RATIO enters,
   so mve_c units cancel).

4. Monthly change (paper's eq. 7 — note: NOT the naive first difference of g_t):
   dg_t = (m_t / m_1) * (1/N_t^common) * sum_i (gamma_{i,t} - gamma_{i,t-1}),
   summed over stocks with a valid, filter-passing gamma in BOTH t and t-1. The
   paper differences WITHIN stock first, then averages, to kill composition noise.

5. Innovations (paper's eq. 8):  dg_t = a + b * dg_{t-1} + c * g_{t-1} + u_t.
   - Full-sample replica (ps_innov_fullsample_replica): coefficients fitted once on
     all months, residuals u_t — methodologically what WRDS publishes; used for the
     sanity correlation against the published series.
   - Point-in-time (ps_innov_pit): at each month t with at least BURN_IN (60)
     usable observations, refit eq. 8 on months <= t only and stamp the residual AT
     t from that fit. Earlier stamps are never revised. Months before the burn-in
     are NaN.
   Scaling for comparability with the published WRDS series: NONE (INNOV_SCALE =
   1.0). This was determined empirically, not from folklore: with the raw eq.-8
   residual, std(published ps_innov) / std(replica) = 1.006 (both ~0.056), and
   dividing by 100 — a convention sometimes attributed to the paper — leaves the
   replica 100x too small against ff.liq_ps. The validation step prints the std
   ratio every run so a units regression would surface immediately.

Known deviations from the paper (documented gaps, expect imperfect correlation):
   - Market return: Ken French's daily mkt (mktrf + rf, from dailyFF.parquet) proxies
     the CRSP value-weighted market return the paper uses. Near-identical but not
     bit-equal, and FF includes Nasdaq in the market portfolio.
   - The paper says "more than 15 observations"; the task fixes >= 15 pairs — adopted.
   - CRSP's current daily file differs from the 2003 vintage (restatements, delist
     handling); the published series also evolves as Stambaugh updates it.
   - Sample starts 1962-08 (paper's start; AMEX volume coverage begins mid-1962) even
     though our daily file reaches back to 1926.

Infrastructure: read-only imports from p1_daily_kernels (iter_buckets bucket
streaming, _stock_offsets, _month_codes, load_daily_ff). The per-stock-month gamma
regressions run in a numba kernel using np.linalg.lstsq with an explicit rank check.

Output artifact: Data/ps_innov_pit.parquet with columns
   [time_avail_m, ps_innov_pit, ps_innov_fullsample_replica]
where time_avail_m is the month of the innovation (month-start timestamp, same
convention as the published monthlyLiquidity.parquet). The value for month t is
computable at the close of month t; downstream users must lag as they would the
published series.

Run:  python p1_kern_ps_liquidity.py   (from Factor_Construction/)
"""

from __future__ import annotations

from pathlib import Path

import numba
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from p1_daily_kernels import _month_codes, _stock_offsets, iter_buckets, load_daily_ff

ROOT = Path(__file__).resolve().parent
MONTHLY_CRSP = ROOT / "Open_Source_Asset_Pricing" / "Signals" / "pyData" / "Intermediate" / "monthlyCRSP.parquet"
PUBLISHED = ROOT / "Open_Source_Asset_Pricing" / "Signals" / "pyData" / "Intermediate" / "monthlyLiquidity.parquet"
ARTIFACT = ROOT / "Data" / "ps_innov_pit.parquet"

MIN_PAIRS = 15  # >= 15 valid (d, d+1) pairs per stock-month (task spec; paper: "more than 15")
SEPT_2001_MCODE = int(np.datetime64("2001-09", "M").astype(np.int64))
SEPT_2001_MIN_PAIRS = 11  # official series' 9/11-closure exception (replication literature)
PRICE_MIN = 5.0  # $5 <= |prc| <= $1000 at end of month t-1 (paper section II.B)
PRICE_MAX = 1000.0
EXCHCD_KEEP = (1, 2)  # NYSE, AMEX — the paper's universe (Nasdaq volume not comparable)
SHRCD_KEEP = (10, 11)  # ordinary common shares
SAMPLE_START_MCODE = int(np.datetime64("1962-08", "M").astype(np.int64))  # paper's first month
BURN_IN = 60  # months of eq.-8 observations required before the first PIT innovation
INNOV_SCALE = 1.0  # raw eq.-8 residual already matches WRDS ps_innov units (std ratio 1.006, see validate())
MIN_STOCKS = 30  # loud sanity floor on cross-section size; never expected to bind post-1962


# ---------------------------------------------------------------------------
# Step 1: per stock-month gamma via numba (exact lstsq, rank-checked)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _gamma_bucket(
    starts: np.ndarray,
    months: np.ndarray,
    cal_pos: np.ndarray,
    ret: np.ndarray,
    mkt_d: np.ndarray,
    dvol: np.ndarray,
    min_month: int,
    min_pairs_by_month: np.ndarray,
    out_stock: np.ndarray,
    out_month: np.ndarray,
    out_gamma: np.ndarray,
    out_pairs: np.ndarray,
) -> int:
    """Per stock-month PS gamma over one bucket (rows sorted permno, time_d).

    cal_pos: row's index on the trading calendar, -1 if the date is off-calendar.
    mkt_d:   market return aligned to the row's date (NaN if off-calendar).
    dvol:    dollar volume in $ millions (NaN when price or volume is missing).
             Day d must have dvol > 0 — the official positive-volume screen; a
             zero-volume day never supplies regressors (its price is a bid/ask
             midpoint, see module docstring).
    min_pairs_by_month: required pair count per month, indexed by (mcode -
    min_month) — MIN_PAIRS everywhere except the 2001-09 exception.
    Months outside the array's range are skipped (no market data there anyway).

    A pair (row k, row k+1) of the same stock is valid iff both rows share the
    month, sit on consecutive calendar positions, and ret[k], ret[k+1], mkt_d[k],
    mkt_d[k+1], dvol[k] are all finite. Months with >= min_pairs valid pairs get an
    OLS fit y = [1, r_d, sign(r_d - mkt_d) * dvol_d] via lstsq; only full-rank
    (rank 3) fits emit gamma (coefficient on the signed-volume column).
    """

    n_out = 0
    x_buf = np.empty((31, 3), dtype=np.float64)  # max trading days in a month
    y_buf = np.empty(31, dtype=np.float64)
    for s in range(starts.shape[0] - 1):
        i = starts[s]
        block_end = starts[s + 1]
        while i < block_end:
            m = months[i]
            j = i
            while j < block_end and months[j] == m:
                j += 1
            # a month with n pairs has n+1 rows; skip small/early months cheaply
            if min_month <= m < min_month + min_pairs_by_month.shape[0]:
                min_pairs = min_pairs_by_month[m - min_month]
            else:
                min_pairs = np.int64(2**31)  # outside the market calendar: never emit
            if j - i >= min_pairs + 1:
                n = 0
                for k in range(i, j - 1):
                    ck = cal_pos[k]
                    if ck < 0 or cal_pos[k + 1] != ck + 1:
                        continue
                    r0 = ret[k]
                    r1 = ret[k + 1]
                    v = dvol[k]
                    mk0 = mkt_d[k]
                    mk1 = mkt_d[k + 1]
                    if not (
                        np.isfinite(r0)
                        and np.isfinite(r1)
                        and np.isfinite(v)
                        and v > 0.0  # official positive-volume screen (PS 2019, sec. 2.3)
                        and np.isfinite(mk0)
                        and np.isfinite(mk1)
                    ):
                        continue
                    re0 = r0 - mk0
                    sgn = 0.0
                    if re0 > 0.0:
                        sgn = 1.0
                    elif re0 < 0.0:
                        sgn = -1.0
                    x_buf[n, 0] = 1.0
                    x_buf[n, 1] = r0
                    x_buf[n, 2] = sgn * v
                    y_buf[n] = r1 - mk1
                    n += 1
                if n >= min_pairs:
                    sol, _res, rank, _sv = np.linalg.lstsq(x_buf[:n], y_buf[:n])
                    if rank == 3:
                        out_stock[n_out] = s
                        out_month[n_out] = m
                        out_gamma[n_out] = sol[2]
                        out_pairs[n_out] = n
                        n_out += 1
            i = j
    return n_out


def build_gamma_panel() -> pd.DataFrame:
    """Stream all daily buckets -> long frame [permno, mcode, gamma, n_pairs].

    mcode is the integer month code (months since 1970-01, from _month_codes),
    restricted to >= 1962-08. No universe filters yet — those need monthlyCRSP and
    are applied in build_aggregate_series so the daily pass stays single-purpose.
    """

    ff_days, fac = load_daily_ff()
    mkt = np.ascontiguousarray(fac[:, 0] + fac[:, 3])  # mktrf + rf = total market return

    # required pairs per month: MIN_PAIRS, except the official 2001-09 exception
    ff_mcodes = ff_days.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)
    last_mcode = int(ff_mcodes.max())
    min_pairs_by_month = np.full(last_mcode - SAMPLE_START_MCODE + 1, MIN_PAIRS, dtype=np.int64)
    min_pairs_by_month[SEPT_2001_MCODE - SAMPLE_START_MCODE] = SEPT_2001_MIN_PAIRS

    parts: list[pd.DataFrame] = []
    for bucket in iter_buckets(["ret", "vol", "prc"]):
        permno = bucket["permno"].to_numpy(np.int64)
        time_d = bucket["time_d"].to_numpy()
        days = time_d.astype("datetime64[D]").astype(np.int64)
        months = _month_codes(time_d)
        ret = bucket["ret"].to_numpy(np.float64)
        vol = bucket["vol"].to_numpy(np.float64)
        prc = bucket["prc"].to_numpy(np.float64)

        pos = np.searchsorted(ff_days, days)
        pos_c = np.minimum(pos, ff_days.shape[0] - 1)
        on_cal = ff_days[pos_c] == days
        cal_pos = np.where(on_cal, pos_c, np.int64(-1))
        mkt_d = np.where(on_cal, mkt[pos_c], np.nan)
        # negative prc is CRSP's bid/ask-midpoint flag -> abs; NaN propagates -> pair invalid
        dvol = np.abs(prc) * vol / 1e6

        starts, ids = _stock_offsets(permno)
        # each emit consumes a month segment of >= min(min_pairs)+1 rows -> hard capacity bound
        cap = ret.shape[0] // (int(min_pairs_by_month.min()) + 1) + 1
        out_stock = np.empty(cap, np.int64)
        out_month = np.empty(cap, np.int64)
        out_gamma = np.empty(cap, np.float64)
        out_pairs = np.empty(cap, np.int64)
        n = _gamma_bucket(
            starts, months, cal_pos, ret, mkt_d, dvol,
            SAMPLE_START_MCODE, min_pairs_by_month,
            out_stock, out_month, out_gamma, out_pairs,
        )
        parts.append(
            pd.DataFrame(
                {
                    "permno": ids[out_stock[:n]],
                    "mcode": out_month[:n].copy(),
                    "gamma": out_gamma[:n].copy(),
                    "n_pairs": out_pairs[:n].copy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 2: universe filters from monthlyCRSP (month t gated by the t-1 row)
# ---------------------------------------------------------------------------


def load_month_filters() -> pd.DataFrame:
    """[permno, mcode, mve_lag]: stock-months PASSING the paper's t-1 filters.

    The monthlyCRSP row stamped time_avail_m = month s carries end-of-s price,
    exchange code, share code and market value; it gates gamma month t = s + 1.
    mve_lag is mve_c at end of t-1 — the per-stock weight inside m_t.
    """

    mc = pq.read_table(
        MONTHLY_CRSP,
        columns=["permno", "time_avail_m", "prc", "exchcd", "shrcd", "mve_c"],
    ).to_pandas()
    price = mc["prc"].abs()
    keep = (
        price.ge(PRICE_MIN)
        & price.le(PRICE_MAX)
        & mc["exchcd"].isin(EXCHCD_KEEP)
        & mc["shrcd"].isin(SHRCD_KEEP)
        & mc["mve_c"].gt(0)
    ).to_numpy()
    mcode = _month_codes(mc["time_avail_m"].to_numpy()) + 1  # t-1 row gates month t
    out = pd.DataFrame(
        {
            "permno": mc["permno"].to_numpy(np.int64)[keep],
            "mcode": mcode[keep],
            "mve_lag": mc["mve_c"].to_numpy(np.float64)[keep],
        }
    )
    if out.duplicated(["permno", "mcode"]).any():
        raise ValueError("monthlyCRSP has duplicate (permno, month) rows — investigate upstream")
    return out


# ---------------------------------------------------------------------------
# Step 3+4: aggregate level g_t and within-stock scaled change dg_t (eq. 7)
# ---------------------------------------------------------------------------


def build_aggregate_series(gammas: pd.DataFrame, filters: pd.DataFrame) -> pd.DataFrame:
    """Monthly frame [mcode, gamma_hat, n_stocks, m_total, g, dg, n_common].

    g_t  = (m_t/m_1) * mean_i gamma_{i,t}                       (scaled level)
    dg_t = (m_t/m_1) * mean_{i in t and t-1} (gamma_{i,t} - gamma_{i,t-1})  (eq. 7)
    """

    panel = gammas.merge(filters, on=["permno", "mcode"], how="inner")
    if panel.empty:
        raise ValueError("no stock-month passed the PS filters — inputs are broken")

    agg = (
        panel.groupby("mcode")
        .agg(gamma_hat=("gamma", "mean"), n_stocks=("gamma", "size"), m_total=("mve_lag", "sum"))
        .reset_index()
        .sort_values("mcode", ignore_index=True)
    )
    mcodes = agg["mcode"].to_numpy()
    expected = np.arange(mcodes[0], mcodes[-1] + 1)
    if not np.array_equal(mcodes, expected):
        missing = np.setdiff1d(expected, mcodes)
        raise ValueError(f"gap in monthly series — missing month codes {missing[:10]}")
    if int(agg["n_stocks"].min()) < MIN_STOCKS:
        bad = agg.loc[agg["n_stocks"] < MIN_STOCKS, "mcode"].tolist()
        raise ValueError(f"cross-section thinner than {MIN_STOCKS} stocks in months {bad} — investigate")

    m1 = float(agg["m_total"].iloc[0])
    agg["g"] = agg["m_total"] / m1 * agg["gamma_hat"]

    lagged = panel[["permno", "mcode", "gamma"]].copy()
    lagged["mcode"] = lagged["mcode"] + 1
    common = panel[["permno", "mcode", "gamma"]].merge(
        lagged, on=["permno", "mcode"], suffixes=("", "_lag")
    )
    dstat = (
        (common["gamma"] - common["gamma_lag"])
        .groupby(common["mcode"])
        .agg(["mean", "size"])
        .rename(columns={"mean": "dgamma_mean", "size": "n_common"})
        .reset_index()
    )
    agg = agg.merge(dstat, on="mcode", how="left")
    have_dg = agg.index > 0  # first month has no t-1 gammas by construction
    if agg.loc[have_dg, "dgamma_mean"].isna().any():
        raise ValueError("within-stock difference missing for a non-initial month — investigate")
    if int(agg.loc[have_dg, "n_common"].min()) < MIN_STOCKS:
        raise ValueError(f"fewer than {MIN_STOCKS} stocks in a within-stock difference month")
    agg["dg"] = agg["m_total"] / m1 * agg["dgamma_mean"]
    return agg


# ---------------------------------------------------------------------------
# Step 5: eq.-8 innovations — full-sample replica and expanding-window PIT
# ---------------------------------------------------------------------------


def fit_innovations(agg: pd.DataFrame, burn_in: int = BURN_IN) -> tuple[np.ndarray, np.ndarray]:
    """(ps_innov_pit, ps_innov_fullsample_replica), aligned to agg rows, x INNOV_SCALE.

    Eq. 8: dg_t = a + b*dg_{t-1} + c*g_{t-1} + u_t. Observations exist for row
    index t >= 2 (dg needs t-1; the lagged regressor needs t-2). The replica fits
    once on all observations (the published series' methodology). The PIT column
    refits on observations s <= t and stamps ONLY the residual at t; a stamp needs
    >= burn_in observations. Rows without a value carry NaN.
    """

    g = agg["g"].to_numpy(np.float64)
    dg = agg["dg"].to_numpy(np.float64)
    t_len = len(agg)
    if not np.isfinite(g).all():
        raise ValueError("non-finite scaled level g_t")
    if not np.isfinite(dg[1:]).all():
        raise ValueError("non-finite dg_t after the first month")

    y = dg[2:]  # y[k] is month index t = k + 2
    x = np.column_stack([np.ones(t_len - 2), dg[1:-1], g[1:-1]])

    sol, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < 3:
        raise ValueError("full-sample eq.-8 design is rank-deficient")
    innov_full = np.full(t_len, np.nan)
    innov_full[2:] = (y - x @ sol) * INNOV_SCALE

    innov_pit = np.full(t_len, np.nan)
    for t in range(2, t_len):
        n_obs = t - 1  # observations at month indices 2..t
        if n_obs < burn_in:
            continue
        sol_t, _, rank_t, _ = np.linalg.lstsq(x[:n_obs], y[:n_obs], rcond=None)
        if rank_t < 3:
            raise ValueError(f"eq.-8 design rank-deficient at expanding-window month index {t}")
        innov_pit[t] = (y[t - 2] - x[t - 2] @ sol_t) * INNOV_SCALE
    return innov_pit, innov_full


# ---------------------------------------------------------------------------
# Validation + artifact
# ---------------------------------------------------------------------------


def validate(out: pd.DataFrame) -> None:
    """Print correlations/scale vs the published series and PIT-vs-replica diagnostics."""

    pub = pq.read_table(PUBLISHED).to_pandas()
    merged = out.merge(pub, on="time_avail_m", how="inner")
    ok = merged["ps_innov_fullsample_replica"].notna() & merged["ps_innov"].notna()
    rep = merged.loc[ok, "ps_innov_fullsample_replica"].to_numpy()
    pb = merged.loc[ok, "ps_innov"].to_numpy()
    corr_rep = float(np.corrcoef(rep, pb)[0, 1])
    ratio = float(np.std(pb, ddof=1) / np.std(rep, ddof=1))
    print(f"[validate] overlap with published series : {ok.sum()} months "
          f"({merged.loc[ok, 'time_avail_m'].min():%Y-%m} .. {merged.loc[ok, 'time_avail_m'].max():%Y-%m})")
    print(f"[validate] corr(replica, published)      : {corr_rep:+.4f}")
    print(f"[validate] std(published)/std(replica)   : {ratio:.4f}  "
          f"(post INNOV_SCALE={INNOV_SCALE}; ~1 means units match the published series)")
    print(f"[validate] std published={np.std(pb, ddof=1):.5f}  replica={np.std(rep, ddof=1):.5f}")

    both = out["ps_innov_pit"].notna() & out["ps_innov_fullsample_replica"].notna()
    pit = out.loc[both, "ps_innov_pit"].to_numpy()
    rep2 = out.loc[both, "ps_innov_fullsample_replica"].to_numpy()
    corr_pit_rep = float(np.corrcoef(pit, rep2)[0, 1])
    print(f"[validate] corr(PIT, replica)            : {corr_pit_rep:+.4f}  over {both.sum()} months")

    okp = merged["ps_innov_pit"].notna() & merged["ps_innov"].notna()
    corr_pit_pub = float(
        np.corrcoef(merged.loc[okp, "ps_innov_pit"], merged.loc[okp, "ps_innov"])[0, 1]
    )
    print(f"[validate] corr(PIT, published)          : {corr_pit_pub:+.4f}  over {okp.sum()} months")

    div = out.loc[both].copy()
    div["abs_diff"] = (div["ps_innov_pit"] - div["ps_innov_fullsample_replica"]).abs()
    top = div.nlargest(10, "abs_diff")[["time_avail_m", "ps_innov_pit", "ps_innov_fullsample_replica", "abs_diff"]]
    print("[validate] largest PIT vs replica divergences:")
    print(top.to_string(index=False))


def main() -> pd.DataFrame:
    print("[1/4] per stock-month gamma regressions (numba, bucket-streamed) ...")
    gammas = build_gamma_panel()
    print(f"      {len(gammas):,} stock-month gammas (pre-filter), "
          f"{gammas['permno'].nunique():,} stocks")

    print("[2/4] universe filters from monthlyCRSP (NYSE/AMEX, shrcd 10/11, $5-$1000 at t-1) ...")
    filters = load_month_filters()
    agg = build_aggregate_series(gammas, filters)
    print(f"      {len(agg)} months "
          f"({np.datetime64(int(agg['mcode'].iloc[0]), 'M')} .. {np.datetime64(int(agg['mcode'].iloc[-1]), 'M')}), "
          f"median cross-section {int(agg['n_stocks'].median())} stocks")

    print("[3/4] eq.-8 innovations: full-sample replica + expanding-window PIT ...")
    innov_pit, innov_full = fit_innovations(agg)

    out = pd.DataFrame(
        {
            "time_avail_m": agg["mcode"].to_numpy().astype("datetime64[M]").astype("datetime64[ns]"),
            "ps_innov_pit": innov_pit,
            "ps_innov_fullsample_replica": innov_full,
        }
    )

    print("[4/4] validation vs published series ...")
    validate(out)

    out.to_parquet(ARTIFACT, index=False)
    print(f"[done] wrote {ARTIFACT}  ({len(out)} rows, "
          f"PIT non-null from {out.loc[out['ps_innov_pit'].notna(), 'time_avail_m'].min():%Y-%m})")
    return out


if __name__ == "__main__":
    main()

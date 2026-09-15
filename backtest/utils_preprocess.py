"""
utils_preprocess.py
Helper functions for label construction, missing-value filling,
and standardization.
"""

import numpy as np
import pandas as pd


# ============================================================
# Label construction
# ============================================================

def label_current_period_return(return_data, stock_list, train_dates, delay=0):
    """
    Label = one-period return at row t + delay.

    delay lets the caller align factors to a stored return panel whose row
    labels do not represent immediately tradable periods.
    """
    label = return_data.shift(-delay).reindex(index=train_dates, columns=stock_list)
    return label, int(delay)


def label_future_return(return_data, stock_list, train_dates, n=1, delay=1):
    """
    Label = compound stock return from row t + delay through t + delay + n - 1.
    """
    if n < 1:
        raise ValueError('n must be at least 1 for future_return.')

    ret = return_data.reindex(columns=stock_list)
    future_ret = pd.DataFrame(
        1.0,
        index=ret.index,
        columns=ret.columns,
    )
    for offset in range(delay, delay + n):
        future_ret = future_ret * (1.0 + ret.shift(-offset))
    future_ret = future_ret - 1.0
    label = future_ret.reindex(index=train_dates)
    return label, int(delay + n - 1)


def label_future_spread(return_data, stock_list, train_dates,
                        n=1, benchmark=None, delay=1):
    """
    Label = compound stock return minus compound benchmark return,
    both from row t + delay through t + delay + n - 1.

    If benchmark is None, uses equal-weight cross-sectional mean each period.
    """
    if n < 1:
        raise ValueError('n must be at least 1 for future_spread.')

    ret = return_data.reindex(columns=stock_list)

    future_ret = pd.DataFrame(
        1.0,
        index=ret.index,
        columns=ret.columns,
    )
    for offset in range(delay, delay + n):
        future_ret = future_ret * (1.0 + ret.shift(-offset))
    future_ret = future_ret - 1.0

    if benchmark is None:
        bench_ret = ret.mean(axis=1)
    else:
        bench_ret = benchmark
    future_bench = pd.Series(1.0, index=bench_ret.index)
    for offset in range(delay, delay + n):
        future_bench = future_bench * (1.0 + bench_ret.shift(-offset))
    future_bench = future_bench - 1.0

    label = future_ret.sub(future_bench, axis=0).reindex(index=train_dates)
    return label, int(delay + n - 1)


LABEL_REGISTRY = {
    'current_period_return': label_current_period_return,
    'future_return': label_future_return,
    'future_spread': label_future_spread,
}


# ============================================================
# Missing-value filling
# ============================================================

def fill_median_cs(df):
    """
    Cross-sectional fill: for each date (row), replace NaN with the
    median across stocks on that date.

    Returns the filled DataFrame.
    """
    return df.T.fillna(df.median(axis=1)).T


def fill_median_ts(df):
    """
    Time-series fill: for each stock (column), replace NaN with the
    median across dates for that stock.

    Returns (filled_df, column_medians).
    column_medians are saved for use at test time.
    """
    col_medians = df.median(axis=0)
    filled = df.fillna(col_medians)
    return filled, col_medians


def fill_test_factors(test_x, fill_method, fill_medians=None):
    """
    Fill missing values in a test feature matrix (stocks x factors).

    Parameters
    ----------
    test_x : DataFrame, shape (stocks, factors)
    fill_method : str, 'fill_median_cs' or 'fill_median_ts'
    fill_medians : dict[str, dict]
        Per-factor fallback values from training.
        Each entry may contain:
        - 'ts': per-stock time-series medians
        - 'global': factor-level scalar median
    """
    test_x = test_x.copy()

    if fill_method == 'fill_median_ts' and fill_medians is not None:
        for fname in test_x.columns:
            if fname not in fill_medians:
                continue
            ts_med = fill_medians[fname].get('ts')
            if ts_med is None:
                continue

            mask = test_x[fname].isna()
            if mask.any():
                test_x.loc[mask, fname] = ts_med.reindex(test_x.index[mask]).values

    if fill_method in ('fill_median_cs', 'fill_median_ts'):
        test_x = test_x.fillna(test_x.median())

        if fill_medians is not None:
            for fname in test_x.columns:
                if fname not in fill_medians:
                    continue
                global_med = fill_medians[fname].get('global')
                if global_med is None or pd.isna(global_med):
                    continue

                mask = test_x[fname].isna()
                if mask.any():
                    test_x.loc[mask, fname] = global_med

        return test_x

    return test_x


# ============================================================
# Standardization
# ============================================================

def fit_standardize(factors_dic, selected_factors):
    """
    Pooled z-score: one mean + one std per factor across all
    (date x stock) observations in the training window.

    Returns (standardized factors_dic, means dict, stds dict).
    Modifies factors_dic in-place for memory efficiency.
    """
    means = {}
    stds = {}
    for fname in selected_factors:
        vals = factors_dic[fname].values.ravel()
        valid = vals[np.isfinite(vals)]
        m = float(np.mean(valid)) if len(valid) > 0 else 0.0
        s = float(np.std(valid, ddof=1)) if len(valid) > 1 else 1.0
        if s == 0:
            s = 1.0
        means[fname] = m
        stds[fname] = s
        factors_dic[fname] = (factors_dic[fname] - m) / s
    return factors_dic, means, stds


def apply_standardize(test_x, means, stds):
    """Apply saved training-set z-score parameters to test features."""
    test_x = test_x.copy()
    for fname in test_x.columns:
        if fname in means:
            test_x[fname] = (test_x[fname] - means[fname]) / stds[fname]
    return test_x


def cs_rank_transform(df):
    """
    Per-date cross-sectional rank transform to [-1, 1].

    For each date (row), non-missing values are ranked (average ties) and
    mapped to 2*rank/(n+1) - 1, so every date has the same bounded, centered
    feature distribution regardless of the factor's level or dispersion that
    month. NaN stays NaN. Uses only same-date information, so the transform
    needs no fitted parameters and cannot leak across dates.
    """
    ranks = df.rank(axis=1, method='average', na_option='keep')
    counts = ranks.notna().sum(axis=1)
    return (2.0 * ranks.div(counts + 1.0, axis=0)) - 1.0


def cs_rank_transform_cross_section(test_x):
    """
    Same transform for a single-date feature matrix (stocks x factors):
    each factor column is one date's cross-section.
    """
    ranks = test_x.rank(axis=0, method='average', na_option='keep')
    counts = ranks.notna().sum(axis=0)
    return (2.0 * ranks.div(counts + 1.0, axis=1)) - 1.0

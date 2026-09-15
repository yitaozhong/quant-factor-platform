"""
utils_backtest.py
Numba-accelerated backtest kernel and performance statistics.
"""

import numba
import numpy as np
import pandas as pd


@numba.jit(nopython=True, cache=True)
def _backtest_core(weights, returns, rebalance_mask):
    """
    Compute portfolio returns and one-way turnover on a CONTIGUOUS period grid.

    The kernel maintains the actually-held (drifted) weight vector across
    periods. On rows where rebalance_mask is True, the book is traded to the
    target row of `weights` and turnover is measured against the drifted
    holdings (compounded across any preceding no-rebalance rows). On rows
    where it is False (a period between rebalances — e.g. a skipped rolling
    window), the book is held: its P&L is booked and turnover is zero.
    `weights` rows on no-rebalance periods are ignored.
    """
    num_periods, num_stocks = weights.shape
    portfolio_returns = np.zeros(num_periods)
    turnover = np.zeros(num_periods)
    current = np.zeros(num_stocks)

    for period_idx in range(num_periods):
        if period_idx > 0:
            capital_base = 1.0 + portfolio_returns[period_idx - 1]
            if abs(capital_base) < 1e-12:
                capital_base = 1.0
            for stock_idx in range(num_stocks):
                previous_return = returns[period_idx - 1, stock_idx]
                if np.isnan(previous_return):
                    previous_return = 0.0
                current[stock_idx] = (
                    current[stock_idx] * (1.0 + previous_return) / capital_base
                )

        if rebalance_mask[period_idx]:
            traded = 0.0
            for stock_idx in range(num_stocks):
                traded += abs(weights[period_idx, stock_idx] - current[stock_idx])
                current[stock_idx] = weights[period_idx, stock_idx]
            turnover[period_idx] = traded / 2.0

        period_return = 0.0
        for stock_idx in range(num_stocks):
            stock_return = returns[period_idx, stock_idx]
            if np.isnan(stock_return):
                stock_return = 0.0
            period_return += current[stock_idx] * stock_return

        portfolio_returns[period_idx] = period_return

    return portfolio_returns, turnover


def compute_performance(port_ret, turnover, tc_rate, periods_per_year=12):
    """Compute backtest performance metrics."""
    if len(port_ret) == 0:
        return {
            'adjusted_returns': np.array([], dtype=np.float64),
            'cumulative_returns': np.array([], dtype=np.float64),
            'drawdown': np.array([], dtype=np.float64),
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_turnover': 0.0,
        }

    adjusted_returns = port_ret - tc_rate * turnover
    cumulative_returns = np.cumprod(1.0 + adjusted_returns)

    num_periods = len(adjusted_returns)
    annualized_return = (
        cumulative_returns[-1] ** (periods_per_year / num_periods) - 1.0
    )

    average_return = float(np.mean(adjusted_returns))
    return_std = (
        float(np.std(adjusted_returns, ddof=1))
        if num_periods > 1 else 0.0
    )
    sharpe_ratio = (
        average_return / return_std * np.sqrt(periods_per_year)
        if return_std > 0 else 0.0
    )

    # The peak includes the 1.0 inception baseline, so a drawdown that starts
    # at the very first period is measured instead of silently truncated.
    running_peak = np.maximum(np.maximum.accumulate(cumulative_returns), 1.0)
    drawdown = (cumulative_returns - running_peak) / running_peak
    max_drawdown = float(np.min(drawdown))
    avg_turnover = float(np.mean(turnover))

    return {
        'adjusted_returns': adjusted_returns,
        'cumulative_returns': cumulative_returns,
        'drawdown': drawdown,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'avg_turnover': avg_turnover,
    }


def yearly_table(adjusted_return_series, turnover_series, periods_per_year=12):
    """Build a yearly performance table."""
    if adjusted_return_series.empty:
        return pd.DataFrame(
            columns=[
                'Year',
                'Annual Return',
                'Sharpe Ratio',
                'Max Drawdown',
                'Avg Turnover',
            ]
        )

    rows = []
    for year in sorted(adjusted_return_series.index.year.unique()):
        year_mask = adjusted_return_series.index.year == year
        year_returns = adjusted_return_series.loc[year_mask].to_numpy()
        year_turnover = turnover_series.loc[year_mask].to_numpy()
        if len(year_returns) == 0:
            continue

        year_cumulative = np.cumprod(1.0 + year_returns)
        # Same inception-baseline rule per calendar year (start-of-year = 1.0).
        year_peak = np.maximum(np.maximum.accumulate(year_cumulative), 1.0)
        year_drawdown = (year_cumulative - year_peak) / year_peak

        annual_return = float(year_cumulative[-1] - 1.0)
        year_std = (
            float(np.std(year_returns, ddof=1))
            if len(year_returns) > 1 else 0.0
        )
        sharpe_ratio = (
            float(np.mean(year_returns)) / year_std * np.sqrt(periods_per_year)
            if year_std > 0 else 0.0
        )

        rows.append({
            'Year': int(year),
            'Annual Return': round(annual_return, 4),
            'Sharpe Ratio': round(sharpe_ratio, 4),
            'Max Drawdown': round(float(np.min(year_drawdown)), 4),
            'Avg Turnover': round(float(np.mean(year_turnover)), 4),
        })

    return pd.DataFrame(rows)

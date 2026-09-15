"""
build_monthly_end_return.py
Build a monthly stock-return panel from CRSP-style daily returns.

The output panel uses first-of-month timestamps to stay index-compatible with
the factor pickles, but each row represents the compounded return realized over
that calendar month from prior month-end to current month-end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = Path(r'D:\Quant\Data\Stock\crsp_ret.pkl')
DEFAULT_TEMPLATE_PATH = PROJECT_ROOT / 'Data' / 'Market_Data' / 'monthly_return.pkl'
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / 'Data' / 'Market_Data' / 'monthly_end_return.pkl'


def build_month_end_return_panel(daily_returns: pd.DataFrame, template_panel: pd.DataFrame | None = None):
    """
    Convert CRSP-style daily returns into a month-end-to-month-end wide panel.

    Parameters
    ----------
    daily_returns : DataFrame
        Must contain columns PERMNO, date, RET.
    template_panel : DataFrame, optional
        If provided, the result is reindexed to the template's index/columns.

    Returns
    -------
    DataFrame
        Wide panel indexed by month-start timestamps and columns=PERMNO.
    """
    required = {'PERMNO', 'date', 'RET'}
    missing = required.difference(daily_returns.columns)
    if missing:
        raise ValueError(f'daily_returns is missing required columns: {sorted(missing)}')

    x = daily_returns.loc[:, ['PERMNO', 'date', 'RET']].copy()
    x['PERMNO'] = pd.to_numeric(x['PERMNO'], errors='raise').astype(np.int64)
    x['date'] = pd.to_datetime(x['date'])
    x['RET'] = pd.to_numeric(x['RET'], errors='coerce')
    x['month'] = x['date'].values.astype('datetime64[M]')
    x['gross'] = 1.0 + x['RET'].fillna(0.0)

    monthly = (
        x.groupby(['month', 'PERMNO'], sort=False)
        .agg(
            gross_prod=('gross', 'prod'),
            obs_count=('RET', 'size'),
            valid_count=('RET', 'count'),
        )
    )
    monthly['monthly_ret'] = monthly['gross_prod'] - 1.0
    monthly.loc[monthly['valid_count'] != monthly['obs_count'], 'monthly_ret'] = np.nan

    panel = monthly['monthly_ret'].unstack('PERMNO').sort_index()
    panel.index = pd.DatetimeIndex(panel.index, name='month')

    if template_panel is not None:
        panel = panel.reindex(index=template_panel.index, columns=template_panel.columns)

    return panel


def build_month_end_return_file(raw_path=DEFAULT_RAW_PATH, template_path=DEFAULT_TEMPLATE_PATH, output_path=DEFAULT_OUTPUT_PATH):
    raw_path = Path(raw_path)
    template_path = None if template_path is None else Path(template_path)
    output_path = Path(output_path)

    print(f'Loading daily returns from: {raw_path}')
    daily_returns = pd.read_pickle(raw_path)

    template_panel = None
    if template_path is not None:
        print(f'Loading alignment template from: {template_path}')
        template_panel = pd.read_pickle(template_path)

    panel = build_month_end_return_panel(daily_returns, template_panel=template_panel)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_pickle(output_path)

    valid_fraction = float(panel.notna().mean().mean()) if panel.size else 0.0
    print(f'Wrote: {output_path}')
    print(f'Shape: {panel.shape}')
    print(f'Index range: {panel.index.min()} to {panel.index.max()}')
    print(f'Average non-missing fraction: {valid_fraction:.4f}')

    return panel


def parse_args():
    parser = argparse.ArgumentParser(description='Build monthly_end_return.pkl from CRSP daily returns.')
    parser.add_argument('--raw-path', default=str(DEFAULT_RAW_PATH))
    parser.add_argument('--template-path', default=str(DEFAULT_TEMPLATE_PATH))
    parser.add_argument('--output-path', default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build_month_end_return_file(
        raw_path=args.raw_path,
        template_path=args.template_path,
        output_path=args.output_path,
    )

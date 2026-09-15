"""
run_models_p1.py
Run the P1-factor backtest with each model and build the comparison report.

Usage (from Project_backtest/):
    python run_models_p1.py ridge      # one model
    python run_models_p1.py lgbm
    python run_models_p1.py mlp
    python run_models_p1.py compare    # table + rank-IC across finished runs

Every model runs on the identical configuration (same factors, return panel,
windows, screens, costs); only model_class/model_params differ, so the
comparison isolates the model choice.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models import LightGBMRegressor, RidgeRegressor, TorchMLPRegressor
from quant_strategy_ml import Quant_Strategy_ML
from run_backtest import build_monthly_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent / 'output'

# Data vintages. 'v2' = library built on the Dec-2024 WRDS data (test years 2010-2024);
# '2025' = library rebuilt on the 2026-08-17 WRDS refresh (CRSP through Dec-2025,
# Compustat/IBES restated), test years 2010-2025. Set with env P1_VINTAGE.
import os
VINTAGE = os.environ.get('P1_VINTAGE', '2025')
VINTAGES = {
    'v2': {
        'factor_path': PROJECT_ROOT / 'Data' / 'factor_pkls_p1',
        'return_path': PROJECT_ROOT / 'Data' / 'Market_Data' / 'monthly_end_return_crsp.pkl',
        'end_period': '2024-12-01',
        'suffix': 'v2',
    },
    '2025': {
        'factor_path': PROJECT_ROOT / 'Data' / 'factor_pkls_p1_2025',
        'return_path': PROJECT_ROOT / 'Data' / 'Market_Data' / 'monthly_end_return_crsp_2025.pkl',
        'end_period': '2025-12-01',
        'suffix': '2025',
    },
}
V = VINTAGES[VINTAGE]
FACTOR_PATH = V['factor_path']
RETURN_PATH = V['return_path']

# Tradability screen (env P1_SCREEN=1): require prior-month |price| >= $5 at the
# signal date, i.e. P1 'Price' (= log|prc|) >= ln(5). Motivation (2026-08-17):
# the unscreened LightGBM book had median |prc| ~$2-3 and median cap ~$25-50M;
# its 2025 return (+130%) came largely from single positions returning 10-20x
# (e.g. permno 84819 +2,225% in 2025-05) — untradeable at any real size.
import math
SCREEN = os.environ.get('P1_SCREEN', '0') == '1'
SCREEN_CFG = {'Price': math.log(5.0)} if SCREEN else {}
SCREEN_TAG = '_p5' if SCREEN else ''

MODELS = {
    'ridge': (RidgeRegressor, {'alpha': 5.0, 'fit_intercept': True}),
    'lgbm': (LightGBMRegressor, {}),
    'mlp': (TorchMLPRegressor, {}),
}
RUN_NAMES = {key: f"p1_{key}_{V['suffix']}{SCREEN_TAG}" for key in MODELS}


def build_config(model_key):
    model_class, model_params = MODELS[model_key]
    config = build_monthly_config(run_name=RUN_NAMES[model_key])
    config.update({
        'Factor_Path': FACTOR_PATH,
        'Return_Path': RETURN_PATH,
        'end_period': V['end_period'],
        'model_class': model_class,
        'model_params': model_params,
        'visualize': False,
        'tradability_screen': SCREEN_CFG,

        # v2 methodology (validated by the 2026-08-14 single-window probes):
        # - future_spread(n=1): the label is the stock's next-month return
        #   MINUS the cross-sectional mean, so pooled models fit which stocks
        #   beat their peers, not which calendar months were good. With the
        #   raw-return label the ridge predictions were systematically
        #   anti-predictive (rank-IC -0.025) despite raw factors carrying
        #   stable |IC| ~ 0.07.
        # - cs_rank features: per-date cross-sectional rank to [-1, 1]
        #   removes factor-level drift across time (probe: 2024 IC -0.027 ->
        #   +0.029 with both changes).
        'label_function': 'future_spread',
        'label_params': {'n': 1},
        'standardize': True,
        'standardize_mode': 'cs_rank',
    })
    return config


def run_model(model_key):
    strategy = Quant_Strategy_ML(build_config(model_key))
    strategy.run()
    print(f"\n[{model_key}] artifacts: {strategy.run_output_dir}")
    return strategy


def _rank_ic_series(predicted, true_values):
    """Per-signal-date Spearman correlation between predictions and realized
    next-period returns (both panels are signal-date indexed)."""
    ic = {}
    for date in predicted.index:
        pred_row = predicted.loc[date].dropna()
        true_row = true_values.loc[date].reindex(pred_row.index).dropna()
        pred_row = pred_row.reindex(true_row.index)
        if len(pred_row) < 30:
            continue
        ic[date] = pred_row.rank().corr(true_row.rank())
    return pd.Series(ic).sort_index()


def compare():
    rows = []
    ic_panels = {}
    for model_key, run_name in RUN_NAMES.items():
        run_dir = OUTPUT_ROOT / run_name
        if not (run_dir / 'run_summary.json').exists():
            print(f'[{model_key}] no finished run at {run_dir} - skipped')
            continue

        artifacts = Quant_Strategy_ML.load_saved_run(run_dir)
        summary = artifacts['run_summary']
        predicted = artifacts['adjusted_predicted_labels']
        true_values = artifacts['true_values']
        ic = _rank_ic_series(predicted, true_values)
        ic_panels[model_key] = ic

        rows.append({
            'model': model_key,
            'ann_return': summary['annualized_return'],
            'sharpe': summary['sharpe_ratio'],
            'max_drawdown': summary['max_drawdown'],
            'avg_turnover': summary['avg_turnover'],
            'long_ann_return': summary['long_only']['annualized_return'],
            'long_sharpe': summary['long_only']['sharpe_ratio'],
            'rank_ic_mean': float(ic.mean()),
            'rank_ic_ir': float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) > 0 else np.nan,
            'skipped_windows': summary.get('skipped_window_count', 0),
        })

    if not rows:
        print('No finished runs to compare.')
        return None

    table = pd.DataFrame(rows).set_index('model')
    table.to_csv(OUTPUT_ROOT / f"p1_model_comparison_{V['suffix']}{SCREEN_TAG}.csv")
    pd.DataFrame(ic_panels).to_pickle(OUTPUT_ROOT / f"p1_model_rank_ic_{V['suffix']}{SCREEN_TAG}.pkl")

    print('\nModel comparison (long-short after costs unless noted):')
    print(table.round(4).to_string())
    print(f"\nSaved: {OUTPUT_ROOT / 'p1_model_comparison.csv'}")
    return table


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in (*MODELS, 'compare'):
        print(f'Usage: python run_models_p1.py [{" | ".join((*MODELS, "compare"))}]')
        sys.exit(1)
    if sys.argv[1] == 'compare':
        compare()
    else:
        run_model(sys.argv[1])

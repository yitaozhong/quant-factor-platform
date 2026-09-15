"""
run_backtest.py
Concrete runner for the monthly cross-sectional stock-selection backtest.
"""

import re
from datetime import datetime
from pathlib import Path

from models import RidgeRegressor
from quant_strategy_ml import Quant_Strategy_ML


def resolve_project_root():
    """
    Resolve the Algo_Trading project root for both scripts and notebooks.
    """
    candidates = []

    file_path = globals().get('__file__')
    if file_path is not None:
        file_path = Path(file_path).resolve()
        candidates.extend([file_path.parent, file_path.parent.parent])

    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.extend(cwd.parents)

    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            ordered_candidates.append(candidate)

    for candidate in ordered_candidates:
        if (
            (candidate / 'Data' / 'factor_pkls').exists()
            and (candidate / 'Project_backtest').exists()
        ):
            return candidate
        if (
            candidate.name == 'Project_backtest'
            and (candidate.parent / 'Data' / 'factor_pkls').exists()
        ):
            return candidate.parent

    raise FileNotFoundError(
        'Could not locate the Algo_Trading project root. '
        'Run the notebook from inside the project or pass an explicit project_root.'
    )


def next_daily_run_name(output_root, prefix='backtest', now=None):
    """
    Generate names like backtest_20260404_1, backtest_20260404_2, ...
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    date_stamp = (datetime.now() if now is None else now).strftime('%Y%m%d')
    pattern = re.compile(rf'^{re.escape(prefix)}_{date_stamp}_(\d+)$')

    next_index = 1
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            next_index = max(next_index, int(match.group(1)) + 1)

    return f'{prefix}_{date_stamp}_{next_index}'


PROJECT_ROOT = resolve_project_root()
DATA_ROOT = PROJECT_ROOT / 'Data'
PROJECT_BACKTEST_DIR = PROJECT_ROOT / 'Project_backtest'


def build_monthly_config(project_root=None, run_name=None):
    project_root = PROJECT_ROOT if project_root is None else Path(project_root)
    data_root = project_root / 'Data'
    project_backtest_dir = project_root / 'Project_backtest'
    output_root = project_backtest_dir / 'output'

    if run_name is None:
        run_name = next_daily_run_name(output_root)

    return {
        'Factor_Path': data_root / 'factor_pkls',
        'Return_Path': data_root / 'Market_Data' / 'monthly_end_return.pkl',

        # Monthly sample with a 10-year train window and annual retraining.
        'start_period': '2000-01-01',
        'end_period': '2024-12-01',
        'train_window_periods': 120,
        'train_interval_periods': 12,

        # Screen for seasoned stocks with enough factor and return history.
        'stock_valid_period_frac': 0.60,
        'stock_valid_factor_frac': 0.35,
        'ret_valid_period_frac': 0.60,

        # Keep factors that are broadly available, then rank by IC / IR.
        'factor_valid_stock_frac': 0.30,
        'factor_valid_period_frac': 0.60,
        'min_ic_obs': 200,
        'factor_filter': {
            'Rank_IC': {'threshold': 0.01},
            'Rank_IR': {'threshold': 0.15, 'Rank_threshold': 40},
        },

        # OSP monthly signals are tradeable by month-end, so with a clean
        # month-end return panel the first executable return sits at row t + 1.
        'signal_delay_periods': 1,
        'label_function': 'current_period_return',
        'label_params': {},

        # Final train-date screen after the label is defined.
        'valid_stock_frac': 0.25,
        'valid_factor_frac': 0.70,

        # Cross-sectional monthly factors are usually best filled cross-sectionally.
        'fill_method': 'fill_median_cs',
        'standardize': True,

        # Self-contained ridge regression avoids external dependencies and scales
        # better than tree boosting on large flat monthly panels.
        'model_class': RidgeRegressor,
        'model_params': {
            'alpha': 5.0,
            'fit_intercept': True,
        },

        # Require most selected factors to be present before test-time filling.
        'valid_feature_frac': 0.75,
        'test_stock_from_training_only': True,

        # Leave prediction post-processing off by default.
        'process_dictionary': {},

        'number_stock_pick': 100,
        'exchange_frac': 0.20,

        'transaction_cost_rate': 0.001,
        'periods_per_year': 12,

        # Persist every run so results can be inspected later without rerunning.
        'save_artifacts': True,
        'output_root': output_root,
        'run_name': run_name,

        # plotly is optional in this environment.
        'visualize': True,
    }


def run_backtest_notebook(project_root=None, config_overrides=None):
    """
    Notebook-friendly entry point.

    Example
    -------
    from run_backtest import run_backtest_notebook
    strategy = run_backtest_notebook()
    """
    config = build_monthly_config(project_root=project_root)
    if config_overrides:
        config.update(config_overrides)

    strategy = Quant_Strategy_ML(config)
    strategy.run()
    return strategy


if __name__ == '__main__':
    strategy = Quant_Strategy_ML(build_monthly_config())
    strategy.run()

    if strategy.backtest_results:
        print('\nYearly Performance:')
        print(strategy.backtest_results['yearly_table'].to_string(index=False))

    

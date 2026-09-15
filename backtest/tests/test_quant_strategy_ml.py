from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from quant_strategy_ml import Quant_Strategy_ML
from build_monthly_end_return import build_month_end_return_panel
from run_backtest import next_daily_run_name
from utils_backtest import _backtest_core


class SimpleLinearModel:
    def fit(self, x_value, y_value):
        x_with_intercept = np.column_stack([np.ones(len(x_value)), x_value])
        self.beta = np.linalg.lstsq(x_with_intercept, y_value, rcond=None)[0]
        return self

    def predict(self, x_value):
        x_with_intercept = np.column_stack([np.ones(len(x_value)), x_value])
        return x_with_intercept @ self.beta


def build_synthetic_data():
    dates = pd.date_range('2020-01-01', periods=6, freq='MS')
    stocks = [10001, 10002, 10003, 10004]

    quality = pd.DataFrame(
        [
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 2.0, 1.0, 3.0],
            [2.0, 4.0, 3.0, 1.0],
            [3.0, 1.0, 4.0, 2.0],
            [1.0, 4.0, 2.0, 3.0],
        ],
        index=dates,
        columns=stocks,
    )
    noise = pd.DataFrame(
        [
            [1.0, 4.0, 2.0, 3.0],
            [2.0, 1.0, 4.0, 3.0],
            [3.0, 1.0, 4.0, 2.0],
            [4.0, 2.0, 1.0, 3.0],
            [2.0, 3.0, 1.0, 4.0],
            [3.0, 4.0, 1.0, 2.0],
        ],
        index=dates,
        columns=stocks,
    )
    returns = quality.sub(quality.mean(axis=1), axis=0) / 100.0

    return dates, stocks, quality, noise, returns


def make_config(factor_dir, return_path, dates, **overrides):
    config = {
        'Factor_Path': factor_dir,
        'Return_Path': return_path,
        'start_period': str(dates[0].date()),
        'end_period': str(dates[-1].date()),
        'train_window_periods': 4,
        'train_interval_periods': 1,
        'stock_valid_period_frac': 0.5,
        'stock_valid_factor_frac': 0.5,
        'ret_valid_period_frac': 0.5,
        'factor_valid_period_frac': 0.5,
        'factor_valid_stock_frac': 0.5,
        'factor_filter': {},
        'min_ic_obs': 3,
        'label_function': 'current_period_return',
        'label_params': {},
        'valid_stock_frac': 0.5,
        'valid_factor_frac': 0.5,
        'fill_method': 'fill_median_cs',
        'standardize': False,
        'model_class': SimpleLinearModel,
        'model_params': {},
        'valid_feature_frac': 0.5,
        'process_dictionary': {},
        'number_stock_pick': 1,
        'exchange_frac': 1.0,
        'transaction_cost_rate': 0.0,
        'periods_per_year': 12,
    }
    config.update(overrides)
    return config


def prepare_first_training_window(strategy):
    strategy.set_train_test_window()
    strategy._current_train_dates = strategy.train_period_lists[0]
    strategy.load_factors()
    strategy.available_stock()
    strategy._reindex_to_stocks()


class QuantStrategyMLTests(unittest.TestCase):
    def setUp(self):
        self.factor_dir = Path('mock_factors')
        self.return_path = Path('mock_monthly_return.pkl')
        self.artifact_root = PROJECT_DIR / 'tests_output_artifacts'
        (
            self.dates,
            self.stocks,
            self.quality,
            self.noise,
            self.returns,
        ) = build_synthetic_data()

    def tearDown(self):
        shutil.rmtree(self.artifact_root, ignore_errors=True)

    def fake_read_pickle(self, path):
        path_str = str(path)
        if path_str.endswith('quality.pkl'):
            return self.quality.copy()
        if path_str.endswith('noise.pkl'):
            return self.noise.copy()
        if path_str.endswith('mock_monthly_return.pkl'):
            return self.returns.copy()
        raise KeyError(path_str)

    def build_strategy(self, **overrides):
        config = make_config(self.factor_dir, self.return_path, self.dates, **overrides)
        factor_listing = ['quality.pkl', 'noise.pkl']
        return (
            patch('quant_strategy_ml.os.listdir', return_value=factor_listing),
            patch('quant_strategy_ml.pd.read_pickle', side_effect=self.fake_read_pickle),
            config,
        )

    def test_future_return_excludes_last_n_periods(self):
        listdir_patch, read_pickle_patch, config = self.build_strategy(
            label_function='future_return',
            label_params={'n': 2},
        )

        with listdir_patch, read_pickle_patch:
            strategy = Quant_Strategy_ML(config)
            prepare_first_training_window(strategy)
            strategy.factor_selection()
            strategy.label_construction()
            strategy._compute_future_excluded_window()

        self.assertEqual(
            strategy.future_excluded_train_window,
            strategy._current_train_dates[:-2],
        )

        first_date = strategy._current_train_dates[0]
        first_stock = strategy.stock_list[0]
        expected = (
            (1.0 + self.returns.loc[strategy._current_train_dates[1], first_stock])
            * (1.0 + self.returns.loc[strategy._current_train_dates[2], first_stock])
            - 1.0
        )
        self.assertTrue(np.isclose(strategy.label.loc[first_date, first_stock], expected))

    def test_factor_selection_rank_threshold_keeps_top_factor(self):
        listdir_patch, read_pickle_patch, config = self.build_strategy(
            factor_filter={'Rank_IC': {'Rank_threshold': 1}},
        )

        with listdir_patch, read_pickle_patch:
            strategy = Quant_Strategy_ML(config)
            prepare_first_training_window(strategy)
            strategy.factor_selection()

        self.assertEqual(strategy.selected_factors, ['quality'])

    def test_backtest_core_uses_drifted_turnover(self):
        weights = np.array([
            [0.5, -0.5],
            [0.5, -0.5],
        ])
        returns = np.array([
            [0.10, 0.00],
            [0.00, 0.00],
        ])

        portfolio_returns, turnover = _backtest_core(
            weights, returns, np.ones(len(weights), dtype=np.bool_)
        )
        expected_turnover = 0.5 * (
            abs(0.5 - (0.5 * 1.10 / 1.05))
            + abs(-0.5 - (-0.5 * 1.00 / 1.05))
        )

        self.assertTrue(np.isclose(portfolio_returns[0], 0.05))
        self.assertTrue(np.isclose(turnover[1], expected_turnover))

    def test_build_month_end_return_panel_compounds_with_missing_guard(self):
        daily = pd.DataFrame(
            {
                'PERMNO': [1, 1, 1, 1, 2, 2, 2],
                'date': [
                    '2020-01-30', '2020-01-31',
                    '2020-02-03', '2020-02-28',
                    '2020-01-15', '2020-01-31', '2020-02-28',
                ],
                'RET': [0.10, 0.00, 0.05, -0.02, 0.03, np.nan, 0.04],
            }
        )
        template = pd.DataFrame(
            index=pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01']),
            columns=[1, 2, 3],
            dtype=float,
        )

        panel = build_month_end_return_panel(daily, template_panel=template)

        self.assertEqual(list(panel.index), list(template.index))
        self.assertEqual(list(panel.columns), list(template.columns))
        self.assertTrue(np.isclose(panel.loc[pd.Timestamp('2020-01-01'), 1], 0.10))
        self.assertTrue(
            np.isclose(
                panel.loc[pd.Timestamp('2020-02-01'), 1],
                (1.05 * 0.98) - 1.0,
            )
        )
        self.assertTrue(np.isnan(panel.loc[pd.Timestamp('2020-01-01'), 2]))
        self.assertTrue(np.isclose(panel.loc[pd.Timestamp('2020-02-01'), 2], 0.04))
        self.assertTrue(np.isnan(panel.loc[pd.Timestamp('2020-03-01'), 1]))
        self.assertTrue(np.isnan(panel.loc[pd.Timestamp('2020-01-01'), 3]))

    def test_execution_weights_shift_forward_when_signal_delay_is_one(self):
        listdir_patch, read_pickle_patch, config = self.build_strategy(
            signal_delay_periods=1,
            label_function='future_return',
            label_params={'n': 1},
        )

        with listdir_patch, read_pickle_patch:
            strategy = Quant_Strategy_ML(config)
            strategy.set_train_test_window()

        strategy.portfolio_weights = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            index=self.dates[:3],
            columns=['A', 'B'],
        )
        execution_weights = strategy._build_execution_weights()

        self.assertEqual(list(execution_weights.index), list(self.dates[1:4]))
        self.assertTrue(np.allclose(execution_weights.iloc[0].to_numpy(), [1.0, 0.0]))
        self.assertTrue(np.allclose(execution_weights.iloc[1].to_numpy(), [0.0, 1.0]))

    def test_select_test_stock_can_restrict_to_training_stocks(self):
        listdir_patch, read_pickle_patch, config = self.build_strategy(
            test_stock_from_training_only=True,
        )

        with listdir_patch, read_pickle_patch:
            strategy = Quant_Strategy_ML(config)

        strategy.selected_factors = ['quality', 'noise']
        strategy.stock_list = [10001, 10002]
        test_x = pd.DataFrame(
            {
                'quality': [1.0, 2.0, 3.0],
                'noise': [1.0, np.nan, 4.0],
            },
            index=[10001, 10002, 99999],
        )

        selected = strategy.select_test_stock(test_x)
        self.assertEqual(selected, [10001, 10002])

    def test_current_return_label_uses_configured_signal_delay(self):
        listdir_patch, read_pickle_patch, config = self.build_strategy(
            signal_delay_periods=2,
            label_function='current_period_return',
            label_params={},
        )

        with listdir_patch, read_pickle_patch:
            strategy = Quant_Strategy_ML(config)
            prepare_first_training_window(strategy)
            strategy.factor_selection()
            strategy.label_construction()
            strategy._compute_future_excluded_window()

        first_date = strategy._current_train_dates[0]
        delayed_date = strategy._current_train_dates[2]
        first_stock = strategy.stock_list[0]

        self.assertTrue(
            np.isclose(
                strategy.label.loc[first_date, first_stock],
                self.returns.loc[delayed_date, first_stock],
            )
        )
        self.assertEqual(
            strategy.future_excluded_train_window,
            strategy._current_train_dates[:-2],
        )

    def test_run_generates_predictions_and_backtest(self):
        listdir_patch, read_pickle_patch, config = self.build_strategy()

        with listdir_patch, read_pickle_patch:
            strategy = Quant_Strategy_ML(config)
            strategy.visualize = lambda: None
            strategy.run()

        self.assertEqual(list(strategy.predicted_labels.index), [self.dates[4], self.dates[5]])
        self.assertFalse(strategy.portfolio_weights.empty)
        self.assertIn('annualized_return', strategy.backtest_results)
        self.assertIn('long_only_annualized_return', strategy.backtest_results)
        self.assertIn('short_only_annualized_return', strategy.backtest_results)
        self.assertTrue(np.allclose(
            strategy.backtest_results['portfolio_returns'].to_numpy(),
            (
                strategy.backtest_results['long_only_portfolio_returns']
                + strategy.backtest_results['short_only_portfolio_returns']
            ).to_numpy(),
        ))

    def test_run_saves_artifacts_when_enabled(self):
        listdir_patch, read_pickle_patch, config = self.build_strategy(
            save_artifacts=True,
            output_root=self.artifact_root,
            run_name='unit_test_run',
        )

        with listdir_patch, read_pickle_patch:
            strategy = Quant_Strategy_ML(config)
            strategy.visualize = lambda: None
            strategy.run()

        run_dir = strategy.run_output_dir
        self.assertIsNotNone(run_dir)
        self.assertTrue((run_dir / 'config.json').exists())
        self.assertTrue((run_dir / 'run_summary.json').exists())
        self.assertTrue((run_dir / 'prediction_records.pkl').exists())
        self.assertTrue((run_dir / 'signal_holdings.csv').exists())
        self.assertTrue((run_dir / 'long_only_cumulative_returns.pkl').exists())
        self.assertTrue((run_dir / 'short_only_cumulative_returns.pkl').exists())
        self.assertTrue((run_dir / 'long_only_execution_holdings.csv').exists())
        self.assertTrue((run_dir / 'short_only_execution_holdings.csv').exists())
        self.assertTrue((run_dir / 'windows' / 'window_001' / 'model.pkl').exists())
        self.assertTrue((run_dir / 'windows' / 'window_001' / 'selected_factors.csv').exists())

        saved = Quant_Strategy_ML.load_saved_run(run_dir)
        self.assertIn('prediction_records', saved)
        self.assertIn('signal_holdings', saved)
        self.assertIn('long_only_cumulative_returns', saved)
        self.assertIn('short_only_cumulative_returns', saved)
        self.assertFalse(saved['prediction_records'].empty)

    def test_next_daily_run_name_increments_by_existing_folders(self):
        day_root = self.artifact_root / 'daily_names'
        day_root.mkdir(parents=True, exist_ok=True)
        (day_root / 'backtest_20260404_1').mkdir(exist_ok=True)
        (day_root / 'backtest_20260404_2').mkdir(exist_ok=True)
        (day_root / 'ignore_me').mkdir(exist_ok=True)

        run_name = next_daily_run_name(
            day_root,
            now=pd.Timestamp('2026-04-04 10:00:00').to_pydatetime(),
        )

        self.assertEqual(run_name, 'backtest_20260404_3')


if __name__ == '__main__':
    unittest.main()

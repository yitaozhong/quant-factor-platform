"""
quant_strategy_ml.py
Cross-sectional ML stock selection backtesting system.
"""

import inspect
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from utils_backtest import _backtest_core, compute_performance, yearly_table
from utils_preprocess import (
    LABEL_REGISTRY,
    apply_standardize,
    cs_rank_transform,
    cs_rank_transform_cross_section,
    fill_median_cs,
    fill_median_ts,
    fill_test_factors,
    fit_standardize,
)


class Quant_Strategy_ML:
    """Cross-sectional ML stock selection backtesting system."""

    # --------------------------------------------------------
    # 1. Initialisation
    # --------------------------------------------------------
    def __init__(self, config: dict):
        self.config = dict(config)
        self.factor_file_map = self._discover_factor_files()

        self.return_data = None
        self.trading_date_list = []

        self.train_period_lists = []
        self.test_period_lists = []

        self._current_train_dates = []
        self._return_train = None
        self.factors_train_dic = {}
        self.stock_list = []
        self.selected_factors = []
        self.selected_factor_metrics = pd.DataFrame()
        self.label = None
        self.label_future_periods = 0
        self.future_excluded_train_window = []
        self.adjust_training_window = []
        self.feature_x = None
        self.label_y = None
        self.training_index = None
        self.model = None
        self.train_mean = {}
        self.train_std = {}
        self.fill_medians = {}

        self.predicted_labels = None
        self.adjusted_predicted_labels = None
        self.true_values = None
        self.prediction_records = None
        self.portfolio_weights = None
        self.execution_portfolio_weights = None
        self.signal_holdings = None
        self.execution_holdings = None
        self.backtest_results = {}
        self.run_output_dir = None
        self.window_artifacts = []
        self._screen_panels = {}

    def _discover_factor_files(self):
        factor_path = self.config['Factor_Path']
        factor_file_map = {}

        for file_name in sorted(os.listdir(factor_path)):
            if file_name.endswith('.pkl'):
                factor_name = os.path.splitext(file_name)[0]
                factor_file_map[factor_name] = os.path.join(factor_path, file_name)

        if not factor_file_map:
            raise ValueError(f'No factor pickle files found under {factor_path}.')

        return factor_file_map

    def _reset_window_state(self):
        self.factors_train_dic = {}
        self.stock_list = []
        self.selected_factors = []
        self.selected_factor_metrics = pd.DataFrame()
        self.label = None
        self.label_future_periods = 0
        self.future_excluded_train_window = []
        self.adjust_training_window = []
        self.feature_x = None
        self.label_y = None
        self.training_index = None
        self.model = None
        self.train_mean = {}
        self.train_std = {}
        self.fill_medians = {}
        self._return_train = None
        self.execution_portfolio_weights = None

    @staticmethod
    def _sanitize_numeric_panel(panel):
        return panel.where(np.isfinite(panel), np.nan)

    def _get_signal_delay_periods(self):
        delay = int(self.config.get('signal_delay_periods', 0))
        if delay < 0:
            raise ValueError('signal_delay_periods must be non-negative.')
        return delay

    @staticmethod
    def _drop_last_dates(date_list, num_dates):
        if num_dates <= 0:
            return list(date_list)
        if num_dates >= len(date_list):
            return []
        return list(date_list[:-num_dates])

    def _get_signal_aligned_return_panel(self, signal_dates, delay_periods=None, stock_list=None):
        delay_periods = (
            self._get_signal_delay_periods()
            if delay_periods is None else int(delay_periods)
        )
        stock_list = self.stock_list if stock_list is None else stock_list
        aligned_returns = self.return_data.shift(-delay_periods)
        return aligned_returns.reindex(index=signal_dates, columns=stock_list)

    def _build_execution_weights(self):
        delay_periods = self._get_signal_delay_periods()
        signal_weights = self.portfolio_weights.sort_index()
        if delay_periods == 0:
            return signal_weights.copy()
        if signal_weights.empty:
            return signal_weights.iloc[0:0].copy()

        date_to_position = {
            date: idx for idx, date in enumerate(self.trading_date_list)
        }
        execution_dates = []
        keep_positions = []

        for row_idx, signal_date in enumerate(signal_weights.index):
            signal_position = date_to_position.get(signal_date)
            if signal_position is None:
                continue

            execution_position = signal_position + delay_periods
            if execution_position >= len(self.trading_date_list):
                continue

            execution_dates.append(self.trading_date_list[execution_position])
            keep_positions.append(row_idx)

        execution_weights = signal_weights.iloc[keep_positions].copy()
        execution_weights.index = execution_dates
        return execution_weights

    def _artifacts_enabled(self):
        return bool(self.config.get('save_artifacts', False))

    def _get_output_root(self):
        output_root = self.config.get('output_root')
        if output_root is None:
            output_root = Path(__file__).resolve().parent / 'output'
        return Path(output_root)

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(key): Quant_Strategy_ML._json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [Quant_Strategy_ML._json_safe(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        if isinstance(value, np.datetime64):
            return str(pd.Timestamp(value))
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if inspect.isclass(value):
            return f'{value.__module__}.{value.__name__}'
        if callable(value):
            module = getattr(value, '__module__', None)
            qualname = getattr(value, '__qualname__', None)
            if module and qualname:
                return f'{module}.{qualname}'
            return repr(value)
        return value

    @staticmethod
    def _write_json(path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    @staticmethod
    def _save_pickle(path, obj):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _save_series_csv(path, values, column_name):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.Series(list(values), name=column_name).to_csv(path, index=False)

    @staticmethod
    def _stack_frame(frame, dropna=True):
        try:
            stacked = frame.stack(future_stack=True)
            if dropna:
                stacked = stacked.dropna()
            return stacked
        except TypeError:
            return frame.stack(dropna=dropna)

    @staticmethod
    def _weights_to_holdings(weights, date_label):
        if weights is None or weights.empty:
            return pd.DataFrame(columns=[date_label, 'stock', 'weight', 'side'])

        stacked = Quant_Strategy_ML._stack_frame(weights, dropna=True).rename('weight')
        stacked = stacked[stacked != 0]
        if stacked.empty:
            return pd.DataFrame(columns=[date_label, 'stock', 'weight', 'side'])

        holdings = stacked.reset_index()
        holdings.columns = [date_label, 'stock', 'weight']
        holdings['side'] = np.where(holdings['weight'] > 0, 'long', 'short')
        return holdings.sort_values([date_label, 'side', 'stock']).reset_index(drop=True)

    def _get_execution_date_map(self, signal_dates):
        delay_periods = self._get_signal_delay_periods()
        date_to_position = {
            date: idx for idx, date in enumerate(self.trading_date_list)
        }
        mapping = {}
        for signal_date in signal_dates:
            signal_position = date_to_position.get(signal_date)
            if signal_position is None:
                mapping[signal_date] = pd.NaT
                continue

            execution_position = signal_position + delay_periods
            if execution_position >= len(self.trading_date_list):
                mapping[signal_date] = pd.NaT
                continue

            mapping[signal_date] = self.trading_date_list[execution_position]

        return mapping

    def _get_true_values_for_predictions(self, prediction_frame):
        if prediction_frame is None or prediction_frame.empty:
            return pd.DataFrame()

        true_values = self._get_signal_aligned_return_panel(
            signal_dates=prediction_frame.index.tolist(),
            stock_list=prediction_frame.columns.tolist(),
        )
        return true_values.reindex(index=prediction_frame.index, columns=prediction_frame.columns)

    def _build_prediction_records(self, original_pred, adjusted_pred, true_values):
        if original_pred is None or original_pred.empty:
            return pd.DataFrame(
                columns=[
                    'signal_date',
                    'execution_date',
                    'stock',
                    'original_prediction',
                    'adjusted_prediction',
                    'true_value',
                ]
            )

        original_stack = self._stack_frame(original_pred, dropna=True).rename('original_prediction')
        if original_stack.empty:
            return pd.DataFrame(
                columns=[
                    'signal_date',
                    'execution_date',
                    'stock',
                    'original_prediction',
                    'adjusted_prediction',
                    'true_value',
                ]
            )

        adjusted_stack = self._stack_frame(
            adjusted_pred.reindex_like(original_pred),
            dropna=True,
        )
        true_stack = self._stack_frame(
            true_values.reindex_like(original_pred),
            dropna=False,
        )

        records = original_stack.to_frame()
        records['adjusted_prediction'] = adjusted_stack.reindex(records.index)
        records['true_value'] = true_stack.reindex(records.index)

        signal_dates = records.index.get_level_values(0)
        execution_map = self._get_execution_date_map(signal_dates.unique())
        records['execution_date'] = signal_dates.map(execution_map)

        records = records.reset_index()
        records.columns = [
            'signal_date',
            'stock',
            'original_prediction',
            'adjusted_prediction',
            'true_value',
            'execution_date',
        ]
        return records[
            [
                'signal_date',
                'execution_date',
                'stock',
                'original_prediction',
                'adjusted_prediction',
                'true_value',
            ]
        ]

    def _initialize_run_output(self):
        self.window_artifacts = []
        self.run_output_dir = None

        if not self._artifacts_enabled():
            return

        output_root = self._get_output_root()
        output_root.mkdir(parents=True, exist_ok=True)

        run_name = self.config.get('run_name')
        if not run_name:
            run_name = f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'

        run_dir = output_root / str(run_name)
        suffix = 1
        while run_dir.exists():
            run_dir = output_root / f'{run_name}_{suffix:02d}'
            suffix += 1

        (run_dir / 'windows').mkdir(parents=True, exist_ok=True)
        self.run_output_dir = run_dir

        self._write_json(
            run_dir / 'config.json',
            self._json_safe(self.config),
        )
        self._save_pickle(run_dir / 'config.pkl', self.config)
        (output_root / 'latest_run.txt').write_text(str(run_dir), encoding='utf-8')

    def _save_window_artifacts(self, window_index, train_dates, test_dates, raw_predictions):
        if self.run_output_dir is None:
            return

        window_dir = self.run_output_dir / 'windows' / f'window_{window_index + 1:03d}'
        window_dir.mkdir(parents=True, exist_ok=True)

        true_values = self._get_true_values_for_predictions(raw_predictions)
        window_summary = {
            'status': 'completed',
            'window_index': int(window_index + 1),
            'train_start': str(train_dates[0]) if train_dates else None,
            'train_end': str(train_dates[-1]) if train_dates else None,
            'test_start': str(test_dates[0]) if test_dates else None,
            'test_end': str(test_dates[-1]) if test_dates else None,
            'available_stock_count': int(len(self.stock_list)),
            'selected_factor_count': int(len(self.selected_factors)),
            'training_row_count': int(len(self.training_index)) if self.training_index is not None else 0,
            'prediction_date_count': int(len(raw_predictions.index)),
            'prediction_cell_count': int(raw_predictions.count().sum()) if raw_predictions is not None else 0,
        }

        self._write_json(window_dir / 'summary.json', window_summary)
        self._save_series_csv(window_dir / 'train_dates.csv', train_dates, 'date')
        self._save_series_csv(window_dir / 'adjusted_training_dates.csv', self.adjust_training_window, 'date')
        self._save_series_csv(window_dir / 'test_dates.csv', test_dates, 'date')
        self._save_series_csv(window_dir / 'available_stocks.csv', self.stock_list, 'stock')
        self._save_series_csv(window_dir / 'selected_factors.csv', self.selected_factors, 'factor')
        self.selected_factor_metrics.to_csv(window_dir / 'selected_factor_metrics.csv')
        raw_predictions.to_pickle(window_dir / 'raw_predictions.pkl')
        true_values.to_pickle(window_dir / 'true_values.pkl')
        self._save_pickle(
            window_dir / 'preprocess_state.pkl',
            {
                'selected_factors': list(self.selected_factors),
                'stock_list': list(self.stock_list),
                'train_mean': self.train_mean,
                'train_std': self.train_std,
                'fill_medians': self.fill_medians,
            },
        )

        try:
            self._save_pickle(window_dir / 'model.pkl', self.model)
        except Exception as exc:
            (window_dir / 'model_save_error.txt').write_text(str(exc), encoding='utf-8')

        self.window_artifacts.append(
            {
                'window_index': int(window_index + 1),
                'window_dir': str(window_dir),
                'test_dates': [str(date) for date in test_dates],
            }
        )

    def _save_skipped_window_artifacts(self, window_index, train_dates, test_dates, error_message):
        if self.run_output_dir is None:
            return

        window_dir = self.run_output_dir / 'windows' / f'window_{window_index + 1:03d}'
        window_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(
            window_dir / 'summary.json',
            {
                'status': 'skipped',
                'window_index': int(window_index + 1),
                'train_start': str(train_dates[0]) if train_dates else None,
                'train_end': str(train_dates[-1]) if train_dates else None,
                'test_start': str(test_dates[0]) if test_dates else None,
                'test_end': str(test_dates[-1]) if test_dates else None,
                'error': error_message,
            },
        )

        self.window_artifacts.append(
            {
                'window_index': int(window_index + 1),
                'window_dir': str(window_dir),
                'test_dates': [str(date) for date in test_dates],
                'status': 'skipped',
                'error': error_message,
            }
        )

    @staticmethod
    def _extract_book_weights(weights, side):
        if weights is None or weights.empty:
            return pd.DataFrame(index=weights.index, columns=weights.columns) if weights is not None else pd.DataFrame()

        values = weights.to_numpy(dtype=np.float64, copy=True)
        if side == 'long':
            values = np.where(values > 0.0, values, 0.0)
            gross = values.sum(axis=1, keepdims=True)
        elif side == 'short':
            values = np.where(values < 0.0, values, 0.0)
            gross = -values.sum(axis=1, keepdims=True)
        else:
            raise ValueError(f'Unsupported side: {side}')

        normalized = np.divide(
            values,
            gross,
            out=np.zeros_like(values),
            where=gross > 0.0,
        )
        return pd.DataFrame(normalized, index=weights.index, columns=weights.columns)

    @staticmethod
    def _compute_leg_performance(weights, returns, dates, tc_rate, periods_per_year,
                                 rebalance_mask=None):
        if rebalance_mask is None:
            rebalance_mask = np.ones(len(weights), dtype=np.bool_)
        port_ret, turnover = _backtest_core(
            weights.values.astype(np.float64),
            returns.values.astype(np.float64),
            np.asarray(rebalance_mask, dtype=np.bool_),
        )
        perf = compute_performance(port_ret, turnover, tc_rate, periods_per_year)

        adjusted_returns = pd.Series(perf['adjusted_returns'], index=dates)
        turnover_series = pd.Series(turnover, index=dates)

        return {
            'weights': weights,
            'portfolio_returns': pd.Series(port_ret, index=dates),
            'adjusted_returns': adjusted_returns,
            'cumulative_returns': pd.Series(perf['cumulative_returns'], index=dates),
            'turnover': turnover_series,
            'drawdown': pd.Series(perf['drawdown'], index=dates),
            'annualized_return': perf['annualized_return'],
            'sharpe_ratio': perf['sharpe_ratio'],
            'max_drawdown': perf['max_drawdown'],
            'avg_turnover': perf['avg_turnover'],
            'yearly_table': yearly_table(adjusted_returns, turnover_series, periods_per_year),
        }

    @staticmethod
    def _add_leg_results(container, leg_name, leg_result):
        prefix = '' if leg_name == 'long_short' else f'{leg_name}_'
        for key, value in leg_result.items():
            if key == 'weights':
                continue
            container[f'{prefix}{key}'] = value

    def _finalize_run_views(self):
        self.true_values = self._get_true_values_for_predictions(self.predicted_labels)
        self.prediction_records = self._build_prediction_records(
            self.predicted_labels,
            self.adjusted_predicted_labels,
            self.true_values,
        )
        self.signal_holdings = self._weights_to_holdings(
            self.portfolio_weights,
            'signal_date',
        )
        self.execution_holdings = self._weights_to_holdings(
            self.execution_portfolio_weights,
            'execution_date',
        )

    def _save_run_artifacts(self):
        if self.run_output_dir is None:
            return

        self.predicted_labels.to_pickle(self.run_output_dir / 'predicted_labels.pkl')
        self.adjusted_predicted_labels.to_pickle(self.run_output_dir / 'adjusted_predicted_labels.pkl')
        self.true_values.to_pickle(self.run_output_dir / 'true_values.pkl')
        self.prediction_records.to_pickle(self.run_output_dir / 'prediction_records.pkl')
        self.portfolio_weights.to_pickle(self.run_output_dir / 'portfolio_weights_signal.pkl')
        self.execution_portfolio_weights.to_pickle(self.run_output_dir / 'portfolio_weights_execution.pkl')
        self.signal_holdings.to_csv(self.run_output_dir / 'signal_holdings.csv', index=False)
        self.execution_holdings.to_csv(self.run_output_dir / 'execution_holdings.csv', index=False)

        leg_results = self.backtest_results.get('leg_results', {})
        for leg_name in ('long_only', 'short_only'):
            leg_result = leg_results.get(leg_name)
            if leg_result is None:
                continue

            leg_result['weights'].to_pickle(self.run_output_dir / f'{leg_name}_portfolio_weights.pkl')
            self._weights_to_holdings(
                leg_result['weights'],
                'execution_date',
            ).to_csv(
                self.run_output_dir / f'{leg_name}_execution_holdings.csv',
                index=False,
            )

        for key, value in self.backtest_results.items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(self.run_output_dir / f'{key}.csv', index=True)
            elif isinstance(value, pd.Series):
                value.to_pickle(self.run_output_dir / f'{key}.pkl')

        run_summary = {
            'run_output_dir': str(self.run_output_dir),
            'skipped_window_count': int(getattr(self, 'skipped_window_count', 0)),
            'unrebalanced_period_count': int(self.backtest_results.get('unrebalanced_periods', 0)),
            'prediction_date_count': int(len(self.predicted_labels.index)),
            'prediction_cell_count': int(self.predicted_labels.count().sum()),
            'signal_holding_count': int(len(self.signal_holdings)),
            'execution_holding_count': int(len(self.execution_holdings)),
            'annualized_return': float(self.backtest_results.get('annualized_return', np.nan)),
            'sharpe_ratio': float(self.backtest_results.get('sharpe_ratio', np.nan)),
            'max_drawdown': float(self.backtest_results.get('max_drawdown', np.nan)),
            'avg_turnover': float(self.backtest_results.get('avg_turnover', np.nan)),
            'long_only': {
                'annualized_return': float(self.backtest_results.get('long_only_annualized_return', np.nan)),
                'sharpe_ratio': float(self.backtest_results.get('long_only_sharpe_ratio', np.nan)),
                'max_drawdown': float(self.backtest_results.get('long_only_max_drawdown', np.nan)),
                'avg_turnover': float(self.backtest_results.get('long_only_avg_turnover', np.nan)),
            },
            'short_only': {
                'annualized_return': float(self.backtest_results.get('short_only_annualized_return', np.nan)),
                'sharpe_ratio': float(self.backtest_results.get('short_only_sharpe_ratio', np.nan)),
                'max_drawdown': float(self.backtest_results.get('short_only_max_drawdown', np.nan)),
                'avg_turnover': float(self.backtest_results.get('short_only_avg_turnover', np.nan)),
            },
            'window_artifacts': self.window_artifacts,
        }
        self._write_json(self.run_output_dir / 'run_summary.json', run_summary)

        for meta in self.window_artifacts:
            if meta.get('status') == 'skipped':
                continue

            window_dir = Path(meta['window_dir'])
            test_dates = pd.to_datetime(meta['test_dates'])
            adjusted_slice = self.adjusted_predicted_labels.reindex(index=test_dates).dropna(how='all')
            records_slice = self.prediction_records.loc[
                self.prediction_records['signal_date'].isin(test_dates)
            ].copy()
            signal_weights_slice = self.portfolio_weights.reindex(index=test_dates).dropna(how='all')

            execution_dates = (
                records_slice['execution_date']
                .dropna()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
            execution_weights_slice = self.execution_portfolio_weights.reindex(index=execution_dates).dropna(how='all')

            adjusted_slice.to_pickle(window_dir / 'adjusted_predictions.pkl')
            records_slice.to_pickle(window_dir / 'prediction_records.pkl')
            signal_weights_slice.to_pickle(window_dir / 'signal_weights.pkl')
            execution_weights_slice.to_pickle(window_dir / 'execution_weights.pkl')
            self._weights_to_holdings(signal_weights_slice, 'signal_date').to_csv(
                window_dir / 'signal_holdings.csv',
                index=False,
            )
            self._weights_to_holdings(execution_weights_slice, 'execution_date').to_csv(
                window_dir / 'execution_holdings.csv',
                index=False,
            )

    @staticmethod
    def load_saved_run(run_dir):
        run_dir = Path(run_dir)
        artifacts = {}

        json_files = {
            'config': 'config.json',
            'run_summary': 'run_summary.json',
        }
        for key, file_name in json_files.items():
            path = run_dir / file_name
            if path.exists():
                artifacts[key] = json.loads(path.read_text(encoding='utf-8'))

        pickle_files = {
            'predicted_labels': 'predicted_labels.pkl',
            'adjusted_predicted_labels': 'adjusted_predicted_labels.pkl',
            'true_values': 'true_values.pkl',
            'prediction_records': 'prediction_records.pkl',
            'portfolio_weights_signal': 'portfolio_weights_signal.pkl',
            'portfolio_weights_execution': 'portfolio_weights_execution.pkl',
            'portfolio_returns': 'portfolio_returns.pkl',
            'adjusted_returns': 'adjusted_returns.pkl',
            'cumulative_returns': 'cumulative_returns.pkl',
            'turnover': 'turnover.pkl',
            'drawdown': 'drawdown.pkl',
            'long_only_portfolio_returns': 'long_only_portfolio_returns.pkl',
            'long_only_adjusted_returns': 'long_only_adjusted_returns.pkl',
            'long_only_cumulative_returns': 'long_only_cumulative_returns.pkl',
            'long_only_turnover': 'long_only_turnover.pkl',
            'long_only_drawdown': 'long_only_drawdown.pkl',
            'long_only_portfolio_weights': 'long_only_portfolio_weights.pkl',
            'short_only_portfolio_returns': 'short_only_portfolio_returns.pkl',
            'short_only_adjusted_returns': 'short_only_adjusted_returns.pkl',
            'short_only_cumulative_returns': 'short_only_cumulative_returns.pkl',
            'short_only_turnover': 'short_only_turnover.pkl',
            'short_only_drawdown': 'short_only_drawdown.pkl',
            'short_only_portfolio_weights': 'short_only_portfolio_weights.pkl',
        }
        for key, file_name in pickle_files.items():
            path = run_dir / file_name
            if path.exists():
                artifacts[key] = pd.read_pickle(path)

        yearly_table_path = run_dir / 'yearly_table.csv'
        if yearly_table_path.exists():
            artifacts['yearly_table'] = pd.read_csv(yearly_table_path)

        long_only_yearly_table_path = run_dir / 'long_only_yearly_table.csv'
        if long_only_yearly_table_path.exists():
            artifacts['long_only_yearly_table'] = pd.read_csv(long_only_yearly_table_path)

        short_only_yearly_table_path = run_dir / 'short_only_yearly_table.csv'
        if short_only_yearly_table_path.exists():
            artifacts['short_only_yearly_table'] = pd.read_csv(short_only_yearly_table_path)

        signal_holdings_path = run_dir / 'signal_holdings.csv'
        if signal_holdings_path.exists():
            artifacts['signal_holdings'] = pd.read_csv(signal_holdings_path)

        execution_holdings_path = run_dir / 'execution_holdings.csv'
        if execution_holdings_path.exists():
            artifacts['execution_holdings'] = pd.read_csv(execution_holdings_path)

        long_only_execution_holdings_path = run_dir / 'long_only_execution_holdings.csv'
        if long_only_execution_holdings_path.exists():
            artifacts['long_only_execution_holdings'] = pd.read_csv(long_only_execution_holdings_path)

        short_only_execution_holdings_path = run_dir / 'short_only_execution_holdings.csv'
        if short_only_execution_holdings_path.exists():
            artifacts['short_only_execution_holdings'] = pd.read_csv(short_only_execution_holdings_path)

        return artifacts

    # --------------------------------------------------------
    # 2. Training / testing window setup
    # --------------------------------------------------------
    def set_train_test_window(self):
        """Load return panel and build rolling train/test windows."""
        self.return_data = pd.read_pickle(self.config['Return_Path']).sort_index()
        if not isinstance(self.return_data.index, pd.DatetimeIndex):
            self.return_data.index = pd.to_datetime(self.return_data.index)
        self.return_data = self._sanitize_numeric_panel(self.return_data)
        self.trading_date_list = self.return_data.index.tolist()

        start = pd.Timestamp(self.config['start_period'])
        end = pd.Timestamp(self.config['end_period'])
        train_len = int(self.config['train_window_periods'])
        interval = int(self.config['train_interval_periods'])

        dates = [d for d in self.trading_date_list if start <= d <= end]
        if len(dates) <= train_len:
            raise ValueError('The date range is too short for the requested rolling window.')

        self.train_period_lists = []
        self.test_period_lists = []

        i = 0
        while i + train_len < len(dates):
            train = dates[i: i + train_len]
            t_start = i + train_len
            t_end = min(t_start + interval, len(dates))
            test = dates[t_start: t_end]
            if test:
                self.train_period_lists.append(train)
                self.test_period_lists.append(test)
            i += interval

        print(f"Number of rolling windows: {len(self.train_period_lists)}")

    # --------------------------------------------------------
    # 4. Load training factors
    # --------------------------------------------------------
    def load_factors(self):
        """Load all factor panels and reindex to current training window."""
        self.factors_train_dic = {}
        for factor_name, factor_file in self.factor_file_map.items():
            factor_panel = pd.read_pickle(factor_file)
            self.factors_train_dic[factor_name] = self._sanitize_numeric_panel(
                factor_panel.reindex(index=self._current_train_dates)
            )
            del factor_panel

    # --------------------------------------------------------
    # 5. Stock availability screening
    # --------------------------------------------------------
    def available_stock(self):
        """Select stocks with sufficient valid data during training."""
        frac_period = self.config['stock_valid_period_frac']
        frac_factor = self.config['stock_valid_factor_frac']
        frac_ret = self.config['ret_valid_period_frac']

        delay_periods = self._get_signal_delay_periods()
        observable_dates = self._drop_last_dates(
            self._current_train_dates,
            delay_periods,
        )

        n_periods = len(observable_dates)
        all_stocks = self.return_data.columns
        if n_periods == 0:
            raise ValueError('The current training window is too short for the configured signal delay.')

        # (n_factors x n_stocks) bool — factor-period validity per stock
        factor_pass_count = np.zeros(len(all_stocks), dtype=np.int32)
        for factor_df in self.factors_train_dic.values():
            valid_frac = (
                factor_df.reindex(index=observable_dates, columns=all_stocks).notna().sum(axis=0) / n_periods
            )
            factor_pass_count += valid_frac.to_numpy() >= frac_period

        factor_pass = factor_pass_count >= frac_factor * len(self.factors_train_dic)

        aligned_returns = self._get_signal_aligned_return_panel(
            observable_dates,
            delay_periods=delay_periods,
            stock_list=all_stocks,
        )
        ret_valid = (aligned_returns.notna().sum(axis=0).to_numpy() / n_periods) >= frac_ret

        self.stock_list = all_stocks[factor_pass & ret_valid].tolist()
        if not self.stock_list:
            raise ValueError('No stocks passed the training availability screen.')
        print(f"Number of available stocks: {len(self.stock_list)}")

    # --------------------------------------------------------
    # 6. Reindex to selected stocks
    # --------------------------------------------------------
    def _reindex_to_stocks(self):
        for name in self.factors_train_dic:
            self.factors_train_dic[name] = (
                self.factors_train_dic[name].reindex(columns=self.stock_list)
            )
        self._return_train = self.return_data.reindex(
            index=self._current_train_dates, columns=self.stock_list
        )

    def _compute_factor_metric_table(self):
        delay_periods = self._get_signal_delay_periods()
        signal_dates = self._drop_last_dates(self._current_train_dates, delay_periods)
        if not signal_dates:
            return pd.DataFrame(columns=['Rank_IC', 'Rank_IR'])

        aligned_returns = self._get_signal_aligned_return_panel(signal_dates)
        rank_return = aligned_returns.rank(axis=1, method='average', na_option='keep')
        min_ic_obs = int(self.config.get('min_ic_obs', 5))
        metric_rows = {}

        for factor_name, factor_df in self.factors_train_dic.items():
            factor_signal = factor_df.reindex(index=signal_dates)
            pair_counts = (factor_signal.notna() & aligned_returns.notna()).sum(axis=1)
            ic_series = (
                factor_signal.rank(axis=1, method='average', na_option='keep')
                .corrwith(rank_return, axis=1)
                .loc[lambda series: pair_counts.loc[series.index] >= min_ic_obs]
                .dropna()
            )
            if ic_series.empty:
                continue

            mean_ic = float(ic_series.mean())
            std_ic = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
            metric_rows[factor_name] = {
                'Rank_IC': abs(mean_ic),
                'Rank_IR': abs(mean_ic / std_ic) if std_ic > 0 else 0.0,
            }

        if not metric_rows:
            return pd.DataFrame(columns=['Rank_IC', 'Rank_IR'])

        return pd.DataFrame.from_dict(metric_rows, orient='index')

    # --------------------------------------------------------
    # 7. Factor selection
    # --------------------------------------------------------
    def factor_selection(self):
        """Two-stage factor screen: validity then IC/IR metrics."""
        frac_period = self.config['factor_valid_period_frac']
        frac_stock = self.config['factor_valid_stock_frac']
        observable_dates = self._drop_last_dates(
            self._current_train_dates,
            self._get_signal_delay_periods(),
        )
        n_periods = len(observable_dates)
        n_stocks = len(self.stock_list)
        if n_periods == 0:
            raise ValueError('The current training window is too short for the configured signal delay.')

        valid = {}
        for factor_name, factor_df in self.factors_train_dic.items():
            valid_per_stock = factor_df.reindex(index=observable_dates).notna().sum(axis=0) / n_periods
            if (valid_per_stock >= frac_period).sum() / n_stocks >= frac_stock:
                valid[factor_name] = factor_df
        self.factors_train_dic = valid
        if not self.factors_train_dic:
            raise ValueError('No factors passed the validity screen.')

        factor_filter = self.config.get('factor_filter', {})
        if not factor_filter:
            self.selected_factors = sorted(self.factors_train_dic.keys())
            self.selected_factor_metrics = pd.DataFrame(index=self.selected_factors)
            print(f"Number of valid factors: {len(self.selected_factors)}")
            return

        metric_table = self._compute_factor_metric_table()
        surviving_factors = metric_table.index.tolist()
        if not surviving_factors:
            raise ValueError('No factors had enough valid observations for Rank IC / Rank IR.')

        for metric_name, metric_cfg in factor_filter.items():
            if metric_name not in metric_table.columns:
                raise ValueError(f'Unsupported factor filter metric: {metric_name}')

            threshold = metric_cfg.get('threshold')
            if threshold is not None:
                surviving_factors = [
                    factor_name
                    for factor_name in surviving_factors
                    if metric_table.at[factor_name, metric_name] >= threshold
                ]

            rank_threshold = metric_cfg.get('Rank_threshold')
            if rank_threshold is not None and surviving_factors:
                surviving_factors = (
                    metric_table.loc[surviving_factors]
                    .sort_values(metric_name, ascending=False)
                    .index[:int(rank_threshold)]
                    .tolist()
                )

            if not surviving_factors:
                break

        if not surviving_factors:
            raise ValueError('No factors survived the configured factor_filter rules.')

        self.selected_factors = surviving_factors
        self.selected_factor_metrics = metric_table.loc[self.selected_factors]
        self.factors_train_dic = {
            factor_name: self.factors_train_dic[factor_name]
            for factor_name in self.selected_factors
        }
        print(f"Number of valid factors: {len(self.selected_factors)}")

    # --------------------------------------------------------
    # 8. Label construction
    # --------------------------------------------------------
    def label_construction(self):
        """Build training labels from config."""
        label_function = self.config['label_function']
        label_params = dict(self.config.get('label_params', {}))
        delay_periods = self._get_signal_delay_periods()

        if callable(label_function):
            label_builder = label_function
        elif label_function in LABEL_REGISTRY:
            label_builder = LABEL_REGISTRY[label_function]
        else:
            raise ValueError(f'Unsupported label function: {label_function}')

        if delay_periods > 0 and 'delay' not in label_params:
            signature = inspect.signature(label_builder)
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if 'delay' in signature.parameters or accepts_kwargs:
                label_params['delay'] = delay_periods

        self.label, self.label_future_periods = label_builder(
            self.return_data,
            self.stock_list,
            self._current_train_dates,
            **label_params,
        )
        self.label = self.label.reindex(
            index=self._current_train_dates,
            columns=self.stock_list,
        )
        self.label = self._sanitize_numeric_panel(self.label)

    # --------------------------------------------------------
    # 9. Future-excluded training window
    # --------------------------------------------------------
    def _compute_future_excluded_window(self):
        if self.label_future_periods == 0:
            self.future_excluded_train_window = self._current_train_dates[:]
        else:
            n = self.label_future_periods
            if n >= len(self._current_train_dates):
                raise ValueError('The label look-ahead horizon is longer than the training window.')
            self.future_excluded_train_window = self._current_train_dates[:-n]

    # --------------------------------------------------------
    # 10. Valid training period selection
    #     (runs BEFORE filling so the check is on raw missingness)
    # --------------------------------------------------------
    def valid_train_period(self):
        """Keep dates where enough factors x stocks are non-missing."""
        valid_stock_frac = self.config['valid_stock_frac']
        valid_factor_frac = self.config['valid_factor_frac']
        n_stocks = len(self.stock_list)
        n_factors = len(self.selected_factors)
        dates = self.future_excluded_train_window
        if n_factors == 0:
            raise ValueError('Factor selection must run before valid_train_period.')

        # (n_dates x n_factors) bool matrix
        validity = np.zeros((len(dates), n_factors), dtype=bool)
        for j, fname in enumerate(self.selected_factors):
            df = self.factors_train_dic[fname].reindex(dates)
            non_miss_frac = df.notna().sum(axis=1).values / n_stocks
            validity[:, j] = non_miss_frac >= valid_stock_frac

        mask = validity.sum(axis=1) >= valid_factor_frac * n_factors
        self.adjust_training_window = [d for d, m in zip(dates, mask) if m]
        if not self.adjust_training_window:
            raise ValueError("No valid training dates remain. "
                             "Loosen valid_stock_frac / valid_factor_frac.")

    # --------------------------------------------------------
    # 11. Missing-value handling  (after valid_train_period)
    # --------------------------------------------------------
    def _handle_missing_values(self):
        fill_method = self.config.get('fill_method', 'fill_median_cs')
        dates = self.adjust_training_window

        self.fill_medians = {}
        for fname in self.selected_factors:
            df = self.factors_train_dic[fname].reindex(
                index=dates, columns=self.stock_list
            )
            factor_values = df.to_numpy(dtype=np.float64, copy=False)
            if np.isnan(factor_values).all():
                raise ValueError(f'Factor {fname} is entirely missing in the adjusted training window.')

            global_median = float(np.nanmedian(factor_values))
            if fill_method == 'fill_median_cs':
                filled = fill_median_cs(df).fillna(global_median)
                self.fill_medians[fname] = {'global': global_median, 'ts': None}
            elif fill_method == 'fill_median_ts':
                filled, col_med = fill_median_ts(df)
                filled = filled.fillna(global_median)
                self.fill_medians[fname] = {
                    'global': global_median,
                    'ts': col_med.fillna(global_median),
                }
            elif fill_method in (None, 'none'):
                filled = df
                self.fill_medians[fname] = {'global': global_median, 'ts': None}
            else:
                raise ValueError(f'Unsupported fill_method: {fill_method}')
            self.factors_train_dic[fname] = filled

        self.label = self.label.reindex(index=dates, columns=self.stock_list)

        print(f"Training window: {dates[0]} to {dates[-1]}")
        print(f"Total training periods: {len(dates)}")

    # --------------------------------------------------------
    # 12. Standardisation
    # --------------------------------------------------------
    def _standardize(self):
        mode = self.config.get('standardize_mode', 'pooled')
        if mode == 'pooled':
            self.factors_train_dic, self.train_mean, self.train_std = (
                fit_standardize(self.factors_train_dic, self.selected_factors)
            )
        elif mode == 'cs_rank':
            # Per-date cross-sectional rank to [-1, 1]: removes factor-level
            # drift across time so a pooled model sees purely cross-sectional
            # information. No fitted parameters — the test path applies the
            # same same-date transform.
            for fname in self.selected_factors:
                self.factors_train_dic[fname] = cs_rank_transform(
                    self.factors_train_dic[fname]
                )
            self.train_mean = {}
            self.train_std = {}
        else:
            raise ValueError(f'Unsupported standardize_mode: {mode}')

    # --------------------------------------------------------
    # 13. Training data preparation
    # --------------------------------------------------------
    def training_preparation(self):
        """Stack panels into (date*stock, factor) flat arrays."""
        dates = self.adjust_training_window
        stocks = self.stock_list
        factors = self.selected_factors
        n_d, n_s, n_f = len(dates), len(stocks), len(factors)

        X = np.empty((n_d * n_s, n_f))
        for i, fname in enumerate(factors):
            X[:, i] = self.factors_train_dic[fname].to_numpy(copy=False).reshape(-1)

        y_flat = self.label.to_numpy(copy=False).reshape(n_d * n_s)

        idx = pd.MultiIndex.from_product(
            [dates, stocks], names=['date', 'stock']
        )
        valid = np.isfinite(y_flat) & np.isfinite(X).all(axis=1)
        if not valid.any():
            raise ValueError('No valid training rows remain after alignment and NaN filtering.')

        self.training_index = idx[valid]
        self.feature_x = pd.DataFrame(X[valid], index=self.training_index, columns=factors)
        self.label_y = pd.Series(y_flat[valid], index=self.training_index, name='label')

    # --------------------------------------------------------
    # 14. Model training
    # --------------------------------------------------------
    def training_function(self):
        """Instantiate and fit the user-specified model."""
        model_class = self.config['model_class']
        model_params = self.config.get('model_params', {})
        self.model = model_class(**model_params)
        self.model.fit(self.feature_x.values, self.label_y.values)

    # --------------------------------------------------------
    # 15. Testing stage
    # --------------------------------------------------------
    def _load_screen_panels(self, dates):
        """Load the tradability_screen panels for the given dates.

        config['tradability_screen'] = {factor_name: minimum_value}. Each named
        panel is a factor file (e.g. the P1 'Price' = |prc| and 'Size' = log
        market cap panels), stamped month t and tradeable at t+1 like every
        other factor, so the screen uses only signal-date-observable data.
        """
        screen = self.config.get('tradability_screen', {})
        self._screen_panels = {}
        for fname in screen:
            if fname not in self.factor_file_map:
                raise ValueError(f'tradability_screen panel not found: {fname}')
            self._screen_panels[fname] = self._sanitize_numeric_panel(
                pd.read_pickle(self.factor_file_map[fname]).reindex(index=dates)
            )

    def select_test_stock(self, test_x, date=None):
        """Keep stocks with >= valid_feature_frac non-missing features and,
        if a tradability_screen is configured, at or above every screen floor
        on this date (a missing screen value is NOT provably tradeable -> out)."""
        frac = self.config.get('valid_feature_frac', 0.5)
        n_f = len(self.selected_factors)
        candidate_index = pd.Index(test_x.index)

        if self.config.get('test_stock_from_training_only', False):
            candidate_index = pd.Index(self.stock_list).intersection(candidate_index)

        candidate_x = test_x.reindex(candidate_index)
        keep = candidate_x.notna().sum(axis=1) >= frac * n_f

        screen = self.config.get('tradability_screen', {})
        if screen:
            if date is None:
                raise ValueError('tradability_screen requires the test date in select_test_stock().')
            for fname, floor in screen.items():
                values = self._screen_panels[fname].loc[date].reindex(candidate_index)
                keep &= values.ge(float(floor)).fillna(False)
        return candidate_x.index[keep].tolist()

    def _test_window(self, test_dates):
        """Predict for every date in the current test window."""
        factors = self.selected_factors
        fill_method = self.config.get('fill_method', 'fill_median_cs')
        do_std = self.config.get('standardize', False)

        # Load selected factors once for the whole test window
        test_factor_data = {}
        for fname in factors:
            df = pd.read_pickle(self.factor_file_map[fname])
            test_factor_data[fname] = self._sanitize_numeric_panel(
                df.reindex(index=test_dates)
            )
            del df

        predictions = {}
        n_stocks_list = []
        self._load_screen_panels(test_dates)

        for date in test_dates:
            # Build feature matrix (stocks x factors)
            test_x = pd.DataFrame(
                {fname: test_factor_data[fname].loc[date]
                 for fname in factors}
            )

            # Stock selection (feature completeness + tradability screen)
            valid_stocks = self.select_test_stock(test_x, date=date)
            if len(valid_stocks) == 0:
                continue
            test_x = test_x.loc[valid_stocks]

            # Fill missing
            test_x = fill_test_factors(
                test_x, fill_method, self.fill_medians
            )
            test_x = self._sanitize_numeric_panel(test_x).dropna(axis=0, how='any')
            if test_x.empty:
                continue
            n_stocks_list.append(len(test_x))

            # Standardise (pooled: training-fitted params; cs_rank: same-date
            # cross-sectional transform, no fitted params by construction)
            if do_std:
                if self.config.get('standardize_mode', 'pooled') == 'cs_rank':
                    test_x = cs_rank_transform_cross_section(test_x)
                else:
                    test_x = apply_standardize(
                        test_x, self.train_mean, self.train_std
                    )

            # Predict
            pred_vals = np.asarray(self.model.predict(test_x[factors].values)).reshape(-1)
            predictions[date] = pd.Series(pred_vals, index=test_x.index)

        del test_factor_data

        if test_dates:
            avg_stk = np.mean(n_stocks_list) if n_stocks_list else 0
            print(f"Test window: {test_dates[0]} to {test_dates[-1]}")
            print(f"Test periods: {len(test_dates)}  |  "
                  f"Avg stocks: {avg_stk:.0f}  |  Factors: {len(factors)}")

        if not predictions:
            return pd.DataFrame()
        return pd.DataFrame(predictions).T.sort_index()

    # --------------------------------------------------------
    # 17. Predicted value adjustment
    # --------------------------------------------------------
    def adjust_pred_value(self):
        """Apply moving average and/or neutralisation."""
        proc = self.config.get('process_dictionary', {})
        pred = self.predicted_labels.sort_index().copy()

        if not proc:
            self.adjusted_predicted_labels = pred
            return

        if 'moving_avg' in proc:
            pred = pred.rolling(
                window=proc['moving_avg'], min_periods=1
            ).mean()

        if 'neutralization' in proc:
            pred = self._neutralize(pred, proc['neutralization'])

        self.adjusted_predicted_labels = pred

    def _neutralize(self, pred, feature_list):
        """Cross-sectional OLS neutralisation per date."""
        neut_data = {}
        for fname in feature_list:
            if fname not in self.factor_file_map:
                raise ValueError(f'Neutralization factor not found: {fname}')
            neut_data[fname] = self._sanitize_numeric_panel(
                pd.read_pickle(self.factor_file_map[fname]).reindex(index=pred.index)
            )

        adjusted = pred.copy()
        for date in pred.index:
            y = pred.loc[date].dropna()
            if len(y) < len(feature_list) + 2:
                continue
            X = pd.DataFrame(
                {fn: neut_data[fn].loc[date].reindex(y.index)
                 for fn in feature_list}
            )
            valid = X.notna().all(axis=1)
            if valid.sum() < len(feature_list) + 2:
                continue
            X_v = np.column_stack([np.ones(valid.sum()),
                                   X[valid].values])
            y_v = y[valid].values
            beta = np.linalg.lstsq(X_v, y_v, rcond=None)[0]
            adjusted.loc[date, y[valid].index] = y_v - X_v @ beta
            # Stocks with a prediction but incomplete neutralization features
            # cannot be put on the residual scale — mixing their raw levels
            # into a cross-section of mean-zero residuals corrupts the ranking,
            # so they are excluded from this date instead.
            uncovered = y.index[~valid]
            if len(uncovered):
                adjusted.loc[date, uncovered] = np.nan

        del neut_data
        return adjusted

    # --------------------------------------------------------
    # 18. Portfolio formation
    # --------------------------------------------------------
    @staticmethod
    def _select_candidates(ranked_stocks, excluded_stocks, target_count, from_bottom=False):
        if target_count <= 0:
            return []

        chosen = []
        iterable = reversed(ranked_stocks) if from_bottom else ranked_stocks
        for stock in iterable:
            if stock in excluded_stocks:
                continue
            chosen.append(stock)
            if len(chosen) == target_count:
                break

        return chosen

    def form_portfolio(self):
        """Long-short equal-weight portfolio with gradual exchange."""
        pred = self.adjusted_predicted_labels.sort_index()
        n_pick = self.config['number_stock_pick']
        exch_frac = self.config['exchange_frac']
        n_exchange = (
            min(n_pick, int(np.ceil(exch_frac * n_pick)))
            if exch_frac > 0 else 0
        )

        dates = pred.index.tolist()
        all_stocks = pred.columns.tolist()
        weights = pd.DataFrame(0.0, index=dates, columns=all_stocks)

        long_holdings = []
        short_holdings = []

        for i, date in enumerate(dates):
            scores = pred.loc[date].dropna().sort_values(ascending=False)

            if len(scores) < 2 * n_pick:
                if i > 0:
                    weights.loc[date] = weights.iloc[i - 1]
                continue

            ranked_stocks = scores.index.tolist()
            score_universe = set(ranked_stocks)

            if i == 0:
                long_holdings = ranked_stocks[:n_pick]
                short_holdings = ranked_stocks[-n_pick:]
            elif n_exchange == 0:
                long_holdings = [stock for stock in long_holdings if stock in score_universe]
                short_holdings = [stock for stock in short_holdings if stock in score_universe]
            else:
                active_l = [stock for stock in long_holdings if stock in score_universe]
                missing_l = {stock for stock in long_holdings if stock not in score_universe}
                worst_l = sorted(active_l, key=lambda stock: scores.at[stock])[:n_exchange]
                long_drop = set(worst_l) | missing_l

                active_s = [stock for stock in short_holdings if stock in score_universe]
                missing_s = {stock for stock in short_holdings if stock not in score_universe}
                worst_s = sorted(
                    active_s,
                    key=lambda stock: scores.at[stock],
                    reverse=True,
                )[:n_exchange]
                short_drop = set(worst_s) | missing_s

                long_holdings = [
                    stock for stock in long_holdings
                    if stock in score_universe and stock not in long_drop
                ]
                short_holdings = [
                    stock for stock in short_holdings
                    if stock in score_universe and stock not in short_drop
                ]

            long_holdings.extend(
                self._select_candidates(
                    ranked_stocks,
                    set(long_holdings) | set(short_holdings),
                    n_pick - len(long_holdings),
                    from_bottom=False,
                )
            )
            short_holdings.extend(
                self._select_candidates(
                    ranked_stocks,
                    set(long_holdings) | set(short_holdings),
                    n_pick - len(short_holdings),
                    from_bottom=True,
                )
            )

            if long_holdings:
                weights.loc[date, long_holdings] = 1.0 / len(long_holdings)
            if short_holdings:
                weights.loc[date, short_holdings] = -1.0 / len(short_holdings)

        self.portfolio_weights = weights

    # --------------------------------------------------------
    # 19. Backtesting
    # --------------------------------------------------------
    def backtest(self):
        """Run numba kernel and compute performance statistics."""
        signal_weights = self.portfolio_weights
        if signal_weights is None or signal_weights.empty:
            raise ValueError('Portfolio weights are empty. Run form_portfolio() first.')
        weights = self._build_execution_weights()
        self.execution_portfolio_weights = weights
        if weights.empty:
            raise ValueError('No executable portfolio weights remain after applying signal_delay_periods.')

        # Expand to the CONTIGUOUS trading-date grid between the first and last
        # execution date. Periods without a rebalance (skipped windows, dropped
        # test dates) keep the book invested: the kernel drifts holdings and
        # books their P&L there, with zero turnover. Without this, gap-period
        # returns silently vanish from every headline statistic.
        rebalance_dates = set(weights.index)
        first_exec, last_exec = weights.index[0], weights.index[-1]
        grid = [d for d in self.trading_date_list if first_exec <= d <= last_exec]
        rebalance_mask = np.array([d in rebalance_dates for d in grid], dtype=np.bool_)
        weights = weights.reindex(index=grid).ffill()

        dates = weights.index
        stocks = weights.columns
        ppy = self.config.get('periods_per_year', 12)
        tc = self.config.get('transaction_cost_rate', 0.0)
        n_gap = int(len(grid) - rebalance_mask.sum())

        ret = self.return_data.reindex(index=dates, columns=stocks)
        leg_results = {
            'long_short': self._compute_leg_performance(
                weights, ret, dates, tc, ppy, rebalance_mask,
            ),
            'long_only': self._compute_leg_performance(
                self._extract_book_weights(weights, 'long'),
                ret,
                dates,
                tc,
                ppy,
                rebalance_mask,
            ),
            'short_only': self._compute_leg_performance(
                self._extract_book_weights(weights, 'short'),
                ret,
                dates,
                tc,
                ppy,
                rebalance_mask,
            ),
        }

        self.backtest_results = {'leg_results': leg_results}
        for leg_name, leg_result in leg_results.items():
            self._add_leg_results(self.backtest_results, leg_name, leg_result)
        self.backtest_results['unrebalanced_periods'] = n_gap

        perf = leg_results['long_short']

        print(f"\n{'=' * 40}")
        print("Backtest Results")
        print(f"{'=' * 40}")
        print(f"Long-Short Annualized Return: {perf['annualized_return']:.4f}")
        print(f"Long-Short Sharpe Ratio:      {perf['sharpe_ratio']:.4f}")
        print(f"Long-Short Max Drawdown:      {perf['max_drawdown']:.4f}")
        print(f"Long-Short Avg Turnover:      {perf['avg_turnover']:.4f}")
        if n_gap:
            print(f"Held-through periods without a rebalance: {n_gap}")
        skipped = getattr(self, 'skipped_window_count', 0)
        if skipped:
            print(f"WARNING: {skipped} rolling window(s) were skipped during training; "
                  f"their months are covered by holding the prior book.")

    # --------------------------------------------------------
    # 20. Visualisation
    # --------------------------------------------------------
    def visualize(self):
        """Plotly charts + yearly performance table."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as exc:
            raise ImportError(
                'plotly is required for visualize(). Install plotly before rendering charts.'
            ) from exc

        res = self.backtest_results
        leg_series = {
            'Long-Short': {
                'cumulative_returns': res['cumulative_returns'],
                'drawdown': res['drawdown'],
                'turnover': res['turnover'],
                'yearly_table': res['yearly_table'],
            },
            'Long Only': {
                'cumulative_returns': res['long_only_cumulative_returns'],
                'drawdown': res['long_only_drawdown'],
                'turnover': res['long_only_turnover'],
                'yearly_table': res['long_only_yearly_table'],
            },
            'Short Only': {
                'cumulative_returns': res['short_only_cumulative_returns'],
                'drawdown': res['short_only_drawdown'],
                'turnover': res['short_only_turnover'],
                'yearly_table': res['short_only_yearly_table'],
            },
        }
        dates = res['cumulative_returns'].index
        colors = {
            'Long-Short': '#5B6CFF',
            'Long Only': '#1F9D55',
            'Short Only': '#D64545',
        }

        # --- Charts ---
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=('Cumulative Return', 'Drawdown', 'Turnover'),
            vertical_spacing=0.08,
        )

        for leg_name, series_map in leg_series.items():
            fig.add_trace(go.Scatter(
                x=dates,
                y=series_map['cumulative_returns'].values,
                mode='lines',
                name=f'{leg_name} Cumulative',
                line=dict(color=colors[leg_name]),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=dates,
                y=series_map['drawdown'].values,
                mode='lines',
                name=f'{leg_name} Drawdown',
                line=dict(color=colors[leg_name]),
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=dates,
                y=series_map['turnover'].values,
                mode='lines',
                name=f'{leg_name} Turnover',
                line=dict(color=colors[leg_name]),
            ), row=3, col=1)

        fig.update_layout(
            height=900, width=1000,
            title_text='Portfolio Backtest Results',
            showlegend=True,
        )
        fig.show()

        # --- Yearly table ---
        table_frames = []
        for leg_name, series_map in leg_series.items():
            yearly = series_map['yearly_table'].copy()
            rename_map = {
                column: f'{leg_name} {column}'
                for column in yearly.columns
                if column != 'Year'
            }
            table_frames.append(yearly.rename(columns=rename_map))

        combined_table = table_frames[0]
        for yearly in table_frames[1:]:
            combined_table = combined_table.merge(yearly, on='Year', how='outer')
        combined_table = combined_table.sort_values('Year').reset_index(drop=True)

        print('\nYearly Performance:')
        print(combined_table.to_string(index=False))

        fig_t = go.Figure(data=[go.Table(
            header=dict(values=list(combined_table.columns)),
            cells=dict(values=[combined_table[c] for c in combined_table.columns]),
        )])
        fig_t.update_layout(title='Yearly Performance Table')
        fig_t.show()

    # --------------------------------------------------------
    # Orchestrator
    # --------------------------------------------------------
    def run(self):
        """Execute the full rolling-window backtest pipeline."""
        self.set_train_test_window()
        self._initialize_run_output()
        all_predictions = []
        self.skipped_window_count = 0

        for w in range(len(self.train_period_lists)):
            print(f"\n{'=' * 60}")
            print(f"Rolling window {w + 1}/{len(self.train_period_lists)}")
            print(f"{'=' * 60}")

            self._reset_window_state()
            self._current_train_dates = self.train_period_lists[w]
            test_dates = self.test_period_lists[w]

            # --- Training pipeline ---
            try:
                self.load_factors()
                self.available_stock()
                self._reindex_to_stocks()
                self.factor_selection()
                self.label_construction()
                self._compute_future_excluded_window()
                self.valid_train_period()        # on raw NaN, before fill
                self._handle_missing_values()    # fill + reindex
                if self.config.get('standardize', False):
                    self._standardize()
                self.training_preparation()
                self.training_function()
            except ValueError as e:
                print(f"Skipping window {w + 1}: {e}")
                self.skipped_window_count += 1
                self._save_skipped_window_artifacts(
                    w,
                    self._current_train_dates,
                    test_dates,
                    str(e),
                )
                self.factors_train_dic = {}
                continue

            # --- Testing pipeline ---
            window_pred = self._test_window(test_dates)
            self._save_window_artifacts(
                w,
                self._current_train_dates,
                test_dates,
                window_pred,
            )
            if not window_pred.empty:
                all_predictions.append(window_pred)

            # Free training-window memory
            self.factors_train_dic = {}

        # --- Post-processing ---
        if all_predictions:
            self.predicted_labels = (
                pd.concat(all_predictions)
                .sort_index()
                .loc[lambda df: ~df.index.duplicated(keep='last')]
            )
        else:
            self.predicted_labels = pd.DataFrame()
            self.adjusted_predicted_labels = pd.DataFrame()
            self.true_values = pd.DataFrame()
            self.prediction_records = pd.DataFrame(
                columns=[
                    'signal_date',
                    'execution_date',
                    'stock',
                    'original_prediction',
                    'adjusted_prediction',
                    'true_value',
                ]
            )
            self.portfolio_weights = pd.DataFrame()
            self.execution_portfolio_weights = pd.DataFrame()
            self.signal_holdings = pd.DataFrame(
                columns=['signal_date', 'stock', 'weight', 'side']
            )
            self.execution_holdings = pd.DataFrame(
                columns=['execution_date', 'stock', 'weight', 'side']
            )
            if self.run_output_dir is not None:
                self._write_json(
                    self.run_output_dir / 'run_summary.json',
                    {
                        'status': 'no_predictions',
                        'run_output_dir': str(self.run_output_dir),
                        'window_artifacts': self.window_artifacts,
                    },
                )
            print("WARNING: no predictions were generated.")
            return

        self.adjust_pred_value()
        self.form_portfolio()
        self.backtest()
        self._finalize_run_views()
        self._save_run_artifacts()
        if self.config.get('visualize', True):
            self.visualize()

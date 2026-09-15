"""
test_audit_fixes.py

Reproducing tests for the defects confirmed by the 2026-08-14 audit of
Project_backtest against Codex_test/project_description.ipynb. Written
BEFORE the fixes (CODING_RULES: bugs get a reproducing test first):

1. Gap months between execution rows must book the held portfolio's P&L,
   with zero turnover in gap months and multi-period drift at the next
   rebalance (audit findings quant_strategy_ml.py:1313 / utils_backtest.py:37).
2. Max drawdown must be measured against the 1.0 inception baseline
   (utils_backtest.py:95) and per calendar year in the yearly table
   (utils_backtest.py:133).
3. Cross-sectional neutralization must not mix mean-zero residuals with
   raw-level predictions on the same date: stocks lacking neutralization
   features get NaN (quant_strategy_ml.py:1190).
4. A single signal row with delay=1 still maps to a valid execution row
   (quant_strategy_ml.py:138 guard).
5. Timing regression guards: signal-aligned returns and label construction
   at signal_delay_periods=1 pair factor row t with return row t+1.
"""

from pathlib import Path
import shutil
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from quant_strategy_ml import Quant_Strategy_ML
from utils_backtest import compute_performance, yearly_table
from utils_preprocess import label_current_period_return


def _write_minimal_factor_dir(factor_dir, dates, stocks):
    factor_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for name in ('alpha', 'beta_f'):
        panel = pd.DataFrame(
            rng.normal(size=(len(dates), len(stocks))),
            index=dates,
            columns=stocks,
        )
        panel.to_pickle(factor_dir / f'{name}.pkl')


class GapBookingTests(unittest.TestCase):
    """Audit scenario: execution rows Feb, Mar, Jun; gap months Apr, May."""

    def setUp(self):
        self.workdir = Path('audit_fix_artifacts')
        self.workdir.mkdir(exist_ok=True)
        self.dates = pd.date_range('2020-01-01', periods=7, freq='MS')
        self.stocks = [1, 2]

        returns = pd.DataFrame(
            [
                [0.00, 0.00],   # Jan
                [0.02, 0.00],   # Feb
                [0.03, -0.05],  # Mar
                [0.30, -0.10],  # Apr  (gap)
                [0.20, 0.00],   # May  (gap)
                [0.01, 0.02],   # Jun
                [0.00, 0.00],   # Jul
            ],
            index=self.dates,
            columns=self.stocks,
        )
        self.return_path = self.workdir / 'ret.pkl'
        returns.to_pickle(self.return_path)

        self.factor_dir = self.workdir / 'factors'
        _write_minimal_factor_dir(self.factor_dir, self.dates, self.stocks)

        config = {
            'Factor_Path': self.factor_dir,
            'Return_Path': self.return_path,
            'signal_delay_periods': 1,
            'transaction_cost_rate': 0.0,
            'periods_per_year': 12,
            'save_artifacts': False,
        }
        self.strategy = Quant_Strategy_ML(config)
        self.strategy.return_data = returns
        self.strategy.trading_date_list = list(self.dates)

        # Signal rows Jan, Feb, May -> execution rows Feb, Mar, Jun.
        self.strategy.portfolio_weights = pd.DataFrame(
            [
                [0.6, 0.4],
                [0.6, 0.4],
                [0.5, 0.5],
            ],
            index=[self.dates[0], self.dates[1], self.dates[4]],
            columns=self.stocks,
        )

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_gap_months_book_held_pnl(self):
        self.strategy.backtest()
        res = self.strategy.backtest_results

        port = res['portfolio_returns']
        turn = res['turnover']

        expected_dates = [self.dates[1], self.dates[2], self.dates[3],
                          self.dates[4], self.dates[5]]
        self.assertEqual(list(port.index), expected_dates,
                         'gap months Apr/May must appear in the booked return series')

        # Independent full-precision reference: hold-and-drift accounting.
        # Feb: enter [.6,.4]; Mar: rebalance back to [.6,.4]; Apr/May: hold;
        # Jun: rebalance to [.5,.5]. Between rebalances weights drift with
        # each stock's return and the book books drifted P&L.
        rets = {
            'feb': np.array([0.02, 0.00]), 'mar': np.array([0.03, -0.05]),
            'apr': np.array([0.30, -0.10]), 'may': np.array([0.20, 0.00]),
            'jun': np.array([0.01, 0.02]),
        }
        exp_port, exp_turn = [], []
        cur = np.array([0.6, 0.4])
        exp_turn.append(np.abs(cur).sum() / 2)                   # entry
        exp_port.append(cur @ rets['feb'])
        for month, target in [('mar', np.array([0.6, 0.4])),
                              ('apr', None), ('may', None),
                              ('jun', np.array([0.5, 0.5]))]:
            prev = {'mar': 'feb', 'apr': 'mar', 'may': 'apr', 'jun': 'may'}[month]
            cur = cur * (1 + rets[prev]) / (1 + exp_port[-1])    # drift
            if target is None:
                exp_turn.append(0.0)
            else:
                exp_turn.append(np.abs(target - cur).sum() / 2)
                cur = target
            exp_port.append(cur @ rets[month])

        np.testing.assert_allclose(port.to_numpy(), exp_port, rtol=0, atol=1e-12)
        np.testing.assert_allclose(turn.to_numpy(), exp_turn, rtol=0, atol=1e-12)

    def test_annualization_uses_calendar_periods(self):
        self.strategy.backtest()
        res = self.strategy.backtest_results
        # Five calendar months Feb..Jun -> exponent 12/5, not 12/3.
        cum = float(np.prod(1.0 + res['portfolio_returns'].to_numpy()))
        expected = cum ** (12 / 5) - 1.0
        self.assertAlmostEqual(res['annualized_return'], expected, places=10)


class DrawdownBaselineTests(unittest.TestCase):
    def test_max_drawdown_from_inception(self):
        perf = compute_performance(
            np.array([-0.10, 0.05]),
            np.zeros(2),
            tc_rate=0.0,
            periods_per_year=12,
        )
        self.assertAlmostEqual(perf['max_drawdown'], -0.10, places=12)
        np.testing.assert_allclose(perf['drawdown'], [-0.10, -0.055], atol=1e-12)

    def test_yearly_table_drawdown_from_year_start(self):
        dates = pd.date_range('2021-01-01', periods=3, freq='MS')
        adj = pd.Series([-0.20, 0.10, 0.05], index=dates)
        turn = pd.Series(0.0, index=dates)
        table = yearly_table(adj, turn, periods_per_year=12)
        self.assertAlmostEqual(float(table.loc[0, 'Max Drawdown']), -0.20, places=12)


class NeutralizationScaleTests(unittest.TestCase):
    def setUp(self):
        self.workdir = Path('audit_fix_neut')
        self.workdir.mkdir(exist_ok=True)
        self.dates = pd.date_range('2020-01-01', periods=2, freq='MS')
        self.stocks = [1, 2, 3, 4, 5, 6]

        self.factor_dir = self.workdir / 'factors'
        self.factor_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(11)
        style = pd.DataFrame(
            rng.normal(size=(2, 6)), index=self.dates, columns=self.stocks
        )
        # stock 6 has NO style value on the first date
        style.iloc[0, 5] = np.nan
        style.to_pickle(self.factor_dir / 'style.pkl')

        ret = pd.DataFrame(0.0, index=self.dates, columns=self.stocks)
        self.return_path = self.workdir / 'ret.pkl'
        ret.to_pickle(self.return_path)

        config = {
            'Factor_Path': self.factor_dir,
            'Return_Path': self.return_path,
            'process_dictionary': {'neutralization': ['style']},
            'save_artifacts': False,
        }
        self.strategy = Quant_Strategy_ML(config)
        self.strategy.predicted_labels = pd.DataFrame(
            [[10.0, 20.0, 30.0, 40.0, 50.0, 999.0],
             [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]],
            index=self.dates,
            columns=self.stocks,
        )

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_uncovered_stock_gets_nan_not_raw_level(self):
        self.strategy.adjust_pred_value()
        adjusted = self.strategy.adjusted_predicted_labels

        # Covered stocks on date 0 are residuals (cross-sectional mean ~ 0).
        covered = adjusted.iloc[0, :5]
        self.assertAlmostEqual(float(covered.mean()), 0.0, places=8)

        # Stock 6 has no style exposure on date 0: leaving its raw level
        # (999.0) would dominate every residual. It must be NaN.
        self.assertTrue(np.isnan(adjusted.iloc[0, 5]))

        # Date 1 has full coverage: everything is a residual.
        self.assertAlmostEqual(float(adjusted.iloc[1].mean()), 0.0, places=8)


class ExecutionGuardTests(unittest.TestCase):
    def setUp(self):
        self.workdir = Path('audit_fix_guard')
        self.workdir.mkdir(exist_ok=True)
        self.dates = pd.date_range('2020-01-01', periods=3, freq='MS')
        self.stocks = [1, 2]

        ret = pd.DataFrame(0.01, index=self.dates, columns=self.stocks)
        self.return_path = self.workdir / 'ret.pkl'
        ret.to_pickle(self.return_path)
        self.factor_dir = self.workdir / 'factors'
        _write_minimal_factor_dir(self.factor_dir, self.dates, self.stocks)

        config = {
            'Factor_Path': self.factor_dir,
            'Return_Path': self.return_path,
            'signal_delay_periods': 1,
            'save_artifacts': False,
        }
        self.strategy = Quant_Strategy_ML(config)
        self.strategy.return_data = ret
        self.strategy.trading_date_list = list(self.dates)
        self.strategy.portfolio_weights = pd.DataFrame(
            [[1.0, -1.0]], index=[self.dates[0]], columns=self.stocks
        )

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_single_signal_row_still_executes(self):
        execution = self.strategy._build_execution_weights()
        self.assertEqual(len(execution), 1)
        self.assertEqual(execution.index[0], self.dates[1])


class TimingRegressionGuards(unittest.TestCase):
    """These pass on the current code; they pin the delay=1 convention."""

    def test_signal_aligned_returns_shift(self):
        dates = pd.date_range('2020-01-01', periods=4, freq='MS')
        ret = pd.DataFrame(
            {1: [0.01, 0.02, 0.03, 0.04]},
            index=dates,
        )
        workdir = Path('audit_fix_timing')
        workdir.mkdir(exist_ok=True)
        try:
            ret_path = workdir / 'ret.pkl'
            ret.to_pickle(ret_path)
            factor_dir = workdir / 'factors'
            _write_minimal_factor_dir(factor_dir, dates, [1])

            strategy = Quant_Strategy_ML({
                'Factor_Path': factor_dir,
                'Return_Path': ret_path,
                'signal_delay_periods': 1,
                'save_artifacts': False,
            })
            strategy.return_data = ret
            strategy.trading_date_list = list(dates)
            strategy.stock_list = [1]

            aligned = strategy._get_signal_aligned_return_panel(list(dates[:3]))
            np.testing.assert_allclose(
                aligned[1].to_numpy(), [0.02, 0.03, 0.04], atol=1e-15
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_label_delay_one_and_future_exclusion(self):
        dates = pd.date_range('2020-01-01', periods=4, freq='MS')
        ret = pd.DataFrame({1: [0.01, 0.02, 0.03, 0.04]}, index=dates)
        label, future_periods = label_current_period_return(
            ret, [1], list(dates), delay=1
        )
        self.assertEqual(future_periods, 1)
        np.testing.assert_allclose(
            label[1].to_numpy()[:3], [0.02, 0.03, 0.04], atol=1e-15
        )
        self.assertTrue(np.isnan(label[1].to_numpy()[3]))


if __name__ == '__main__':
    unittest.main()


class CrossSectionalRankTests(unittest.TestCase):
    """standardize_mode='cs_rank': per-date rank mapping to [-1, 1].

    The transform uses only same-date cross-sectional information, so it is
    inherently leak-free and needs no fitted parameters at test time.
    """

    def test_panel_transform_properties(self):
        from utils_preprocess import cs_rank_transform

        dates = pd.date_range('2020-01-01', periods=2, freq='MS')
        panel = pd.DataFrame(
            [[10.0, 20.0, 30.0, np.nan],
             [5.0, 1.0, 3.0, 2.0]],
            index=dates, columns=[1, 2, 3, 4],
        )
        out = cs_rank_transform(panel)

        # date 0: ranks 1,2,3 of 3 -> 2*r/(n+1)-1 = [-0.5, 0.0, 0.5]; NaN kept
        np.testing.assert_allclose(out.iloc[0, :3], [-0.5, 0.0, 0.5], atol=1e-12)
        self.assertTrue(np.isnan(out.iloc[0, 3]))
        # date 1: ranks of [5,1,3,2] = [4,1,3,2] of 4 -> [0.6,-0.6,0.2,-0.2]
        np.testing.assert_allclose(out.iloc[1], [0.6, -0.6, 0.2, -0.2], atol=1e-12)

    def test_monotone_invariance(self):
        from utils_preprocess import cs_rank_transform

        rng = np.random.default_rng(5)
        panel = pd.DataFrame(rng.normal(size=(3, 50)),
                             index=pd.date_range('2020-01-01', periods=3, freq='MS'),
                             columns=range(50))
        a = cs_rank_transform(panel)
        b = cs_rank_transform(np.exp(panel * 3.0))
        pd.testing.assert_frame_equal(a, b)

    def test_test_time_matches_panel_transform(self):
        from utils_preprocess import cs_rank_transform, cs_rank_transform_cross_section

        rng = np.random.default_rng(9)
        dates = pd.date_range('2020-01-01', periods=2, freq='MS')
        stocks = list(range(30))
        panels = {
            'f1': pd.DataFrame(rng.normal(size=(2, 30)), index=dates, columns=stocks),
            'f2': pd.DataFrame(rng.normal(size=(2, 30)), index=dates, columns=stocks),
        }
        # test_x as the pipeline builds it: stocks x factors on one date
        test_x = pd.DataFrame({name: panels[name].loc[dates[1]] for name in panels})
        out = cs_rank_transform_cross_section(test_x)
        for name in panels:
            np.testing.assert_allclose(
                out[name].to_numpy(),
                cs_rank_transform(panels[name]).loc[dates[1]].to_numpy(),
                atol=1e-12,
            )


class TradabilityScreenTests(unittest.TestCase):
    """tradability_screen: drop stocks below price / market-cap floors at test time.

    The screen reads factor panels named in the config (e.g. the P1 'Price' and
    'Size' factors, stamped month t and tradeable at t+1 like every other factor),
    so it uses only same-date-observable information.
    """

    def setUp(self):
        self.workdir = Path('audit_fix_screen')
        self.workdir.mkdir(exist_ok=True)
        self.dates = pd.date_range('2020-01-01', periods=3, freq='MS')
        self.stocks = [1, 2, 3, 4]
        fdir = self.workdir / 'factors'; fdir.mkdir(exist_ok=True)
        pd.DataFrame(1.0, index=self.dates, columns=self.stocks).to_pickle(fdir / 'alpha.pkl')
        # screen panels: stock 3 cheap, stock 4 tiny
        pd.DataFrame([[10, 20, 2.0, 30]] * 3, index=self.dates, columns=self.stocks).to_pickle(fdir / 'Price.pkl')
        pd.DataFrame([[9.0, 9.0, 9.0, 3.0]] * 3, index=self.dates, columns=self.stocks).to_pickle(fdir / 'Size.pkl')
        ret = pd.DataFrame(0.0, index=self.dates, columns=self.stocks)
        self.return_path = self.workdir / 'ret.pkl'; ret.to_pickle(self.return_path)
        self.strategy = Quant_Strategy_ML({
            'Factor_Path': fdir, 'Return_Path': self.return_path, 'save_artifacts': False,
            'valid_feature_frac': 0.5,
            'tradability_screen': {'Price': 5.0, 'Size': 4.0},   # min |price| 5, min log-size 4
        })
        self.strategy.selected_factors = ['alpha']
        self.strategy.stock_list = self.stocks
        self.strategy._load_screen_panels(self.dates)

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_screen_drops_cheap_and_tiny(self):
        test_x = pd.DataFrame({'alpha': [1.0, 1.0, 1.0, 1.0]}, index=self.stocks)
        kept = self.strategy.select_test_stock(test_x, date=self.dates[1])
        self.assertEqual(kept, [1, 2])

    def test_missing_screen_value_excludes(self):
        # a stock with NaN in a screen panel is not provably tradeable -> excluded
        self.strategy._screen_panels['Price'].loc[self.dates[1], 1] = np.nan
        test_x = pd.DataFrame({'alpha': [1.0] * 4}, index=self.stocks)
        kept = self.strategy.select_test_stock(test_x, date=self.dates[1])
        self.assertEqual(kept, [2])

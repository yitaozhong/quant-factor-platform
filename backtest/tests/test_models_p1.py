"""
test_models_p1.py
Interface and sanity tests for the three backtest models.

Each model must: learn an obvious linear signal (rank correlation with truth),
be deterministic under its seed, and fail loudly on misuse.
"""

from pathlib import Path
import importlib.util
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models import LightGBMRegressor, RidgeRegressor, TorchMLPRegressor

HAS_LIGHTGBM = importlib.util.find_spec('lightgbm') is not None
HAS_TORCH = importlib.util.find_spec('torch') is not None


def make_linear_dataset(n_rows=4000, n_features=8, noise=0.05, seed=3):
    rng = np.random.default_rng(seed)
    x_value = rng.normal(size=(n_rows, n_features))
    beta = np.linspace(1.0, -1.0, n_features)
    y_value = x_value @ beta + noise * rng.normal(size=n_rows)
    return x_value, y_value


def rank_corr(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


class ModelContractTests(unittest.TestCase):
    def _check_learns_signal(self, model):
        x_value, y_value = make_linear_dataset()
        split = 3000
        model.fit(x_value[:split], y_value[:split])
        predictions = np.asarray(model.predict(x_value[split:])).reshape(-1)
        self.assertEqual(len(predictions), len(y_value[split:]))
        self.assertGreater(rank_corr(predictions, y_value[split:]), 0.8)

    def test_ridge_learns_signal(self):
        self._check_learns_signal(RidgeRegressor(alpha=1.0))

    @unittest.skipUnless(HAS_LIGHTGBM, 'lightgbm not installed')
    def test_lightgbm_learns_signal(self):
        self._check_learns_signal(LightGBMRegressor(n_estimators=150))

    @unittest.skipUnless(HAS_TORCH, 'torch not installed')
    def test_mlp_learns_signal(self):
        self._check_learns_signal(
            TorchMLPRegressor(hidden_layers=(32, 16), max_epochs=60, patience=8)
        )

    @unittest.skipUnless(HAS_LIGHTGBM, 'lightgbm not installed')
    def test_lightgbm_deterministic(self):
        x_value, y_value = make_linear_dataset()
        pred_a = LightGBMRegressor(n_estimators=60).fit(x_value, y_value).predict(x_value)
        pred_b = LightGBMRegressor(n_estimators=60).fit(x_value, y_value).predict(x_value)
        np.testing.assert_array_equal(pred_a, pred_b)

    @unittest.skipUnless(HAS_TORCH, 'torch not installed')
    def test_mlp_deterministic(self):
        x_value, y_value = make_linear_dataset(n_rows=1500)
        model_kwargs = dict(hidden_layers=(16,), max_epochs=15, patience=5)
        pred_a = TorchMLPRegressor(**model_kwargs).fit(x_value, y_value).predict(x_value)
        pred_b = TorchMLPRegressor(**model_kwargs).fit(x_value, y_value).predict(x_value)
        np.testing.assert_array_equal(pred_a, pred_b)

    def test_predict_before_fit_raises(self):
        x_value = np.zeros((3, 2))
        for model in (RidgeRegressor(), LightGBMRegressor(), TorchMLPRegressor()):
            with self.assertRaises(ValueError):
                model.predict(x_value)

    def test_empty_fit_raises(self):
        empty_x = np.zeros((0, 2))
        empty_y = np.zeros(0)
        models = [RidgeRegressor()]
        if HAS_LIGHTGBM:
            models.append(LightGBMRegressor())
        if HAS_TORCH:
            models.append(TorchMLPRegressor())
        for model in models:
            with self.assertRaises(ValueError):
                model.fit(empty_x, empty_y)


if __name__ == '__main__':
    unittest.main()

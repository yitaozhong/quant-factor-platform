"""
models.py
Models for the backtesting pipeline. All models expose the same interface:

    model = ModelClass(**model_params)
    model.fit(x_2d_float, y_1d_float)
    predictions = model.predict(x_2d_float)

RidgeRegressor is self-contained numpy. LightGBMRegressor and TorchMLPRegressor
import their heavy dependency lazily inside fit(), so importing this module
never requires lightgbm or torch.
"""

import numpy as np


class RidgeRegressor:
    """
    Closed-form ridge regression with a scikit-learn-like interface.

    This is a practical default for large monthly cross-sectional panels:
    training cost depends mainly on the feature dimension, not the number of
    rows, once the normal equations are formed.
    """

    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, x_value, y_value):
        x_value = np.asarray(x_value, dtype=np.float64)
        y_value = np.asarray(y_value, dtype=np.float64).reshape(-1)
        if x_value.ndim != 2:
            raise ValueError('x_value must be a 2D array.')
        if len(x_value) != len(y_value):
            raise ValueError('x_value and y_value must have the same number of rows.')
        if len(x_value) == 0:
            raise ValueError('Cannot fit RidgeRegressor on an empty dataset.')

        if self.fit_intercept:
            x_mean = x_value.mean(axis=0)
            y_mean = float(y_value.mean())
            x_centered = x_value - x_mean
            y_centered = y_value - y_mean
        else:
            x_mean = np.zeros(x_value.shape[1], dtype=np.float64)
            y_mean = 0.0
            x_centered = x_value
            y_centered = y_value

        gram_matrix = x_centered.T @ x_centered
        ridge_matrix = gram_matrix + self.alpha * np.eye(x_value.shape[1], dtype=np.float64)
        rhs = x_centered.T @ y_centered

        self.coef_ = np.linalg.solve(ridge_matrix, rhs)
        self.intercept_ = y_mean - float(x_mean @ self.coef_)
        return self

    def predict(self, x_value):
        if self.coef_ is None:
            raise ValueError('RidgeRegressor must be fit before predict().')

        x_value = np.asarray(x_value, dtype=np.float64)
        if x_value.ndim != 2:
            raise ValueError('x_value must be a 2D array.')

        return x_value @ self.coef_ + self.intercept_


class LightGBMRegressor:
    """
    Gradient-boosted tree regressor (LightGBM) with the pipeline interface.

    Defaults are conservative for monthly cross-sectional panels (~hundreds of
    thousands of rows, tens of features): moderate depth via num_leaves, strong
    minimum leaf size against noise-fitting, row/column subsampling, and a
    fixed seed so runs are reproducible. The tree count is fixed (no early
    stopping) because fit() receives one flat training matrix; validation-based
    stopping would need a date-aware split, which the pipeline does not pass in.
    """

    def __init__(self, **params):
        defaults = {
            'n_estimators': 400,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 200,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1,
        }
        defaults.update(params)
        self.params = defaults
        self.model = None

    def fit(self, x_value, y_value):
        from lightgbm import LGBMRegressor

        x_value = np.asarray(x_value, dtype=np.float64)
        y_value = np.asarray(y_value, dtype=np.float64).reshape(-1)
        if x_value.ndim != 2:
            raise ValueError('x_value must be a 2D array.')
        if len(x_value) != len(y_value):
            raise ValueError('x_value and y_value must have the same number of rows.')
        if len(x_value) == 0:
            raise ValueError('Cannot fit LightGBMRegressor on an empty dataset.')

        self.model = LGBMRegressor(**self.params)
        self.model.fit(x_value, y_value)
        return self

    def predict(self, x_value):
        if self.model is None:
            raise ValueError('LightGBMRegressor must be fit before predict().')
        x_value = np.asarray(x_value, dtype=np.float64)
        if x_value.ndim != 2:
            raise ValueError('x_value must be a 2D array.')
        return self.model.predict(x_value)


class TorchMLPRegressor:
    """
    Multi-layer perceptron (PyTorch) with the pipeline interface.

    Architecture: input -> hidden_layers with ReLU + dropout -> 1 output.
    Training: Adam + MSE, mini-batches, early stopping on a time-tail
    validation split (the LAST val_fraction of rows — the training matrix is
    ordered by date, so the validation set is the most recent slice of the
    training window; no information from outside the training window is used).
    All randomness is seeded; the best-validation weights are restored after
    stopping. Inputs are assumed standardized by the pipeline (standardize=True).
    """

    def __init__(self, hidden_layers=(64, 32), dropout=0.1, learning_rate=1e-3,
                 batch_size=8192, max_epochs=200, patience=15, val_fraction=0.1,
                 weight_decay=1e-5, random_state=42):
        self.hidden_layers = tuple(int(h) for h in hidden_layers)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self.model = None
        self._torch = None

    def _build_network(self, torch, n_features):
        layers = []
        width_in = n_features
        for width_out in self.hidden_layers:
            layers.append(torch.nn.Linear(width_in, width_out))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(self.dropout))
            width_in = width_out
        layers.append(torch.nn.Linear(width_in, 1))
        return torch.nn.Sequential(*layers)

    def fit(self, x_value, y_value):
        import torch

        self._torch = torch
        x_value = np.asarray(x_value, dtype=np.float32)
        y_value = np.asarray(y_value, dtype=np.float32).reshape(-1)
        if x_value.ndim != 2:
            raise ValueError('x_value must be a 2D array.')
        if len(x_value) != len(y_value):
            raise ValueError('x_value and y_value must have the same number of rows.')
        if len(x_value) == 0:
            raise ValueError('Cannot fit TorchMLPRegressor on an empty dataset.')

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        n_rows = len(x_value)
        n_val = max(1, int(round(self.val_fraction * n_rows)))
        if n_val >= n_rows:
            raise ValueError('val_fraction leaves no training rows.')
        x_train, y_train = x_value[:-n_val], y_value[:-n_val]
        x_val = torch.from_numpy(x_value[-n_val:])
        y_val = torch.from_numpy(y_value[-n_val:])

        self.model = self._build_network(torch, x_value.shape[1])
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = torch.nn.MSELoss()

        best_val = np.inf
        best_state = None
        bad_epochs = 0

        for _epoch in range(self.max_epochs):
            self.model.train()
            order = rng.permutation(len(x_train))
            for start in range(0, len(order), self.batch_size):
                batch_idx = order[start:start + self.batch_size]
                xb = torch.from_numpy(x_train[batch_idx])
                yb = torch.from_numpy(y_train[batch_idx])
                optimizer.zero_grad()
                loss = loss_fn(self.model(xb).squeeze(-1), yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model(x_val).squeeze(-1), y_val))

            if val_loss < best_val - 1e-9:
                best_val = val_loss
                best_state = {
                    key: tensor.detach().clone()
                    for key, tensor in self.model.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict(self, x_value):
        if self.model is None:
            raise ValueError('TorchMLPRegressor must be fit before predict().')
        torch = self._torch
        x_value = np.asarray(x_value, dtype=np.float32)
        if x_value.ndim != 2:
            raise ValueError('x_value must be a 2D array.')
        with torch.no_grad():
            out = self.model(torch.from_numpy(x_value)).squeeze(-1)
        return out.numpy().astype(np.float64)

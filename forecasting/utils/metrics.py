import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


def _prepare_targets(y_true, y_pred, drop_zero=False):
    """Convert inputs to float arrays and drop rows with NaNs or zero targets when requested."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if drop_zero:
        mask &= y_true != 0
    if not mask.any():
        return np.array([], dtype=float), np.array([], dtype=float)
    return y_true[mask], y_pred[mask]


def rmse(y_true, y_pred):
    y_true_clean, y_pred_clean = _prepare_targets(y_true, y_pred)
    if y_true_clean.size == 0:
        return float("nan")
    return math.sqrt(mean_squared_error(y_true_clean, y_pred_clean))


def mae(y_true, y_pred):
    y_true_clean, y_pred_clean = _prepare_targets(y_true, y_pred)
    if y_true_clean.size == 0:
        return float("nan")
    return mean_absolute_error(y_true_clean, y_pred_clean)


def mape(y_true, y_pred):
    y_true_clean, y_pred_clean = _prepare_targets(y_true, y_pred, drop_zero=True)
    if y_true_clean.size == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100)

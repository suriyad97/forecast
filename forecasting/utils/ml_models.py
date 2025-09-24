import numpy as np
import pandas as pd
from typing import Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None


try:
    from prophet import Prophet
except ImportError:  # pragma: no cover - optional dependency
    Prophet = None


MODEL_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "linearregression": {},
    "lasso": {
        "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
    },
    "elasticnet": {
        "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.2, 0.5, 0.8],
    },
    "prophet": {},
    "randomforest": {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [None, 6, 10, 16],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "xgboost": {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.03, 0.05, 0.08],
        "max_depth": [3, 4, 6],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.5, 1.0, 1.5],
    },
    "lightgbm": {
        "n_estimators": [300, 500, 700],
        "learning_rate": [0.03, 0.05, 0.08],
        "num_leaves": [31, 63, 127],
        "min_child_samples": [10, 20, 40],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
    },
}


def _normalize_model_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())

class ProphetRegressor:
    def __init__(self, **kwargs):
        if Prophet is None:
            raise ImportError("Prophet is not installed. Run pip install prophet to enable it.")
        self._init_kwargs = kwargs or {}
        self.model: Any | None = None
        self.regressors: list[str] = []
        self.fitted_ = False

    def fit(self, X, y):
        if Prophet is None:
            raise ImportError("Prophet is not installed. Run pip install prophet to enable it.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index)
        self.regressors = list(X.columns)
        self.model = Prophet(**self._init_kwargs)
        for reg in self.regressors:
            self.model.add_regressor(reg)
        train_df = X.copy()
        train_df["y"] = y.to_numpy()
        train_df = train_df.reset_index()
        index_name = X.index.name or "index"
        train_df = train_df.rename(columns={index_name: "ds"})
        train_df = train_df.sort_values("ds")
        fit_df = train_df[["ds"] + self.regressors + ["y"]]
        self.model.fit(fit_df)
        self.fitted_ = True
        return self

    def predict(self, X):
        if not self.fitted_ or self.model is None:
            raise ValueError("ProphetRegressor must be fitted before calling predict().")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        predict_df = X.copy()
        predict_df = predict_df.reset_index()
        index_name = X.index.name or "index"
        predict_df = predict_df.rename(columns={index_name: "ds"})
        predict_df = predict_df.sort_values("ds")
        future = predict_df[["ds"]].copy()
        for reg in self.regressors:
            if reg in predict_df.columns:
                future[reg] = predict_df[reg].to_numpy()
            else:
                future[reg] = 0.0
        forecast = self.model.predict(future)
        return forecast["yhat"].to_numpy()

    def get_params(self, deep: bool = False):
        return {**self._init_kwargs}

    def set_params(self, **params):
        self._init_kwargs.update(params)
        return self

    def __repr__(self):
        return f"ProphetRegressor(kwargs={self._init_kwargs})"

def make_lag_matrix(series: pd.Series, lags=(1, 2, 3, 6, 12), extra_df: pd.DataFrame | None = None):
    df = pd.DataFrame({"y": series})
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    if extra_df is not None:
        df = df.join(extra_df, how="left")
    df = df.dropna()
    X = df.drop(columns=["y"])
    y = df["y"]
    return X, y, df


def iterative_forecast(
    model,
    last_known_y: pd.Series,
    steps: int,
    lags=(1, 2, 3, 6, 12),
    future_exog: pd.DataFrame | None = None,
):
    preds: list[float] = []
    idx = []
    for _ in range(steps):
        next_idx = last_known_y.index[-1] + pd.offsets.MonthBegin(1)
        row = {}
        for lag in lags:
            if len(last_known_y) >= lag:
                row[f"lag_{lag}"] = last_known_y.iloc[-lag]
            else:
                row[f"lag_{lag}"] = last_known_y.iloc[-1]
        if future_exog is not None and next_idx in future_exog.index:
            for col in future_exog.columns:
                row[col] = future_exog.loc[next_idx, col]
        X_new = pd.DataFrame([row], index=[next_idx])
        yhat = float(model.predict(X_new)[0])
        preds.append(yhat)
        idx.append(next_idx)
        last_known_y = pd.concat([last_known_y, pd.Series([yhat], index=[next_idx])])
    return pd.Series(preds, index=idx, name="forecast")


def get_ml_model(name: str):
    key = _normalize_model_name(name)
    if key in {"linearregression", "linear"}:
        return LinearRegression()
    if key == "lasso":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=1.0, max_iter=5000, random_state=42)),
        ])
    if key in {"elasticnet", "elastic"}:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000, random_state=42)),
        ])
    if key == "prophet":
        return ProphetRegressor()
    if key == "xgboost":
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
            objective="reg:squarederror",
        )
    if key == "lightgbm":
        return LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="rmse",
            verbosity=-1,
        )
    if key == "randomforest":
        return RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")

def _compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = np.abs((y_true - y_pred) / denom)
    return float(np.nanmean(mape) * 100.0)


def _choose_cv_splits(n_samples: int, requested: int) -> int:
    if n_samples < 8:
        return 0
    max_possible = max(2, n_samples // 4)
    return int(min(requested, max_possible))


def _run_random_search(model, params: dict[str, list], X_train: pd.DataFrame, y_train: pd.Series, n_iter: int, cv_splits: int):
    if not params or cv_splits < 2:
        fitted = model.fit(X_train, y_train)
        return fitted, None
    total_combinations = 1
    for values in params.values():
        total_combinations *= len(values)
    effective_iter = min(n_iter, total_combinations)
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=effective_iter,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        n_jobs=-1,
        random_state=42,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search


def _build_feature_importance(model, X_sample: pd.DataFrame) -> pd.DataFrame | None:
    if shap is None or X_sample.empty:
        return None
    try:
        explainer = shap.TreeExplainer(model)
        shap_result = explainer(X_sample)
        values = getattr(shap_result, "values", shap_result)
        if isinstance(values, list):  # xgboost returns list for multi-output
            values = values[0]
        mean_abs = np.mean(np.abs(values), axis=0)
        importance = (
            pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        importance["relative_importance_pct"] = (
            importance["mean_abs_shap"] / importance["mean_abs_shap"].sum() * 100.0
        )
        return importance
    except Exception:
        return None


def train_ml_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    model_names: list[str] | None = None,
    cv_splits: int = 3,
    n_iter: int = 15,
) -> dict[str, dict]:
    if model_names is None:
        model_names = ["RandomForest", "XGBoost", "LightGBM"]

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, index=X_train.index)

    if X_test is None:
        X_test = pd.DataFrame()
    elif not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    if y_test is None:
        y_test = pd.Series(dtype=float)
    elif not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test, index=X_test.index)

    results: dict[str, dict] = {}
    cv_choice = _choose_cv_splits(len(X_train), cv_splits)

    for name in model_names:
        key = _normalize_model_name(name)
        display_name = name
        try:
            base_model = get_ml_model(name)
        except (ValueError, ImportError) as exc:
            results[display_name] = {"error": str(exc)}
            continue
        param_grid = MODEL_PARAM_GRIDS.get(key, {})
        try:
            fitted_model, search = _run_random_search(base_model, param_grid, X_train, y_train, n_iter, cv_choice)
        except Exception as exc:
            results[display_name] = {"error": str(exc)}
            continue

        train_pred = fitted_model.predict(X_train)
        train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
        train_mae = float(mean_absolute_error(y_train, train_pred))
        train_mape = _compute_mape(y_train.to_numpy(), np.asarray(train_pred))

        test_metrics = {}
        test_pred_series = pd.Series(dtype=float)
        if not X_test.empty and not y_test.empty:
            test_pred = fitted_model.predict(X_test)
            test_pred_series = pd.Series(test_pred, index=X_test.index)
            test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
            test_mae = float(mean_absolute_error(y_test, test_pred))
            test_mape = _compute_mape(y_test.to_numpy(), np.asarray(test_pred))
            test_metrics = {
                "rmse": test_rmse,
                "mae": test_mae,
                "mape": test_mape,
            }

        cv_summary = None
        if search is not None:
            best_score = float(-search.best_score_)
            cv_summary = {
                "cv_metric": "MAE",
                "cv_best_score": best_score,
                "best_params": search.best_params_,
            }
        else:
            cv_summary = {
                "cv_metric": "MAE",
                "cv_best_score": train_mae,
                "best_params": getattr(fitted_model, "get_params", lambda: {})(),
            }

        shap_sample = X_train.iloc[-min(len(X_train), 200) :].copy()
        importance = _build_feature_importance(fitted_model, shap_sample)

        results[display_name] = {
            "model": fitted_model,
            "cv": cv_summary,
            "train_metrics": {
                "rmse": train_rmse,
                "mae": train_mae,
                "mape": train_mape,
            },
            "test_metrics": test_metrics,
            "shap_importance": importance,
            "test_predictions": test_pred_series,
        }

    return results


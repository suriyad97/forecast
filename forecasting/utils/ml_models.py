import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def make_lag_matrix(series: pd.Series, lags=(1,2,3,6,12), extra_df: pd.DataFrame=None):
    df = pd.DataFrame({ "y": series })
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    if extra_df is not None:
        df = df.join(extra_df, how="left")
    df = df.dropna()
    X = df.drop(columns=["y"])
    y = df["y"]
    return X, y, df

def iterative_forecast(model, last_known_y: pd.Series, steps:int, lags=(1,2,3,6,12),
                       future_exog: pd.DataFrame=None):
    preds = []
    idx = []
    for i in range(steps):
        next_idx = last_known_y.index[-1] + pd.offsets.MonthBegin(1)
        row = {}
        for lag in lags:
            if len(last_known_y) >= lag:
                row[f"lag_{lag}"] = last_known_y.iloc[-lag]
            else:
                row[f"lag_{lag}"] = last_known_y.iloc[-1]
        if future_exog is not None and next_idx in future_exog.index:
            for c in future_exog.columns:
                row[c] = future_exog.loc[next_idx, c]
        X_new = pd.DataFrame([row], index=[next_idx])
        yhat = float(model.predict(X_new)[0])
        preds.append(yhat)
        idx.append(next_idx)
        last_known_y = pd.concat([last_known_y, pd.Series([yhat], index=[next_idx])])
    return pd.Series(preds, index=idx, name="forecast")

def get_ml_model(name:str):
    name = name.lower()
    if name == "xgboost":
        return XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
    if name == "lightgbm":
        return LGBMRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=-1,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
    if name == "randomforest":
        return RandomForestRegressor(
            n_estimators=600, max_depth=None, random_state=42, n_jobs=-1
        )
    raise ValueError(f"Unknown model: {name}")

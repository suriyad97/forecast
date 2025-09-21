import pandas as pd


def make_lags(df, target_col='premium', lags=(1, 2, 3, 6, 12)):
    out = df.copy()
    for lag in lags:
        out[f'lag_{lag}'] = out[target_col].shift(lag)
    return out


def add_time_features(df):
    out = df.copy()
    out['year'] = out.index.year
    out['month'] = out.index.month
    return out


def make_rolling(df, target_col='premium', windows=(3, 6)):
    out = df.copy()
    for w in windows:
        out[f'roll_mean_{w}'] = out[target_col].rolling(w).mean()
    return out


def encode_categoricals(df: pd.DataFrame, encoding_map: dict[str, str]) -> pd.DataFrame:
    """Encode categorical columns according to the provided strategy per column."""
    encoded = df.copy()
    for col, method in encoding_map.items():
        if col not in encoded.columns:
            continue
        series = encoded[col]
        if method == 'One-Hot':
            include_na = series.isna().any()
            dummies = pd.get_dummies(series, prefix=col, dtype=float, dummy_na=include_na)
            encoded = encoded.drop(columns=[col])
            encoded = pd.concat([encoded, dummies], axis=1)
        elif method == 'Label':
            labels, _ = pd.factorize(series, sort=True, use_na_sentinel=True)
            encoded[col] = pd.Series(labels, index=series.index).astype(float)
        elif method == 'Frequency':
            tmp = series.fillna('__NA__')
            freq = tmp.value_counts(normalize=True)
            encoded[col] = tmp.map(freq).astype(float)
        else:
            raise ValueError(f"Unknown encoding method: {method}")
    return encoded

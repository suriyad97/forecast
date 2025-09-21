import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype


def load_data(file):
    return pd.read_csv(file)


def aggregate_monthly(df, date_col='policy_issue_date', value_col='premium',
                      product_col='product', channel_col='channel', dayfirst=True):
    # If already monthly aggregated (has 'year_month'), respect it
    if date_col not in df.columns and 'year_month' in df.columns:
        out = df.copy()
        out['year_month'] = pd.to_datetime(out['year_month'])
        return out.set_index('year_month').sort_index()

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst, errors='coerce')
    df = df.dropna(subset=[date_col, value_col])
    df['year_month'] = df[date_col].dt.to_period('M').dt.to_timestamp()
    group_cols = ['year_month']
    if product_col in df.columns:
        group_cols.append(product_col)
    if channel_col in df.columns:
        group_cols.append(channel_col)
    out = df.groupby(group_cols, as_index=False, observed=False)[value_col].sum()
    return out.sort_values('year_month').set_index('year_month')


def train_test_split_monthly(series, split_date=None, train_fraction=0.8):
    series = series.sort_index()
    if split_date is not None:
        cutoff = pd.to_datetime(split_date)
        train = series.loc[:cutoff]
        test = series.loc[cutoff + pd.offsets.MonthBegin(1):]
    else:
        n = len(series)
        split_idx = int(n * train_fraction)
        train = series.iloc[:split_idx]
        test = series.iloc[split_idx:]
    return train, test


def identify_exogenous_candidates(df: pd.DataFrame, target_col: str = 'premium'):
    """Return candidate exogenous columns partitioned by data type."""
    numeric_cols, categorical_cols, datetime_cols = [], [], []
    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]
        if is_numeric_dtype(series):
            numeric_cols.append(col)
        elif is_datetime64_any_dtype(series):
            datetime_cols.append(col)
        else:
            categorical_cols.append(col)
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols,
    }


def build_exog_matrix(df: pd.DataFrame, numeric_cols=None, categorical_cols=None):
    """Return a numeric design matrix for the requested exogenous columns."""
    numeric_cols = list(numeric_cols or [])
    categorical_cols = list(categorical_cols or [])
    if not numeric_cols and not categorical_cols:
        return None

    frames = []
    if numeric_cols:
        frames.append(df[numeric_cols])
    if categorical_cols:
        cat_df = pd.get_dummies(
            df[categorical_cols].astype('category'),
            prefix=categorical_cols,
            drop_first=True,
            dtype=float,
        )
        frames.append(cat_df)

    exog = pd.concat(frames, axis=1)
    exog = exog.replace([np.inf, -np.inf], np.nan)
    exog = exog.loc[:, exog.notna().any()]  # drop all-NaN columns
    return exog

import streamlit as st
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_object_dtype, is_bool_dtype
from utils.features import make_lags, add_time_features, make_rolling, encode_categoricals

st.title("Feature Engineering")

if "raw_df" not in st.session_state or "monthly_df" not in st.session_state:
    st.warning("Upload a dataset on the landing or ingestion page before engineering features.")
    st.stop()

if "train_split_ratio" not in st.session_state:
    st.session_state["train_split_ratio"] = 0.8
    st.session_state["train_split_label"] = "80% Train / 20% Test"

raw_df = st.session_state["raw_df"].copy()
monthly = st.session_state["monthly_df"].copy().sort_index().asfreq("MS")
st.write("Base series length:", len(monthly))
split_options = {
    "70% Train / 30% Test": 0.70,
    "75% Train / 25% Test": 0.75,
    "80% Train / 20% Test": 0.80,
    "85% Train / 15% Test": 0.85,
    "90% Train / 10% Test": 0.90,
}
default_split_label = st.session_state.get("train_split_label", "80% Train / 20% Test")
split_label = st.selectbox(
    "Train/Test split",
    options=list(split_options.keys()),
    index=list(split_options.keys()).index(default_split_label) if default_split_label in split_options else 2,
)
split_ratio = split_options[split_label]
st.session_state["train_split_ratio"] = split_ratio
st.session_state["train_split_label"] = split_label


# ----- Identify categorical features from raw data -----
raw_categorical_cols = []
raw_unique_lookup: dict[str, int] = {}
for col in raw_df.columns:
    if col == "premium":
        continue
    series = raw_df[col]
    if is_object_dtype(series) or is_categorical_dtype(series) or is_bool_dtype(series):
        raw_categorical_cols.append(col)
        raw_unique_lookup[col] = int(series.nunique(dropna=True))
raw_categorical_cols = sorted(dict.fromkeys(raw_categorical_cols))

if raw_categorical_cols:
    st.subheader("Categorical Feature Overview (Raw Data)")
    overview_rows = []
    for col in raw_categorical_cols:
        ser = raw_df[col]
        unique_vals = raw_unique_lookup.get(col, int(ser.nunique(dropna=True)))
        mode_series = ser.dropna().mode()
        most_common = mode_series.iloc[0] if not mode_series.empty else "-"
        overview_rows.append({
            "column": col,
            "unique_values": unique_vals,
            "most_common": most_common,
        })
    cat_overview_df = pd.DataFrame(overview_rows).sort_values("unique_values", ascending=False)
    st.dataframe(cat_overview_df, use_container_width=True)

    st.subheader("Select Categorical Exogenous Factors")
    exog_help = "These columns will be treated as categorical drivers across feature engineering and modeling pages."
    selected_exog_cols = st.multiselect(
        "Categorical columns",
        options=raw_categorical_cols,
        default=raw_categorical_cols,
        help=exog_help,
        format_func=lambda c: f"{c} (unique={raw_unique_lookup.get(c, 0)})"
    )
    st.session_state["selected_categorical_exog"] = selected_exog_cols
else:
    st.info("No categorical columns detected in the uploaded dataset.")
    selected_exog_cols = []

# ----- Lag / rolling selections -----
lags = st.multiselect("Select lags", [1, 2, 3, 6, 12], default=[1, 2, 3, 6, 12])
rolls = st.multiselect("Select rolling windows", [3, 6], default=[3, 6])

# ----- Determine categorical candidates available for encoding -----
categorical_candidates = []
unique_counts: dict[str, int] = {}
for col in monthly.columns:
    if col == "premium":
        continue
    series = monthly[col]
    if is_object_dtype(series) or is_categorical_dtype(series) or is_bool_dtype(series):
        categorical_candidates.append(col)
        unique_counts[col] = int(series.nunique(dropna=True))

selected_exog_in_monthly = [col for col in selected_exog_cols if col in monthly.columns]
missing_exog = sorted(set(selected_exog_cols) - set(selected_exog_in_monthly))
if missing_exog:
    st.warning(
        "The following categorical drivers are not present in the monthly aggregation and will be skipped in feature encoding: "
        + ", ".join(missing_exog)
    )

categorical_candidates = sorted(set(categorical_candidates).union(selected_exog_in_monthly))
for col in categorical_candidates:
    if col not in unique_counts:
        if col in raw_unique_lookup:
            unique_counts[col] = raw_unique_lookup[col]
        elif col in raw_df.columns:
            unique_counts[col] = int(raw_df[col].nunique(dropna=True))
        else:
            unique_counts[col] = int(monthly[col].nunique(dropna=True))

encoding_plan: dict[str, str] = {}
if categorical_candidates:
    st.subheader("Categorical Encoding")
    display_labels = {col: f"{col} ({unique_counts.get(col, 0)} unique)" for col in categorical_candidates}
    selected_cats = st.multiselect(
        "Columns to encode",
        options=categorical_candidates,
        default=categorical_candidates,
        format_func=lambda name: display_labels.get(name, name),
    )
    for col in selected_cats:
        uniques = unique_counts.get(col, 0)
        if uniques <= 10:
            default_method = "One-Hot"
        elif uniques <= 30:
            default_method = "Label"
        else:
            default_method = "Frequency"
        encoding_plan[col] = st.selectbox(
            f"Encoding for {col} ({uniques} unique values)",
            options=["One-Hot", "Label", "Frequency"],
            index=["One-Hot", "Label", "Frequency"].index(default_method),
            key=f"encoding_plan_{col}",
        )
else:
    st.info("No categorical columns detected in the monthly aggregation.")

# ----- Build feature dataset -----
lagged = make_lags(monthly, lags=tuple(lags) if lags else (1, 2, 3, 6, 12))
lagged = add_time_features(lagged)
lagged = make_rolling(lagged, windows=tuple(rolls) if rolls else (3, 6))
lagged = lagged.replace([float('inf'), float('-inf')], pd.NA).sort_index()

if len(lagged) >= 2:
    split_idx = max(1, int(len(lagged) * split_ratio))
    if split_idx >= len(lagged):
        split_idx = len(lagged) - 1
    fe_train = lagged.iloc[:split_idx]
    fe_test = lagged.iloc[split_idx:]
else:
    fe_train = lagged
    fe_test = lagged.iloc[0:0]

if encoding_plan:
    fe_train = encode_categoricals(fe_train, encoding_plan)
    fe_test = encode_categoricals(fe_test, encoding_plan)

fe = pd.concat([fe_train, fe_test]).sort_index()

st.session_state['features_train_df'] = fe_train
st.session_state['features_test_df'] = fe_test

train_start = fe_train.index.min().strftime('%Y-%m') if len(fe_train) else '-'
train_end = fe_train.index.max().strftime('%Y-%m') if len(fe_train) else '-'
test_start = fe_test.index.min().strftime('%Y-%m') if len(fe_test) else '-'
test_end = fe_test.index.max().strftime('%Y-%m') if len(fe_test) else '-'

st.markdown(f"**Train window:** {train_start} -> {train_end} (n={len(fe_train)})")
st.markdown(f"**Test window:** {test_start} -> {test_end} (n={len(fe_test)})")

st.subheader('Engineered Features (head)')
st.dataframe(fe.head(15))

csv = fe.to_csv(index=True).encode()
st.download_button('Download features CSV', data=csv, file_name='features_dataset.csv', mime='text/csv')

st.session_state['features_df'] = fe



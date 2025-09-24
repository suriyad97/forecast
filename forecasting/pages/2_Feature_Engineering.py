import numpy as np
import pandas as pd
import re
import streamlit as st
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

from utils.data import load_data
from utils.features import add_time_features, encode_categoricals, make_lags, make_rolling
from utils.ml_models import train_ml_models

try:  # optional dependency surface for SHAP messaging
    import shap  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully downstream
    shap = None
FEATURE_SESSION_KEYS_TO_CLEAR = [
    "raw_df",
    "monthly_df",
    "monthly_source_name",
    "features_df",
    "features_train_df",
    "features_test_df",
    "feature_exogenous_cols",
    "feature_target_internal",
    "feature_target_label",
    "selected_products",
    "selected_channels",
    "age_bucket_column",
    "selected_categorical_exog",
    "ml_training_results",
    "monthly_train_series",
    "product_column_name",
    "channel_column_name",
    "target_column_name",
    "ingestion_product_filter",
    "ingestion_channel_filter",
]


def _clear_feature_state() -> None:
    for key in FEATURE_SESSION_KEYS_TO_CLEAR:
        st.session_state.pop(key, None)
    st.session_state["landing_has_data"] = False
    st.session_state["landing_upload_active"] = False
    st.session_state["feature_upload_active"] = False
    st.session_state.pop("feature_engineering_uploader", None)



SAFE_FEATURE_PATTERN = re.compile(r"[\[\]<>]")


def _sanitize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    used: set[str] = set()
    for col in df.columns:
        safe = SAFE_FEATURE_PATTERN.sub("_", str(col))
        safe = re.sub(r"_+", "_", safe).strip("_")
        if not safe:
            safe = "feature"
        candidate = safe
        suffix = 1
        while candidate in used:
            suffix += 1
            candidate = f"{safe}_{suffix}"
        rename_map[col] = candidate
        used.add(candidate)
    if any(original != new for original, new in rename_map.items()):
        df = df.rename(columns=rename_map)
    return df

st.title("Feature Engineering")

uploaded_fresh = st.file_uploader(
    "Upload CSV (policy-level or monthly aggregated)",
    type=["csv"],
    key="feature_engineering_uploader",
)
if uploaded_fresh is not None:
    try:
        new_df = load_data(uploaded_fresh)
        st.session_state["raw_df"] = new_df
        st.session_state["monthly_source_name"] = getattr(uploaded_fresh, "name", "uploaded_file")
        st.session_state["landing_has_data"] = True
        st.session_state["feature_upload_active"] = True
        st.session_state["landing_upload_active"] = False
        st.success("Dataset loaded for feature engineering.")
    except Exception as exc:  # pragma: no cover - user facing
        st.error(f"Failed to read the uploaded file: {exc}")
        _clear_feature_state()
else:
    if st.session_state.get("feature_upload_active"):
        _clear_feature_state()

if "raw_df" not in st.session_state:
    st.warning("Upload a dataset here or on the landing page before engineering features.")
    st.stop()

raw_df = st.session_state["raw_df"].copy()


# ----- Column overview & classification -----
column_profile: list[dict[str, object]] = []
categorical_cols: list[str] = []
numeric_cols: list[str] = []
datetime_cols: list[str] = []

for col in raw_df.columns:
    series = raw_df[col]
    if is_datetime64_any_dtype(series):
        classified = "datetime"
        datetime_cols.append(col)
    elif is_numeric_dtype(series):
        classified = "numeric"
        numeric_cols.append(col)
    elif is_bool_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype) or is_object_dtype(series):
        classified = "categorical"
        categorical_cols.append(col)
    else:
        classified = "categorical"
        categorical_cols.append(col)

    column_profile.append(
        {
            "column": col,
            "dtype": str(series.dtype),
            "role": classified,
            "unique": int(series.nunique(dropna=True)),
            "missing_pct": round(series.isna().mean() * 100.0, 2),
        }
    )

profile_df = pd.DataFrame(column_profile).sort_values("column")
st.subheader("Column Type Overview")
st.dataframe(profile_df, use_container_width=True)

if not numeric_cols:
    st.error("No numeric columns detected. At least one numeric target is required.")
    st.stop()

# ----- Target & date selections -----
default_target = "premium" if "premium" in numeric_cols else numeric_cols[0]
default_target_idx = numeric_cols.index(default_target) if default_target in numeric_cols else 0
target_col = st.selectbox("Target column", options=numeric_cols, index=default_target_idx)
st.session_state["target_column_name"] = target_col

if datetime_cols:
    default_date = "policy_issue_date" if "policy_issue_date" in datetime_cols else datetime_cols[0]
else:
    inferred_dates = [c for c in raw_df.columns if "date" in c.lower()]
    datetime_cols = inferred_dates
    default_date = inferred_dates[0] if inferred_dates else None

if not datetime_cols:
    st.error("Unable to locate a usable date column. Please add one before proceeding.")
    st.stop()

default_date_idx = datetime_cols.index(default_date) if default_date in datetime_cols else 0
date_col = st.selectbox("Date column", options=datetime_cols, index=default_date_idx)
dayfirst = st.checkbox("Treat dates as day-first", value=True)

if categorical_cols:
    st.subheader("Categorical Drivers")
    default_categorical = [
        col for col in categorical_cols
        if col not in {target_col, date_col}
    ]
    selected_categorical = st.multiselect(
        "Categorical columns to engineer",
        options=sorted(categorical_cols),
        default=sorted(default_categorical) if default_categorical else sorted(categorical_cols),
        help="Dominant category per month is retained and encoded downstream.",
    )
else:
    st.info("No categorical columns detected in the uploaded dataset.")
    selected_categorical = []

st.session_state["selected_categorical_exog"] = selected_categorical
# ----- Product / channel filters -----
filter_expander = st.expander("Product & Channel Filters", expanded=True)
with filter_expander:
    product_col_options = ["(none)"] + categorical_cols
    default_product_idx = product_col_options.index("product") if "product" in product_col_options else 0
    product_col_choice = st.selectbox("Product column", options=product_col_options, index=default_product_idx)

    channel_candidates = [c for c in categorical_cols if c != product_col_choice]
    channel_col_options = ["(none)"] + channel_candidates
    default_channel_idx = channel_col_options.index("channel") if "channel" in channel_col_options else 0
    channel_col_choice = st.selectbox("Channel column", options=channel_col_options, index=default_channel_idx)

    filtered_df = raw_df.copy()
    if product_col_choice != "(none)":
        product_values = sorted(filtered_df[product_col_choice].dropna().unique().tolist())
        selected_products = st.multiselect(
            "Products to include",
            options=product_values,
            default=product_values,
        )
        if selected_products:
            filtered_df = filtered_df[filtered_df[product_col_choice].isin(selected_products)]
        st.session_state["selected_products"] = selected_products
    else:
        selected_products = []

    if channel_col_choice != "(none)":
        channel_values = sorted(filtered_df[channel_col_choice].dropna().unique().tolist())
        selected_channels = st.multiselect(
            "Channels to include",
            options=channel_values,
            default=channel_values,
        )
        if selected_channels:
            filtered_df = filtered_df[filtered_df[channel_col_choice].isin(selected_channels)]
        st.session_state["selected_channels"] = selected_channels
    else:
        selected_channels = []

st.session_state["product_column_name"] = None if product_col_choice == "(none)" else product_col_choice
st.session_state["channel_column_name"] = None if channel_col_choice == "(none)" else channel_col_choice

if filtered_df.empty:
    st.error("No rows left after applying the selected filters.")
    st.stop()

st.write(f"Filtered dataset size: {len(filtered_df):,d} rows")

# ----- Age buckets -----
age_candidates = [col for col in numeric_cols if "age" in col.lower()]
age_expander = st.expander("Age Buckets", expanded=False)
age_bucket_col = None
age_bucket_labels: list[str] = []
with age_expander:
    default_age_idx = age_candidates.index(age_candidates[0]) + 1 if age_candidates else 0
    age_selection = st.selectbox(
        "Age column for bucketing",
        options=["(none)"] + age_candidates,
        index=default_age_idx,
        help="Automatically derive age buckets that can be used as categorical drivers.",
    )
    if age_selection != "(none)":
        age_bucket_col = age_selection
        age_bins = [0, 25, 35, 45, 55, 65, np.inf]
        age_bucket_labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
        filtered_df["_age_bucket"] = pd.cut(
            filtered_df[age_bucket_col],
            bins=age_bins,
            labels=age_bucket_labels,
            include_lowest=True,
            right=False,
        )
        st.session_state["age_bucket_column"] = age_bucket_col
        st.info("Age buckets derived and will be added as categorical exogenous factors.")
    else:
        filtered_df.pop("_age_bucket", None)
        st.session_state.pop("age_bucket_column", None)


# ----- Monthly aggregation -----
work_df = filtered_df.copy()
work_df[date_col] = pd.to_datetime(work_df[date_col], dayfirst=dayfirst, errors="coerce")
work_df = work_df.dropna(subset=[date_col, target_col])
work_df["year_month"] = work_df[date_col].dt.to_period("M").dt.to_timestamp()

numeric_exog = [col for col in numeric_cols if col not in {target_col, date_col}]
if age_bucket_col is not None and age_bucket_col in numeric_exog:
    numeric_exog.remove(age_bucket_col)

agg_dict: dict[str, str] = {target_col: "sum"}
for col in numeric_exog:
    agg_dict[col] = "mean"

monthly = work_df.groupby("year_month", as_index=True).agg(agg_dict)

if "_age_bucket" in work_df.columns:
    bucket_counts = (
        work_df.dropna(subset=["_age_bucket"])
        .groupby(["year_month", "_age_bucket"])
        .size()
        .unstack(fill_value=0)
    )
    bucket_counts.columns = [f"age_bucket_{str(label)}" for label in bucket_counts.columns]
    monthly = monthly.join(bucket_counts, how="left").fillna(0)
    age_bucket_labels = list(bucket_counts.columns)

categorical_mode_columns: list[str] = []
categorical_sources: list[str] = []
if selected_categorical:
    categorical_sources.extend(selected_categorical)
if "product_col_choice" in locals() and product_col_choice not in {None, "(none)"}:
    categorical_sources.append(product_col_choice)
if "channel_col_choice" in locals() and channel_col_choice not in {None, "(none)"}:
    categorical_sources.append(channel_col_choice)
if "_age_bucket" in work_df.columns:
    categorical_sources.append("_age_bucket")

categorical_sources = [col for col in dict.fromkeys(categorical_sources) if col in work_df.columns]
for col in categorical_sources:
    mode_series = (
        work_df.groupby("year_month")[col]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    )
    feature_name = "age_bucket_mode" if col == "_age_bucket" else f"{col}_mode"
    monthly[feature_name] = mode_series
    categorical_mode_columns.append(feature_name)

for col in categorical_mode_columns:
    series = monthly[col]
    if isinstance(series.dtype, pd.CategoricalDtype):
        if "__MISSING__" not in series.cat.categories:
            series = series.cat.add_categories(["__MISSING__"])
    monthly[col] = series.fillna("__MISSING__").astype(str)

monthly = monthly.sort_index()

internal_target = "premium"
if target_col != internal_target:
    monthly = monthly.rename(columns={target_col: internal_target})

st.subheader("Monthly Aggregation Snapshot")
st.dataframe(monthly.tail(12), use_container_width=True)

if internal_target not in monthly.columns or monthly[internal_target].sum() == 0:
    st.warning("Target column has zero total value across the filtered dataset. Check filters and selections.")

# ----- Train/Test configuration -----
ratio_default = float(st.session_state.get("train_split_ratio", 0.8))
train_ratio = st.slider("Train fraction", min_value=0.6, max_value=0.95, value=ratio_default, step=0.05)
st.session_state["train_split_ratio"] = train_ratio
train_label = f"{int(train_ratio * 100):02d}% Train / {int((1 - train_ratio) * 100):02d}% Test"
st.session_state["train_split_label"] = train_label
st.write(f"Using split: {train_label}")

lag_options = [1, 2, 3, 6, 12, 24]
roll_options = [3, 6, 12]
default_lags = [1, 2, 3, 6, 12]
default_rolls = [3, 6]
selected_lags = st.multiselect("Lag features", lag_options, default=default_lags)
selected_rolls = st.multiselect("Rolling windows", roll_options, default=default_rolls)

if not selected_lags:
    selected_lags = default_lags
if not selected_rolls:
    selected_rolls = default_rolls

# ----- Feature creation -----
feature_base = make_lags(monthly, target_col=internal_target, lags=tuple(selected_lags))
feature_base = add_time_features(feature_base)
feature_base = make_rolling(feature_base, target_col=internal_target, windows=tuple(selected_rolls))
feature_base = feature_base.replace([float("inf"), float("-inf")], np.nan).dropna()

categorical_candidates: list[str] = []
unique_counts: dict[str, int] = {}
for col in feature_base.columns:
    if col == internal_target:
        continue
    series = feature_base[col]
    if is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype) or is_bool_dtype(series):
        categorical_candidates.append(col)
        unique_counts[col] = int(series.nunique(dropna=True))

st.subheader("Categorical Encoding Plan")
encoding_plan: dict[str, str] = {}
if categorical_candidates:
    rec_rows: list[dict[str, object]] = []

    def _recommend(col_name: str, uniques: int) -> tuple[str, str]:
        series = feature_base[col_name]
        if is_bool_dtype(series) or uniques <= 6:
            return "One-Hot", "Low cardinality"
        if uniques <= 25:
            return "Label", "Moderate cardinality"
        return "Frequency", "High cardinality"

    for col in categorical_candidates:
        uniques = unique_counts.get(col, int(feature_base[col].nunique(dropna=True)))
        method, rationale = _recommend(col, uniques)
        rec_rows.append(
            {
                "column": col,
                "unique": uniques,
                "recommended": method,
                "rationale": rationale,
            }
        )
        encoding_plan[col] = st.selectbox(
            f"Encoding for {col}",
            options=["One-Hot", "Label", "Frequency"],
            index=["One-Hot", "Label", "Frequency"].index(method),
            key=f"encoding_plan_{col}",
        )

    st.dataframe(pd.DataFrame(rec_rows), use_container_width=True)
else:
    st.info("No categorical columns detected after feature creation.")

feature_df = feature_base.copy()
if encoding_plan:
    feature_df = encode_categoricals(feature_df, encoding_plan)

feature_df = feature_df.sort_index()
feature_df = _sanitize_feature_columns(feature_df)

if len(feature_df) < 4:
    st.error("Not enough rows after feature engineering. Relax filters or reduce lag depth.")
    st.stop()

split_idx = max(1, int(len(feature_df) * train_ratio))
if split_idx >= len(feature_df):
    split_idx = len(feature_df) - 1

features_train = feature_df.iloc[:split_idx]
features_test = feature_df.iloc[split_idx:]

feature_cols = [col for col in feature_df.columns if col != internal_target]

st.session_state["features_df"] = feature_df
st.session_state["features_train_df"] = features_train
st.session_state["features_test_df"] = features_test
st.session_state["feature_exogenous_cols"] = feature_cols
st.session_state["feature_target_internal"] = internal_target
st.session_state["feature_target_label"] = target_col
st.caption("Engineered features have been handed off to the modeling & metrics section.")

train_window = (
    features_train.index.min().strftime("%Y-%m"),
    features_train.index.max().strftime("%Y-%m"),
)
test_window = (
    features_test.index.min().strftime("%Y-%m") if not features_test.empty else "-",
    features_test.index.max().strftime("%Y-%m") if not features_test.empty else "-",
)

st.subheader("Train/Test Windows")
st.markdown(f"**Train window:** {train_window[0]} -> {train_window[1]} (n={len(features_train)})")
st.markdown(f"**Test window:** {test_window[0]} -> {test_window[1]} (n={len(features_test)})")

preview_expander = st.expander("Preview Engineered Datasets", expanded=False)
with preview_expander:
    st.write("Train dataset (first 15 rows)")
    st.dataframe(features_train.head(15), use_container_width=True)
    st.write("Test dataset (first 15 rows)")
    if features_test.empty:
        st.write("Test set is empty after split.")
    else:
        st.dataframe(features_test.head(15), use_container_width=True)
    st.caption(f"Exogenous/driver columns ({len(feature_cols)}): {', '.join(feature_cols) if feature_cols else 'None'}")

csv_bytes = feature_df.to_csv(index=True).encode()
st.download_button(
    "Download engineered features",
    data=csv_bytes,
    file_name="features_dataset.csv",
    mime="text/csv",
)

# ----- Recommendations based on data profile -----
high_cardinality = [col for col, count in unique_counts.items() if count > 50]
recommendations: list[str] = []
if high_cardinality:
    recommendations.append(
        "Frequency encoding applied to high-cardinality columns to keep the feature matrix compact."
    )
if len(features_train) < 24:
    recommendations.append("Consider expanding the training window for more robust lag estimates.")
if not recommendations:
    recommendations.append("Current configuration looks balanced across lags, rolling means, and encoders.")

rec_expander = st.expander("Recommendations", expanded=False)
with rec_expander:
    for rec in recommendations:
        st.markdown(f"- {rec}")
# ----- Model training, hyper-parameter tuning, CV, SHAP -----
st.info("Next step: open the Models & Metrics page to select algorithms and evaluate them on this engineered dataset.")



import streamlit as st
import pandas as pd
from utils.data import load_data, aggregate_monthly
from utils.eda_view import render_descriptive_analysis

st.title("Ingestion & EDA")

if "train_split_ratio" not in st.session_state:
    st.session_state["train_split_ratio"] = 0.8
    st.session_state["train_split_label"] = "80% Train / 20% Test"
\nuploaded = st.file_uploader("Upload CSV (policy-level or monthly aggregated)", type=["csv"], key="ingestion_uploader")

if uploaded is not None:
    try:
        raw_df = load_data(uploaded)
        st.session_state["raw_df"] = raw_df
        st.session_state["monthly_source_name"] = getattr(uploaded, "name", "uploaded_file")
    except Exception as exc:
        st.error(f"Failed to read the uploaded file: {exc}")
        for key in ["raw_df", "monthly_df", "features_df", "monthly_source_name", "ingestion_product_filter", "ingestion_channel_filter"]:
            st.session_state.pop(key, None)
        raw_df = None
else:
    raw_df = st.session_state.get("raw_df")

if raw_df is not None:
    if st.sidebar.button("Remove uploaded dataset"):
        for key in ["raw_df", "monthly_df", "features_df", "monthly_source_name", "ingestion_product_filter", "ingestion_channel_filter"]:
            st.session_state.pop(key, None)
        st.experimental_rerun()

    st.sidebar.header("Segment Filters")
    product_options = [""] + sorted(list(raw_df.get("product", pd.Series([])).dropna().unique()))
    channel_options = [""] + sorted(list(raw_df.get("channel", pd.Series([])).dropna().unique()))
    product_default = st.session_state.get("ingestion_product_filter", "")
    channel_default = st.session_state.get("ingestion_channel_filter", "")
    product_index = product_options.index(product_default) if product_default in product_options else 0
    channel_index = channel_options.index(channel_default) if channel_default in channel_options else 0
    product = st.sidebar.selectbox("Product (optional)", options=product_options, index=product_index, key="ingestion_product")
    channel = st.sidebar.selectbox("Channel (optional)", options=channel_options, index=channel_index, key="ingestion_channel")
    st.session_state["ingestion_product_filter"] = product
    st.session_state["ingestion_channel_filter"] = channel

    monthly = aggregate_monthly(raw_df, date_col="policy_issue_date", value_col="premium", dayfirst=True)
    if product:
        monthly = monthly[monthly.get("product") == product]
    if channel:
        monthly = monthly[monthly.get("channel") == channel]

    monthly = monthly.sort_index().asfreq("MS")
    st.session_state["monthly_df"] = monthly

    if len(monthly) >= 2:
        split_ratio = st.session_state.get("train_split_ratio", 0.8)
        split_ratio = min(max(split_ratio, 0.5), 0.95)
        split_point = max(1, int(len(monthly) * split_ratio))
        if split_point >= len(monthly):
            split_point = len(monthly) - 1
        train_series = monthly.iloc[:split_point]["premium"].dropna() if "premium" in monthly.columns else None
        test_series = monthly.iloc[split_point:]["premium"].dropna() if "premium" in monthly.columns else None
        st.session_state["monthly_train_series"] = train_series
        st.session_state["monthly_test_series"] = test_series
    else:
        st.session_state["monthly_train_series"] = None
        st.session_state["monthly_test_series"] = None

    render_descriptive_analysis(raw_df, monthly)
else:
    st.info("Upload a CSV file to begin exploring your data.")


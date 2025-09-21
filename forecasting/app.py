import streamlit as st
import pandas as pd
from utils.data import load_data, aggregate_monthly
from utils.eda_view import render_sales_overview

st.set_page_config(page_title="Premium Forecasting", layout="wide")
st.title("Premium Forecasting Suite")

st.write(
    "Upload your historical premium data to generate a business-friendly snapshot. "
    "Switch to the dedicated pages for feature engineering, model evaluation, and detailed EDA."
)

if "train_split_ratio" not in st.session_state:
    st.session_state["train_split_ratio"] = 0.8
    st.session_state["train_split_label"] = "80% Train / 20% Test"

uploaded
 = st.file_uploader(
    "Upload CSV (policy-level or monthly aggregated)",
    type=["csv"],
    key="landing_uploader",
)

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
    if st.button("Remove uploaded dataset", type="secondary"):
        for key in ["raw_df", "monthly_df", "features_df", "monthly_source_name", "ingestion_product_filter", "ingestion_channel_filter"]:
            st.session_state.pop(key, None)
        st.experimental_rerun()

    try:
        monthly_df = aggregate_monthly(raw_df, date_col="policy_issue_date", value_col="premium", dayfirst=True)
        monthly_df = monthly_df.sort_index().asfreq("MS")
        st.session_state["monthly_df"] = monthly_df

        idx = monthly_df.index.dropna()
        years_available = sorted(idx.year.unique().tolist()) if len(idx) else []
        months_available = sorted(idx.month.unique().tolist()) if len(idx) else []

        if years_available and months_available:
            col_years, col_months = st.columns(2)
            selected_years = col_years.multiselect("Filter by Year", years_available, default=years_available)
            selected_months = col_months.multiselect(
                "Filter by Month",
                months_available,
                default=months_available,
                format_func=lambda m: pd.Timestamp(year=2000, month=m, day=1).strftime("%b"),
            )
        else:
            selected_years, selected_months = years_available, months_available

        render_sales_overview(raw_df, monthly_df, years=selected_years, months=selected_months)
    except Exception as exc:
        st.error(
            "Unable to build the monthly aggregation required for analysis. Please confirm the dataset includes 'policy_issue_date' and 'premium' columns.\n"
            f"Details: {exc}"
        )
else:
    st.info("Upload a CSV file to unlock the overview dashboard.")

st.markdown("---")
st.write("Quick navigation:")
st.page_link("pages/1_Ingestion_and_EDA.py", label="Ingestion & EDA")
st.page_link("pages/2_Feature_Engineering.py", label="Feature Engineering")
st.page_link("pages/3_🤖_Models_and_Metrics.py", label="Models & Metrics")
st.page_link("pages/4_🔮_Forecast_Explorer.py", label="Forecast Explorer")

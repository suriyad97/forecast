import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data import identify_exogenous_candidates, build_exog_matrix
from utils.models import (
    fit_ses,
    fit_holt,
    fit_hw_add,
    fit_hw_mul,
    fit_sarima,
    fit_sarimax,
    MODEL_KEY_TO_LABEL,
    MODEL_LABEL_TO_KEY,
)
from utils.ml_models import get_ml_model, make_lag_matrix, iterative_forecast

st.title("Forecast Explorer - Refit on Full Data & Predict Future")

if "monthly_df" not in st.session_state:
    st.warning("Please upload data on the landing page first.")
    st.stop()

EXOG_MODELS = {"SARIMAX_exog", "XGBoost", "LightGBM", "RandomForest"}


df = st.session_state["monthly_df"].copy().sort_index().asfreq("MS")
series = df["premium"]

model_label = st.selectbox("Model", list(MODEL_KEY_TO_LABEL.values()))
model_choice = MODEL_LABEL_TO_KEY[model_label]
seasonal_periods = st.number_input("Seasonal Periods", min_value=2, value=12)
horizon = st.number_input("Forecast Horizon (months)", min_value=1, value=6)

future_index = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

# ----- Exogenous selection -----
exog_numeric_selected: list[str] = []
exog_categorical_selected: list[str] = []
full_exog_history = None
future_exog = None
future_mode = "Repeat last known"

if model_choice in EXOG_MODELS:
    st.subheader("Exogenous Drivers")
    candidates = identify_exogenous_candidates(df)
    numeric_options = candidates["numeric"]
    categorical_options = candidates["categorical"]

    if not numeric_options and not categorical_options:
        st.info("No additional columns available as exogenous drivers.")
    else:
        exog_numeric_selected = st.multiselect(
            "Numeric columns",
            numeric_options,
            default=numeric_options,
        )
        exog_categorical_selected = st.multiselect(
            "Categorical columns",
            categorical_options,
        )

        if exog_numeric_selected or exog_categorical_selected:
            future_mode = st.radio("Future exog values", ["Repeat last known", "Zero"], index=0)

            # Build a raw frame with future values before encoding
            future_raw = pd.DataFrame(index=future_index)
            for col in exog_numeric_selected:
                if future_mode == "Repeat last known":
                    last_val = df[col].dropna().iloc[-1] if df[col].dropna().size else 0.0
                    future_raw[col] = last_val
                else:
                    future_raw[col] = 0.0
            for col in exog_categorical_selected:
                if future_mode == "Repeat last known":
                    last_series = df[col].dropna()
                    last_val = last_series.iloc[-1] if not last_series.empty else pd.NA
                    future_raw[col] = last_val
                else:
                    future_raw[col] = pd.NA

            combined_raw = pd.concat([df, future_raw], axis=0)
            encoded = build_exog_matrix(
                combined_raw,
                numeric_cols=exog_numeric_selected,
                categorical_cols=exog_categorical_selected,
            )
            if encoded is not None:
                full_exog_history = encoded.loc[df.index].ffill().fillna(0.0)
                future_exog = encoded.loc[future_index].fillna(0.0)
        else:
            st.warning("Select at least one column to use as an exogenous driver.")

# ----- Fit on FULL history and forecast future -----
if model_choice == "SES":
    model = fit_ses(series)
    yhat = model.forecast(horizon)
elif model_choice == "Holt":
    model = fit_holt(series)
    yhat = model.forecast(horizon)
elif model_choice == "HW_Add":
    model = fit_hw_add(series, seasonal_periods=seasonal_periods)
    yhat = model.forecast(horizon)
elif model_choice == "HW_Mul":
    model = fit_hw_mul(series, seasonal_periods=seasonal_periods)
    yhat = model.forecast(horizon)
elif model_choice == "SARIMA(1,1,1)(1,1,1)s":
    model = fit_sarima(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_periods))
    yhat = model.forecast(steps=horizon)
elif model_choice == "SARIMAX_exog":
    if full_exog_history is None or future_exog is None:
        st.error("Select and configure exogenous columns before running SARIMAX.")
        st.stop()
    model = fit_sarimax(
        series,
        full_exog_history,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, seasonal_periods),
    )
    yhat = model.forecast(steps=horizon, exog=future_exog)
else:
    lag_selection = st.multiselect(
        "Lags (ML only)",
        [1, 2, 3, 6, 12],
        default=[1, 2, 3, 6, 12],
    )
    lag_tuple = tuple(lag_selection) if lag_selection else (1, 2, 3, 6, 12)
    extra = full_exog_history if full_exog_history is not None else None
    X_all, y_all, _ = make_lag_matrix(series, lags=lag_tuple, extra_df=extra)
    model = get_ml_model(model_choice)
    model.fit(X_all, y_all)
    yhat_series = iterative_forecast(
        model,
        series,
        steps=horizon,
        lags=lag_tuple,
        future_exog=future_exog,
    )
    yhat = yhat_series.values

forecast = pd.Series(yhat, index=future_index, name="forecast")

st.subheader("Future Forecast")
st.dataframe(forecast.rename("premium").to_frame())

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=series.index, y=series.values, name="History", mode="lines", line=dict(color="#264653"))
)
fig.add_trace(
    go.Scatter(
        x=forecast.index,
        y=forecast.values,
        name=f"Forecast (+{horizon})",
        mode="lines+markers",
        line=dict(color="#e76f51"),
        marker=dict(size=8),
    )
)
fig.update_layout(
    template="plotly_white",
    title=f"{model_label} - Refit on Full History & Forecast",
    xaxis_title="Month",
    yaxis_title="Premium",
)
st.plotly_chart(fig, use_container_width=True)

st.download_button(
    "Download forecast CSV",
    data=forecast.rename("premium").to_csv().encode(),
    file_name="future_forecast.csv",
    mime="text/csv",
)


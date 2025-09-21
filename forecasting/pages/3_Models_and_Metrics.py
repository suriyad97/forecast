import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data import (
    train_test_split_monthly,
    identify_exogenous_candidates,
    build_exog_matrix,
)
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
from utils.metrics import rmse, mae, mape
from utils.ml_models import get_ml_model, make_lag_matrix

st.title("Models & Metrics - Evaluate on Test Only")

if "monthly_df" not in st.session_state:
    st.warning("Please complete Ingestion & EDA first.")
    st.stop()

EXOG_MODELS = {"SARIMAX_exog", "XGBoost", "LightGBM", "RandomForest"}


df = st.session_state["monthly_df"].copy().sort_index().asfreq("MS")
series = df["premium"]

# ----- Controls -----
st.sidebar.header("Evaluation Setup")
train_series_cached = st.session_state.get("monthly_train_series")
train_fraction_default = st.session_state.get("train_split_ratio", 0.8)
default_split = (
    train_series_cached.index[-1].strftime("%Y-%m")
    if isinstance(train_series_cached, pd.Series) and len(train_series_cached) > 0
    else (
        series.index[int(len(series) * train_fraction_default)].strftime("%Y-%m")
        if len(series) > 3
        else series.index[-1].strftime("%Y-%m")
    )
)
split_date = st.sidebar.text_input("Train end date (YYYY-MM):", value=default_split)
seasonal_periods = st.sidebar.number_input("Seasonal Periods", min_value=2, value=12)

model_label = st.sidebar.selectbox(
    "Choose a model to evaluate",
    list(MODEL_KEY_TO_LABEL.values()),
)
model_choice = MODEL_LABEL_TO_KEY[model_label]

exog_numeric_selected: list[str] = []
exog_categorical_selected: list[str] = []
full_exog = None

if model_choice in EXOG_MODELS:
    st.sidebar.subheader("Exogenous Drivers")
    candidates = identify_exogenous_candidates(df)
    numeric_options = candidates["numeric"]
    categorical_options = candidates["categorical"]

    if not numeric_options and not categorical_options:
        st.sidebar.info("No additional columns available as exogenous drivers.")
    else:
        exog_numeric_selected = st.sidebar.multiselect(
            "Numeric columns",
            numeric_options,
            default=numeric_options,
        )
        exog_categorical_selected = st.sidebar.multiselect(
            "Categorical columns",
            categorical_options,
        )

        if exog_numeric_selected or exog_categorical_selected:
            full_exog = build_exog_matrix(
                df,
                numeric_cols=exog_numeric_selected,
                categorical_cols=exog_categorical_selected,
            )
            if full_exog is not None:
                full_exog = full_exog.sort_index().ffill().fillna(0.0)
        else:
            st.sidebar.warning("Select at least one column to use as an exogenous driver.")

# ----- Split -----
try:
    train, test = train_test_split_monthly(series, split_date=f"{split_date}-01")
except Exception:
    train, test = train_test_split_monthly(series, train_fraction=train_fraction_default)

st.write(f"Train: {train.index.min().date()} - {train.index.max().date()}  (n={len(train)})")
st.write(
    f"Test : {test.index.min().date() if len(test)>0 else None} - "
    f"{test.index.max().date() if len(test)>0 else None}  (n={len(test)})"
)

if len(test) == 0:
    st.info("Not enough test data to evaluate. Adjust the split date.")
    st.stop()

# ----- Fit chosen model on train and forecast test -----
yhat = None
if model_choice == "SES":
    model = fit_ses(train)
    yhat = model.forecast(len(test))
elif model_choice == "Holt":
    model = fit_holt(train)
    yhat = model.forecast(len(test))
elif model_choice == "HW_Add":
    model = fit_hw_add(train, seasonal_periods=seasonal_periods)
    yhat = model.forecast(len(test))
elif model_choice == "HW_Mul":
    model = fit_hw_mul(train, seasonal_periods=seasonal_periods)
    yhat = model.forecast(len(test))
elif model_choice == "SARIMA(1,1,1)(1,1,1)s":
    model = fit_sarima(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_periods))
    yhat = model.forecast(steps=len(test))
elif model_choice == "SARIMAX_exog":
    if full_exog is None:
        st.error("Select at least one exogenous column.")
        st.stop()
    model = fit_sarimax(
        train,
        full_exog.loc[train.index],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, seasonal_periods),
    )
    yhat = model.forecast(steps=len(test), exog=full_exog.loc[test.index])
else:  # ML models
    extra = full_exog if full_exog is not None else None
    lag_selection = st.sidebar.multiselect(
        "Lags (ML only)",
        [1, 2, 3, 6, 12],
        default=[1, 2, 3, 6, 12],
    )
    lag_tuple = tuple(lag_selection) if lag_selection else (1, 2, 3, 6, 12)
    X_all, y_all, _ = make_lag_matrix(
        series,
        lags=lag_tuple,
        extra_df=extra,
    )
    X_train = X_all.loc[X_all.index <= train.index.max()]
    y_train = y_all.loc[y_all.index <= train.index.max()]
    X_test = X_all.loc[(X_all.index >= test.index.min()) & (X_all.index <= test.index.max())]

    model = get_ml_model(model_choice)
    model.fit(X_train, y_train)
    yhat = pd.Series(model.predict(X_test), index=X_test.index)

# ----- Metrics -----
if isinstance(yhat, pd.Series):
    yhat_series = yhat.reindex(test.index)
else:
    yhat_series = pd.Series(yhat, index=test.index)

aligned = pd.concat([test.rename("actual"), yhat_series.rename("forecast")], axis=1).dropna()

st.subheader("Test Metrics")
if aligned.empty:
    st.warning("No overlapping observations between actuals and forecast to score.")
else:
    st.write(
        {
            "RMSE": float(rmse(aligned["actual"], aligned["forecast"])),
            "MAE": float(mae(aligned["actual"], aligned["forecast"])),
            "MAPE": float(mape(aligned["actual"], aligned["forecast"])),
        }
    )

# ----- Plot -----
st.subheader("Forecast vs Actuals (Test Window)")
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=train.index, y=train.values, name="Train", mode="lines", line=dict(color="#8ecae6"))
)
fig.add_trace(
    go.Scatter(x=test.index, y=test.values, name="Test", mode="lines", line=dict(color="#023047"))
)
fig.add_trace(
    go.Scatter(x=yhat_series.index, y=yhat_series.values, name=model_label, mode="lines+markers", line=dict(color="#fb8500"))
)
fig.update_layout(
    template="plotly_white",
    title=f"Test Forecast - {model_label}",
    xaxis_title="Month",
    yaxis_title="Premium",
)
st.plotly_chart(fig, use_container_width=True)

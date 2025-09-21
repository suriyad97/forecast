# Streamlit Time Series Forecasting (Insurance Premiums)

This app lets you:
- Ingest policy-level data and aggregate to monthly premiums (product × channel)
- Run EDA (tables, plots, ACF/PACF)
- Engineer features (lags/rolling)
- Fit statistical models (SES, Holt, Holt-Winters Add/Mul, SARIMA, SARIMAX)
- Fit ML models (XGBoost, LightGBM, RandomForest) with lagged features
- Evaluate on a test window (RMSE/MAE/MAPE)
- Retrain on full history and forecast future months
- Explore forecasts product/channel-wise
- Generate **synthetic data** from a default 4-year schema

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Expected Input Columns
- `policy_issue_date` (first of month; format auto-parsed)
- `premium` (numeric)
- Optional: `product`, `channel`, `gender`, `age`, `location`, `benefit_period`, etc.
Or upload a pre-aggregated monthly file with `year_month`, `premium` (+ optional `product`, `channel`).

## Pages
1. **🧰 Synthetic Data Generator** – default 4-year schema, saves to `data/synthetic_data.csv`.
2. **📥 Ingestion & EDA** – upload/aggregate, segment filters, plots, ACF/PACF.
3. **🧪 Feature Engineering** – Lags and rolling features.
4. **🤖 Models & Metrics** – ETS/Holt/HW Add/Mul, SARIMA/SARIMAX, XGB/LightGBM/RF; metrics & plots.
5. **🔮 Forecast Explorer** – Full retrain and future forecasting; exogenous + ML iterative forecast.

# Streamlit Time Series Forecasting (Insurance Premiums)

This app lets you:
- Upload policy-level data and aggregate to monthly premiums (product x channel)
- Review core EDA visuals on the landing page (tables, plots, ACF/PACF)
- Engineer features (lags, rolling stats, categorical encodings)
- Fit statistical models (SES, Holt, Holt-Winters Add/Mul, SARIMA, SARIMAX)
- Fit ML models (XGBoost, LightGBM, RandomForest) with lagged features
- Evaluate on a test window (RMSE/MAE/MAPE)
- Retrain on full history and forecast future months
- Explore forecasts product/channel-wise

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
1. **Feature Engineering** - Upload, filter, and engineer time-series features with automated encoding plus model training, tuning, and SHAP diagnostics.
2. **Models & Metrics** - Evaluate statistical and ML models on the hold-out window with configurable exogenous drivers.
3. **Forecast Explorer** - Refit on the full history and generate forward-looking forecasts with optional drivers.
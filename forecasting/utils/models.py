from collections import OrderedDict
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

MODEL_OPTIONS = [
    ("SES", "Simple Exponential Smoothing (SES)"),
    ("Holt", "Holt's Linear Trend"),
    ("HW_Add", "Holt-Winters Additive"),
    ("HW_Mul", "Holt-Winters Multiplicative"),
    ("SARIMA(1,1,1)(1,1,1)s", "Seasonal ARIMA (1,1,1)(1,1,1)s"),
    ("SARIMAX_exog", "Seasonal ARIMAX with Exogenous Drivers"),
    ("XGBoost", "XGBoost Regressor"),
    ("LightGBM", "LightGBM Regressor"),
    ("RandomForest", "Random Forest Regressor"),
]

MODEL_KEY_TO_LABEL = OrderedDict(MODEL_OPTIONS)
MODEL_LABEL_TO_KEY = OrderedDict((label, key) for key, label in MODEL_OPTIONS)


def fit_ses(train):
    return SimpleExpSmoothing(train).fit()


def fit_holt(train):
    return ExponentialSmoothing(train, trend='add').fit()


def fit_hw_add(train, seasonal_periods=12):
    return ExponentialSmoothing(train, trend='add', seasonal='add',
                                seasonal_periods=seasonal_periods).fit()


def fit_hw_mul(train, seasonal_periods=12):
    return ExponentialSmoothing(train, trend='add', seasonal='mul',
                                seasonal_periods=seasonal_periods).fit()


def fit_sarima(train, order=(1,1,1), seasonal_order=(1,1,1,12)):
    return SARIMAX(train, order=order, seasonal_order=seasonal_order,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


def fit_sarimax(train_y, train_exog, order=(1,1,1), seasonal_order=(1,1,1,12)):
    return SARIMAX(train_y, exog=train_exog, order=order, seasonal_order=seasonal_order,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

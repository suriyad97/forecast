import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import (
    train_test_split_monthly,
    identify_exogenous_candidates,
    build_exog_matrix,
)
from utils.metrics import mae, mape, rmse
from utils.ml_models import get_ml_model, make_lag_matrix, train_ml_models
from utils.models import (
    fit_holt,
    fit_hw_add,
    fit_hw_mul,
    fit_sarima,
    fit_sarimax,
    fit_ses,
    MODEL_KEY_TO_LABEL,
    MODEL_LABEL_TO_KEY,
)

st.title("Models & Metrics")

feature_tab, classical_tab = st.tabs([
    "Feature-based ML",
    "Classical Baselines",
])

with feature_tab:
    st.subheader("Feature-Based Model Training")
    required_keys = {
        "features_train_df",
        "feature_exogenous_cols",
        "feature_target_internal",
        "feature_target_label",
    }
    ready_for_training = required_keys.issubset(st.session_state.keys())

    if not ready_for_training:
        st.warning(
            "Run the Feature Engineering page first to build the training dataset before launching models here."
        )
    else:
        features_train = st.session_state["features_train_df"].copy()
        features_test = st.session_state.get("features_test_df", pd.DataFrame()).copy()
        feature_cols = st.session_state.get("feature_exogenous_cols", [])
        internal_target = st.session_state["feature_target_internal"]
        target_label = st.session_state["feature_target_label"]
        selected_model_names = st.session_state.get("model_name_selection", ["RandomForest", "XGBoost", "LightGBM"])

        st.caption(
            f"Using engineered features from the Feature Engineering page (target: {target_label})."
        )

        if features_train.empty:
            st.error("Training dataset is empty. Revisit the Feature Engineering page to regenerate it.")
        else:
            train_window = (
                features_train.index.min().strftime("%Y-%m"),
                features_train.index.max().strftime("%Y-%m"),
            )
            test_window = (
                features_test.index.min().strftime("%Y-%m") if not features_test.empty else "-",
                features_test.index.max().strftime("%Y-%m") if not features_test.empty else "-",
            )

            st.write(
                f"Train window: {train_window[0]} 	 {train_window[1]} (n={len(features_train)})"
            )
            st.write(
                f"Test window: {test_window[0]} 	 {test_window[1]} (n={len(features_test)})"
            )

            model_group_map: dict[str, list[str]] = {
                "Linear Regression": ["Linear Regression"],
                "Lasso": ["Lasso"],
                "Elastic Net": ["Elastic Net"],
                "Prophet": ["Prophet"],
                "Boosting models": ["RandomForest", "XGBoost", "LightGBM"],
            }

            default_labels = st.session_state.get("model_group_selection", ["Boosting models"])
            model_group_selection = st.multiselect(
                "Models to train",
                options=list(model_group_map.keys()),
                default=default_labels if default_labels else ["Boosting models"],
                help="Pick which algorithms to include in the training run.",
            )
            if not model_group_selection:
                model_group_selection = ["Boosting models"]

            resolved_models: list[str] = []
            for label in model_group_selection:
                resolved_models.extend(model_group_map[label])
            resolved_models = list(dict.fromkeys(resolved_models))

            st.session_state["model_group_selection"] = model_group_selection
            st.session_state["model_name_selection"] = resolved_models

            if resolved_models:
                st.caption(f"Training will include: {', '.join(resolved_models)}")
            else:
                st.caption("No models selected yet.")

            run_training = st.button("Run training & evaluation", type="primary")
            if run_training:
                if not resolved_models:
                    st.warning("Select at least one model before running training.")
                else:
                    X_train = features_train[feature_cols]
                    y_train = features_train[internal_target]
                    X_test = features_test[feature_cols] if not features_test.empty else pd.DataFrame()
                    y_test = (
                        features_test[internal_target]
                        if (not features_test.empty and internal_target in features_test.columns)
                        else pd.Series(dtype=float)
                    )

                    with st.spinner("Running cross-validation, tuning, and SHAP analysis..."):
                        training_results = train_ml_models(
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            model_names=resolved_models,
                        )

                    st.session_state["ml_training_results"] = training_results
                    st.success(
                        "Model training complete using the engineered feature dataset. Review results below."
                    )

            if "ml_training_results" in st.session_state:
                results = st.session_state["ml_training_results"]
                features_test = st.session_state.get("features_test_df", pd.DataFrame())
                y_test_series = (
                    features_test[internal_target]
                    if internal_target in features_test.columns
                    else pd.Series(dtype=float)
                )

                for model_name, payload in results.items():
                    if "error" in payload:
                        st.error(f"{model_name}: {payload.get('error')}")
                        continue

                    st.markdown(f"### {model_name}")
                    cv_info = payload.get("cv", {})
                    train_metrics = payload.get("train_metrics", {})
                    test_metrics = payload.get("test_metrics", {})

                    metrics_table = pd.DataFrame(
                        {
                            "Train": train_metrics,
                            "Test": test_metrics if test_metrics else {k: np.nan for k in train_metrics},
                        }
                    )
                    st.dataframe(metrics_table, use_container_width=True)

                    if cv_info:
                        st.write(
                            {
                                "CV metric": cv_info.get("cv_metric"),
                                "CV best score": cv_info.get("cv_best_score"),
                                "Best params": cv_info.get("best_params"),
                            }
                        )

                    shap_importance = payload.get("shap_importance")
                    if shap_importance is not None and not shap_importance.empty:
                        st.write("Top SHAP features")
                        st.dataframe(shap_importance.head(15), use_container_width=True)
                    else:
                        st.info(
                            "SHAP importance unavailable for this model configuration. Install the shap package for tree models if needed."
                        )

                    test_pred = payload.get("test_predictions")
                    if test_pred is not None and not test_pred.empty and not features_test.empty:
                        comparison = pd.concat(
                            [
                                y_test_series.rename("actual"),
                                test_pred.rename("prediction"),
                            ],
                            axis=1,
                        ).dropna()
                        if not comparison.empty:
                            st.line_chart(comparison)

                    st.divider()

with classical_tab:
    st.subheader("Classical Baseline Evaluation")
    if "monthly_df" not in st.session_state:
        st.warning("Upload data on the landing page to unlock classical model evaluation.")
    else:
        df = st.session_state["monthly_df"].copy().sort_index().asfreq("MS")
        series = df["premium"]

        EXOG_MODELS = {"SARIMAX_exog", "XGBoost", "LightGBM", "RandomForest"}

        st.sidebar.header("Evaluation Setup")
        train_series_cached = st.session_state.get("monthly_train_series")
        train_fraction_default = st.session_state.get("train_split_ratio", 0.8)
        if isinstance(train_series_cached, pd.Series) and len(train_series_cached) > 0:
            default_split = train_series_cached.index[-1].strftime("%Y-%m")
        elif len(series) > 3:
            default_split = series.index[int(len(series) * train_fraction_default)].strftime("%Y-%m")
        else:
            default_split = series.index[-1].strftime("%Y-%m")

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

        try:
            train, test = train_test_split_monthly(series, split_date=f"{split_date}-01")
        except Exception:
            train, test = train_test_split_monthly(series, train_fraction=train_fraction_default)

        st.write(
            f"Train: {train.index.min().date()} - {train.index.max().date()}  (n={len(train)})"
        )
        st.write(
            f"Test : {test.index.min().date() if len(test)>0 else None} - {test.index.max().date() if len(test)>0 else None}  (n={len(test)})"
        )

        if len(test) == 0:
            st.info("Not enough test data to evaluate. Adjust the split date.")
        else:
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
                    yhat = None
                else:
                    model = fit_sarimax(
                        train,
                        full_exog.loc[train.index],
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, seasonal_periods),
                    )
                    yhat = model.forecast(steps=len(test), exog=full_exog.loc[test.index])
            else:
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

            if yhat is not None:
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

                st.subheader("Forecast vs Actuals (Test Window)")
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=train.index, y=train.values, name="Train", mode="lines", line=dict(color="#8ecae6"))
                )
                fig.add_trace(
                    go.Scatter(x=test.index, y=test.values, name="Test", mode="lines", line=dict(color="#023047"))
                )
                fig.add_trace(
                    go.Scatter(
                        x=yhat_series.index,
                        y=yhat_series.values,
                        name=model_label,
                        mode="lines+markers",
                        line=dict(color="#fb8500"),
                    )
                )
                fig.update_layout(
                    template="plotly_white",
                    title=f"Test Forecast - {model_label}",
                    xaxis_title="Month",
                    yaxis_title="Premium",
                )
                st.plotly_chart(fig, use_container_width=True)


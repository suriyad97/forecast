import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf, adfuller


def _plot_series(series: pd.Series, title: str, y_label: str = "Premium", show_trend: bool = True) -> None:
    series = series.dropna()
    if series.empty:
        st.info(f"No data available to plot for {title}.")
        return
    idx = pd.to_datetime(series.index)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=series.values,
            mode="lines+markers",
            name=title,
            line=dict(color="#1d3557", width=2),
            marker=dict(size=6),
        )
    )
    if show_trend and len(series) >= 2:
        x_numeric = np.arange(len(series))
        mask = ~np.isnan(series.values)
        if mask.any() and mask.sum() >= 2:
            coeffs = np.polyfit(x_numeric[mask], series.values[mask], 1)
            trend = coeffs[0] * x_numeric + coeffs[1]
            fig.add_trace(
                go.Scatter(
                    x=idx,
                    y=trend,
                    mode="lines",
                    name="Linear Trend",
                    line=dict(color="#264653", width=2, dash="dot"),
                )
            )
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode="x unified",
        legend_title="Series",
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_bar(series: pd.Series, title: str, axis_label: str, orientation: str = "v") -> None:
    bar_df = series.reset_index(name="value")
    if orientation == "h":
        fig = px.bar(
            bar_df,
            x="value",
            y=bar_df.columns[0],
            orientation="h",
            template="plotly_white",
            title=title,
            text_auto=".2s",
        )
        fig.update_layout(xaxis_title="Value", yaxis_title=axis_label)
    else:
        fig = px.bar(
            bar_df,
            x=bar_df.columns[0],
            y="value",
            orientation="v",
            template="plotly_white",
            title=title,
            text_auto=".2s",
        )
        fig.update_layout(xaxis_title=axis_label, yaxis_title="Value")
    fig.update_traces(marker_color="#264653")
    st.plotly_chart(fig, use_container_width=True)


def _plot_heatmap(df: pd.DataFrame, title: str, text_format: str = ".2f", zmin=None, zmax=None, colorscale="Viridis"):
    fig = px.imshow(
        df,
        text_auto=text_format,
        color_continuous_scale=colorscale,
        aspect="auto",
        zmin=zmin,
        zmax=zmax,
        title=title,
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


def _plot_acf_pacf(series: pd.Series, title_prefix: str) -> dict:
    series = series.dropna()
    result = {"acf": [], "pacf": []}
    if len(series) < 2:
        st.info(f"Not enough data points to compute ACF/PACF for {title_prefix}.")
        return result
    nlags = min(36, max(1, len(series) // 2))
    if nlags < 1:
        st.info(f"Not enough data points to compute ACF/PACF for {title_prefix}.")
        return result
    conf_level = 1.96 / np.sqrt(len(series))
    lags = list(range(1, nlags + 1))
    acf_values = sm_acf(series, nlags=nlags, fft=False)[1: nlags + 1]
    fig_acf = go.Figure(
        data=[go.Bar(x=lags, y=acf_values, marker_color="#2a9d8f")],
        layout=go.Layout(template="plotly_white", title=f"ACF - {title_prefix}")
    )
    fig_acf.update_layout(
        xaxis_title="Lag",
        yaxis_title="Autocorrelation",
        shapes=[
            dict(type="line", x0=0.5, x1=nlags + 0.5, y0=conf_level, y1=conf_level, line=dict(dash="dash", color="#e76f51")),
            dict(type="line", x0=0.5, x1=nlags + 0.5, y0=-conf_level, y1=-conf_level, line=dict(dash="dash", color="#e76f51")),
            dict(type="line", x0=0.5, x1=nlags + 0.5, y0=0, y1=0, line=dict(color="#264653")),
        ],
    )
    st.plotly_chart(fig_acf, use_container_width=True)
    result["acf"] = [lag for lag, val in zip(lags, acf_values) if abs(val) > conf_level]
    pacf_values = sm_pacf(series, nlags=nlags, method="yw")[1: nlags + 1]
    fig_pacf = go.Figure(
        data=[go.Bar(x=lags, y=pacf_values, marker_color="#e9c46a")],
        layout=go.Layout(template="plotly_white", title=f"PACF - {title_prefix}")
    )
    fig_pacf.update_layout(
        xaxis_title="Lag",
        yaxis_title="Partial Autocorrelation",
        shapes=[
            dict(type="line", x0=0.5, x1=nlags + 0.5, y0=conf_level, y1=conf_level, line=dict(dash="dash", color="#e76f51")),
            dict(type="line", x0=0.5, x1=nlags + 0.5, y0=-conf_level, y1=-conf_level, line=dict(dash="dash", color="#e76f51")),
            dict(type="line", x0=0.5, x1=nlags + 0.5, y0=0, y1=0, line=dict(color="#264653")),
        ],
    )
    st.plotly_chart(fig_pacf, use_container_width=True)
    result["pacf"] = [lag for lag, val in zip(lags, pacf_values) if abs(val) > conf_level]
    return result


def _run_adf(series: pd.Series, label: str) -> dict:
    series = series.dropna().astype(float)
    result = {
        "Series": label,
        "ADF Statistic": np.nan,
        "p-value": np.nan,
        "Lags Used": np.nan,
        "Observations": len(series),
        "Critical Values": None,
    }
    if len(series) < 3:
        return result
    try:
        adf_stat, p_value, usedlag, nobs, critical_vals, _ = adfuller(series, autolag="AIC")
        result.update(
            {
                "ADF Statistic": adf_stat,
                "p-value": p_value,
                "Lags Used": usedlag,
                "Observations": nobs,
                "Critical Values": critical_vals,
            }
        )
    except Exception:
        pass
    return result


def _display_adf_summary(results: list[dict]) -> None:
    if not results:
        return
    summary = pd.DataFrame(
        [
            {
                "Series": res.get("Series"),
                "ADF Statistic": res.get("ADF Statistic"),
                "p-value": res.get("p-value"),
                "Lags Used": res.get("Lags Used"),
                "Observations": res.get("Observations"),
            }
            for res in results
        ]
    )
    st.dataframe(summary, use_container_width=True)
    crit_vals = results[-1].get("Critical Values") or results[0].get("Critical Values")
    if crit_vals:
        st.dataframe(
            pd.DataFrame(
                {
                    "Significance": list(crit_vals.keys()),
                    "Critical Value": list(crit_vals.values()),
                }
            ),
            use_container_width=True,
        )


def _describe_seasonality(acf_pacf_info: dict, period_hint: int | None = None) -> None:
    acf_lags = [lag for lag in acf_pacf_info.get("acf", []) if lag > 1]
    pacf_lags = [lag for lag in acf_pacf_info.get("pacf", []) if lag > 1]
    if not acf_lags and not pacf_lags:
        st.info("ACF/PACF do not show significant seasonal spikes beyond lag 1.")
        return
    msg_parts = []
    if acf_lags:
        msg_parts.append(
            "ACF spikes at lags: " + ", ".join(str(l) for l in acf_lags)
        )
    if pacf_lags:
        msg_parts.append(
            "PACF spikes at lags: " + ", ".join(str(l) for l in pacf_lags)
        )
    if msg_parts:
        st.info("; ".join(msg_parts))
    if period_hint:
        seasonal_lags = [lag for lag in set(acf_lags + pacf_lags) if abs(lag - period_hint) <= 1]
        if seasonal_lags:
            st.success(
                "Seasonality around lag "
                + str(period_hint)
                + " detected (lags: "
                + ", ".join(str(l) for l in seasonal_lags)
                + ")."
            )


def _plot_seasonal_subseries(series: pd.Series) -> None:
    series = series.dropna()
    if series.empty:
        st.info("Insufficient data for seasonal subseries plot.")
        return
    df = series.to_frame(name="premium")
    df["year"] = df.index.year
    df["month_label"] = df.index.strftime("%b")
    fig = px.line(
        df,
        x="month_label",
        y="premium",
        color="year",
        line_group="year",
        title="Seasonal Subseries (monthly profiles by year)",
        markers=True,
        template="plotly_white",
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Premium")
    st.plotly_chart(fig, use_container_width=True)


def _plot_distribution(series: pd.Series, title: str) -> None:
    series = series.dropna()
    if series.empty:
        st.info("Not enough data for distribution analysis.")
        return
    fig = px.histogram(
        series,
        x=series.values,
        nbins=min(40, max(10, len(series) // 5)),
        marginal="box",
        opacity=0.8,
        title=title,
        template="plotly_white",
    )
    fig.update_layout(xaxis_title="Value", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)


def _plot_rolling_stats(series: pd.Series, windows: tuple[int, ...] = (3, 6, 12)) -> None:
    series = series.dropna()
    if series.empty:
        st.info("Not enough data for rolling statistics.")
        return
    fig = go.Figure()
    idx = pd.to_datetime(series.index)
    fig.add_trace(
        go.Scatter(x=idx, y=series.values, mode="lines", name="Series", line=dict(color="#1d3557"))
    )
    palette = ["#ff9f1c", "#2ec4b6", "#e71d36", "#cbf3f0"]
    for i, window in enumerate(windows):
        if window >= len(series):
            continue
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=rolling_mean,
                mode="lines",
                name=f"Mean ({window})",
                line=dict(color=palette[i % len(palette)], dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=rolling_std,
                mode="lines",
                name=f"Std ({window})",
                line=dict(color=palette[i % len(palette)], dash="dot"),
            )
        )
    fig.update_layout(
        template="plotly_white",
        title="Rolling Mean & Std",
        xaxis_title="Date",
        yaxis_title="Value",
    )
    st.plotly_chart(fig, use_container_width=True)


def _detect_outliers(series: pd.Series, z_thresh: float = 3.0) -> pd.DataFrame:
    series = series.dropna()
    if series.empty:
        return pd.DataFrame(columns=["date", "premium", "zscore"])
    zscores = stats.zscore(series.values, nan_policy="omit")
    mask = np.abs(zscores) >= z_thresh
    return pd.DataFrame(
        {
            "date": series.index[mask],
            "premium": series.values[mask],
            "zscore": zscores[mask],
        }
    )


def render_descriptive_analysis(raw: pd.DataFrame, monthly: pd.DataFrame) -> None:
    st.subheader("Dataset Summary")
    total_rows = len(raw)
    time_min = pd.to_datetime(raw.get("policy_issue_date"), dayfirst=True, errors="coerce").min()
    time_max = pd.to_datetime(raw.get("policy_issue_date"), dayfirst=True, errors="coerce").max()
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{total_rows:,}")
    if pd.notna(time_min) and pd.notna(time_max):
        date_range_value = f"{time_min:%d-%b-%Y} → {time_max:%d-%b-%Y}"
    else:
        date_range_value = "-"
    with c2:
        st.markdown("<div style='font-size:0.80rem;color:#666;'>Date Range</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.9rem;font-weight:700;'>{date_range_value}</div>", unsafe_allow_html=True)
    c3.metric("Columns", len(raw.columns))
    st.dataframe(raw.head(10))

    st.subheader("Basic Data Describe")
    numeric_describe = raw.select_dtypes(include=[np.number]).describe().T
    st.dataframe(numeric_describe, use_container_width=True)

    st.subheader("Monthly Aggregation Snapshot")
    st.dataframe(monthly.head(12))

    if "premium" not in monthly.columns or monthly["premium"].dropna().empty:
        st.info("Premium series not available for advanced diagnostics.")
        return

    premium_series = monthly["premium"].dropna()

    st.subheader("Step 1: Trend and Seasonality")
    inferred_freq = pd.infer_freq(premium_series.index)
    period_guess = 12
    if inferred_freq:
        if inferred_freq.startswith("Q"):
            period_guess = 4
        elif inferred_freq.startswith("W"):
            period_guess = 52
        elif inferred_freq.startswith("D"):
            period_guess = 7
    seasonal_flag = False
    if len(premium_series) >= period_guess * 2:
        try:
            decomposition = seasonal_decompose(
                premium_series,
                model="additive",
                period=period_guess,
                extrapolate_trend="freq",
            )
            fig_dec = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=("Trend", "Seasonal", "Residual"),
            )
            fig_dec.add_trace(
                go.Scatter(
                    x=decomposition.trend.index,
                    y=decomposition.trend,
                    mode="lines",
                    name="Trend",
                    line=dict(color="#1d3557"),
                ),
                row=1,
                col=1,
            )
            fig_dec.add_trace(
                go.Scatter(
                    x=decomposition.seasonal.index,
                    y=decomposition.seasonal,
                    mode="lines",
                    name="Seasonal",
                    line=dict(color="#457b9d"),
                ),
                row=2,
                col=1,
            )
            fig_dec.add_trace(
                go.Scatter(
                    x=decomposition.resid.index,
                    y=decomposition.resid,
                    mode="lines",
                    name="Residual",
                    line=dict(color="#e63946"),
                ),
                row=3,
                col=1,
            )
            fig_dec.update_layout(template="plotly_white", height=600, showlegend=False)
            st.plotly_chart(fig_dec, use_container_width=True)
            seasonal_strength = np.nanmax(decomposition.seasonal) - np.nanmin(decomposition.seasonal)
            baseline = np.nanmean(premium_series)
            if baseline and seasonal_strength / baseline > 0.1:
                seasonal_flag = True
        except Exception as exc:
            st.info(f"Unable to decompose series: {exc}")
    else:
        st.info(
            f"Need at least {period_guess * 2} observations to run seasonal decomposition; currently have {len(premium_series)}."
        )

    _plot_series(premium_series, "Monthly Premium Level")

    st.subheader("Step 2: Seasonality Diagnostics")
    if len(premium_series) >= period_guess:
        monthly_avg = premium_series.groupby(premium_series.index.month).mean()
        month_labels = [pd.Timestamp(year=2000, month=int(m), day=1).strftime("%b") for m in monthly_avg.index]
        seasonality_series = pd.Series(monthly_avg.values, index=month_labels, name="Seasonality")
        _plot_bar(seasonality_series, "Average Premium by Calendar Month", "Month", orientation="h")
        if seasonal_flag or (seasonality_series.max() - seasonality_series.min()) / (seasonality_series.mean() or 1) > 0.1:
            st.success("Pronounced seasonality detected across calendar months.")
        else:
            st.info("Seasonality appears mild across calendar months.")
        _plot_seasonal_subseries(premium_series)
    else:
        st.info("Not enough observations to compute seasonal averages.")

    st.subheader("Step 3: Autocorrelation Structure")
    acf_pacf_level = _plot_acf_pacf(premium_series, "Level Series")
    _describe_seasonality(acf_pacf_level, period_hint=period_guess)

    st.subheader("Step 4: Stationarity Workflow")
    adf_results: list[dict] = []
    step_counter = 1
    current_series = premium_series

    def evaluate_series(series: pd.Series, label: str, show_trend: bool = True) -> dict:
        nonlocal step_counter, current_series
        st.markdown(f"**Step 4.{step_counter}: {label}**")
        _plot_series(series, f"{label} Series", show_trend=show_trend)
        acf_info = _plot_acf_pacf(series, label)
        _describe_seasonality(acf_info, period_hint=period_guess)
        res = _run_adf(series, label)
        adf_results.append(res)
        p_value = res.get("p-value", np.nan)
        if np.isnan(p_value):
            st.info("ADF test could not be computed for this series.")
        elif p_value < 0.05:
            st.success("Series is stationary (ADF p-value < 0.05).")
        else:
            st.warning("Series remains non-stationary (ADF p-value >= 0.05).")
        step_counter += 1
        current_series = series
        return res

    level_result = evaluate_series(current_series, "Level")

    if level_result.get("p-value", np.nan) >= 0.05:
        log_result = None
        if (current_series > 0).all():
            log_series = np.log(current_series)
            log_result = evaluate_series(log_series, "Log Transform", show_trend=True)
            current_series = log_series
        else:
            st.info("Log transform skipped because the series contains non-positive values.")

        diff1_result = None
        if (log_result or level_result).get("p-value", np.nan) >= 0.05:
            diff1 = current_series.diff().dropna()
            if not diff1.empty:
                diff1_result = evaluate_series(diff1, "First Difference", show_trend=False)
            else:
                st.info("First difference produced no valid observations. Skipping differencing diagnostics.")

            if diff1_result and diff1_result.get("p-value", np.nan) >= 0.05:
                diff2 = diff1.diff().dropna()
                if not diff2.empty:
                    evaluate_series(diff2, "Second Difference", show_trend=False)
                else:
                    st.info("Second difference produced no valid observations.")

    _display_adf_summary(adf_results)

    st.subheader("Step 5: Rolling Diagnostics & Outliers")
    _plot_rolling_stats(premium_series)
    outliers_df = _detect_outliers(premium_series)
    if not outliers_df.empty:
        st.warning("Potential premium spikes detected (|z| >= 3).")
        st.dataframe(outliers_df, use_container_width=True)
    else:
        st.info("No extreme outliers detected in the premium series (|z| >= 3).")

    st.subheader("Step 6: Distribution & Residual Checks")
    _plot_distribution(premium_series, "Premium Distribution (Level)")
    if adf_results:
        last_stationary = next((res for res in reversed(adf_results) if res.get("p-value") < 0.05), adf_results[-1])
        label = last_stationary.get("Series", "Stationary")
        stationary_series = current_series if label != "Level" else premium_series
        _plot_distribution(stationary_series, f"Distribution - {label}")

    st.subheader("Step 7: Correlation Matrix (Numeric Features)")
    numeric_monthly = monthly.select_dtypes(include=[np.number]).dropna(how="all", axis=1)
    if len(numeric_monthly.columns) >= 2:
        corr = numeric_monthly.corr()
        _plot_heatmap(corr, "Correlation (Monthly Aggregation)", text_format=".2f", zmin=-1, zmax=1, colorscale="RdBu")
    else:
        st.info("Not enough numeric columns to compute correlations.")


def render_sales_overview(raw: pd.DataFrame, monthly: pd.DataFrame, years=None, months=None) -> None:
    years_set = set(years) if years else None
    months_set = set(months) if months else None
    monthly_filtered = monthly.copy()
    monthly_filtered.index = pd.to_datetime(monthly_filtered.index, errors="coerce")
    monthly_filtered = monthly_filtered[monthly_filtered.index.notna()].sort_index()
    if years_set:
        monthly_filtered = monthly_filtered[monthly_filtered.index.year.isin(years_set)]
    if months_set:
        monthly_filtered = monthly_filtered[monthly_filtered.index.month.isin(months_set)]
    if "premium" not in monthly_filtered.columns:
        st.warning("No premium history available for the selected filters.")
        return
    mask = monthly_filtered["premium"].notna()
    if not mask.any():
        st.warning("No premium history available for the selected filters.")
        return
    premium_series = monthly_filtered.loc[mask, "premium"]
    premium_series.index = monthly_filtered.index[mask]
    premium_series = premium_series.groupby(premium_series.index).sum().sort_index()
    if premium_series.empty:
        st.warning("No premium history available for the selected filters.")
        return
    raw_filtered = raw.copy()
    raw_filtered["policy_issue_date"] = pd.to_datetime(raw_filtered.get("policy_issue_date"), dayfirst=True, errors="coerce")
    raw_filtered["premium"] = pd.to_numeric(raw_filtered.get("premium"), errors="coerce")
    raw_filtered = raw_filtered.dropna(subset=["policy_issue_date", "premium"])
    if years_set:
        raw_filtered = raw_filtered[raw_filtered["policy_issue_date"].dt.year.isin(years_set)]
    if months_set:
        raw_filtered = raw_filtered[raw_filtered["policy_issue_date"].dt.month.isin(months_set)]
    if raw_filtered.empty:
        st.warning("No policy records match the selected filters.")
        return
    total_premium = float(premium_series.sum())
    avg_monthly = float(premium_series.mean())
    latest_value = float(premium_series.iloc[-1])
    if len(premium_series) >= 2:
        prev_value = premium_series.iloc[-2]
        change_pct = ((latest_value - prev_value) / prev_value * 100) if abs(prev_value) > 1e-9 else float("nan")
    else:
        change_pct = float("nan")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Premium", f"{total_premium:,.0f}")
    c2.metric("Avg Monthly Premium", f"{avg_monthly:,.0f}")
    delta_label = f"{change_pct:+.1f}%" if not np.isnan(change_pct) else "-"
    c3.metric("Latest Month Premium", f"{latest_value:,.0f}", delta_label)
    st.caption("Figures reflect the filtered dataset. Jump to Ingestion & EDA for deeper diagnostics.")
    st.markdown("**Premium Trend**")
    _plot_series(premium_series, "Premium Trend")
    charts = []
    if "product" in raw_filtered.columns:
        prod = raw_filtered.dropna(subset=["product"]).groupby("product", observed=False)["premium"].sum().sort_values(ascending=False)
        if not prod.empty:
            charts.append(("Premium by Product", prod.head(8), "product"))
    if "channel" in raw_filtered.columns:
        ch = raw_filtered.dropna(subset=["channel"]).groupby("channel", observed=False)["premium"].sum().sort_values(ascending=False)
        if not ch.empty:
            charts.append(("Premium by Channel", ch.head(8), "channel"))
    if "location" in raw_filtered.columns:
        loc = raw_filtered.dropna(subset=["location"]).groupby("location", observed=False)["premium"].sum().sort_values(ascending=False)
        if not loc.empty:
            charts.append(("Premium by Location", loc.head(10), "location"))
    if "benefit_period" in raw_filtered.columns:
        ben = raw_filtered.dropna(subset=["benefit_period"]).groupby("benefit_period", observed=False)["premium"].sum().sort_values(ascending=False)
        if not ben.empty:
            charts.append(("Premium by Benefit Period", ben.head(10), "benefit_period"))
    if "pins_gender" in raw_filtered.columns:
        gender = raw_filtered.dropna(subset=["pins_gender"]).groupby("pins_gender", observed=False)["premium"].sum().sort_values(ascending=False)
        if not gender.empty:
            charts.append(("Premium by Gender", gender, "pins_gender"))
    if "pin_age" in raw_filtered.columns:
        age_vals = pd.to_numeric(raw_filtered["pin_age"], errors="coerce")
        age_bins = [0, 25, 35, 45, 55, 65, 75, 200]
        age_labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
        age_bucket = pd.cut(age_vals, bins=age_bins, labels=age_labels, right=False)
        age_premium = raw_filtered.assign(age_bucket=age_bucket).groupby("age_bucket", observed=False)["premium"].sum().reindex(age_labels)
        if age_premium.notna().any():
            charts.append(("Premium by Age Band", age_premium.fillna(0), "age_bucket"))
    for title, series, name in charts:
        st.markdown(f"**{title}**")
        _plot_bar(series, title, name, orientation="h")
    insights = []
    recommendations = []
    if len(premium_series) >= 2 and not np.isnan(change_pct):
        insights.append(f"Latest month premium moved {change_pct:+.1f}% versus the prior month.")
    annual_series = premium_series.resample("YE").sum()
    if len(annual_series) >= 2:
        last_year = annual_series.index[-1].year
        prior_year = annual_series.index[-2].year
        annual_change = annual_series.iloc[-1] - annual_series.iloc[-2]
        base = annual_series.iloc[-2]
        annual_pct = (annual_change / base * 100) if abs(base) > 1e-9 else float("nan")
        insights.append(
            f"Year-to-date {last_year} premium totals {annual_series.iloc[-1]:,.0f} ({annual_pct:+.1f}% vs {prior_year})."
        )
        if not np.isnan(annual_pct) and annual_pct < 0:
            recommendations.append(
                "Investigate the drivers behind the year-over-year premium decline and adjust sales plans accordingly."
            )
    if charts:
        top_series = charts[0][1]
        if top_series.sum() > 0:
            top_name = top_series.index[0]
            share = top_series.iloc[0] / top_series.sum()
            insights.append(f"{top_name} contributes {share:.0%} of premium within its segment.")
    if not insights:
        insights.append("No notable trends detected for the selected filters.")
    if not recommendations:
        recommendations.append("Maintain the current strategy and monitor detailed EDA for emerging trends.")
    st.subheader("Insights")
    for item in insights:
        st.markdown(f"- {item}")
    st.subheader("Recommendations")
    for item in recommendations:
        st.markdown(f"- {item}")

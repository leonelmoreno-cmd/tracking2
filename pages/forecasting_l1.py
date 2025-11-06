# pages/forecasting_l1.py
import io
import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
import plotly.express as px
from prophet.plot import plot_components_plotly  # interactive components

# Page config
st.set_page_config(page_title="Forecasting L1", layout="wide",
                   initial_sidebar_state="expanded")

# ============================================================
#  HEADER
# ============================================================
def _header():
    st.title("Forecasting L1 – Variable: Traffic from Google - Weekly Forecast with Prophet")
    st.markdown(
        """
        Upload your weekly time series (CSV) → clean & validate → fit a Prophet model with US-holiday effects → forecast next N weeks (selectable) with confidence intervals.
        
        **Steps:**
        1. Upload CSV file (date + value columns).  
        2. Choose horizon (4, 8, 12, 16 weeks).  
        3. Choose handling for missing weeks (forward-fill / interpolate / warn).  
        4. Set model hyperparameters (changepoint_prior_scale).  
        5. Fit model → view forecast + components + diagnostics.  
        """
    )
    uploaded_file = st.file_uploader(
        "Upload CSV (must contain a date column and a numeric value column)",
        type=["csv", "tsv"]
    )
    return uploaded_file


# ============================================================
#  DATA VALIDATION
# ============================================================
def _runtime_checks(df: pd.DataFrame) -> bool:
    ok = True
    if df.empty:
        st.error("Uploaded file is empty.")
        return False

    if not {"ds", "y"}.issubset(df.columns):
        st.error("DataFrame must have columns named 'ds' (date) and 'y' (value).")
        return False

    if df.shape[0] < 30:
        st.warning(f"Only {df.shape[0]} weekly observations found. Forecast quality may be degraded.")

    df_idx = df.set_index("ds").sort_index()
    diffs = df_idx.index.to_series().diff().dropna()
    gaps = diffs.dt.days[df_idx.index.to_series().diff().dt.days > 10]
    if len(gaps) > 0:
        st.warning(
            f"Detected {len(gaps)} gaps larger than ~10 days between weekly dates: "
            f"{gaps.unique()[:3].tolist()} …"
        )
    return ok


# ============================================================
#  DATA PREPARATION
# ============================================================
def _prepare_data(df_raw: pd.DataFrame, missing_method: str = "warn") -> pd.DataFrame:
    df = df_raw.copy()

    # Ensure at least two columns
    if len(df.columns) < 2:
        st.error(
            f"Expected at least 2 columns (date + value) but found {len[df.columns]}. "
            "Please upload a file with a date column and a value column."
        )
        st.stop()

    # Take only first two columns as (date, value)
    df = df.iloc[:, :2].copy()
    df.columns = ["ds", "y"]

    # Types
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # Clean
    df = df.dropna(subset=["ds", "y"])
    df = df.sort_values("ds").reset_index(drop=True)

    # Missing handling
    if missing_method == "forward-fill":
        df["y"] = df["y"].ffill()
    elif missing_method == "interpolate":
        df["y"] = df["y"].interpolate()
    else:
        if df["y"].isna().any():
            st.warning(
                "Missing values detected after loading. "
                "Consider using forward-fill or interpolation method."
            )

    df = df.dropna(subset=["y"]).reset_index(drop=True)
    return df


# ============================================================
#  MODEL FITTING & FORECAST
# ============================================================
def _fit_model(df: pd.DataFrame, changepoint_prior_scale: float, interval_width: float = 0.95) -> Prophet:
    m = Prophet(
        interval_width=interval_width,
        weekly_seasonality=False,   # weekly disabled for weekly data
        yearly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
    )
    m.add_country_holidays(country_name="US")
    m.fit(df)
    return m


def _make_future_and_predict(m: Prophet, freq: str, periods: int):
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return future, forecast


# ============================================================
#  PLOTS
# ============================================================
def _plot_forecast_interactive(df: pd.DataFrame, forecast: pd.DataFrame):
    fig = go.Figure()

    # Confidence band
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"], y=forecast["yhat_lower"], mode="lines",
            line=dict(width=0), name="Lower bound", hovertemplate="Lower: %{y:.2f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"], y=forecast["yhat_upper"], mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor="rgba(0,0,0,0.15)",
            name="Confidence interval", hovertemplate="Upper: %{y:.2f}<extra></extra>",
        )
    )

    # Forecast line
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast (yhat)",
            hovertemplate="Date: %{x}<br>Forecast: %{y:.2f}<extra></extra>",
        )
    )

    # Actuals
    fig.add_trace(
        go.Scatter(
            x=df["ds"], y=df["y"], mode="markers+lines", name="Actual",
            hovertemplate="Date: %{x}<br>Actual: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Forecast vs Actual (with confidence band)",
        xaxis_title="Date", yaxis_title="Value", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")


def _plot_forecast_future_only(df: pd.DataFrame, forecast: pd.DataFrame):
    """Future only forecast with band; marks last observed date."""
    last_ds = pd.to_datetime(df["ds"].max())
    fc = forecast[forecast["ds"] > last_ds].copy()
    if fc.empty:
        st.info("No future points found in the forecast (future dataframe is empty).")
        return

    fig = go.Figure()

    # Confidence band (future)
    fig.add_trace(
        go.Scatter(
            x=fc["ds"], y=fc["yhat_lower"], mode="lines", line=dict(width=0),
            name="Lower bound", hovertemplate="Lower: %{y:.2f}<extra></extra>", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc["ds"], y=fc["yhat_upper"], mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(0,0,0,0.15)", name="Confidence interval",
            hovertemplate="Upper: %{y:.2f}<extra></extra>",
        )
    )

    # Forecast line (future only)
    fig.add_trace(
        go.Scatter(
            x=fc["ds"], y=fc["yhat"], mode="lines", name="Forecast (yhat)",
            hovertemplate="Date: %{x}<br>Forecast: %{y:.2f}<extra></extra>",
        )
    )

    # Last observed marker & vline
    last_row = df.loc[df["ds"] == last_ds]
    if not last_row.empty:
        fig.add_trace(
            go.Scatter(
                x=last_row["ds"], y=last_row["y"], mode="markers", name="Last actual",
                marker=dict(size=9),
                hovertemplate="Date: %{x}<br>Actual: %{y:.2f}<extra></extra>",
            )
        )
    x_line = last_ds.to_pydatetime()
    fig.add_vline(x=x_line, line_dash="dot", line_color="gray")
    fig.add_annotation(x=x_line, y=1, yref="paper", text="Last actual",
                       showarrow=False, xanchor="left")

    fig.update_layout(
        title="Future Forecast (from last actual onward)",
        xaxis_title="Date", yaxis_title="Value", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")


def _plot_components(m: Prophet, forecast: pd.DataFrame):
    """Show Prophet seasonality/trend components (Plotly, interactive)."""
    fig = plot_components_plotly(m, forecast)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
#  CV HELPERS & QUALITY GATE
# ============================================================
def _compute_cv_metrics(m: Prophet, df: pd.DataFrame, horizon_weeks: int):
    """Run CV with guardrails. Return ((df_p, df_cv), message)."""
    n_weeks = df.shape[0]
    if n_weeks < horizon_weeks * 3:
        return None, f"Not enough history for cross-validation (need ~{horizon_weeks*3}, have {n_weeks})."

    initial_weeks = max(int(n_weeks * 0.6), horizon_weeks * 2)
    max_initial = max(n_weeks - horizon_weeks - 1, 1)
    initial_weeks = min(initial_weeks, max_initial)
    if initial_weeks <= 0 or (initial_weeks + horizon_weeks) >= n_weeks:
        return None, ("Cross-validation skipped: insufficient span after adjusting windows "
                      f"(n={n_weeks}, horizon={horizon_weeks}, initial={initial_weeks}).")

    period_weeks = max(horizon_weeks // 2, 1)
    initial = f"{initial_weeks}W"
    period = f"{period_weeks}W"
    horizon = f"{horizon_weeks}W"

    try:
        df_cv = cross_validation(
            m, initial=initial, period=period, horizon=horizon, parallel="processes"
        )
        df_p = performance_metrics(df_cv)
        return (df_p, df_cv), None
    except Exception as e:
        return None, f"Cross-validation failed: {e}"


def _quality_gate_rmse(df_p: pd.DataFrame, df_hist: pd.DataFrame,
                       horizon_weeks: int, rmse_ratio_threshold: float = 0.35):
    """
    Return (ok, msg, rmse_val, median_y).
    ok=False if RMSE > threshold * median(y) at the row whose horizon is closest to target.
    """
    median_y = float(df_hist["y"].median())
    df_tmp = df_p.copy()
    df_tmp["horizon_td"] = pd.to_timedelta(df_tmp["horizon"])
    target = pd.Timedelta(weeks=horizon_weeks)
    idx = (df_tmp["horizon_td"] - target).abs().idxmin()
    rmse_val = float(df_tmp.loc[idx, "rmse"])

    limit = rmse_ratio_threshold * median_y
    if rmse_val > limit:
        msg = (f"Forecast hidden: RMSE={rmse_val:.3f} is greater than "
               f"{rmse_ratio_threshold:.0%} of median(y)={median_y:.3f} "
               f"(limit={limit:.3f}).")
        return False, msg, rmse_val, median_y
    return True, "", rmse_val, median_y


# ---------- Plotly CV metric helper ----------
def _plot_cv_metric_plotly(df_p: pd.DataFrame, metric: str = "rmse"):
    """Interactive Plotly version of the Prophet CV metric plot."""
    df_plot = df_p.copy()
    df_plot["horizon_days"] = pd.to_timedelta(df_plot["horizon"]).dt.days
    fig = px.line(
        df_plot,
        x="horizon_days",
        y=metric,
        markers=True,
        title=f"Cross-Validation Performance – {metric.upper()} vs Horizon (days)",
        labels={"horizon_days": "Horizon (days)", metric: metric.upper()},
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(hovermode="x unified", height=450)
    return fig


# ============================================================
#  DIAGNOSTICS (robust)
# ============================================================
def _diagnostics_section(m: Prophet, df: pd.DataFrame, horizon_weeks: int, df_p=None, df_cv=None):
    st.subheader("Model Diagnostics")
    st.markdown("This section shows cross-validation and residuals to evaluate forecast performance.")

    # Residuals on fitted history
    hist = df.copy()[["ds", "y"]]
    forecast_hist = m.predict(hist[["ds"]])
    hist = hist.merge(forecast_hist[["ds", "yhat"]], how="left", on="ds")
    hist["residual"] = hist["y"] - hist["yhat"]

    fig_res = px.histogram(hist, x="residual", title="Residual Distribution")
    st.plotly_chart(fig_res, width="stretch")

    fig_scatter = px.scatter(hist, x="yhat", y="residual", title="Residual vs Forecast")
    st.plotly_chart(fig_scatter, width="stretch")

    # Cross-validation (reuse if provided)
    if df_p is None or df_cv is None:
        n_weeks = df.shape[0]
        if n_weeks < horizon_weeks * 3:
            st.info(
                f"Not enough history for cross-validation (need at least ~{horizon_weeks*3} weeks, have {n_weeks})."
            )
            return

        initial_weeks = max(int(n_weeks * 0.6), horizon_weeks * 2)
        max_initial = max(n_weeks - horizon_weeks - 1, 1)
        initial_weeks = min(initial_weeks, max_initial)

        if initial_weeks <= 0 or (initial_weeks + horizon_weeks) >= n_weeks:
            st.info(
                "Cross-validation skipped: insufficient span after adjusting the initial window. "
                f"(n={n_weeks}, horizon={horizon_weeks}, initial={initial_weeks})"
            )
            return

        period_weeks = max(horizon_weeks // 2, 1)
        initial = f"{initial_weeks}W"
        period = f"{period_weeks}W"
        horizon = f"{horizon_weeks}W"

        st.markdown(f"### Cross-validation (initial={initial}, period={period}, horizon={horizon})")

        try:
            with st.spinner("Running cross validation (this may take some time)…"):
                df_cv = cross_validation(
                    m,
                    initial=initial,
                    period=period,
                    horizon=horizon,
                    parallel="processes",
                )
                df_p = performance_metrics(df_cv)
        except ValueError as e:
            st.info(f"Cross-validation skipped due to window constraints: {e}")
            return
        except Exception as e:
            st.warning(f"Cross-validation failed: {e}")
            return
    else:
        st.markdown("### Cross-validation (precomputed)")

    st.dataframe(df_p, width="stretch")

    # Plotly, metric-selectable CV plot
    metric_choice = st.selectbox(
        "Select CV metric to visualize",
        options=["rmse", "mae", "mape", "coverage", "mse", "mdape", "smape"],
        index=0,
    )
    try:
        fig_cv = _plot_cv_metric_plotly(df_p, metric=metric_choice)
        st.plotly_chart(fig_cv, use_container_width=True)
    except Exception as e:
        st.info(f"Could not render CV plot: {e}")


# ============================================================
#  MAIN
# ============================================================
def main():
    uploaded_file = _header()
    if uploaded_file is None:
        st.info("Please upload a CSV file to start forecasting.")
        return

    # Robust Google Trends CSV Reader
    try:
        raw_bytes = uploaded_file.read()
        text = raw_bytes.decode("utf-8-sig", errors="replace")

        # Remove empty lines
        lines = [ln for ln in text.splitlines() if ln.strip() != ""]

        # Drop metadata like "Category: All categories"
        if lines and lines[0].strip().lower().startswith("category:"):
            lines = lines[1:]

        # Find header line that starts with "Week"
        header_idx = 0
        for i, ln in enumerate(lines):
            if ln.strip().lower().startswith("week"):
                header_idx = i
                break

        # Rebuild from header line
        content = "\n".join(lines[header_idx:])

        # Autodetect separator (tab/comma/semicolon)
        raw = pd.read_csv(io.StringIO(content), sep=None, engine="python")

    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        st.stop()

    if raw.shape[1] < 2:
        st.error("Expected at least 2 columns (date + value) but found 1. Please check your CSV file.")
        st.stop()

    # Sidebar
    st.sidebar.header("Forecasting settings")
    horizon_weeks = st.sidebar.selectbox("Forecast horizon (weeks)", options=[4, 8, 12, 16], index=2)
    missing_method = st.sidebar.selectbox("Missing-week handling", options=["warn", "forward-fill", "interpolate"], index=0)
    changepoint_prior_scale = st.sidebar.slider(
        "Changepoint prior scale (trend flexibility)", 0.001, 0.5, 0.1, 0.01
    )
    interval_width = st.sidebar.slider("Prediction interval width", 0.80, 0.99, 0.95, 0.01)

    # Prepare Data
    try:
        df_prepared = _prepare_data(raw, missing_method=missing_method)
    except Exception as e:
        st.error(f"Data preparation failed: {e}")
        st.stop()

    if not _runtime_checks(df_prepared):
        st.stop()

    # Fit Model
    with st.spinner("Fitting Prophet model…"):
        model = _fit_model(df_prepared, changepoint_prior_scale, interval_width)

    # ---- Quality gate: CV -> RMSE vs median(y) rule (35%) ----
    with st.spinner("Evaluating model quality (cross-validation)…"):
        cv_result, cv_msg = _compute_cv_metrics(model, df_prepared, horizon_weeks)

    if cv_result is None:
        st.info(cv_msg)
        st.stop()

    df_p, df_cv = cv_result
    ok, reason, rmse_val, median_y = _quality_gate_rmse(
        df_p, df_prepared, horizon_weeks, rmse_ratio_threshold=0.35
    )

    if not ok:
        st.error(reason)
        st.caption("Tip: Provide more history, reduce volatility, tune changepoint_prior_scale, "
                   "or adjust seasonality/holidays.")
        st.write("Cross-validation metrics:")
        st.dataframe(df_p, width="stretch")
        # Show Plotly RMSE curve for transparency
        try:
            st.plotly_chart(_plot_cv_metric_plotly(df_p, metric="rmse"), use_container_width=True)
        except Exception:
            pass
        return  # block display below this line

    # ---- Passed quality gate → proceed to forecast & plots ----
    freq = "W-MON"  # keep weekly Monday future grid for consistency
    with st.spinner(f"Forecasting next {horizon_weeks} weeks…"):
        future, forecast = _make_future_and_predict(model, freq=freq, periods=horizon_weeks)

    _plot_forecast_interactive(df_prepared, forecast)   # history + future
    _plot_forecast_future_only(df_prepared, forecast)   # future only
    _plot_components(model, forecast)

    # Download
    csv_out = forecast.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full forecast CSV",
        csv_out,
        file_name=f"forecast_prophet_{horizon_weeks}w.csv",
        mime="text/csv"
    )

    # Diagnostics (reuse CV results)
    _diagnostics_section(model, df_prepared, horizon_weeks, df_p=df_p, df_cv=df_cv)


if __name__ == "__main__":
    main()

# pages/forecasting_l1.py
import io
import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(page_title="Forecasting L1", layout="wide",
                   initial_sidebar_state="expanded")

def _header():
    st.title("Forecasting L1 – Weekly Forecast with Prophet")
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
    uploaded_file = st.file_uploader("Upload CSV (must contain a date column and a numeric value column)", type=["csv", "tsv"])
    return uploaded_file

def _runtime_checks(df: pd.DataFrame) -> bool:
    """
    Perform runtime data-quality checks:
    - date column parseable, value numeric
    - at least minimal number of points (say 30 weeks)
    - check for consistent weekly frequency / missing weeks
    - check for outliers (optionally)
    Returns True if passes basic checks, else False (and displays warnings).
    """
    ok = True
    if df.empty:
        st.error("Uploaded file is empty.")
        return False

    # Check ds and y columns
    if not {"ds", "y"}.issubset(df.columns):
        st.error("DataFrame must have columns named 'ds' (date) and 'y' (value).")
        return False

    # Check minimum length
    if df.shape[0] < 30:
        st.warning(f"Only {df.shape[0]} weekly observations found. Forecast quality may be degraded.")

    # Check for weekly frequency gaps
    df_idx = df.set_index("ds").sort_index()
    diffs = df_idx.index.to_series().diff().dropna()
    # convert to days
    gaps = diffs.dt.days[df_idx.index.to_series().diff().dt.days > 10]
    if len(gaps) > 0:
        st.warning(f"Detected {len(gaps)} gaps larger than ~10 days between weekly dates: {gaps.unique()[:3].tolist()} …")
    return ok

def _prepare_data(df_raw: pd.DataFrame, missing_method: str = "warn") -> pd.DataFrame:
    import streamlit as st  # asegúrate que esté disponible
    
    df = df_raw.copy()
    
    # Ensure we have at least two columns (date + value)
    if len(df.columns) < 2:
        st.error(f"Expected at least 2 columns (date + value) but found {len(df.columns)}. Please upload a file with a date column and a value column.")
        st.stop()
    
    # Select only the first two columns and rename them
    df = df.iloc[:, :2].copy()
    df.columns = ["ds", "y"]
    
    # Convert types
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    
    # Drop rows with missing date or value
    df = df.dropna(subset=["ds", "y"])
    df = df.sort_values("ds").reset_index(drop=True)
    
    # Resample to weekly frequency (Monday)
    df = df.set_index("ds").resample("W-MON").mean().reset_index()
    
    # Handle missing weeks according to method
    if missing_method == "forward‐fill":
        df["y"] = df["y"].ffill()
    elif missing_method == "interpolate":
        df["y"] = df["y"].interpolate()
    else:
        # missing_method == "warn"
        if df["y"].isna().any():
            st.warning("Missing values detected after weekly resampling. Consider using forward-fill or interpolation method.")
    
    # Drop remaining NaNs in y
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    return df


def _fit_model(df: pd.DataFrame, changepoint_prior_scale: float, interval_width: float=0.95) -> Prophet:
    m = Prophet(interval_width=interval_width,
                weekly_seasonality=True,
                yearly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=changepoint_prior_scale)
    m.add_country_holidays(country_name='US')
    # Optionally: add custom monthly seasonality, extra regressors etc.
    m.fit(df)
    return m

def _make_future_and_predict(m: Prophet, freq: str, periods: int) -> pd.DataFrame:
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return future, forecast

def _plot_forecast_interactive(df: pd.DataFrame, forecast: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast (yhat)'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower bound', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper bound', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Actual', mode='markers+lines'))
    fig.update_layout(title='Forecast vs Actual', xaxis_title='Date', yaxis_title='Value', height=500)
    st.plotly_chart(fig, use_container_width=True)

def _plot_components(m: Prophet, forecast: pd.DataFrame):
    fig_comp = m.plot_components(forecast)
    st.pyplot(fig_comp)

def _diagnostics_section(m: Prophet, df: pd.DataFrame, horizon_weeks: int):
    st.subheader("Model Diagnostics")
    st.markdown("This section shows cross-validation and residuals to evaluate forecast performance.")

    # Compute residuals on fitted history
    hist = df.copy().rename(columns={'ds':'ds','y':'y'})
    forecast_hist = m.predict(hist[['ds']])
    hist = hist.merge(forecast_hist[['ds','yhat']], how='left', on='ds')
    hist['residual'] = hist['y'] - hist['yhat']

    fig_res = px.histogram(hist, x='residual', title='Residual Distribution')
    st.plotly_chart(fig_res, use_container_width=True)

    fig_scatter = px.scatter(hist, x='yhat', y='residual', title='Residual vs Forecast')
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Cross-validation (only if enough history)
    if df.shape[0] >= (horizon_weeks * 3):  # heuristic: at least 3× horizon
        st.markdown("### Cross-validation for horizon = {horizon_weeks} weeks".format(horizon_weeks=horizon_weeks))
        initial = f"{int(df.shape[0] - horizon_weeks)}W"
        period = f"{int(horizon_weeks//2)}W" if horizon_weeks//2 >=1 else "1W"
        horizon = f"{horizon_weeks}W"
        with st.spinner("Running cross validation (this may take some time)…"):
            df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon, parallel="processes")
            df_p = performance_metrics(df_cv)
        st.dataframe(df_p, use_container_width=True)
        fig_cv = plot_cross_validation_metric(df_cv, metric='rmse')
        st.pyplot(fig_cv)
    else:
        st.info(f"Not enough history for cross-validation (need at least ~{horizon_weeks*3} weeks, have {df.shape[0]}).")

def main():
    uploaded_file = _header()
    if uploaded_file is None:
        st.info("Please upload a CSV file to start forecasting.")
        return

    try:
        raw = pd.read_csv(io.BytesIO(uploaded_file.read()))
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        st.stop()

    st.subheader("Raw Data Preview")
    st.dataframe(raw.head(), use_container_width=True)

    # Sidebar controls
    st.sidebar.header("Forecasting settings")
    horizon_weeks = st.sidebar.selectbox("Forecast horizon (weeks)", options=[4,8,12,16], index=2)
    missing_method = st.sidebar.selectbox("Missing-week handling", options=["warn","forward-fill","interpolate"], index=0)
    changepoint_prior_scale = st.sidebar.slider("Changepoint prior scale (trend flexibility)", min_value=0.001, max_value=0.5, value=0.1, step=0.01)
    interval_width = st.sidebar.slider("Prediction interval width", min_value=0.80, max_value=0.99, value=0.95, step=0.01)

    # Prepare data
    try:
        df_prepared = _prepare_data(raw, missing_method=missing_method)
    except Exception as e:
        st.error(f"Data preparation failed: {e}")
        st.stop()

    if not _runtime_checks(df_prepared):
        st.stop()

    st.subheader("Prepared Weekly Data")
    st.dataframe(df_prepared.head(), use_container_width=True)

    # Fit model
    with st.spinner("Fitting Prophet model…"):
        model = _fit_model(df_prepared, changepoint_prior_scale=changepoint_prior_scale,
                           interval_width=interval_width)

    # Forecast
    freq = "W-MON"  # weekly Monday
    with st.spinner(f"Forecasting next {horizon_weeks} weeks…"):
        future, forecast = _make_future_and_predict(model, freq=freq, periods=horizon_weeks)

    st.subheader("Forecast Results (next {horizon_weeks} weeks)".format(horizon_weeks=horizon_weeks))
    st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(horizon_weeks),
                 use_container_width=True)

    # Plots
    _plot_forecast_interactive(df_prepared, forecast)
    _plot_components(model, forecast)

    # Download forecast
    csv_out = forecast.to_csv(index=False).encode('utf-8')
    st.download_button("Download full forecast CSV", csv_out,
                       file_name=f"forecast_prophet_{horizon_weeks}w.csv", mime="text/csv")

    # Diagnostics
    _diagnostics_section(model, df_prepared, horizon_weeks)

if __name__ == "__main__":
    main()

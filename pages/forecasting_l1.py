# pages/forecasting_l1.py
import io
import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
import plotly.express as px

# Page config (optional: you might already set at top)
st.set_page_config(page_title="Forecasting L1", layout="wide")

def _header():
    st.title("Forecasting L1 – Weekly Forecast with Prophet")
    st.caption("Upload your weekly data (CSV with date + value) → model via Prophet → forecast next 12 weeks with confidence interval and holiday effects (USA).")

    uploaded_file = st.file_uploader("Upload CSV (must contain a date column and a value column)", type=["csv", "tsv"])
    st.markdown("**Expected CSV format:**\n- A column for date (any name) that can be parsed to datetime\n- A column for the numeric value to forecast\n- Data must be at weekly frequency or convertible to weekly aggregated series")
    return uploaded_file

def _validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    # 1. infer date column & value column (you may ask user to pick)
    # For simplicity we assume first column is date, second is value
    df = df.copy()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])
    df = df.sort_values("ds")
    # Check frequency: weekly
    # Resample if necessary
    df = df.set_index("ds").resample("W-MON").mean().reset_index()  # Monday weekly
    return df

def _fit_prophet(df: pd.DataFrame, interval_width: float=0.95) -> Prophet:
    m = Prophet(interval_width=interval_width,
                weekly_seasonality=True,
                yearly_seasonality=True,
                daily_seasonality=False)
    m.add_country_holidays(country_name='US')
    # You can optionally add monthly or custom seasonalities if you think helpful:
    # m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.fit(df)
    return m

def _make_future_and_predict(m: Prophet, periods: int = 12, freq: str='W-MON') -> pd.DataFrame:
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return forecast

def _plot_forecast(df: pd.DataFrame, forecast: pd.DataFrame):
    fig = m.plot(forecast)  # note: returns a matplotlib figure
    st.pyplot(fig)

    # but maybe better to use Plotly for interactive
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
    fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower bound', line=dict(dash='dash')))
    fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper bound', line=dict(dash='dash')))
    fig2.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Actual', mode='markers+lines'))
    fig2.update_layout(title='Forecast vs Actual', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig2, use_container_width=True)

    # Plot components
    fig_comp = m.plot_components(forecast)
    st.pyplot(fig_comp)

def main():
    uploaded_file = _header()
    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            df_raw = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")
            st.stop()

        st.write("Raw data preview:")
        st.dataframe(df_raw.head(), use_container_width=True)

        # Validate & prep
        try:
            df = _validate_and_prepare(df_raw)
        except Exception as e:
            st.error(f"Data preparation failed: {e}")
            st.stop()

        st.write("Prepared weekly data (after resampling):")
        st.dataframe(df.head(), use_container_width=True)

        if df.shape[0] < 30:
            st.warning("Warning: less than ~30 weekly points available. Forecast quality may be degraded.")

        # Fit model
        with st.spinner("Fitting Prophet model..."):
            m = _fit_prophet(df)

        # Forecast next 12 weeks
        forecast = _make_future_and_predict(m, periods=12, freq='W-MON')

        st.subheader("Forecast results (next 12 weeks)")
        st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(12), use_container_width=True)

        # Plot results
        _plot_forecast(df, forecast)

        # Download option
        csv = forecast.to_csv(index=False).encode('utf-8')
        st.download_button("Download forecast CSV", csv, file_name="forecast_prophet_12weeks.csv", mime="text/csv")

        # Optional: Cross‐validation metrics
        with st.expander("Show cross-validation metrics"):
            st.write("Running cross-validation (initial ~80%, horizon 12 weeks)...")
            df_cv = cross_validation(m, initial=f'{int(len(df)*0.8)}W', period='1W', horizon='12W')
            df_p = performance_metrics(df_cv)
            st.write(df_p)
            fig_cv = plot_cross_validation_metric(df_cv, metric='rmse')
            st.pyplot(fig_cv)

    else:
        st.info("Please upload a CSV file to start forecasting.")


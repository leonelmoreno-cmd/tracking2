# pages/traffic.py
#!/usr/bin/env python
# coding: utf-8

import io
import re
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Page config ----------
st.set_page_config(page_title="Demand Analysis JC", layout="wide")
st.title("Demand Analysis JC")
st.caption("Google Trends (US, last 5y, en-US) → STL (LOESS) → Plotly → Better decisions")

# --- Guard: urllib3<2 recomendado para pytrends 4.9.2 ---
try:
    import urllib3
    from packaging import version
    if version.parse(urllib3.__version__) >= version.parse("2.0.0"):
        st.error(
            "Incompatible urllib3 version detected "
            f"({urllib3.__version__}). Please pin urllib3<2 in requirements.txt."
        )
        st.stop()
except Exception:
    pass

from pytrends.request import TrendReq
from statsmodels.tsa.seasonal import STL
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


# ---------- UI + top matter ----------
def _header():
    kw = st.text_input("Keyword (required for Request mode)", value="", placeholder="e.g., rocket stove")
    col_req, col_csv = st.columns(2)
    request_clicked = col_req.button("Request")
    uploaded_file = col_csv.file_uploader("Choose Google Trends CSV", type=["csv", "tsv"])
    upload_clicked = col_csv.button("Upload CSV")
    return kw, request_clicked, uploaded_file, upload_clicked


# ---------- Config ----------
HL = "en-US"
TZ = 360
TIMEFRAME = "today 5-y"
GEO = "US"


# ---------- Helpers ----------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_trends(keyword: str) -> pd.DataFrame:
    pytrends = TrendReq(
        hl=HL,
        tz=TZ,
        timeout=(10, 25),
        retries=2,
        backoff_factor=0.1,
    )
    pytrends.build_payload([keyword], timeframe=TIMEFRAME, geo=GEO)
    df = pytrends.interest_over_time()
    if df.empty:
        return df
    if len(df) > 0:
        df = df.iloc[:-1]
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def infer_period(dt_index: pd.DatetimeIndex) -> int:
    if len(dt_index) < 3:
        return 12
    deltas = np.diff(dt_index.values).astype("timedelta64[D]").astype(int)
    med = int(np.median(deltas))
    if med <= 1:
        return 7
    elif med <= 7:
        return 52
    else:
        return 12


def build_figure(df_plot: pd.DataFrame, title_kw: str) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=("Original", "Trend", "Seasonal", "Residual")
    )
    fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["original"], name="Original", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["trend"],   name="Trend",   mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["seasonal"],name="Seasonal",mode="lines"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["remainder"],name="Residual",mode="lines"), row=4, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)
    fig.update_layout(height=900, title_text=f"STL Decomposition — {title_kw} — Google Trends (US, last 5y)")
    return fig


def _looks_like_header(cols: list[str]) -> bool:
    if not cols or len(cols) < 2:
        return False
    return cols[0].strip().lower() in {"week", "semana", "date", "fecha"}


def _clean_keyword_label(raw: str) -> str:
    label = raw.strip()
    label = re.sub(r":\s*$", "", label)
    label = re.sub(r":\s*\(", " (", label)
    return label


def parse_trends_csv(file_bytes: bytes) -> tuple[pd.DataFrame, str]:
    text = file_bytes.decode("utf-8-sig", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    header_idx = -1

    for i, ln in enumerate(lines):
        for sep in [",", ";", "\t"]:
            parts = [p.strip() for p in ln.split(sep)]
            if len(parts) >= 2 and _looks_like_header(parts):
                header_idx = i
                break
        if header_idx != -1:
            break

    if header_idx == -1:
        raise ValueError("Could not locate a header row (expected columns like Week/Semana/Date/Fecha).")

    content = "\n".join(lines[header_idx:])
    df_raw = pd.read_csv(io.StringIO(content), sep=None, engine="python")

    date_candidates = [c for c in df_raw.columns if c.strip().lower() in {"week", "semana", "date", "fecha"}]
    if not date_candidates:
        raise ValueError("Date column not found (looking for Week/Semana/Date/Fecha).")
    date_col = date_candidates[0]

    value_cols = [c for c in df_raw.columns if c != date_col]
    if not value_cols:
        raise ValueError("Value column not found (expected a keyword column).")

    chosen = None
    for col in value_cols:
        non_na = pd.to_numeric(df_raw[col], errors="coerce")
        if non_na.notna().sum() > 0:
            chosen = col
            break
    chosen = chosen or value_cols[0]

    series_label = _clean_keyword_label(str(chosen))

    df = pd.DataFrame({
        "date": pd.to_datetime(df_raw[date_col], errors="coerce"),
        series_label: pd.to_numeric(df_raw[chosen], errors="coerce"),
    }).dropna(subset=["date"]).sort_values("date")

    df = df.dropna(subset=[series_label]).set_index("date")
    return df, series_label


def run_stl_pipeline(df: pd.DataFrame, series_name: str):
    if df.empty:
        st.warning("No data available after parsing. Please verify the file or keyword.")
        st.stop()
    if series_name not in df.columns:
        st.error(f"Column '{series_name}' not found in the dataset.")
        st.stop()

    y = df[series_name].astype(float)
    if y.size < 10:
        st.error("Not enough points for STL decomposition. Provide more history.")
        st.stop()

    period = infer_period(y.index)
    try:
        res = STL(y, period=period, robust=True).fit()
    except Exception as e:
        st.error(f"STL decomposition failed: {e}")
        st.stop()

    df_plot = pd.DataFrame({
        "date": y.index,
        "original": y.values,
        "trend": res.trend,
        "seasonal": res.seasonal,
        "remainder": res.resid
    })

    # ---- Main decomposition figure ----
    fig = build_figure(df_plot, series_name)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "*How to read this:* the **Trend** shows the long-term direction, **Seasonal** the recurring intra-year pattern, "
        "and **Residual** the short-term irregular component (noise/outliers). Values are relative Google Trends interest."
    )

    with st.expander("Show data (original/trend/seasonal/residual)"):
        st.dataframe(df_plot, use_container_width=True)
        st.download_button(
            "Download CSV",
            df_plot.to_csv(index=False).encode("utf-8"),
            file_name=f"stl_{series_name.strip().replace(' ','_')}.csv",
            mime="text/csv"
        )

    # =========================
    # Additional analyses
    # =========================
    st.markdown("## Additional seasonal analyses")

    # ---- Chart 2: Average seasonal pattern by ISO week ----
    semana_mean = (
        df_plot.assign(iso_week=df_plot["date"].dt.isocalendar().week.astype(int))
               .groupby("iso_week", as_index=False)["seasonal"].mean()
               .rename(columns={"seasonal": "mean_seasonal"})
               .sort_values("iso_week")
               .reset_index(drop=True)
    )
    fig_week = px.line(
        semana_mean, x="iso_week", y="mean_seasonal",
        title="Average seasonal pattern by ISO week (1–53)"
    )
    fig_week.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_week.update_xaxes(dtick=4, title="ISO week (1–53)")
    fig_week.update_yaxes(title="Seasonal value")
    st.plotly_chart(fig_week, use_container_width=True)
    st.markdown(
        "*Interpretation:* this line shows the **average seasonal effect** at each ISO week across all years. "
        "Peaks indicate weeks that are typically above the baseline; troughs indicate below-baseline weeks."
    )

    # ---- Chart 3: Average seasonal pattern by month ----
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_map = dict(zip(range(1, 13), month_labels))
    mes_mean = (
        df_plot.assign(month_num=df_plot["date"].dt.month)
               .groupby("month_num", as_index=False)["seasonal"].mean()
               .rename(columns={"seasonal": "mean_seasonal"})
    )
    mes_mean["month_lab"] = mes_mean["month_num"].map(month_map)
    fig_month_mean = px.line(
        mes_mean, x="month_lab", y="mean_seasonal", markers=True,
        title="Average seasonal pattern by month"
    )
    fig_month_mean.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_month_mean.update_xaxes(title="Month")
    fig_month_mean.update_yaxes(title="Seasonal value")
    st.plotly_chart(fig_month_mean, use_container_width=True)
    st.markdown(
        "*Interpretation:* this summarizes the **typical monthly seasonality**. "
        "Use it to spot which months are usually stronger or weaker relative to the yearly baseline."
    )

    # ---- Chart 4: Seasonal distribution by month (box plot) ----
    df_box = df_plot.assign(
        month_num=df_plot["date"].dt.month,
        month_lab=lambda d: d["month_num"].map(month_map)
    )
    fig_box = px.box(
        df_box, x="month_lab", y="seasonal",
        title="Seasonal distribution by month"
    )
    fig_box.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_box.update_xaxes(title="Month")
    fig_box.update_yaxes(title="Seasonal value")
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown(
        "*Interpretation:* boxes show the **spread of seasonal values** for each month across years. "
        "Wider boxes or longer whiskers mean more variability; outliers capture unusual months."
    )

    # ---- Chart 5: Year-over-year seasonality comparison (last 3–4 years) ----
    df_plot = df_plot.copy()
    df_plot["year"] = df_plot["date"].dt.year
    df_plot["iso_week"] = df_plot["date"].dt.isocalendar().week.astype(int)

    max_year = int(df_plot["year"].max())
    start_year = max(max_year - 3, int(df_plot["year"].min()))
    df_last_years = df_plot.query("@start_year <= year <= @max_year").copy()

    fig_yoy = px.line(
        df_last_years, x="iso_week", y="seasonal", color="year",
        title=f"Seasonality compared by year ({start_year}–{max_year})"
    )
    fig_yoy.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_yoy.update_xaxes(title="ISO week", dtick=4)
    fig_yoy.update_yaxes(title="Seasonal value")
    st.plotly_chart(fig_yoy, use_container_width=True)
    st.markdown(
        "*Interpretation:* this compares **seasonal curves across recent years** on the same ISO-week axis. "
        "Look for alignment (stable seasonality) or divergences (shifts in timing or magnitude)."
    )


def _run_request_mode(kw: str):
    if not kw.strip():
        st.error("Please enter a keyword to use Request mode.")
        st.stop()
    with st.spinner("Fetching Google Trends…"):
        df = fetch_trends(kw.strip())
    if df.empty:
        st.warning("No data returned by Google Trends for this keyword/timeframe/geo.")
        st.stop()
    col_name = kw.strip()
    if col_name not in df.columns:
        st.error(f"Column '{col_name}' not found in Trends result.")
        st.stop()
    run_stl_pipeline(df, col_name)


def _run_upload_mode(uploaded_file):
    if uploaded_file is None:
        st.error("Please choose a CSV file exported from Google Trends.")
        st.stop()
    file_bytes = uploaded_file.read()
    df_csv, series_label = parse_trends_csv(file_bytes)
    run_stl_pipeline(df_csv, series_label)


# ---------- Entry point ----------
def main():
    kw, request_clicked, uploaded_file, upload_clicked = _header()
    if request_clicked:
        _run_request_mode(kw)
    elif upload_clicked:
        _run_upload_mode(uploaded_file)

    st.markdown(
        """
        <small>
        Data source: Google Trends via <code>pytrends</code> • Decomposition: <code>statsmodels.STL</code> • Charts: Plotly • Host: Streamlit Community Cloud
        </small>
        """,
        unsafe_allow_html=True
    )

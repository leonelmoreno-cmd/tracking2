import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Competitors Price Tracker",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# -------------------------------
# Data loading
# -------------------------------
@st.cache_data
def fetch_data():
    url = "https://raw.githubusercontent.com/leonelmoreno-cmd/tracking2/main/data/synthethic3.csv"
    df = pd.read_csv(url)
    return df

@st.cache_data
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["week_number"] = df["date"].dt.isocalendar().week
    df = df.sort_values(by=["asin", "week_number"])
    df["discount"] = df.apply(
        lambda row: "Discounted" if pd.notna(row.get("product_original_price")) else "No Discount",
        axis=1
    )
    df["price_change"] = df.groupby("asin")["product_price"].pct_change() * 100
    return df

# -------------------------------
# NEW: Overview chart (by brand)
# -------------------------------
def create_overview_graph(df: pd.DataFrame, brands_to_plot=None, use_markers=True) -> go.Figure:
    # Optional brand filter
    if brands_to_plot is not None and len(brands_to_plot) > 0:
        df = df[df["brand"].isin(brands_to_plot)]

    # Global ranges and weeks
    weeks = sorted(df["week_number"].unique())
    max_price = float(np.nanmax([df["product_price"].max(), df["product_original_price"].max()]))
    min_week = df["week_number"].min()
    max_week = df["week_number"].max()

    fig = go.Figure()
    trace_mode = "lines+markers" if use_markers else "lines"

    # One line per brand (avg price per week in case of multiple ASINs per brand)
    brand_week = (
        df.sort_values("date")
          .groupby(["brand", "week_number"], as_index=False)["product_price"].mean()
    )
    for brand, g in brand_week.groupby("brand"):
        fig.add_trace(go.Scatter(
            x=g["week_number"],
            y=g["product_price"],
            mode=trace_mode,
            name=str(brand),
            hovertemplate=(
                "Brand: %{text}<br>" +
                "Price: $%{y:.2f}<br>" +
                "Week: %{x}<extra></extra>"
            ),
            text=g["brand"],
            showlegend=True
        ))

    # Axes and layout
    fig.update_yaxes(range=[0, max_price], title_text="Product Price (USD)")
    fig.update_xaxes(
        range=[min_week, max_week],
        tickmode="linear", tick0=min_week, dtick=1,
        title_text="Week Number"
    )
    fig.update_layout(
        height=420,
        hovermode="x unified",
        legend_title_text="Brand",
        margin=dict(l=20, r=20, t=30, b=40)
    )
    return fig

# -------------------------------
# Subplots per ASIN
# -------------------------------
def create_price_graph(df: pd.DataFrame) -> go.Figure:
    asins = df["asin"].dropna().unique()
    num_asins = len(asins)

    # Grid: 3 columns
    cols = 3 if num_asins >= 3 else max(1, num_asins)
    rows = int(np.ceil(num_asins / cols))

    fig = make_subplots(
        rows=rows, cols=cols, shared_xaxes=True,
        vertical_spacing=0.15, horizontal_spacing=0.06,
        subplot_titles=[
            f"<a href='{df[df['asin']==asin]['product_url'].iloc[0]}' target='_blank' "
            f"style='color:#FFFBFE; text-decoration:none;'>{df[df['asin']==asin]['brand'].iloc[0]} - {asin}</a>"
            for asin in asins
        ]
    )

    max_price = float(np.nanmax([df["product_price"].max(), df["product_original_price"].max()]))
    min_week = df["week_number"].min()
    max_week = df["week_number"].max()

    # Ensure x tick labels are visible on all subplots
    fig.for_each_xaxis(lambda ax: ax.update(showticklabels=True))

    for i, asin in enumerate(asins):
        asin_data = df[df["asin"] == asin].sort_values("date")
        if asin_data.empty:
            continue

        dashed = "dot" if (asin_data["discount"] == "Discounted").any() else "solid"
        r = i // cols + 1
        c = i % cols + 1

        fig.add_trace(
            go.Scatter(
                x=asin_data["week_number"],
                y=asin_data["product_price"],
                mode="lines+markers",
                name=str(asin),
                line=dict(dash=dashed),
                hovertemplate=(
                    "ASIN: %{text}<br>" +
                    "Price: $%{y:.2f}<br>" +
                    "Week: %{x}<br>" +
                    "Price Change: %{customdata:.2f}%<br>" +
                    "<extra></extra>"
                ),
                text=asin_data["asin"],
                customdata=asin_data["price_change"],
                showlegend=False
            ),
            row=r, col=c
        )

    # Uniform scales and integer week ticks
    fig.update_yaxes(range=[0, max_price])
    fig.update_xaxes(range=[min_week, max_week])
    fig.for_each_xaxis(lambda ax: ax.update(tickmode="linear", tick0=min_week, dtick=1))

    fig.update_layout(
        height=max(400, 280 * rows),
        xaxis_title="Week Number",
        yaxis_title="Product Price (USD)",
        margin=dict(l=20, r=20, t=50, b=50)
    )
    return fig

# -------------------------------
# Main UI
# -------------------------------
df = fetch_data()
prepared_df = prepare_data(df)

# Last update
last_update = prepared_df["date"].max()
last_update_str = last_update.strftime("%Y-%m-%d") if pd.notna(last_update) else "N/A"

# Title + Subtitle
st.markdown(
    f"""
    <div style="text-align:center;">
        <h1 style="font-size: 36px; margin-bottom: 4px;">Competitors Price Tracker</h1>
        <h3 style="color:#666; font-weight:400; margin-top:0;">Last update: {last_update_str}</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# -------- Overview (by brand) --------
st.subheader("Overview — All Brands")

all_brands = sorted(prepared_df["brand"].dropna().unique().tolist())
use_markers = st.checkbox("Show markers on overview", value=False)
selected_brands = st.multiselect("Brands to display (overview)", options=all_brands, default=all_brands)

overview_fig = create_overview_graph(prepared_df, brands_to_plot=selected_brands, use_markers=use_markers)
st.plotly_chart(overview_fig, use_container_width=True)

# -------- Subplots by ASIN --------
price_graph = create_price_graph(prepared_df)
st.plotly_chart(price_graph, use_container_width=True)

# -------------------------------
# Table + filters
# -------------------------------
st.subheader("Detailed Product Information")

asin_options = ["All"] + sorted(prepared_df["asin"].dropna().unique().tolist())
discount_options = ["All", "Discounted", "No Discount"]

asin_filter = st.selectbox("Filter by ASIN", options=asin_options, index=0)
discount_filter = st.selectbox("Filter by Discount Status", options=discount_options, index=0)

filtered_df = prepared_df.copy()
if asin_filter != "All":
    filtered_df = filtered_df[filtered_df["asin"] == asin_filter]
if discount_filter != "All":
    filtered_df = filtered_df[filtered_df["discount"] == discount_filter]

st.dataframe(filtered_df)

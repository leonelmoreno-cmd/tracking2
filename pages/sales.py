# pages/sales.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from components.common import set_page_config, fetch_data, prepare_data
from components.basket_utils import resolve_active_basket
from components.basket_and_toggle_section import render_basket_and_toggle
from components.header import display_header

# =============================================================
# Utilities — Simulated Sales
# =============================================================

def simulate_sales(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    rng = np.random.default_rng(seed)
    work = df.copy()

    work["units_sold"] = rng.integers(1, 6, size=len(work))
    noise = rng.uniform(0.8, 1.2, size=len(work))

    price = pd.to_numeric(work["product_price"], errors="coerce").clip(lower=0).fillna(0)
    work["sales_amount"] = (price * work["units_sold"] * noise).astype(float)

    return work

# =============================================================
# Sales Overview Section — Filters + Highlights
# =============================================================

def render_sales_overview_section(df: pd.DataFrame, period: str):
    st.header("Data simulada, falta conexión con JungleScout, página solo para fines de layout", divider="gray")
    st.subheader("Sales — Overview")
    st.caption("Filter below. Metrics and chart are based on simulated sales data.")

    left_col, right_col = st.columns([0.7, 2.3], gap="large")

    all_brands = sorted(df["brand"].dropna().unique().tolist())
    date_min, date_max = df["date"].dropna().min(), df["date"].dropna().max()

    with left_col:
        st.caption("Select brands to filter the overview chart.")
        selected_brands = st.multiselect(
            "Brands to display (sales)",
            options=all_brands,
            default=all_brands,
            help="Brands to compare in the sales chart."
        )

        overview_date_range = None
        if pd.notna(date_min) and pd.notna(date_max):
            overview_date_range = st.date_input(
                "Filter by date (sales)",
                value=(date_min.date(), date_max.date()),
                min_value=date_min.date(),
                max_value=date_max.date(),
                help="Choose a start/end date for the overview."
            )

    with right_col:
        df_overview = df.copy()
        if overview_date_range:
            dstart, dend = (
                overview_date_range if isinstance(overview_date_range, tuple) else (overview_date_range, overview_date_range)
            )
            df_overview = df_overview[(df_overview["date"].dt.date >= dstart) & (df_overview["date"].dt.date <= dend)]
        if selected_brands:
            df_overview = df_overview[df_overview["brand"].isin(selected_brands)]

        st.markdown("### Sales Highlights")
        if df_overview.empty:
            st.info("No data for current filters.")
        else:
            grouped = df_overview.groupby("brand", dropna=False).agg(
                total_sales=("sales_amount", "sum"),
                avg_sales=("sales_amount", "mean"),
                total_units=("units_sold", "sum"),
                avg_units=("units_sold", "mean")
            ).reset_index()

            # Sales metrics
            top_sales = grouped.loc[grouped["total_sales"].idxmax()]
            low_sales = grouped.loc[grouped["total_sales"].idxmin()]
            avg_sales = grouped["avg_sales"].mean()

            # Variation metrics (by brand)
            variations = grouped["total_sales"].pct_change().fillna(0)
            max_var = variations.max() * 100
            min_var = variations.min() * 100
            avg_var = variations.mean() * 100

            # Units metrics
            top_units = grouped.loc[grouped["total_units"].idxmax()]
            low_units = grouped.loc[grouped["total_units"].idxmin()]
            avg_units_val = grouped["avg_units"].mean()

            col1, col2, col3 = st.columns(3)
            # Sales
            col1.metric("Highest Sales", f"{top_sales['brand']} (${top_sales['total_sales']:,.0f})")
            col1.metric("Lowest Sales", f"{low_sales['brand']} (${low_sales['total_sales']:,.0f})")
            col1.metric("Average Sales", f"${avg_sales:,.0f}")

            # Variations
            col2.metric("Highest Variation", f"{max_var:.1f}%")
            col2.metric("Lowest Variation", f"{min_var:.1f}%")
            col2.metric("Average Variation", f"{avg_var:.1f}%")

            # Units
            col3.metric("Most Units Sold", f"{top_units['brand']} ({top_units['total_units']:,})")
            col3.metric("Least Units Sold", f"{low_units['brand']} ({low_units['total_units']:,})")
            col3.metric("Average Units Sold", f"{avg_units_val:,.1f}")

    return df_overview, selected_brands, period

# =============================================================
# Sales Overview Graph (line chart by brand)
# =============================================================

def create_sales_overview_graph(df: pd.DataFrame, brands_to_plot=None, week_range=None, use_markers=False, period: str = "week") -> go.Figure:
    if brands_to_plot:
        df = df[df["brand"].isin(brands_to_plot)]
    if week_range is not None and period == "week":
        wk_min, wk_max = week_range
        df = df[(df["week_number"] >= wk_min) & (df["week_number"] <= wk_max)]

    group_key = "date" if period == "day" else "week_number"
    brand_period = df.groupby(["brand", group_key], as_index=False)["sales_amount"].sum()

    fig = go.Figure()
    trace_mode = "lines+markers" if use_markers else "lines"

    for brand, g in brand_period.groupby("brand"):
        fig.add_trace(
            go.Scatter(
                x=g[group_key],
                y=g["sales_amount"],
                mode=trace_mode,
                name=str(brand),
                hovertemplate=(
                    "Brand: %{text}<br>Sales: $%{y:,.0f}<br>" + ("Date: %{x|%Y-%m-%d}" if period == "day" else "Week: %{x}") + "<extra></extra>"
                ),
                text=g["brand"]
            )
        )

    fig.update_yaxes(title_text="Sales Amount (USD)")
    fig.update_xaxes(title_text=("Date" if period == "day" else "Week Number"))

    fig.update_layout(
        title="Sales Overview — Total Sales by Brand",
        height=420,
        hovermode="x unified",
        legend_title_text="Brand",
        margin=dict(l=20, r=20, t=50, b=40)
    )
    return fig

# =============================================================
# Sales Breakdown — Subplots by Brand (line charts)
# =============================================================

def render_sales_breakdown(df: pd.DataFrame, period: str = "week"):
    st.header("Sales Breakdown by Brand")

    if df.empty:
        st.info("No data available.")
        return

    brands = df["brand"].dropna().unique()
    num_brands = len(brands)
    cols = 3 if num_brands >= 3 else max(1, num_brands)
    rows = int(np.ceil(num_brands / cols))

    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, subplot_titles=[str(b) for b in brands])

    for i, brand in enumerate(brands):
        brand_data = df[df["brand"] == brand].sort_values("date")
        if brand_data.empty:
            continue

        if period == "day":
            x_vals = brand_data["date"]
            hover_x = "Date: %{x|%Y-%m-%d}"
        else:
            x_vals = brand_data["week_number"]
            hover_x = "Week: %{x}"

        r = i // cols + 1
        c = i % cols + 1

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=brand_data["sales_amount"],
                mode="lines+markers",
                name=str(brand),
                hovertemplate=f"Brand: {brand}<br>Sales: $%{{y:,.0f}}<br>{hover_x}<extra></extra>",
                showlegend=False
            ),
            row=r, col=c
        )

    fig.update_layout(
        height=max(400, 280 * rows),
        margin=dict(l=20, r=20, t=50, b=50),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show sales table"):
        if period == "day":
            df["xlabel"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            df["xlabel"] = df["week_number"].astype(int)
        df_table = df.pivot_table(index="xlabel", columns="brand", values="sales_amount", aggfunc="sum").sort_index()
        st.dataframe(df_table)

# =============================================================
# MAIN PAGE
# =============================================================

def main():
    set_page_config()
    DEFAULT_BASKET = "synthethic3.csv"
    active_basket_name, active_url, name_to_url = resolve_active_basket(DEFAULT_BASKET)

    df = fetch_data(active_url)
    prepared_df = prepare_data(df, basket_name=active_basket_name)
    display_header(prepared_df)

    period, active_basket_name = render_basket_and_toggle(name_to_url, active_basket_name, DEFAULT_BASKET)
    active_url = name_to_url.get(active_basket_name, active_url)

    df = fetch_data(active_url)
    prepared_df = prepare_data(df, basket_name=active_basket_name)
    sales_df = simulate_sales(prepared_df)

    st.header("Sales Overview")
    if sales_df is None or sales_df.empty:
        st.warning("No data available. Load data first.")
        return

    df_overview, selected_brands, period = render_sales_overview_section(sales_df, period=period)
    overview_fig = create_sales_overview_graph(df_overview, brands_to_plot=None, period=period)
    st.plotly_chart(overview_fig, use_container_width=True)

    render_sales_breakdown(df_overview, period=period)

if __name__ == "__main__":
    main()

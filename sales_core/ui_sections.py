from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from .week_utils import week_label_iso
from .aggregate import compute_highlights_for_week
from .config import AMAZON_DP_FMT

def render_overview_filters_and_highlights(weekly_df: pd.DataFrame, available_weeks: list[pd.Timestamp]):
    st.subheader("📊 Sales — Overview")
    st.caption("Weekly aggregates (Fri→Thu) from Jungle Scout.")

    left_col, right_col = st.columns([0.7, 2.3], gap="large")

    all_brands = sorted(weekly_df["brand"].fillna("Unknown").unique().tolist())
    week_labels = [f"{we.date()} (W{week_label_iso(we)})" for we in available_weeks]
    label_to_we = dict(zip(week_labels, available_weeks))

    with left_col:
        selected_brands = st.multiselect(
            "Brands to display (sales)",
            options=all_brands,
            default=all_brands,
            help="Brands to compare in the sales chart."
        )
        selected_week_label = st.selectbox(
            "Reporting week (labeled by week-ending Thursday)",
            options=week_labels,
            index=len(week_labels) - 1 if week_labels else 0
        )
        selected_week_end = label_to_we.get(selected_week_label)

    with right_col:
        df_overview = weekly_df.copy()
        if selected_brands:
            df_overview = df_overview[df_overview["brand"].isin(selected_brands)]

        st.markdown("### ✨ Sales Highlights (selected week)")
        if df_overview.empty or selected_week_end is None:
            st.info("No data for current filters.")
        else:
            metrics = compute_highlights_for_week(df_overview, selected_week_end)
            if not metrics:
                st.info("No data for the selected week.")
            else:
                col1, col2, col3 = st.columns(3)
                # Sales
                b_h, v_h = metrics["sales"]["highest"]
                b_l, v_l = metrics["sales"]["lowest"]
                v_avg = metrics["sales"]["average"]
                col1.metric("🏆 Highest Sales", f"{b_h} (${v_h:,.0f})")
                col1.metric("📉 Lowest Sales", f"{b_l} (${v_l:,.0f})")
                col1.metric("📊 Average Sales", f"${v_avg:,.0f}")

                # Variation
                vb_h, vv_h = metrics["variation"]["highest"]
                vb_l, vv_l = metrics["variation"]["lowest"]
                v_avg2 = metrics["variation"]["average"]
                col2.metric("📈 Highest Variation", f"{vv_h:,.1f}% ({vb_h})")
                col2.metric("📉 Lowest Variation", f"{vv_l:,.1f}% ({vb_l})")
                col2.metric("📊 Average Variation", f"{v_avg2:,.1f}%")

                # Units
                ub_m, uv_m = metrics["units"]["most"]
                ub_l, uv_l = metrics["units"]["least"]
                u_avg = metrics["units"]["average"]
                col3.metric("📦 Most Units Sold", f"{ub_m} ({uv_m:,})")
                col3.metric("📦 Least Units Sold", f"{ub_l} ({uv_l:,})")
                col3.metric("📦 Average Units Sold", f"{u_avg:,.1f}")

    return df_overview, selected_brands, selected_week_end

def create_overview_graph(df: pd.DataFrame, brands_to_plot: list[str] | None):
    d = df.copy()
    if brands_to_plot:
        d = d[d["brand"].isin(brands_to_plot)]
    agg = d.groupby(["brand", "week_end"], as_index=False)["sales_amount"].sum()

    fig = go.Figure()
    for brand, g in agg.groupby("brand"):
        g = g.sort_values("week_end")
        fig.add_trace(
            go.Scatter(
                x=g["week_end"],
                y=g["sales_amount"],
                mode="lines+markers",
                name=str(brand),
                hovertemplate="Brand: %{text}<br>Sales: $%{y:,.0f}<br>Week end: %{x|%Y-%m-%d}<extra></extra>",
                text=g["brand"],
            )
        )
    fig.update_yaxes(title_text="Sales Amount (USD)")
    fig.update_xaxes(title_text="Week Ending (Thursday)")
    fig.update_layout(
        title="📈 Sales Overview — Total Sales by Brand (Weekly Fri→Thu)",
        height=420,
        hovermode="x unified",
        legend_title_text="Brand",
        margin=dict(l=20, r=20, t=50, b=40)
    )
    return fig

def render_breakdown(df: pd.DataFrame):
    st.header("🔎 Sales Breakdown by Brand")
    if df.empty:
        st.info("No data available.")
        return

    brands = df["brand"].dropna().unique()
    num_brands = len(brands)
    cols = 3 if num_brands >= 3 else max(1, num_brands)
    rows = int(np.ceil(num_brands / cols))

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=rows, cols=cols, shared_xaxes=True,
        subplot_titles=[str(b) for b in brands],
        vertical_spacing=0.12, horizontal_spacing=0.08
    )

    for i, brand in enumerate(brands):
        g = (
            df[df["brand"] == brand]
            .groupby(["brand", "week_end"], as_index=False)["sales_amount"].sum()
            .sort_values("week_end")
        )
        if g.empty: 
            continue
        r = i // cols + 1; c = i % cols + 1
        fig.add_trace(
            go.Scatter(
                x=g["week_end"], y=g["sales_amount"],
                mode="lines+markers", name=str(brand),
                hovertemplate=f"Brand: {brand}<br>Sales: $%{{y:,.0f}}<br>Week end: %{{x|%Y-%m-%d}}<extra></extra>",
                showlegend=False
            ),
            row=r, col=c
        )

    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(showticklabels=True, ticks="outside", ticklen=4, tickformat="%Y-%m-%d", tickangle=-45, row=r, col=c)

    fig.update_layout(
        height=max(400, 280 * rows),
        margin=dict(l=20, r=20, t=50, b=80),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 Show weekly sales table (with ASIN links)"):
        table = df.copy()
        table["week_end"] = table["week_end"].dt.strftime("%Y-%m-%d")
        table["product_url"] = table["asin"].map(lambda a: AMAZON_DP_FMT.format(asin=a))
        st.dataframe(table[["week_end","brand","asin","units_sold","sales_amount","avg_price","product_url"]],
                     use_container_width=True)

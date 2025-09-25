import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.evolution_utils import (
    _aggregate_by_period,
    _has_discount_by_asin,
    _common_layout,
    _hover_template,
)

# ------------------------------------------------------------
# Price % Variation — overall ASINs in one chart
# ------------------------------------------------------------

def plot_price_variation_by_asin(df: pd.DataFrame, period: str = "day") -> None:
    """
    Creates an interactive figure for price percentage variation across all ASINs.
    The chart displays the ASINs in the y-axis and price percentage variation in the x-axis.
    """
    # Aggregate data based on the selected period (day or week)
    dfp = _aggregate_by_period(df, period)
    # Get the latest date or week based on the period
    if period == "day":
        latest_date = dfp["x"].max()
        df_latest = dfp[dfp["x"] == latest_date]
    elif period == "week":
        latest_week = dfp["iso_week"].dt.isocalendar().week.max()
        df_latest = dfp[dfp["iso_week"].dt.isocalendar().week == latest_week]
    else:
        st.warning("Invalid period specified")
        return

    if df_latest.empty:
        st.info("No data available for the selected period.")
        return

    # --- Prepare the data for plotting ---
    df_latest = df_latest.dropna(subset=["price_change"])  # Ensure we only plot rows with price changes

    # Create a list of ASINs and brands
    asin_with_brand = df_latest.apply(lambda row: f"{row['brand']} — {row['asin']}", axis=1)

    # Create a horizontal bar chart for all ASINs
    fig = go.Figure()

    # Add bars for price changes
    fig.add_trace(go.Bar(
        y=asin_with_brand,  # y-axis: ASIN and brand
        x=df_latest["price_change"],  # x-axis: price change percentage
        orientation="h",
        marker=dict(color=np.where(df_latest["price_change"] >= 0, "green", "red")),  # Green for positive, Red for negative
        hovertemplate=_hover_template("ASIN", "Price % change", show_pct=True, period=period),
    ))

    # Apply a common layout
    _common_layout(
        fig,
        nrows=1,
        title=f"Price Percentage Variation (by ASIN) - {period.capitalize()}",
        y_title="ASIN — Brand",
        x_title="Price Change (%)",
        y_min=-100,  # Ensure we show negative percentages clearly
        y_max=100,  # Ensure we show positive percentages clearly
        period=period,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Collapsed table with average price change per ASIN
    with st.expander("Show price % variation table"):
        tbl = (
            df_latest.pivot_table(
                index="asin",
                columns="brand",
                values="price_change",
                aggfunc="mean"
            ).sort_index()
        )
        st.dataframe(tbl)

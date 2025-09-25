import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from components.evolution_utils import (
    _aggregate_by_period,
    _has_discount_by_asin,
    _common_layout,
    _hover_template,
    _dash_for_asin,
)

# ------------------------------------------------------------
# Price % Variation — per ASIN subplots (grid layout)
# ------------------------------------------------------------

def plot_price_variation_by_asin(df: pd.DataFrame, period: str = "day") -> None:
    """
    Creates an interactive subplot figure for price percentage variation by ASIN (grid layout).
    Each subplot title is clickable: 'brand - asin' links to product_url.
    """
    dfp = _aggregate_by_period(df, period)
    asins = sorted(dfp["asin"].unique().tolist())
    n = len(asins)

    if n == 0:
        st.info("No price variation data to display.")
        return

    # --- grid config ---
    max_cols = 3
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))

    # global y-range: from 0 up to max positive change
    max_up = float(dfp["price_change"].max(skipna=True)) if dfp["price_change"].notna().any() else 0.0
    y_min = min(0.0, dfp["price_change"].min(skipna=True))  # Ensure negative side is included
    y_max = max(0.0, max_up)

    # Separate positive and negative price changes
    df_positive = dfp[dfp["price_change"] > 0]
    df_negative = dfp[dfp["price_change"] < 0]

    # Discount mapping
    discount_map = _has_discount_by_asin(dfp)

    # Subplot titles with clickable brand + asin
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=True,
        subplot_titles=[
            f"<a href='{df[df['asin'] == asin]['product_url'].iloc[0]}' target='_blank' "
            f"style='color:#FFFBFE; text-decoration:none;'>{dfp[dfp['asin'] == asin]['brand'].iloc[0]} - {asin}</a>"
            if "brand" in dfp.columns and "product_url" in df.columns
            else f"ASIN {asin}"
            for asin in asins
        ],
        vertical_spacing=0.16,
        horizontal_spacing=0.08,
    )

    # map asin → (row, col)
    pos_map = {}
    for i, asin in enumerate(asins):
        r = i // cols + 1
        c = i % cols + 1
        pos_map[asin] = (r, c)

    hover_tmpl = _hover_template("ASIN", "Price % change", show_pct=True, period=period)

    # Add traces for positive and negative variations
    for asin in asins:
        g_pos = df_positive[df_positive["asin"] == asin].sort_values("x")
        g_neg = df_negative[df_negative["asin"] == asin].sort_values("x")

        if g_pos.empty and g_neg.empty:
            continue

        # customdata: [asin, xlabel, price_change]
        customdata_pos = pd.DataFrame({
            "asin": g_pos["asin"].astype(str),
            "xlabel": g_pos["xlabel"].astype(str),
            "pct": g_pos["price_change"].astype(float),
        }).to_numpy()

        customdata_neg = pd.DataFrame({
            "asin": g_neg["asin"].astype(str),
            "xlabel": g_neg["xlabel"].astype(str),
            "pct": g_neg["price_change"].astype(float),
        }).to_numpy()

        r, c = pos_map[asin]

        if not g_pos.empty:
            fig.add_trace(
                go.Bar(
                    x=g_pos["price_change"],
                    y=g_pos["xlabel"],
                    orientation="h",
                    name=f"Positive - {asin}",
                    marker=dict(color="green"),
                    hovertemplate=hover_tmpl,
                    customdata=customdata_pos,
                ),
                row=r, col=c
            )

        if not g_neg.empty:
            fig.add_trace(
                go.Bar(
                    x=g_neg["price_change"],
                    y=g_neg["xlabel"],
                    orientation="h",
                    name=f"Negative - {asin}",
                    marker=dict(color="red"),
                    hovertemplate=hover_tmpl,
                    customdata=customdata_neg,
                ),
                row=r, col=c
            )

    # Apply common layout
    _common_layout(
        fig,
        nrows=rows,
        title="Price Percentage Variation (by ASIN)",
        y_title="Price Variation (%)",
        y_min=y_min,
        y_max=y_max,
        period=period,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Collapsed table
    with st.expander("Show price % variation table"):
        tbl = (
            dfp.pivot_table(
                index="xlabel",
                columns="asin",
                values="price_change",
                aggfunc="mean"
            ).sort_index()
        )
        st.dataframe(tbl)

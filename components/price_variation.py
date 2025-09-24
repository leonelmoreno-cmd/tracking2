# components/price_variation.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from components.evolution_utils import (
    _aggregate_by_period,
    _has_discount_by_asin,
    _common_layout,
    _annotate_max_per_subplot,
    _hover_template,
    _dash_for_asin,
)

# ------------------------------------------------------------
# Price % Variation — per ASIN subplots
# ------------------------------------------------------------

def plot_price_variation_by_asin(df: pd.DataFrame, period: str = "day") -> None:
    """
    Creates an interactive subplot figure for price percentage variation by ASIN.
    - One row per ASIN.
    - X: day or ISO week (period-aware).
    - Y: price_change (%) (0 .. global max).
    - Line style: dotted if product was ever discounted.
    - Hover: ASIN, Price % change, Date/Week, plus % change.
    - Annotation: max % change in each subplot.
    """
    dfp = _aggregate_by_period(df, period)
    asins = sorted(dfp["asin"].unique().tolist())
    n = len(asins)

    if n == 0:
        st.info("No price variation data to display.")
        return

    # global y-range: from 0 up to max positive change
    max_up = float(dfp["price_change"].max(skipna=True)) if dfp["price_change"].notna().any() else 0.0
    y_min = 0.0
    y_max = max(0.0, max_up)

    discount_map = _has_discount_by_asin(dfp)
    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True,
        subplot_titles=[f"ASIN {a}" for a in asins]
    )

    row_map = {a: i + 1 for i, a in enumerate(asins)}
    hover_tmpl = _hover_template("ASIN", "Price % change", show_pct=True, period=period)

    for asin in asins:
        g = dfp[dfp["asin"] == asin].sort_values("x")

        # customdata: [asin, xlabel, price_change]
        customdata = pd.DataFrame({
            "asin": g["asin"].astype(str),
            "xlabel": g["xlabel"].astype(str),
            "pct": g["price_change"].astype(float),
        }).to_numpy()

        fig.add_trace(
            go.Scatter(
                x=g["x"],
                y=g["price_change"],
                mode="lines+markers",
                line=dict(dash=_dash_for_asin(asin, discount_map), width=2),
                marker=dict(size=5),
                hovertemplate=hover_tmpl,
                customdata=customdata,
                name=f"ASIN {asin}",
            ),
            row=row_map[asin], col=1
        )

    _common_layout(
        fig, n,
        title="Price Percentage Variation (by ASIN)",
        y_title="Price Variation (%)",
        y_min=y_min, y_max=y_max,
        period=period,
    )

    # annotate max % change per subplot
    _annotate_max_per_subplot(
        fig, dfp, ycol="price_change", row_map=row_map, unit_suffix="%"
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

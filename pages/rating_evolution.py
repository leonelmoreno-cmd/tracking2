# components/rating_evolution.py

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
# Rating Evolution — per ASIN subplots
# ------------------------------------------------------------

def plot_rating_evolution_by_asin(df: pd.DataFrame, period: str = "day") -> None:
    """
    Creates an interactive subplot figure for rating evolution by ASIN.
    - One row per ASIN, titles show ASIN codes.
    - X: day or ISO week (period-aware).
    - Y: product_star_rating (0 .. max across all subplots).
    - Line style: dotted if product was ever discounted, else solid.
    - Hover: ASIN, Rating, Date/Week, % change in rating.
    - Annotation: max rating point in each subplot.
    """
    dfp = _aggregate_by_period(df, period)
    dfp = dfp[dfp["product_star_rating"].notna()]
    asins = sorted(dfp["asin"].unique().tolist())
    n = len(asins)

    if n == 0:
        st.info("No rating data to display.")
        return

    # global y-range
    y_max = float(dfp["product_star_rating"].max(skipna=True))
    y_min = 0.0

    discount_map = _has_discount_by_asin(dfp)
    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True,
        subplot_titles=[f"ASIN {a}" for a in asins]
    )

    row_map = {a: i + 1 for i, a in enumerate(asins)}
    hover_tmpl = _hover_template("ASIN", "Rating", show_pct=True, period=period)

    for asin in asins:
        g = dfp[dfp["asin"] == asin].sort_values("x")

        # customdata: [asin, xlabel, rating_change_pct]
        customdata = pd.DataFrame({
            "asin": g["asin"].astype(str),
            "xlabel": g["xlabel"].astype(str),
            "pct": g["rating_change_pct"].astype(float),
        }).to_numpy()

        fig.add_trace(
            go.Scatter(
                x=g["x"],
                y=g["product_star_rating"],
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
        title="Rating Evolution (by ASIN)",
        y_title="Product Star Rating",
        y_min=y_min, y_max=y_max,
        period=period,
    )

    # annotate max rating per subplot
    _annotate_max_per_subplot(
        fig, dfp, ycol="product_star_rating", row_map=row_map
    )

    st.plotly_chart(fig, use_container_width=True)

    # Collapsed table
    with st.expander("Show rating table"):
        tbl = (
            dfp.pivot_table(
                index="xlabel",
                columns="asin",
                values="product_star_rating",
                aggfunc="mean"
            ).sort_index()
        )
        st.dataframe(tbl)

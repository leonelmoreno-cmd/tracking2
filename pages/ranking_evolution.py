# components/ranking_evolution.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from components.evolution_utils import (
    _aggregate_by_period,
    _has_discount_by_asin,
    _common_layout,
    _dash_for_asin,
)

# ------------------------------------------------------------
# Ranking Evolution — per ASIN subplots
# ------------------------------------------------------------

def plot_ranking_evolution_by_asin(df: pd.DataFrame, period: str = "day") -> None:
    """
    Creates an interactive subplot figure for ranking evolution by ASIN.
    - Rank per date/week across products (1 = best rating).
    - One row per ASIN, Y shows rank over time.
    - X: day or ISO week (period-aware).
    - Line style: dotted if product was ever discounted.
    - Hover: ASIN, Rank, Date/Week, plus Rating and Price % change.
    - Annotation: best rank (lowest value) per subplot.
    """
    dfp = _aggregate_by_period(df, period)

    # compute rank per period across ASINs based on rating (descending → 1 is best)
    dfp["rank"] = (
        dfp.groupby("x")["product_star_rating"]
           .rank(method="first", ascending=False)
    )

    asins = sorted(dfp["asin"].unique().tolist())
    n = len(asins)
    if n == 0:
        st.info("No ranking data to display.")
        return

    # y-axis: 1 = best, up to total number of ASINs
    y_min = 1
    y_max = max(1, len(asins))

    discount_map = _has_discount_by_asin(dfp)
    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True,
        subplot_titles=[f"ASIN {a}" for a in asins]
    )

    row_map = {a: i + 1 for i, a in enumerate(asins)}
    period_label = "Week" if period.lower() == "week" else "Date"
    hover_template = (
        "<b>ASIN</b>: %{customdata[0]}"
        "<br><b>Rank</b>: %{y:.0f}"
        f"<br><b>{period_label}</b>: %{{customdata[1]}}"
        "<br><b>Rating</b>: %{customdata[2]:.2f}"
        "<br><b>Price % change</b>: %{customdata[3]:.2f}%"
        "<extra></extra>"
    )

    for asin in asins:
        g = dfp[dfp["asin"] == asin].sort_values("x")

        # customdata: [asin, xlabel, rating, price_change]
        customdata = pd.DataFrame({
            "asin": g["asin"].astype(str),
            "xlabel": g["xlabel"].astype(str),
            "rating": g["product_star_rating"].astype(float),
            "pct_price": g["price_change"].astype(float),
        }).to_numpy()

        fig.add_trace(
            go.Scatter(
                x=g["x"],
                y=g["rank"],
                mode="lines+markers",
                line=dict(dash=_dash_for_asin(asin, discount_map), width=2),
                marker=dict(size=5),
                hovertemplate=hover_template,
                customdata=customdata,
                name=f"ASIN {asin}",
            ),
            row=row_map[asin], col=1
        )

    _common_layout(
        fig, n,
        title="Ranking Evolution (by ASIN)",
        y_title="Rank (1 = best)",
        y_min=y_min, y_max=y_max,
        period=period,
        reverse_y=True  # so that 1 (best) appears at the top
    )

    # annotate "best" (min rank) per subplot
    for asin, g in dfp.groupby("asin"):
        if g["rank"].notna().any():
            idx = g["rank"].idxmin()
            x = g.loc[idx, "x"]
            y = g.loc[idx, "rank"]
            label = g.loc[idx, "xlabel"]
            fig.add_annotation(
                x=x, y=y, row=row_map[asin], col=1,
                text=f"best: {int(y)}<br>{label}",
                xanchor="left", yanchor="top",
                showarrow=True, arrowhead=2, arrowsize=1, ax=20, ay=20
            )

    st.plotly_chart(fig, use_container_width=True)

    # collapsed table
    with st.expander("Show ranking table"):
        tbl = (
            dfp.pivot_table(
                index="xlabel",
                columns="asin",
                values="rank",
                aggfunc="mean"
            ).sort_index()
        )
        st.dataframe(tbl)

import streamlit as st
import numpy as np
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
# Ranking Evolution — per ASINs subplots (grid layout)
# ------------------------------------------------------------

def plot_ranking_evolution_by_asin(df: pd.DataFrame, period: str = "day") -> None:
    """
    Creates an interactive subplot figure for ranking evolution by ASIN, in a grid layout.
    Each subplot title is clickable: 'brand - asin' links to product_url.
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

    # --- grid layout ---
    max_cols = 3
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))

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
        if g.empty:
            continue

        # customdata: [asin, xlabel, rating, price_change]
        customdata = pd.DataFrame({
            "asin": g["asin"].astype(str),
            "xlabel": g["xlabel"].astype(str),
            "rating": g["product_star_rating"].astype(float),
            "pct_price": g["price_change"].astype(float),
        }).to_numpy()

        r, c = pos_map[asin]

        fig.add_trace(
            go.Scatter(
                x=g["x"],
                y=g["rank"],
                mode="lines+markers",
                line=dict(dash=_dash_for_asin(asin, discount_map), width=2),
                marker=dict(size=5),
                hovertemplate=hover_template,
                customdata=customdata,
                name=f"{g['brand'].iloc[0]} - {asin}" if "brand" in g.columns else f"ASIN {asin}",
            ),
            row=r, col=c
        )

    # aplicar diseño común (ajusta con nº de filas reales)
    _common_layout(
        fig,
        nrows=rows,
        title="Ranking Evolution (by ASIN)",
        y_title="Rank (1 = best)",
        y_min=y_min,
        y_max=y_max,
        period=period,
        reverse_y=True,  # so that 1 (best) appears at the top
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

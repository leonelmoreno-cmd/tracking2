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
    Each subplot title is clickable: 'brand - asin - sub_category_name' links to product_url.
    - Y axis: rank suavizado con promedio móvil (1 = mejor).
    - Líneas: verde si mejoran, rojo si empeoran.
    - Tooltip: incluye rank real, Δ rank y sales_volume.
    """
    dfp = _aggregate_by_period(df, period)

    # ordenar y calcular cambio de rank
    dfp = dfp.sort_values(["asin", "x"])
    dfp["rank_delta"] = dfp.groupby("asin")["rank"].diff()

    # suavizado: promedio móvil de 3 periodos
    dfp["rank_smooth"] = dfp.groupby("asin")["rank"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )

    asins = sorted(dfp["asin"].unique().tolist())
    n = len(asins)
    if n == 0:
        st.info("No ranking data to display.")
        return

    # rango del eje Y (según rank real)
    y_min = float(dfp["rank"].min(skipna=True))
    y_max = float(dfp["rank"].max(skipna=True))

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
            f"style='color:#FFFBFE; text-decoration:none;'>"
            f"{dfp[dfp['asin'] == asin]['brand'].iloc[0]} - {asin} - {dfp[dfp['asin'] == asin]['sub_category_name'].iloc[0]}</a>"
            if "brand" in dfp.columns and "product_url" in df.columns and "sub_category_name" in dfp.columns
            else f"ASIN {asin}"
            for asin in asins
        ],
        vertical_spacing=0.16,
        horizontal_spacing=0.08,
    )

    pos_map = {}
    for i, asin in enumerate(asins):
        r = i // cols + 1
        c = i % cols + 1
        pos_map[asin] = (r, c)

    period_label = "Week" if period.lower() == "week" else "Date"
    hover_template = (
        "<b>ASIN</b>: %{customdata[0]}"
        "<br><b>Rank (real)</b>: %{customdata[2]:.0f}"
        "<br><b>Rank (smoothed)</b>: %{y:.0f}"
        f"<br><b>{period_label}</b>: %{{customdata[1]}}"
        "<br><b>Δ rank</b>: %{customdata[3]:+.0f}"
        "<br><b>Sales volume</b>: %{customdata[4]}"
        "<extra></extra>"
    )

    for asin in asins:
        g = dfp[dfp["asin"] == asin].sort_values("x")
        if g.empty:
            continue

        # color: verde si mejoró (rank menor), rojo si empeoró
        if g["rank_delta"].iloc[-1] < 0:
            color = "red"   # empeoró
        else:
            color = "green" # mejoró

        customdata = pd.DataFrame({
            "asin": g["asin"].astype(str),
            "xlabel": g["xlabel"].astype(str),
            "rank": g["rank"].astype(float),
            "delta": g["rank_delta"].fillna(0).astype(float),
            "sales": g.get("sales_volume", pd.Series(index=g.index)).astype(str),
        }).to_numpy()

        r, c = pos_map[asin]

        fig.add_trace(
            go.Scatter(
                x=g["x"],
                y=g["rank_smooth"],  # usamos el suavizado en Y
                mode="lines+markers",
                line=dict(dash=_dash_for_asin(asin, discount_map), width=2, color=color),
                marker=dict(size=5, color=color),
                hovertemplate=hover_template,
                customdata=customdata,
                name=(
                    f"{g['brand'].iloc[0]} - {asin} - {g['sub_category_name'].iloc[0]}"
                    if "brand" in g.columns and "sub_category_name" in g.columns
                    else f"ASIN {asin}"
                ),
            ),
            row=r, col=c
        )

    # aplicar layout común
    _common_layout(
        fig,
        nrows=rows,
        title="Ranking Evolution (by ASIN)",
        y_title="Category Rank (1 = best)",
        y_min=y_min,
        y_max=y_max,
        period=period,
        reverse_y=True,
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

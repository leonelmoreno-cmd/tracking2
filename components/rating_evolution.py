import streamlit as st
import numpy as np
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

def plot_rating_evolution_by_asin_grid(df: pd.DataFrame, period: str = "day") -> None:
    """
    Versión con subplots en cuadrícula (hasta 3 columnas) para evolución de ratings por ASIN.
    Cada título de subplot muestra 'brand - asin'.
    """
    dfp = _aggregate_by_period(df, period)
    # Filtrar solo las filas con rating no nulo
    dfp = dfp[dfp["product_star_rating"].notna()]

    asins = sorted(dfp["asin"].unique().tolist())
    n = len(asins)

    if n == 0:
        st.info("No rating data to display.")
        return

    # determinar número de columnas y filas para la cuadrícula
    max_cols = 3
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))

    # límite global del eje Y
    y_max = 5.0
    y_min = 0.0

    discount_map = _has_discount_by_asin(dfp)

    # Crear la figura con subplots en cuadrícula
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=True,
        subplot_titles=[
            f"{dfp[dfp['asin'] == asin]['brand_x'].iloc[0]} - {asin}" if 'brand_x' in dfp.columns else f"ASIN {asin}"
            for asin in asins
        ],
        vertical_spacing=0.16,
        horizontal_spacing=0.08,
    )

    # Mapa asin → (fila, columna)
    row_map = {}
    for i, asin in enumerate(asins):
        r = i // cols + 1
        c = i % cols + 1
        row_map[asin] = (r, c)

    hover_tmpl = _hover_template("ASIN", "Rating", show_pct=True, period=period)

    for asin in asins:
        g = dfp[dfp["asin"] == asin].sort_values("x")
        if g.empty:
            continue

        # preparar customdata para hover
        customdata = pd.DataFrame({
            "asin": g["asin"].astype(str),
            "xlabel": g["xlabel"].astype(str),
            "pct": g["rating_change_pct"].astype(float),
        }).to_numpy()

        r, c = row_map[asin]

        fig.add_trace(
            go.Scatter(
                x=g["x"],
                y=g["product_star_rating"],
                mode="lines+markers",
                line=dict(dash=_dash_for_asin(asin, discount_map), width=2),
                marker=dict(size=5),
                hovertemplate=hover_tmpl,
                customdata=customdata,
                name=f"{g['brand'].iloc[0]} - {asin}" if "brand" in g.columns else f"ASIN {asin}",
            ),
            row=r, col=c
        )

    # Aplica diseño común
    _common_layout(
        fig,
        nrows=rows,  # número de filas visuales
        title="Rating Evolution (by ASIN)",
        y_title="Product Star Rating",
        y_min=y_min,
        y_max=y_max,
        period=period,
    )
    fig.update_yaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)

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

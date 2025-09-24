# components/percentage_var.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, List

# -------------------------------
# Data preparation (safe to keep here even if your page prepares earlier)
# -------------------------------
@st.cache_data(show_spinner=False)
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Coerce types
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["product_price"] = pd.to_numeric(df.get("product_price"), errors="coerce")
    if "product_original_price" in df.columns:
        df["product_original_price"] = pd.to_numeric(df.get("product_original_price"), errors="coerce")
    df["product_star_rating"] = pd.to_numeric(df.get("product_star_rating"), errors="coerce")

    # ISO week info (for weekly mode)
    iso = df["date"].dt.isocalendar()
    df["week_number"] = iso.week
    df["year"] = iso.year

    # Discount tag (strict: only if original exists AND price < original)
    if "product_original_price" in df.columns:
        df["discount"] = np.where(
            df["product_original_price"].notna() & (df["product_price"] < df["product_original_price"]),
            "Discounted", "No Discount"
        )
    else:
        df["discount"] = "No Discount"

    # Sort and compute price % change
    df = df.sort_values(by=["asin", "date"])
    df["price_change"] = df.groupby("asin")["product_price"].pct_change() * 100

    return df

# -------------------------------
# Helpers
# -------------------------------
def _grid(num_asins: int) -> Tuple[int, int]:
    cols = 3 if num_asins >= 3 else max(1, num_asins)
    rows = int(np.ceil(num_asins / cols)) if num_asins else 1
    return rows, cols

def _x_vals(asin_df: pd.DataFrame, period: str):
    if period == "day":
        return asin_df["date"], "Date", "Date: %{x|%Y-%m-%d}"
    else:
        return asin_df["week_number"], "Week Number", "Week: %{x}"

def _subplot_titles(asins: List[str], df: pd.DataFrame) -> List[str]:
    titles = []
    for asin in asins:
        g = df[df["asin"] == asin]
        label = str(asin)
        if "brand" in g.columns and "product_url" in g.columns and not g.empty:
            try:
                brand = str(g["brand"].iloc[0])
                url = str(g["product_url"].iloc[0])
                label = f"<a href='{url}' target='_blank' style='color:#111; text-decoration:none;'>{brand} - {asin}</a>"
            except Exception:
                label = str(asin)
        titles.append(label)
    return titles

def _fig_base(rows: int, cols: int, titles: List[str]) -> go.Figure:
    fig = make_subplots(
        rows=rows, cols=cols, shared_xaxes=True,
        vertical_spacing=0.12, horizontal_spacing=0.06,
        subplot_titles=titles
    )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=40),
    )
    # Crosshair spikes on all x-axes
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=True, mirror=True)
    fig.update_yaxes(showline=True, mirror=True, zeroline=False)
    return fig

def _last_point_text_trace(x, y, text: str) -> go.Scatter:
    """Single-point transparent marker with text label at last value."""
    return go.Scatter(
        x=[x], y=[y],
        mode="text",
        text=[text],
        textposition="top right",
        hoverinfo="skip",
        showlegend=False
    )

# -------------------------------
# 1) Rating — subplots per ASIN
# -------------------------------
def create_rating_graph(df: pd.DataFrame, period: str = "week") -> go.Figure:
    asins = df["asin"].dropna().unique().tolist()
    n = len(asins)
    rows, cols = _grid(n)
    titles = _subplot_titles(asins, df)
    fig = _fig_base(rows, cols, titles)

    # y-range consistent across all panes
    fig.update_yaxes(range=[0, 5], tick0=0, dtick=1, title_text="Product Star Rating", row=1, col=1)

    # week axis helpers
    if period == "week":
        min_week = int(df["week_number"].min())
        max_week = int(df["week_number"].max())

    for i, asin in enumerate(asins):
        r, c = (i // cols) + 1, (i % cols) + 1
        g = df[df["asin"] == asin].sort_values("date")
        if g.empty:
            continue

        x, x_title, hover_x = _x_vals(g, period)

        # Prepare customdata for hovertemplate
        custom = np.stack([
            g["asin"].astype(str).fillna(""),
            g.get("brand", pd.Series("", index=g.index)).astype(str).fillna(""),
            g["product_price"].astype(float),
            g.get("product_original_price", pd.Series(np.nan, index=g.index)).astype(float),
            g["discount"].astype(str),
            g["price_change"].astype(float),
        ], axis=1)

        fig.add_trace(
            go.Scatter(
                x=x, y=g["product_star_rating"],
                mode="lines+markers",
                name=str(asin),
                line=dict(shape="linear"),
                customdata=custom,
                hovertemplate=(
                    "ASIN: %{customdata[0]}<br>"
                    "Brand: %{customdata[1]}<br>"
                    f"{hover_x}<br>"
                    "Rating: %{y:.2f}★<br>"
                    "Price: $%{customdata[2]:.2f}<br>"
                    "Original: $%{customdata[3]:.2f}<br>"
                    "Discount: %{customdata[4]}<br>"
                    "Δ Price: %{customdata[5]:.2f}%"
                    "<extra></extra>"
                ),
                showlegend=False
            ),
            row=r, col=c
        )

        # Last value annotation (simple text trace)
        last_x = x.iloc[-1]
        last_y = g["product_star_rating"].iloc[-1]
        fig.add_trace(_last_point_text_trace(last_x, last_y, f"{last_y:.2f}★"), row=r, col=c)

    # X settings
    if period == "week":
        fig.update_xaxes(range=[min_week, max_week], tickmode="linear", tick0=min_week, dtick=1, row=rows, col=1)
        x_title = "Week Number"
    else:
        x_title = "Date"

    fig.update_layout(
        height=max(420, 280 * rows),
        xaxis_title=x_title
    )
    return fig

# -------------------------------
# 2) Price % Variation — subplots per ASIN
# -------------------------------
def create_price_variation_graph(df: pd.DataFrame, period: str = "week") -> go.Figure:
    asins = df["asin"].dropna().unique().tolist()
    n = len(asins)
    rows, cols = _grid(n)
    titles = _subplot_titles(asins, df)
    fig = _fig_base(rows, cols, titles)

    # Symmetric y-range around zero for all panes
    max_abs = float(np.nanmax(np.abs(df["price_change"]))) if "price_change" in df.columns else 0.0
    max_abs = 5.0 if not np.isfinite(max_abs) or max_abs == 0 else max_abs
    pad = max_abs * 0.1
    fig.update_yaxes(range=[-(max_abs + pad), (max_abs + pad)], title_text="Price Variation (%)", row=1, col=1, zeroline=True)

    if period == "week":
        min_week = int(df["week_number"].min())
        max_week = int(df["week_number"].max())

    for i, asin in enumerate(asins):
        r, c = (i // cols) + 1, (i % cols) + 1
        g = df[df["asin"] == asin].sort_values("date")
        if g.empty:
            continue

        dashed = "dot" if (g["discount"] == "Discounted").any() else "solid"
        x, x_title, hover_x = _x_vals(g, period)

        custom = np.stack([
            g["asin"].astype(str).fillna(""),
            g.get("brand", pd.Series("", index=g.index)).astype(str).fillna(""),
            g["product_price"].astype(float),
            g.get("product_original_price", pd.Series(np.nan, index=g.index)).astype(float),
            g["discount"].astype(str),
        ], axis=1)

        fig.add_trace(
            go.Scatter(
                x=x, y=g["price_change"],
                mode="lines+markers",
                name=str(asin),
                line=dict(dash=dashed),
                customdata=custom,
                hovertemplate=(
                    "ASIN: %{customdata[0]}<br>"
                    "Brand: %{customdata[1]}<br>"
                    f"{hover_x}<br>"
                    "Δ Price: %{y:.2f}%<br>"
                    "Price: $%{customdata[2]:.2f}<br>"
                    "Original: $%{customdata[3]:.2f}<br>"
                    "Discount: %{customdata[4]}"
                    "<extra></extra>"
                ),
                showlegend=False
            ),
            row=r, col=c
        )

        # Last value label
        if len(x) and len(g["price_change"]):
            last_x = x.iloc[-1]
            last_y = g["price_change"].iloc[-1]
            fig.add_trace(_last_point_text_trace(last_x, last_y, f"{last_y:.2f}%"), row=r, col=c)

    if period == "week":
        fig.update_xaxes(range=[min_week, max_week], tickmode="linear", tick0=min_week, dtick=1, row=rows, col=1)
        x_title = "Week Number"
    else:
        x_title = "Date"

    fig.update_layout(
        height=max(420, 280 * rows),
        xaxis_title=x_title
    )
    return fig

# -------------------------------
# 3) Ranking — subplots per ASIN
# -------------------------------
def create_ranking_graph(df: pd.DataFrame, period: str = "week") -> go.Figure:
    # Compute ranking by date (1 = best rating)
    ranked = df.copy()
    ranked["ranking"] = ranked.groupby("date")["product_star_rating"].rank(method="first", ascending=False)

    asins = ranked["asin"].dropna().unique().tolist()
    n = len(asins)
    rows, cols = _grid(n)
    titles = _subplot_titles(asins, ranked)
    fig = _fig_base(rows, cols, titles)

    # Global rank range (1..N unique ASINs)
    n_asins_global = len(asins)
    fig.update_yaxes(range=[n_asins_global + 0.5, 0.5], title_text="Ranking (1 = best)", row=1, col=1)  # reversed

    if period == "week":
        min_week = int(ranked["week_number"].min())
        max_week = int(ranked["week_number"].max())

    for i, asin in enumerate(asins):
        r, c = (i // cols) + 1, (i % cols) + 1
        g = ranked[ranked["asin"] == asin].sort_values("date")
        if g.empty:
            continue

        x, x_title, hover_x = _x_vals(g, period)

        custom = np.stack([
            g["asin"].astype(str).fillna(""),
            g.get("brand", pd.Series("", index=g.index)).astype(str).fillna(""),
            g["product_star_rating"].astype(float),
        ], axis=1)

        fig.add_trace(
            go.Scatter(
                x=x, y=g["ranking"],
                mode="lines+markers",
                name=str(asin),
                customdata=custom,
                hovertemplate=(
                    "ASIN: %{customdata[0]}<br>"
                    "Brand: %{customdata[1]}<br>"
                    f"{hover_x}<br>"
                    "Rank: %{y:.0f}<br>"
                    "Rating: %{customdata[2]:.2f}★"
                    "<extra></extra>"
                ),
                showlegend=False
            ),
            row=r, col=c
        )

        # Last value label
        last_x = x.iloc[-1]
        last_y = g["ranking"].iloc[-1]
        fig.add_trace(_last_point_text_trace(last_x, last_y, f"#{int(last_y)}"), row=r, col=c)

    if period == "week":
        fig.update_xaxes(range=[min_week, max_week], tickmode="linear", tick0=min_week, dtick=1, row=rows, col=1)
        x_title = "Week Number"
    else:
        x_title = "Date"

    fig.update_layout(
        height=max(420, 280 * rows),
        xaxis_title=x_title
    )
    return fig

# -------------------------------
# Expanders: compact pivots for each section
# -------------------------------
def _rating_pivot(df: pd.DataFrame, period: str) -> pd.DataFrame:
    key = "date" if period == "day" else "week_number"
    p = df.pivot_table(index=key, columns="asin", values="product_star_rating", aggfunc="mean")
    return p.sort_index()

def _price_change_pivot(df: pd.DataFrame, period: str) -> pd.DataFrame:
    key = "date" if period == "day" else "week_number"
    p = df.pivot_table(index=key, columns="asin", values="price_change", aggfunc="mean")
    return p.sort_index()

def _ranking_pivot(df: pd.DataFrame, period: str) -> pd.DataFrame:
    key = "date" if period == "day" else "week_number"
    ranked = df.copy()
    ranked["ranking"] = ranked.groupby("date")["product_star_rating"].rank(method="first", ascending=False)
    p = ranked.pivot_table(index=key, columns="asin", values="ranking", aggfunc="mean")
    return p.sort_index()

# -------------------------------
# Public API
# -------------------------------
def main(df: pd.DataFrame, period: str = "week"):
    """Render the three independent charts + their tables."""
    # 1) Rating Evolution
    st.subheader("Rating Evolution — per ASIN")
    rating_fig = create_rating_graph(df, period=period)
    st.plotly_chart(rating_fig, use_container_width=True)
    with st.expander("Show rating table", expanded=False):
        st.dataframe(_rating_pivot(df, period))

    # 2) Price Percentage Variation
    st.subheader("Price Percentage Variation — per ASIN")
    price_fig = create_price_variation_graph(df, period=period)
    st.plotly_chart(price_fig, use_container_width=True)
    with st.expander("Show price variation table", expanded=False):
        st.dataframe(_price_change_pivot(df, period))

    # 3) Ranking Evolution
    st.subheader("Ranking Evolution — per ASIN (1 = best)")
    ranking_fig = create_ranking_graph(df, period=period)
    st.plotly_chart(ranking_fig, use_container_width=True)
    with st.expander("Show ranking table", expanded=False):
        st.dataframe(_ranking_pivot(df, period))

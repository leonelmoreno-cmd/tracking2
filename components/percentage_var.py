# components/percentage_var.py
import math
from datetime import date as _date
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric types we rely on exist and are numeric."""
    out = df.copy()
    for col in ("product_price", "product_star_rating", "product_original_price"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["date"] = pd.to_datetime(out.get("date"), errors="coerce")
    return out

def _iso_week_cols(df: pd.DataFrame) -> pd.DataFrame:
    iso = df["date"].dt.isocalendar()
    df["iso_year"] = iso.year
    df["iso_week"] = iso.week
    return df

def _week_start(year: int, week: int) -> pd.Timestamp:
    """Return Monday of given ISO week as a pandas Timestamp."""
    return pd.Timestamp(_date.fromisocalendar(int(year), int(week), 1))

def _aggregate_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Build an 'analysis' dataframe for day/week views.
    - day: keep daily rows as-is (sorted).
    - week: aggregate by (asin, iso_year, iso_week) using mean for price & rating.
    Also computes price_change (%) after period aggregation.
    Adds:
      - x (datetime): date or week-start date (for plotting)
      - xlabel (str): formatted label for hover (Date or ISO Week)
    """
    base = _ensure_numeric(df)
    base = _iso_week_cols(base)

    if period.lower() == "week":
        grp = base.groupby(["asin", "iso_year", "iso_week"], as_index=False).agg(
            product_price=("product_price", "mean"),
            product_star_rating=("product_star_rating", "mean"),
            # if any discounted within the week, mark as discounted for style context
            discount_any=("product_original_price", lambda s: pd.notna(s).any())
        )
        # x axis as Monday of ISO week
        grp["x"] = grp.apply(lambda r: _week_start(r["iso_year"], r["iso_week"]), axis=1)
        grp = grp.sort_values(["asin", "x"])
        # price change on aggregated series
        grp["price_change"] = grp.groupby("asin")["product_price"].pct_change() * 100
        # rating pct change (for informative hover)
        grp["rating_change_pct"] = grp.groupby("asin")["product_star_rating"].pct_change() * 100
        grp["xlabel"] = grp.apply(lambda r: f"Week {int(r['iso_year'])}-W{int(r['iso_week']):02d}", axis=1)
        return grp
    else:
        # day
        base = base.sort_values(["asin", "date"]).copy()
        # consistent discount flag per row (presence of original price => treat as discounted row)
        base["discount_any"] = pd.notna(base.get("product_original_price"))
        # changes on daily series
        base["price_change"] = base.groupby("asin")["product_price"].pct_change() * 100
        base["rating_change_pct"] = base.groupby("asin")["product_star_rating"].pct_change() * 100
        base["x"] = base["date"]
        base["xlabel"] = base["date"].dt.strftime("%Y-%m-%d")
        return base

def _has_discount_by_asin(df: pd.DataFrame) -> dict:
    """
    Returns dict asin -> bool indicating if product ever had discount across the WHOLE dataset.
    We treat 'discount' as present if product_original_price is present on any row
    OR if 'discount_any' exists and is True anywhere.
    """
    if "discount_any" in df.columns:
        s = df.groupby("asin")["discount_any"].max()
        return s.astype(bool).to_dict()

    if "product_original_price" in df.columns:
        s = df.groupby("asin")["product_original_price"].apply(lambda s: pd.notna(s).any())
        return s.astype(bool).to_dict()

    # default: assume not discounted
    return {asin: False for asin in df["asin"].unique()}

def _common_layout(fig: go.Figure, nrows: int, title: str, y_title: str,
                   y_min: float, y_max: float, period: str,
                   reverse_y: bool = False) -> None:
    """Apply consistent aesthetics, sizing, and axes settings."""
    height_per_row = 240  # tweakable
    fig.update_layout(
        template="plotly_white",
        height=max(260, int(height_per_row * max(1, nrows))),
        title=title,
        margin=dict(t=60, l=70, r=30, b=50),
        hovermode="x unified",
        showlegend=False,
    )
    # X label once at bottom; shared X for linked interactions
    fig.update_xaxes(title_text="Week" if period.lower() == "week" else "Date")
    # Y range synchronized across all subplots
    # Ranking uses reverse if requested
    fig.update_yaxes(title_text=y_title, range=[y_min, y_max], autorange=False)
    if reverse_y:
        fig.update_yaxes(autorange="reversed")

def _annotate_max_per_subplot(fig: go.Figure, dfp: pd.DataFrame, ycol: str,
                              row_map: dict, unit_suffix: str = "") -> None:
    """
    For each ASIN subplot, add an annotation at its max point.
    row_map: asin -> row index (1-based for plotly).
    """
    for asin, g in dfp.groupby("asin"):
        if g[ycol].notna().any():
            idx = g[ycol].idxmax()
            x = g.loc[idx, "x"]
            y = g.loc[idx, ycol]
            label = g.loc[idx, "xlabel"]
            fig.add_annotation(
                x=x, y=y, row=row_map[asin], col=1,
                text=f"max: {y:.2f}{unit_suffix}<br>{label}",
                xanchor="left", yanchor="bottom",
                showarrow=True, arrowhead=2, arrowsize=1, ax=20, ay=-20
            )

def _hover_template(as_label: str, value_label: str, show_pct: bool, period: str) -> str:
    """
    Build a Plotly hovertemplate.
    We will pass additional fields as customdata when needed.
    """
    period_label = "Week" if period.lower() == "week" else "Date"
    # fields available:
    # - %{customdata[0]} -> ASIN
    # - %{y}            -> value
    # - %{customdata[1]} -> xlabel (date/week)
    # - %{customdata[2]} -> pct change (optional)
    base = (
        f"<b>ASIN</b>: %{{customdata[0]}}"
        f"<br><b>{value_label}</b>: %{{y:.2f}}"
        f"<br><b>{period_label}</b>: %{{customdata[1]}}"
    )
    if show_pct:
        base += "<br><b>% change</b>: %{customdata[2]:.2f}%"
    return base + "<extra></extra>"

def _dash_for_asin(asin: str, discount_map: dict) -> str:
    """Return line dash based on whether asin ever had discount."""
    return "dot" if discount_map.get(asin, False) else "solid"


# ------------------------------------------------------------
# 1) Rating Evolution — per ASIN subplots
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
    asins = dfp["asin"].unique().tolist()
    asins.sort()
    n = len(asins)
    if n == 0:
        st.info("No rating data to display.")
        return

    # global y-range
    y_max = float(dfp["product_star_rating"].max(skipna=True))
    y_min = 0.0

    discount_map = _has_discount_by_asin(dfp)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=[f"ASIN {a}" for a in asins])

    row_map = {a: i+1 for i, a in enumerate(asins)}
    hover_tmpl = _hover_template("ASIN", "Rating", show_pct=True, period=period)

    for asin in asins:
        g = dfp[dfp["asin"] == asin].sort_values("x")
        # customdata: [asin, xlabel, rating_change_pct]
        customdata = pd.DataFrame({
            "asin": g["asin"].astype(str),
            "xlabel": g["xlabel"].astype(str),
            "pct": g["rating_change_pct"].astype(float)
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

    _common_layout(fig, n, title="Rating Evolution (by ASIN)",
                   y_title="Product Star Rating", y_min=y_min, y_max=y_max, period=period)

    # annotate max per subplot
    _annotate_max_per_subplot(fig, dfp, ycol="product_star_rating", row_map=row_map, unit_suffix="")

    st.plotly_chart(fig, use_container_width=True)

    # Collapsed table: pivot by period label vs ASIN
    with st.expander("Show rating table"):
        tbl = (dfp.pivot_table(index="xlabel", columns="asin", values="product_star_rating", aggfunc="mean")
                  .sort_index())
        st.dataframe(tbl)


# ------------------------------------------------------------
# 2) Price % Variation — per ASIN subplots
# ------------------------------------------------------------
def plot_price_variation_by_asin(df: pd.DataFrame, period: str = "day") -> None:
    """
    Creates an interactive subplot figure for price percentage variation by ASIN.
    - One row per ASIN.
    - X: day or ISO week (period-aware).
    - Y: price_change (%) (0 .. global max).
    - Line style: dotted if product was ever discounted.
    - Hover: ASIN, Price % change, Date/Week (also shows current price in hover footer).
    - Annotation: max % change in each subplot.
    """
    dfp = _aggregate_by_period(df, period)
    # We may have NaN on first row of each ASIN; keep them (Plotly ignores NaN)
    asins = sorted(dfp["asin"].unique().tolist())
    n = len(asins)
    if n == 0:
        st.info("No price variation data to display.")
        return

    # global y-range (use absolute max to accommodate negative dips; user asked 0..max)
    # If all negative, fallback to 0..0; otherwise 0..max_positive
    max_up = float(dfp["price_change"].max(skipna=True)) if dfp["price_change"].notna().any() else 0.0
    y_min = 0.0
    y_max = max(0.0, max_up)

    discount_map = _has_discount_by_asin(dfp)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=[f"ASIN {a}" for a in asins])

    row_map = {a: i+1 for i, a in enumerate(asins)}
    hover_tmpl = _hover_template("ASIN", "Price % change", show_pct=False, period=period)
    # We'll add price in hoverfoot by injecting it into the value and leaving % change
    # as a separate field (customdata[2]) if you want to show it. Here we keep main value as price_change.

    for asin in asins:
        g = dfp[dfp["asin"] == asin].sort_values("x")
        customdata = pd.DataFrame({
            "asin": g["asin"].astype(str),
            "xlabel": g["xlabel"].astype(str),
            "pct": g["price_change"].astype(float)
        }).to_numpy()

        fig.add_trace(
            go.Scatter(
                x=g["x"],
                y=g["price_change"],
                mode="lines+markers",
                line=dict(dash=_dash_for_asin(asin, discount_map), width=2),
                marker=dict(size=5),
                hovertemplate=_hover_template("ASIN", "Price % change", show_pct=True, period=period),
                customdata=customdata,
                name=f"ASIN {asin}",
            ),
            row=row_map[asin], col=1
        )

    _common_layout(fig, n, title="Price Percentage Variation (by ASIN)",
                   y_title="Price Variation (%)", y_min=y_min, y_max=y_max, period=period)

    _annotate_max_per_subplot(fig, dfp, ycol="price_change", row_map=row_map, unit_suffix="%")

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show price % variation table"):
        tbl = (dfp.pivot_table(index="xlabel", columns="asin", values="price_change", aggfunc="mean")
                  .sort_index())
        st.dataframe(tbl)


# ------------------------------------------------------------
# 3) Ranking Evolution — per ASIN subplots
# ------------------------------------------------------------
def plot_ranking_evolution_by_asin(df: pd.DataFrame, period: str = "day") -> None:
    """
    Creates an interactive subplot figure for ranking evolution by ASIN.
    - Rank per date/week across products (1 = best rating).
    - One row per ASIN, Y shows rank over time.
    - X: day or ISO week (period-aware).
    - Line style: dotted if product was ever discounted.
    - Hover: ASIN, Rank, Date/Week, also shows rating and price % change if available.
    - Annotation: best rank (i.e., min rank value) per subplot.
    """
    dfp = _aggregate_by_period(df, period)
    # compute rank per period across ASINs based on rating (descending => 1 is best)
    dfp["rank"] = (
        dfp.groupby("x")["product_star_rating"]
           .rank(method="first", ascending=False)
    )

    asins = sorted(dfp["asin"].unique().tolist())
    n = len(asins)
    if n == 0:
        st.info("No ranking data to display.")
        return

    # Y range synchronized: from 1 to number of ASINs (optionally allow 0..n for your spec)
    y_min = 1
    y_max = max(1, len(asins))

    discount_map = _has_discount_by_asin(dfp)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=[f"ASIN {a}" for a in asins])

    row_map = {a: i+1 for i, a in enumerate(asins)}
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

    _common_layout(fig, n, title="Ranking Evolution (by ASIN)",
                   y_title="Rank (1 = best)", y_min=y_min, y_max=y_max,
                   period=period, reverse_y=True)

    # Annotate "best" (min rank) per subplot
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

    with st.expander("Show ranking table"):
        tbl = (dfp.pivot_table(index="xlabel", columns="asin", values="rank", aggfunc="mean")
                  .sort_index())
        st.dataframe(tbl)


# ------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------
def main(df: pd.DataFrame, period: str = "day") -> None:
    """
    Build three independent figures (no dropdown), each with its own table:
    1) Rating Evolution (subplots by ASIN)
    2) Price Percentage Variation (subplots by ASIN)
    3) Ranking Evolution (subplots by ASIN)

    Args:
        df: DataFrame expected to contain at least:
            ['asin', 'date', 'product_price', 'product_star_rating'] and optionally 'product_original_price'
        period: 'day' or 'week' (controls x-axis and aggregation)
    """
    st.subheader("Rating — Evolution by ASIN")
    plot_rating_evolution_by_asin(df, period=period)

    st.subheader("Price — Percentage Variation by ASIN")
    plot_price_variation_by_asin(df, period=period)

    st.subheader("Ranking — Evolution by ASIN")
    plot_ranking_evolution_by_asin(df, period=period)

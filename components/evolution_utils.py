# components/evolution_utils.py

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date as _date

# ------------------------------------------------------------
# Data preparation helpers
# ------------------------------------------------------------

def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric columns exist and are coerced to numeric."""
    out = df.copy()
    for col in ("product_price", "product_star_rating", "product_original_price"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["date"] = pd.to_datetime(out.get("date"), errors="coerce")
    return out

def _iso_week_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add ISO year/week columns."""
    iso = df["date"].dt.isocalendar()
    df["iso_year"] = iso.year
    df["iso_week"] = iso.week
    return df

def _week_start(year: int, week: int) -> pd.Timestamp:
    """Return Monday of given ISO week."""
    return pd.Timestamp(_date.fromisocalendar(int(year), int(week), 1))

def _aggregate_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Build an aggregated DataFrame for analysis.
    - day: keep daily rows (sorted).
    - week: aggregate by (asin, iso_year, iso_week) using mean.
    Adds:
      - x (datetime): date or week start
      - xlabel (str): formatted label for hover
      - price_change, rating_change_pct
    """
    base = _ensure_numeric(df)
    base = _iso_week_cols(base)

    if period.lower() == "week":
        grp = base.groupby(["asin", "iso_year", "iso_week"], as_index=False).agg(
            product_price=("product_price", "mean"),
            product_star_rating=("product_star_rating", "mean"),
            discount_any=("product_original_price", lambda s: pd.notna(s).any())
        )
        grp["x"] = grp.apply(lambda r: _week_start(r["iso_year"], r["iso_week"]), axis=1)
        grp = grp.sort_values(["asin", "x"])
        grp["price_change"] = grp.groupby("asin")["product_price"].pct_change() * 100
        grp["rating_change_pct"] = grp.groupby("asin")["product_star_rating"].pct_change() * 100
        grp["xlabel"] = grp.apply(lambda r: f"Week {int(r['iso_year'])}-W{int(r['iso_week']):02d}", axis=1)
        return grp

    # daily
    base = base.sort_values(["asin", "date"]).copy()
    base["discount_any"] = pd.notna(base.get("product_original_price"))
    base["price_change"] = base.groupby("asin")["product_price"].pct_change() * 100
    base["rating_change_pct"] = base.groupby("asin")["product_star_rating"].pct_change() * 100
    base["x"] = base["date"]
    base["xlabel"] = base["date"].dt.strftime("%Y-%m-%d")
    return base


# ------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------

def _has_discount_by_asin(df: pd.DataFrame) -> dict:
    """
    Return dict {asin: bool} if product ever had discount in dataset.
    """
    if "discount_any" in df.columns:
        return df.groupby("asin")["discount_any"].max().astype(bool).to_dict()

    if "product_original_price" in df.columns:
        return df.groupby("asin")["product_original_price"].apply(lambda s: pd.notna(s).any()).astype(bool).to_dict()

    return {asin: False for asin in df["asin"].unique()}

def _common_layout(fig: go.Figure, nrows: int, title: str, y_title: str,
                   y_min: float, y_max: float, period: str,
                   reverse_y: bool = False) -> None:
    """Apply consistent aesthetics, sizing, and axes settings."""
    height_per_row = 240
    fig.update_layout(
        template="plotly_white",
        height=max(260, int(height_per_row * max(1, nrows))),
        title=title,
        margin=dict(t=60, l=70, r=30, b=50),
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(title_text="Week" if period.lower() == "week" else "Date",showticklabels=True)
    
    fig.update_yaxes(range=[y_min, y_max], autorange=False)
    if reverse_y:
        fig.update_yaxes(autorange="reversed")

def _annotate_max_per_subplot(fig: go.Figure, dfp: pd.DataFrame, ycol: str,
                              row_map: dict, unit_suffix: str = "") -> None:
    """
    For each ASIN subplot, add annotation at max point.
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
    Build Plotly hovertemplate for interactive charts.
    """
    period_label = "Week" if period.lower() == "week" else "Date"
    base = (
        f"<b>ASIN</b>: %{{customdata[0]}}"
        f"<br><b>{value_label}</b>: %{{y:.2f}}"
        f"<br><b>{period_label}</b>: %{{customdata[1]}}"
    )
    if show_pct:
        base += "<br><b>% change</b>: %{customdata[2]:.2f}%"
    return base + "<extra></extra>"

def _dash_for_asin(asin: str, discount_map: dict) -> str:
    """Return line style: dotted if ever discounted, else solid."""
    return "dot" if discount_map.get(asin, False) else "solid"

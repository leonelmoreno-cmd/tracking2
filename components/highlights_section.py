# highlights_section.py
import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------
# Compute highlights
# -------------------------------
def compute_highlights(df: pd.DataFrame, period: str = "week") -> dict:
    """
    Compute highlights metrics for the last period.
    Returns a dictionary with max/min discount, price, and price changes.
    """
    if df.empty:
        return {"label": "N/A"}

    if period == "week":
        last_period = int(df["week_number"].max())
        df_period = df[df["week_number"] == last_period].copy()
        label = f"week {last_period}"
    else:
        last_day_ts = df["date"].max()
        last_day = last_day_ts.date()
        df_period = df[df["date"].dt.date == last_day].copy()
        label = last_day.strftime("%Y-%m-%d")

    # Calculate discount percentage
    df_period["discount_pct"] = np.where(
        df_period["product_original_price"].notna() & (df_period["product_original_price"] != 0),
        (df_period["product_original_price"] - df_period["product_price"]) / df_period["product_original_price"] * 100.0,
        np.nan
    )

    # Extremes for discount and price
    row_max_disc = df_period.loc[df_period["discount_pct"].idxmax()] if df_period["discount_pct"].notna().any() else None
    row_min_disc = df_period.loc[df_period["discount_pct"].idxmin()] if df_period["discount_pct"].notna().any() else None
    row_max_price = df_period.loc[df_period["product_price"].idxmax()] if not df_period["product_price"].isna().all() else None
    row_min_price = df_period.loc[df_period["product_price"].idxmin()] if not df_period["product_price"].isna().all() else None

    # Latest row per brand for price change
    latest_by_brand = df_period.loc[df_period.groupby("brand")["date"].idxmax()] if not df_period.empty else pd.DataFrame()
    row_max_change = latest_by_brand.loc[latest_by_brand["price_change"].idxmax()] if not latest_by_brand.empty and latest_by_brand["price_change"].notna().any() else None
    row_min_change = latest_by_brand.loc[latest_by_brand["price_change"].idxmin()] if not latest_by_brand.empty and latest_by_brand["price_change"].notna().any() else None

    # Average price change
    if not latest_by_brand.empty and latest_by_brand["price_change"].notna().any():
        avg_change = float(np.nanmean(latest_by_brand["price_change"]))
        avg_change_n = int(latest_by_brand["brand"].nunique())
    else:
        avg_change = None
        avg_change_n = 0

    return {
        "label": label,
        "row_max_disc": row_max_disc,
        "row_min_disc": row_min_disc,
        "row_max_price": row_max_price,
        "row_min_price": row_min_price,
        "row_max_change": row_max_change,
        "row_min_change": row_min_change,
        "avg_change": avg_change,
        "avg_change_n": avg_change_n,
    }

# -------------------------------
# Render highlights
# -------------------------------
def render_highlights(df_overview: pd.DataFrame, period: str = "week"):
    st.markdown("### Last period highlights")
    highlights = compute_highlights(df_overview, period=period)
    label = highlights.get("label", "N/A")

    dcol, pcol, ccol = st.columns(3)

    # Column 1: discounts
    with dcol:
        if highlights.get("row_max_disc") is not None:
            st.metric(f"🏷️ Highest discount — {label} — {highlights['row_max_disc']['brand']}",
                      f"{highlights['row_max_disc']['discount_pct']:.1f}%")
        else:
            st.metric(f"🏷️ Highest discount — {label}", "N/A")

        if highlights.get("row_min_disc") is not None:
            st.metric(f"🏷️ Lowest discount — {label} — {highlights['row_min_disc']['brand']}",
                      f"{highlights['row_min_disc']['discount_pct']:.1f}%")
        else:
            st.metric(f"🏷️ Lowest discount — {label}", "N/A")

    # Column 2: prices
    with pcol:
        if highlights.get("row_max_price") is not None:
            st.metric(f"💲 Highest price — {label} — {highlights['row_max_price']['brand']}",
                      f"${highlights['row_max_price']['product_price']:.2f}")
        else:
            st.metric(f"💲 Highest price — {label}", "N/A")

        if highlights.get("row_min_price") is not None:
            st.metric(f"💲 Lowest price — {label} — {highlights['row_min_price']['brand']}",
                      f"${highlights['row_min_price']['product_price']:.2f}")
        else:
            st.metric(f"💲 Lowest price — {label}", "N/A")

    # Column 3: price changes
    with ccol:
        if highlights.get("row_max_change") is not None:
            st.metric(f"🔺 Largest price change — {label} — {highlights['row_max_change']['brand']}",
                      f"{highlights['row_max_change']['price_change']:+.1f}%")
        else:
            st.metric(f"🔺 Largest price change — {label}", "N/A")

        if highlights.get("row_min_change") is not None:
            st.metric(f"🔻 Lowest price change — {label} — {highlights['row_min_change']['brand']}",
                      f"{highlights['row_min_change']['price_change']:+.1f}%")
        else:
            st.metric(f"🔻 Lowest price change — {label}", "N/A")

        # Average price change at the bottom
        if highlights.get("avg_change") is not None:
            st.metric(f"↕ Average price change — {label}",
                      f"{highlights['avg_change']:+.1f}%")
        else:
            st.metric(f"↕ Average price change — {label}", "N/A")

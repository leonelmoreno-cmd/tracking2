# components/best_sellers_section.py
import re
import streamlit as st
import pandas as pd
import plotly.express as px
from components.common import (
    GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH,
    _raw_url_for, fetch_data
)

# -------------------------------
# Step 1: Load subcategory data
# -------------------------------
def load_subcategory_data(active_basket_name: str) -> pd.DataFrame:
    """
    Load the sub-category CSV corresponding to the current basket.
    """
    subcategory_url = _raw_url_for(GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH, active_basket_name)
    df_sub = fetch_data(subcategory_url)
    df_sub["date"] = pd.to_datetime(df_sub["date"], errors="coerce")
    return df_sub

# -------------------------------
# Step 2: Get latest date data
# -------------------------------
def get_latest_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Return only the rows corresponding to the latest date in the subcategory data.
    """
    latest_date = df["date"].max()
    df_latest = df[df["date"] == latest_date].copy()
    return df_latest, latest_date

# -------------------------------
# Step 3: Remove duplicates and get top 10 best sellers
# -------------------------------
def top_10_best_sellers(df_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate ASINs and return the top 10 by ascending 'rank' (1 = best).
    """
    df_cleaned = df_latest.drop_duplicates(subset=["asin"], keep="first")
    df_top = df_cleaned.sort_values("rank", ascending=True).head(10).reset_index(drop=True)
    return df_top

# -------------------------------
# Small helper to sanitize photo URLs
# -------------------------------
def _fix_photo_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    url = re.sub(r"(\.jpg)m($|\b)", r"\1", url, flags=re.IGNORECASE)
    url = re.sub(r"(\.png)m($|\b)", r"\1", url, flags=re.IGNORECASE)
    return url

# -------------------------------
# Step 4/5: Render section with horizontal bar chart + image selection
# -------------------------------
def render_best_sellers_section_with_table(active_basket_name: str):
    st.subheader("Best-sellers Rank")
    st.caption("Top 10 products based on the latest available date from the sub-category file.")

    # Load and prepare
    df_sub = load_subcategory_data(active_basket_name)
    if df_sub.empty:
        st.info("No sub-category data found.")
        return

    if "rank" not in df_sub.columns or "asin" not in df_sub.columns:
        st.warning("Required columns 'rank' and/or 'asin' are missing in the sub-category file.")
        return

    df_latest, latest_date = get_latest_data(df_sub)
    if df_latest.empty:
        st.info("No rows for the latest date.")
        return

    df_top10 = top_10_best_sellers(df_latest)
    if df_top10.empty:
        st.info("No top-10 items could be determined.")
        return

    st.markdown(f"**Latest update:** {latest_date.strftime('%Y-%m-%d')}")

    # Prepare chart data with "score" to invert rank (Rank 1 = largest bar)
    df_chart = df_top10.copy()
    df_chart["score"] = df_chart["rank"].max() + 1 - df_chart["rank"]

    # Layout: chart left, selection + image right
    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        st.markdown("**Top 10 by rank** (Rank 1 = longest bar)")
        fig = px.bar(
            df_chart,
            y="asin", x="score",
            orientation="h",
            text="rank",
            title="Top 10 Best-sellers",
            labels={"asin": "ASIN", "score": "Relative size (inverted rank)", "rank": "Rank"},
        )
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=df_chart.sort_values("rank")["asin"].tolist()
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Selection + product image
    asin_to_label = {}
    for _, row in df_top10.iterrows():
        brand = row.get("brand", "")
        asin = row["asin"]
        label = f"{brand} — {asin}" if isinstance(brand, str) and brand else asin
        asin_to_label[asin] = label

    default_asin = df_top10.iloc[0]["asin"]

    with c2:
        st.markdown("**Show product image**")
        selected_label = st.selectbox(
            "Select a product",
            options=list(asin_to_label.values()),
            index=list(asin_to_label.keys()).index(default_asin)
            if default_asin in asin_to_label else 0
        )
        selected_asin = next(a for a, lbl in asin_to_label.items() if lbl == selected_label)
        row_sel = df_top10[df_top10["asin"] == selected_asin].iloc[0]

        photo_url = _fix_photo_url(row_sel.get("product_photo", ""))
        product_url = row_sel.get("product_url", "")
        title = row_sel.get("product_title", "")
        rank_val = row_sel.get("rank", "")

        if photo_url:
            st.image(photo_url, use_container_width=True, caption=f"Rank {rank_val} — {title[:80]}")
        else:
            st.info("No product photo available for this item.")

        if isinstance(product_url, str) and product_url:
            st.markdown(f"[Open product page]({product_url})")

    # Table in expander
    with st.expander("Top 10 Best-sellers Data"):
        cols_to_show = [
            "asin", "brand", "product_title", "rank",
            "product_price", "product_star_rating", "product_num_ratings",
            "product_url", "product_photo"
        ]
        cols_final = [c for c in cols_to_show if c in df_top10.columns]
        st.dataframe(df_top10[cols_final], use_container_width=True)

# components/best_sellers_section.py
import streamlit as st
import pandas as pd
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
    subcategory_url = _raw_url_for(
        GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH, active_basket_name
    )
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
# Step 3: Top 10 best sellers
# -------------------------------
def top_10_best_sellers(df_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate ASINs and return the top 10 best-selling products based on 'rank'.
    """
    df_cleaned = df_latest.drop_duplicates(subset=["asin"], keep="first")
    df_top = df_cleaned.sort_values("rank", ascending=True).head(10).copy()
    return df_top

# -------------------------------
# Small helper to fix photo URLs
# -------------------------------
def _clean_photo_url(url: str) -> str:
    if isinstance(url, str):
        # common trailing typos like ".jpgm" or ".pngm"
        url = url.replace(".jpgm", ".jpg").replace(".pngm", ".png")
        url = url.replace("http://", "https://")
    return url

# -------------------------------
# Step 4/5: Render in Streamlit
# -------------------------------
def render_best_sellers_section_with_table(active_basket_name: str):
    st.subheader("Best-sellers Rank")
    st.caption("Top 10 products based on the latest available date from the sub-category file.")

    # Load subcategory and latest snapshot
    df_sub = load_subcategory_data(active_basket_name)
    if df_sub.empty:
        st.warning("No sub-category data found.")
        return

    df_latest, latest_date = get_latest_data(df_sub)
    if df_latest.empty:
        st.warning("No rows for the latest date.")
        return

    df_top10 = top_10_best_sellers(df_latest)
    if df_top10.empty:
        st.warning("No products found for Top 10.")
        return

    # Clean photo URL if column exists
    if "product_photo" in df_top10.columns:
        df_top10["product_photo_clean"] = df_top10["product_photo"].apply(_clean_photo_url)
    else:
        df_top10["product_photo_clean"] = ""

    st.markdown(f"**Latest update:** {latest_date.strftime('%Y-%m-%d')}")

    # --- Layout: chart (left) + selection & image (right) ---
    left, right = st.columns([2, 1])

    # Chart data for st.bar_chart (note: st.bar_chart is vertical)
    chart_df = (
        df_top10[["asin", "rank"]]
        .set_index("asin")
        .sort_values("rank", ascending=True)
    )

    with left:
        st.bar_chart(chart_df, use_container_width=True)
        st.caption("Lower rank is better (Rank 1 = best).")

    # Build radio options "Brand - ASIN" if brand exists
    def _option_label(row):
        if "brand" in row.index and pd.notna(row["brand"]):
            return f"{row['brand']} — {row['asin']}"
        return row["asin"]

    labels = [_option_label(r) for _, r in df_top10.iterrows()]
    asin_by_label = {label: row["asin"] for label, (_, row) in zip(labels, df_top10.iterrows())}

    with right:
        selected_label = st.radio("Select a product", options=labels, index=0)
        selected_asin = asin_by_label[selected_label]
        row = df_top10[df_top10["asin"] == selected_asin].iloc[0]

        # Show image (if available)
        photo = row.get("product_photo_clean", "")
        title = row.get("product_title", "")
        url = row.get("product_url", "")
        rank = int(row.get("rank", -1)) if pd.notna(row.get("rank", None)) else None

        if isinstance(photo, str) and len(photo) > 5:
            st.image(photo, use_container_width=True, caption=title if isinstance(title, str) else None)
        else:
            st.info("No product image available for this item.")

        # Quick facts
        st.markdown("**Details**")
        if rank is not None:
            st.metric(label="Rank", value=rank)
        price = row.get("product_price", None)
        rating = row.get("product_star_rating", None)
        reviews = row.get("product_num_ratings", None)

        if pd.notna(price):
            st.write(f"**Price:** ${price:,.2f}")
        if pd.notna(rating):
            st.write(f"**Rating:** {rating}")
        if pd.notna(reviews):
            st.write(f"**Reviews:** {int(reviews):,}")

        if isinstance(url, str) and url:
            st.markdown(f"[Open on Amazon]({url})")

    # Data table in expander
    with st.expander("Top 10 Best-sellers Data"):
        cols_to_show = [
            c for c in [
                "asin", "brand", "product_title", "rank", "product_price",
                "product_star_rating", "product_num_ratings", "product_url", "product_photo_clean"
            ] if c in df_top10.columns
        ]
        st.dataframe(df_top10[cols_to_show], use_container_width=True)

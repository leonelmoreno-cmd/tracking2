# components/best_sellers_section.py
import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from components.common import (
    GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH,
    _raw_url_for, fetch_data
)

# -------------------------------
# Step 1: Load subcategory data
# -------------------------------
def load_subcategory_data(active_basket_name: str) -> pd.DataFrame:
    subcategory_url = _raw_url_for(GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH, active_basket_name)
    df_sub = fetch_data(subcategory_url)
    df_sub["date"] = pd.to_datetime(df_sub["date"], errors="coerce")
    return df_sub

# -------------------------------
# Step 2: Get latest date data
# -------------------------------
def get_latest_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    latest_date = df["date"].max()
    df_latest = df[df["date"] == latest_date].copy()
    return df_latest, latest_date

# -------------------------------
# Step 3: Top 10 best sellers
# -------------------------------
def top_10_best_sellers(df_latest: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df_latest.drop_duplicates(subset=["asin"], keep="first")
    df_top = df_cleaned.sort_values("rank", ascending=True).head(10).reset_index(drop=True)
    return df_top

# -------------------------------
# Helper: sanitize photo URLs
# -------------------------------
def _fix_photo_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    url = re.sub(r"(\.jpg)m($|\b)", r"\1", url, flags=re.IGNORECASE)
    url = re.sub(r"(\.png)m($|\b)", r"\1", url, flags=re.IGNORECASE)
    return url

# -------------------------------
# Step 4: Render section
# -------------------------------
def render_best_sellers_section_with_table(active_basket_name: str):
    st.subheader("Best-sellers Rank")
    st.caption("Top 10 products based on the latest available date from the sub-category file.")

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

    # Score invertido: Rank 1 = barra más larga
    df_chart = df_top10.copy()
    df_chart["score"] = df_chart["rank"].max() + 1 - df_chart["rank"]

    # Etiqueta única ASIN
    df_chart["asin_label"] = df_chart["asin"]

    # Ordenar por rank (1 arriba, 10 abajo)
    df_chart = df_chart.sort_values("rank", ascending=True)

    # Layout: gráfico + selección de imagen
    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        st.markdown("**Top 10 by rank** (Rank 1 = longest bar)")
        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=df_chart["asin_label"],      # eje Y único
            x=df_chart["score"],           # eje X invertido
            text=df_chart["rank"],         # mostrar rank
            textposition="outside",
            orientation="h",
            marker_color="orange",
            customdata=df_chart[[
                "asin", "product_title", "product_price",
                "product_star_rating", "product_num_ratings"
            ]].values,
            hovertemplate=(
                "<b>ASIN:</b> %{customdata[0]}<br>"
                "<b>Rank:</b> %{text}<br>"
                "<b>Title:</b> %{customdata[1]}<br>"
                "<b>Price:</b> $%{customdata[2]:.2f}<br>"
                "<b>Rating:</b> %{customdata[3]}<br>"
                "<b>Reviews:</b> %{customdata[4]}<br>"
                "<extra></extra>"
            )
        ))

        fig.update_layout(
            title="Top 10 Best-sellers",
            xaxis_title="Relative size (Rank 1 = best)",
            yaxis_title="ASIN",
            margin=dict(l=80, r=20, t=50, b=40),
            height=500
        )
        st.plotly_chart(fig, width='stretch')

    # Selección de producto + imagen
    asin_to_label = {}
    for _, row in df_top10.iterrows():
        asin = row["asin"]
        label = asin
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

    # Tabla completa
    with st.expander("Top 10 Best-sellers Data"):
        cols_to_show = [
            "asin", "product_title", "rank",
            "product_price", "product_star_rating", "product_num_ratings",
            "product_url", "product_photo"
        ]
        cols_final = [c for c in cols_to_show if c in df_top10.columns]
        st.dataframe(df_top10[cols_final], use_container_width=True)

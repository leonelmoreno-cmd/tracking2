# pages/home.py
import streamlit as st
from components.common import (
    set_page_config, fetch_data, prepare_data,
    _raw_url_for, GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH
)
from components.basket_utils import resolve_active_basket
from components.basket_and_toggle_section import render_basket_and_toggle
from components.visualization import create_overview_graph
from components.overview_section import render_overview_section
from components.current_basket_gallery import render_current_basket_gallery

def main():
    # Page configuration
    set_page_config()
    DEFAULT_BASKET = "synthethic3.csv"
    active_basket_name, active_url, name_to_url = resolve_active_basket(DEFAULT_BASKET)

    # 🔹 Load and prepare initial data to get last update date
    df = fetch_data(active_url)
    prepared_df = prepare_data(df, basket_name=active_basket_name)
    last_update = prepared_df["date"].max()
    last_update_str = (
        last_update.strftime("%Y-%m-%d")
        if prepared_df["date"].notna().any()
        else "N/A"
    )

    # 🔹 Centered header
    st.markdown(
        f"""
        <div style="text-align:center;">
            <h1 style="font-size: 36px; margin-bottom:-15px;">Competitor Price Monitoring</h1>
            <h6 style="color:#666; font-weight:200; margin-top:0;">
                Last update: {last_update_str} - Developed by Leonel Moreno
            </h6>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 🔹 Basket toggle
    period, active_basket_name = render_basket_and_toggle(
        name_to_url, active_basket_name, DEFAULT_BASKET
    )
    active_url = name_to_url.get(active_basket_name, active_url)

    # 🔹 Reload data if basket changed
    df = fetch_data(active_url)
    prepared_df = prepare_data(df, basket_name=active_basket_name)

    # 🔹 Try to enrich with subcategory CSV (photos, urls, titles, etc.)
    try:
        subcat_url = _raw_url_for(GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH, active_basket_name)
        subcat_df = fetch_data(subcat_url)

        if "asin" in prepared_df.columns and "asin" in subcat_df.columns:
            subcat_cols = [
                "asin", "brand", "product_title", "product_price",
                "product_star_rating", "product_num_ratings",
                "product_url", "product_photo", "rank"
            ]
            subcat_cols = [c for c in subcat_cols if c in subcat_df.columns]
            prepared_df = prepared_df.merge(subcat_df[subcat_cols], on="asin", how="left")
    except Exception as e:
        st.warning(f"Could not load subcategory data: {e}")

    # 🔹 Overview section
    df_overview, selected_brands, period = render_overview_section(
        prepared_df, period=period
    )

    # 🔹 Overview chart
    overview_fig = create_overview_graph(
        df_overview, brands_to_plot=None, period=period
    )
    st.plotly_chart(overview_fig, use_container_width=True)

    # 🔹 Basket gallery
    render_current_basket_gallery(prepared_df, columns=5)

if __name__ == "__main__":
    main()

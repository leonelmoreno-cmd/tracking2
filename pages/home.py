import streamlit as st
from components.common import set_page_config, fetch_data, prepare_data
from components.basket_utils import resolve_active_basket
from components.basket_and_toggle_section import render_basket_and_toggle
from components.visualization import create_overview_graph
from components.overview_section import render_overview_section
from components.percentage_var import main as percentage_var_main

def main():
    # Page-specific config if needed
    set_page_config()

    DEFAULT_BASKET = "synthethic3.csv"
    active_basket_name, active_url, name_to_url = resolve_active_basket(DEFAULT_BASKET)

    df = fetch_data(active_url)
    prepared_df = prepare_data(df)
    last_update = prepared_df["date"].max()
    last_update_str = last_update.strftime("%Y-%m-%d") if prepared_df["date"].notna().any() else "N/A"

    st.markdown(
        f"""
        <div style="text-align:center;">
            <h1 style="font-size: 36px; margin-bottom:-15px;">Competitor Price Monitoring</h1>
            <h6 style="color:#666; font-weight:200; margin-top:0;">Last update: {last_update_str} - Developed by Leonel Team</h6>
        </div>
        """,
        unsafe_allow_html=True
    )

    df_overview, selected_brands, period = render_overview_section(prepared_df, period=None)
    overview_fig = create_overview_graph(prepared_df, brands_to_plot=None, period=period)
    st.plotly_chart(overview_fig, use_container_width=True)

    percentage_var_main(prepared_df)

df_overview, selected_brands, period = render_overview_section(prepared_df, period)

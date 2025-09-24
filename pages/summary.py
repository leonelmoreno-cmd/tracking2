import streamlit as st
from components.common import set_page_config, fetch_data, prepare_data
from components.visualization import create_price_graph
from components.overview_section import render_overview_section
from components.basket_utils import resolve_active_basket
from components.basket_and_toggle_section import render_basket_and_toggle

def main():
    set_page_config()

    DEFAULT_BASKET = "synthethic3.csv"
    active_basket_name, active_url, name_to_url = resolve_active_basket(DEFAULT_BASKET)

    df = fetch_data(active_url)
    prepared_df = prepare_data(df)

    # Basket toggle
    period, active_basket_name = render_basket_and_toggle(name_to_url, active_basket_name, DEFAULT_BASKET)
    active_url = name_to_url.get(active_basket_name, active_url)

    price_fig = create_price_graph(prepared_df, period=period)
    st.plotly_chart(price_fig, use_container_width=True)

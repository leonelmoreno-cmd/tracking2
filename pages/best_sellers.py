import streamlit as st
from components.common import set_page_config, fetch_data, prepare_data
from components.best_sellers_section import render_best_sellers_section_with_table
from components.basket_utils import resolve_active_basket
from components.basket_and_toggle_section import render_basket_and_toggle
from components.header import display_header

def main():
    set_page_config()

    DEFAULT_BASKET = "synthethic3.csv"
    active_basket_name, active_url, name_to_url = resolve_active_basket(DEFAULT_BASKET)

    # Optionally allow the user to toggle baskets
    period, active_basket_name = render_basket_and_toggle(name_to_url, active_basket_name, DEFAULT_BASKET)
    active_url = name_to_url.get(active_basket_name, active_url)

    df = fetch_data(active_url)
    prepared_df = prepare_data(df, basket_name=active_basket_name)

    st.header("Best Sellers")
    display_header(prepared_df)
    render_best_sellers_section_with_table(active_basket_name)

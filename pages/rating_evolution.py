# pages/rating_evolution.py

import streamlit as st
from components.common import set_page_config, fetch_data, prepare_data
from components.basket_utils import resolve_active_basket
from components.basket_and_toggle_section import render_basket_and_toggle
from components.rating_evolution import plot_rating_evolution_by_asin_grid


def main():
    set_page_config()

    # Default CSV
    DEFAULT_BASKET = "synthethic3.csv"
    active_basket_name, active_url, name_to_url = resolve_active_basket(DEFAULT_BASKET)

    # Allow basket toggle + daily/weekly toggle
    period, active_basket_name = render_basket_and_toggle(
        name_to_url, active_basket_name, DEFAULT_BASKET
    )
    active_url = name_to_url.get(active_basket_name, active_url)

    # Load + prepare data
    df = fetch_data(active_url)
    prepared_df = prepare_data(df, basket_name=active_basket_name)

    if prepared_df is None or prepared_df.empty:
        st.warning("No data available. Load data first.")
    else:
        plot_rating_evolution_by_asin_grid(prepared_df, period=period)


if __name__ == "__main__":
    main()

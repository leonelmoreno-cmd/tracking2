import streamlit as st
from components.common import set_page_config, fetch_data, prepare_data
from components.basket_utils import resolve_active_basket
from components.basket_and_toggle_section import render_basket_and_toggle
from components.detailed_table_section import render_detailed_table
from components.percentage_var import main as percentage_var_main
def main():
    set_page_config()

    DEFAULT_BASKET = "synthethic3.csv"
    active_basket_name, active_url, name_to_url = resolve_active_basket(DEFAULT_BASKET)

    period, active_basket_name = render_basket_and_toggle(name_to_url, active_basket_name, DEFAULT_BASKET)
    active_url = name_to_url.get(active_basket_name, active_url)

    df = fetch_data(active_url)
    prepared_df = prepare_data(df)

    percentage_var_main(prepared_df)
    
    st.header("Detailed Product Table")

    if prepared_df is None or prepared_df.empty:
        st.warning("No data available. Load data first.")
    else:
        filtered_df = render_detailed_table(prepared_df)

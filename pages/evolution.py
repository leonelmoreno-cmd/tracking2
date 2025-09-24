# pages/evolution.py
import streamlit as st
from components.common import set_page_config, fetch_data, prepare_data
from components.basket_utils import resolve_active_basket
from components.basket_and_toggle_section import render_basket_and_toggle
from components.percentage_var import main as percentage_var_main


def main():
    set_page_config()

    # CSV por defecto
    DEFAULT_BASKET = "synthethic3.csv"
    active_basket_name, active_url, name_to_url = resolve_active_basket(DEFAULT_BASKET)

    # Permitir cambiar de dataset
    period, active_basket_name = render_basket_and_toggle(
        name_to_url, active_basket_name, DEFAULT_BASKET
    )
    active_url = name_to_url.get(active_basket_name, active_url)

    # Cargar y preparar datos
    df = fetch_data(active_url)
    prepared_df = prepare_data(df, basket_name=active_basket_name)

    # Mostrar la sección de evolución
    st.header("Evolution / Percentage Variation")
    if prepared_df is None or prepared_df.empty:
        st.warning("No data available. Load data first.")
    else:
        percentage_var_main(prepared_df)

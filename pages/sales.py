import streamlit as st
import pandas as pd
import numpy as np
from components.common import set_page_config, fetch_data, prepare_data
from components.basket_utils import resolve_active_basket
from components.basket_and_toggle_section import render_basket_and_toggle


def main():
    set_page_config()

    # CSV por defecto
    DEFAULT_BASKET = "synthethic3.csv"
    active_basket_name, active_url, name_to_url = resolve_active_basket(DEFAULT_BASKET)

    # Toggle de dataset
    period, active_basket_name = render_basket_and_toggle(
        name_to_url, active_basket_name, DEFAULT_BASKET
    )
    active_url = name_to_url.get(active_basket_name, active_url)

    # Cargar y preparar datos
    df = fetch_data(active_url)
    prepared_df = prepare_data(df)

    st.header("📈 Sales Overview")

    if prepared_df is None or prepared_df.empty:
        st.warning("No data available. Load data first.")
        return

    # --- Ejemplo ficticio: generar ventas simuladas por marca ---
    sales_data = (
        prepared_df.groupby("brand")["product_price"]
        .sum()
        .reset_index()
        .rename(columns={"product_price": "total_sales"})
    )
    # Agregar ruido ficticio de ventas
    sales_data["total_sales"] = sales_data["total_sales"] * (1 + np.random.rand(len(sales_data)))

    # Mostrar tabla
    st.subheader("Sales by Brand (simulated)")
    st.dataframe(sales_data, use_container_width=True)

    # Mostrar gráfico de barras
    st.bar_chart(sales_data.set_index("brand")["total_sales"])


if __name__ == "__main__":
    main()


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from components.common import set_page_config, fetch_data, prepare_data
from components.basket_utils import resolve_active_basket
from components.basket_and_toggle_section import render_basket_and_toggle


def main():
    set_page_config()

    # CSV por defecto
    DEFAULT_BASKET = "synthethic3.csv"
    active_basket_name, active_url, name_to_url = resolve_active_basket(DEFAULT_BASKET)

    # Toggle de dataset (y daily/weekly si aplica en otras páginas)
    period, active_basket_name = render_basket_and_toggle(
        name_to_url, active_basket_name, DEFAULT_BASKET
    )
    active_url = name_to_url.get(active_basket_name, active_url)

    # Cargar y preparar datosd
    df = fetch_data(active_url)
    prepared_df = prepare_data(df, basket_name=active_basket_name)

    st.header("Sales Overview")

    if prepared_df is None or prepared_df.empty:
        st.warning("No data available. Load data first.")
        return

    # 1) Agregar ventas simuladas por marca (base: suma de precios)
    sales_data = (
        prepared_df.groupby("brand", dropna=False)["product_price"]
        .sum()
        .reset_index()
        .rename(columns={"product_price": "total_sales"})
    )
    # 2) Agregar un ruido ficticio para simular variación de ventas
    sales_data["total_sales"] = sales_data["total_sales"] * (1 + np.random.rand(len(sales_data)))

    # 3) Ordenar de mayor a menor para el gráfico
    sales_sorted = sales_data.sort_values("total_sales", ascending=False)

    # 4) Gráfico de barras HORIZONTAL y orden controlado
    fig = px.bar(
        sales_sorted,
        x="total_sales",
        y="brand",
        orientation="h",
        title="Sales by Brand (simulated)",
        labels={"total_sales": "Total Sales (simulated)", "brand": "Brand"},
    )
    # Mantener exactamente el orden del DataFrame (mayor → menor)
    fig.update_yaxes(categoryorder="array", categoryarray=sales_sorted["brand"].tolist())
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=40))

    st.plotly_chart(fig, use_container_width=True)

    # 5) Tabla debajo del gráfico
    st.subheader("Sales by Brand (simulated)")
    st.dataframe(sales_sorted, use_container_width=True)


if __name__ == "__main__":
    main()

import streamlit as st
from components.common import set_page_config, fetch_data, prepare_data
from components.basket_utils import resolve_active_basket
from components.basket_and_toggle_section import render_basket_and_toggle
from components.visualization import create_overview_graph
from components.overview_section import render_overview_section

def main():
    set_page_config()

    DEFAULT_BASKET = "synthethic3.csv"
    active_basket_name, active_url, name_to_url = resolve_active_basket(DEFAULT_BASKET)

    st.markdown(
        f"""
        <div style="text-align:center;">
            <h1 style="font-size: 36px; margin-bottom:-15px;">Competitor Price Monitoring</h1>
            <h6 style="color:#666; font-weight:200; margin-top:0;">Last update: {last_update_str} - Developed by Leonel Moreno </h6>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 🔹 Agregar el bloque de toggle + selector (ANTES de cargar datos)
    period, active_basket_name = render_basket_and_toggle(
        name_to_url, active_basket_name, DEFAULT_BASKET
    )
    # Si cambió el basket, usa su URL actualizada
    active_url = name_to_url.get(active_basket_name, active_url)

    # 🔹 Cargar y preparar datos del basket actualmente seleccionado
    df = fetch_data(active_url)
    prepared_df = prepare_data(df)

    # Cabecera con última actualización
    last_update = prepared_df["date"].max()
    last_update_str = last_update.strftime("%Y-%m-%d") if prepared_df["date"].notna().any() else "N/A"

    # 🔹 Pasar el period seleccionado al overview (y usar el DF filtrado que devuelve)
    df_overview, selected_brands, period = render_overview_section(prepared_df, period=period)

    # 🔹 Graficar usando el DF ya filtrado (fecha/marcas) y el periodo correcto
    overview_fig = create_overview_graph(df_overview, brands_to_plot=None, period=period)
    st.plotly_chart(overview_fig, use_container_width=True)

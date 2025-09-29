def main():
    set_page_config()

    DEFAULT_BASKET = "synthethic3.csv"
    active_basket_name, active_url, name_to_url = resolve_active_basket(DEFAULT_BASKET)

    # 1) Toggle primero: periodo y basket definitivos
    period, active_basket_name = render_basket_and_toggle(
        name_to_url, active_basket_name, DEFAULT_BASKET
    )
    active_url = name_to_url.get(active_basket_name, active_url)

    # 2) Cargar y preparar con el basket activo
    df = fetch_data(active_url)
    prepared_df = prepare_data(df, basket_name=active_basket_name)

    display_header(prepared_df)

    # --- Sección 1: Pricing Breakdown por ASIN (gráfico + tabla) ---
    st.header("Breakdown by ASIN")
    price_fig = create_price_graph(prepared_df, period=period)
    st.plotly_chart(price_fig, use_container_width=True)

    with st.expander("Show price table"):
        df_table = prepared_df.copy()
        if period == "day":
            df_table["xlabel"] = pd.to_datetime(
                df_table["date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        else:
            df_table["xlabel"] = df_table["week_number"].astype(int)

        df_table["brand_asin"] = (
            df_table["brand"].fillna("Unknown") + " - " + df_table["asin"].astype(str)
        )

        tbl = (
            df_table.pivot_table(
                index="xlabel",
                columns="brand_asin",
                values="product_price",
                aggfunc="mean"
            ).sort_index()
        )
        st.dataframe(tbl)

    # --- Sección 2: Price Percentage Variation (antes era una página aparte) ---
    st.divider()
    st.header("Price Percentage Variation (by ASIN)")
    if prepared_df is None or prepared_df.empty:
        st.warning("No data available. Load data first.")
    else:
        plot_price_variation_by_asin(prepared_df, period=period)

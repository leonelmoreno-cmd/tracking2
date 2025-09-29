# pages/ranking_evolution.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from components.evolution_utils import (
    _aggregate_by_period,
    _has_discount_by_asin,
    _common_layout,
    _dash_for_asin,
)

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
    display_header(prepared_df)

    st.header("Ranking Evolution (by ASIN)")
    if prepared_df is None or prepared_df.empty:
        st.warning("No data available. Load data first.")
    else:
        # 🔑 pasamos df y period a la función que ahora incluye sub_category_name en los títulos
        plot_ranking_evolution_by_asin(prepared_df, period=period)


if __name__ == "__main__":
    main()

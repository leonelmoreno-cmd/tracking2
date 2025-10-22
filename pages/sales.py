# pages/sales.py
from __future__ import annotations
import streamlit as st
import pandas as pd

from components.common import (
    set_page_config, fetch_data, prepare_data,
    list_repo_csvs,
)
from components.basket_utils import resolve_active_basket
from components.basket_and_toggle_section import render_basket_and_toggle
from components.header import display_header

from sales_core.config import DEFAULT_BASKET
from sales_core.pipeline import run_pipeline
from sales_core.week_utils import four_full_weeks_window
from sales_core.ui_sections import (
    render_overview_filters_and_highlights,
    create_overview_graph,
    render_breakdown,
)

def main():
    set_page_config()

    # Pick active basket
    active_basket_name, active_url, name_to_url = resolve_active_basket(DEFAULT_BASKET)

    # Display header if prior CSV available (non-blocking)
    try:
        df_existing = fetch_data(active_url)
        prepared = prepare_data(df_existing, basket_name=active_basket_name)
        display_header(prepared)
    except Exception:
        st.info("No pre-existing CSV loaded; we will fetch Jungle Scout data now.")

    # Keep your existing basket toggle UI (period is ignored; we are weekly only)
    period, active_basket_name = render_basket_and_toggle(name_to_url, active_basket_name, DEFAULT_BASKET)
    active_url = name_to_url.get(active_basket_name, active_url)

    # Run ETL pipeline (ASINs → JS daily → brand map → weekly → write CSV)
    with st.spinner("Building weekly dataset from Jungle Scout..."):
        weekly = run_pipeline(active_basket_name, competitor_download_url=active_url)

    if weekly is None or weekly.empty:
        st.warning("No weekly data available.")
        return

    # Available weeks for dropdown
    available_weeks = sorted(weekly["week_end"].dropna().unique().tolist())
    st.header("📊 Sales Dashboard")
    st.caption(lambda: f"Window: {four_full_weeks_window()[0].date()} to {four_full_weeks_window()[1].date()}")

    df_overview, selected_brands, selected_week_end = render_overview_filters_and_highlights(weekly, available_weeks)
    fig = create_overview_graph(df_overview, selected_brands or None)
    st.plotly_chart(fig, use_container_width=True)
    render_breakdown(df_overview)

if __name__ == "__main__":
    main()

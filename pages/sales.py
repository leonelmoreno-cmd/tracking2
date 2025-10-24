# pages/sales.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from sales_core.repo_io import read_weekly_csv_remote, read_weekly_csv_local


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

    active_basket_name, active_url, name_to_url = resolve_active_basket(DEFAULT_BASKET)
    period, active_basket_name = render_basket_and_toggle(name_to_url, active_basket_name, DEFAULT_BASKET)

    st.header("📊 Sales Dashboard")

    # 👇 SOLO leemos el CSV generado por el workflow
    try:
        weekly = read_weekly_csv_remote(active_basket_name)   # <- remoto recomendado
        # weekly = read_weekly_csv_local(active_basket_name)  # <- alternativa local
    except Exception as e:
        st.warning(f"Weekly CSV not available for '{active_basket_name}'. "
                   f"Run the GitHub Action (cron Friday or manual dispatch). Details: {e}")
        return

    if weekly is None or weekly.empty:
        st.warning("No weekly data available for this basket.")
        return

available_weeks = sorted(weekly["week_end"].dropna().unique().tolist())

(df_overview,
 selected_brands,
 selected_week_end,
 metric_col,
 metric_y_title,
 metric_hover_y,
 metric_prefix) = render_overview_filters_and_highlights(weekly, available_weeks)

fig = create_overview_graph(
    df_overview,
    selected_brands or None,
    metric_col=metric_col,
    metric_y_title=metric_y_title,
    metric_hover_y=metric_hover_y
)
st.plotly_chart(fig, use_container_width=True)

render_breakdown(
    df_overview,
    metric_col=metric_col,
    metric_y_title=metric_y_title,
    metric_hover_y=metric_hover_y
)

if __name__ == "__main__":
    main()

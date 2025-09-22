import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict
from common import (
    GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH,
    _raw_url_for, fetch_data, prepare_data, list_repo_csvs, set_page_config
)
from visualization import create_overview_graph, create_price_graph
from best_sellers_section import render_best_sellers_section_with_table
from highlights_section import render_highlights
from product_table_section import render_product_table
import percentage_var

# -------------------------------
# Page config
# -------------------------------
set_page_config()

# -------------------------------
# Repo constants
# -------------------------------
DEFAULT_BASKET = "synthethic3.csv"

# -------------------------------
# Resolve active basket (URL & session)
# -------------------------------
csv_items = list_repo_csvs(GITHUB_OWNER, GITHUB_REPO, GITHUB_PATH, GITHUB_BRANCH)
name_to_url: Dict[str, str] = {it["name"]: it["download_url"] for it in csv_items}

qp = st.query_params.to_dict() if hasattr(st, "query_params") else {}
qp_basket = qp.get("basket")
if isinstance(qp_basket, list):
    qp_basket = qp_basket[0] if qp_basket else None

if "basket" not in st.session_state:
    st.session_state["basket"] = qp_basket if qp_basket else DEFAULT_BASKET

active_basket_name = st.session_state["basket"]
active_url = name_to_url.get(
    active_basket_name,
    _raw_url_for(GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH, active_basket_name)
)

# -------------------------------
# Load & prepare data
# -------------------------------
df = fetch_data(active_url)
prepared_df = prepare_data(df)
last_update = prepared_df["date"].max()
last_update_str = last_update.strftime("%Y-%m-%d") if pd.notna(last_update) else "N/A"

# -------------------------------
# Title + Subtitle
# -------------------------------
st.markdown(
    f"""
    <div style="text-align:center;">
        <h1 style="font-size: 36px; margin-bottom:-15px;">Competitor Price Monitoring</h1>
        <h6 style="color:#666; font-weight:200; margin-top:0;">Last update: {last_update_str} - Developed by JC Team</h6>
    </div>
    """,
    unsafe_allow_html=True
)

# ===============================
# Centered container (Current + Change basket + Global toggle)
# ===============================
col1, col2, col3 = st.columns([3, 2, 2], gap="small")

with col1:
    st.markdown(
        f"<div style='text-align:left; margin:4px 0;'>"
        f"<span style='color:#16a34a; font-weight:600;'>Current basket:</span> "
        f"<code style='color:#16a34a;'>{active_basket_name}</code>"
        f"</div>",
        unsafe_allow_html=True
    )

with col2:
    with st.popover("🧺 Change basket"):
        st.caption("Pick a CSV from the list and click Apply.")
        options = list(name_to_url.keys()) if name_to_url else [DEFAULT_BASKET]
        idx = options.index(active_basket_name) if active_basket_name in options else 0
        sel = st.selectbox("File (CSV) in repo", options=options, index=idx, key="basket_select")
        if st.button("Apply", type="primary"):
            st.session_state["basket"] = sel
            if hasattr(st, "query_params"):
                st.query_params["basket"] = sel
            else:
                try:
                    st.experimental_set_query_params(basket=sel)
                except Exception:
                    pass
            st.rerun()

with col3:
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    aggregate_daily = st.toggle(
        "Aggregate by day (instead of week)",
        value=False,
        help="When ON, all charts use daily prices; when OFF, weekly averages."
    )
    period = "day" if aggregate_daily else "week"

# ===============================
# Overview chart
# ===============================
st.subheader("Overview — All Brands")
st.caption("Use the controls below to filter the overview. The metrics summarize the latest period across selected brands.")

left_col, right_col = st.columns([0.7, 2.3], gap="large")
all_brands = sorted(prepared_df["brand"].dropna().unique().tolist())
date_min, date_max = prepared_df["date"].dropna().min(), prepared_df["date"].dropna().max()

with left_col:
    st.caption("Select the brands to filter the overview chart.")
    selected_brands = st.multiselect(
        "Brands to display (overview)",
        options=all_brands,
        default=all_brands,
        help="Select the brands you want to compare in the overview chart."
    )
    overview_date_range = None
    if pd.notna(date_min) and pd.notna(date_max):
        overview_date_range = st.date_input(
            "Filter by date (overview)",
            value=(date_min.date(), date_max.date()),
            min_value=date_min.date(),
            max_value=date_max.date(),
            help="Choose a start/end date to constrain the overview."
        )

with right_col:
    df_overview = prepared_df.copy()
    if overview_date_range:
        dstart, dend = overview_date_range if isinstance(overview_date_range, tuple) else (overview_date_range, overview_date_range)
        df_overview = df_overview[(df_overview["date"].dt.date >= dstart) & (df_overview["date"].dt.date <= dend)]
    if selected_brands:
        df_overview = df_overview[df_overview["brand"].isin(selected_brands)]

    # ✅ Render highlights using the modular component
    render_highlights(df_overview, period=period)

# -------------------------------
# Overview chart with Plotly
# -------------------------------
overview_fig = create_overview_graph(df_overview, brands_to_plot=selected_brands, period=period)
st.plotly_chart(overview_fig, use_container_width=True)

# ===============================
# Percentage variation charts
# ===============================
percentage_var.main(prepared_df)

# ===============================
# Best sellers section
# ===============================
render_best_sellers_section_with_table(active_basket_name)

# ===============================
# Subplots by brand/ASIN
# ===============================
st.subheader("By Brand — Individual ASINs")
st.caption("Each small chart tracks a single ASIN. Subplot titles link to the product pages.")
price_graph = create_price_graph(prepared_df, period=period)
st.plotly_chart(price_graph, use_container_width=True)

# ===============================
# Detailed product table + filters (modular)
# ===============================
from product_table_section import render_product_table
render_product_table(prepared_df, all_brands, date_min, date_max)

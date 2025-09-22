import streamlit as st
from common import set_page_config, fetch_data, prepare_data
from visualization import create_overview_graph, create_price_graph
from best_sellers_section import render_best_sellers_section_with_table
from highlights_section import render_highlights
from overview_section import render_overview_section
from detailed_table_section import render_detailed_table
from basket_utils import resolve_active_basket  # <-- modularizado
from basket_and_toggle_section import render_basket_and_toggle  # <-- nuevo componente
import percentage_var

# -------------------------------
# Page config
# -------------------------------
set_page_config()

# -------------------------------
# Repo constants & Resolve active basket (modular)
# -------------------------------
DEFAULT_BASKET = "synthethic3.csv"
active_basket_name, active_url, name_to_url = resolve_active_basket(DEFAULT_BASKET)

# -------------------------------
# Load & prepare data
# -------------------------------
df = fetch_data(active_url)
prepared_df = prepare_data(df)
last_update = prepared_df["date"].max()
last_update_str = last_update.strftime("%Y-%m-%d") if prepared_df["date"].notna().any() else "N/A"

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
# Modular Basket + Toggle Section
# ===============================
period, active_basket_name = render_basket_and_toggle(name_to_url, active_basket_name, DEFAULT_BASKET)
active_url = name_to_url.get(
    active_basket_name,
    f"https://raw.githubusercontent.com/OWNER/REPO/BRANCH/PATH/{active_basket_name}"  # reemplazar según tu lógica _raw_url_for
)

# ===============================
# Modular Overview + Highlights
# ===============================
df_overview, selected_brands = render_overview_section(prepared_df, period)

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
# Modular Detailed Product Table + Filters
# ===============================
filtered_df = render_detailed_table(prepared_df)

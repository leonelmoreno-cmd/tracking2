import streamlit as st
from common import set_page_config, fetch_data, prepare_data
from visualization import create_overview_graph, create_price_graph
from best_sellers_section import render_best_sellers_section_with_table
from highlights_section import render_highlights
from overview_section import render_overview_section
from detailed_table_section import render_detailed_table
from basket_utils import resolve_active_basket  # <-- modularizado
import percentage_var

# -------------------------------
# Page config
# -------------------------------
set_page_config()

# -------------------------------
# Repo constants
# -------------------------------
DEFAULT_BASKET = "synthethic3.csv"

# ===============================
# Resolve active basket (modular)
# ===============================
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
# Centered container (Current + Change basket + Global toggle)
# ===============================
# Nota: Esta sección aún se puede modularizar si quieres, por ahora se mantiene inline
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

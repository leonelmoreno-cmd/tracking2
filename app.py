# app.py
# app.py
import streamlit as st

# ✅ AQUÍ. FUERA DE main(). ARRIBA DEL TODO.
st.set_page_config(
    page_title="Competitor Price Monitoring",
    page_icon="📊",
    layout="wide",
)
import streamlit as st
from pages.home import main as home_page
from pages.summary import main as summary_page
from pages.best_sellers import main as best_sellers_page
from pages.detailed_table import main as detailed_table_page
from pages.settings import main as settings_page
from pages.traffic import main as traffic_page
from pages.sales import main as sales
from pages.ngram import main as ngram_page
from pages.campaigns_evolution import main as campaigns_evolution_page
from pages.rating_evolution import main as rating_evolution_page
from pages.ranking_evolution import main as ranking_evolution_page
from pages.placements import main as placements
from pages.parameters import main as parameters_page
from pages.ad_product_type import main as ad_product_type_page
from pages.placements_sb import main as placements_sb_page
from pages.placement_analysis import main as placement_analysis_page
def main():


    pages = [
        st.Page(home_page, title="Overview", icon="📊", default=True),
        st.Page(summary_page, title="Pricing Breakdown by ASIN", icon="📈", url_path="summary"),
        # Modularized evolution pages
        st.Page(rating_evolution_page, title="Rating Evolution", icon="⭐", url_path="rating-evolution"),
        st.Page(ranking_evolution_page, title="Ranking Evolution", icon="🏆", url_path="ranking-evolution"),
        # Other existing pages
        st.Page(sales, title="Sales Estimate", icon="💰", url_path="sales"),
        st.Page(best_sellers_page, title="Best Sellers Rank", icon="🥇", url_path="best-sellers"),
        st.Page(detailed_table_page, title="Detailed Table", icon="📋", url_path="detailed-table"),
        st.Page(traffic_page, title="Web Traffic", icon="🌐", url_path="traffic"),
        st.Page(ngram_page, title="N-gram", icon="🔤", url_path="ngram"),
        st.Page(campaigns_evolution_page, title="Campaigns Evolution", icon="🔄", url_path="campaigns-evolution"),
        st.Page(placements,title="Placements SP", icon="🕵️",url_path="placements"),
        st.Page(placements_sb_page, title="Placements SB", icon="🕵️‍♀️", url_path="placements-sb"),
        st.Page(parameters_page, title="Parameters", icon="🧮", url_path="parameters"),
        st.Page(ad_product_type_page, title="Ad product type", icon="🧾", url_path="ad-product-type"),
        st.Page(placement_analysis_page,title="Placement Analysis",icon="📍",url_path="placement-analysis"),
        st.Page(settings_page, title="Settings", icon="⚙️", url_path="settings"),
    ]

    nav = st.navigation(pages, position="sidebar")
    nav.run()


if __name__ == "__main__":
    main()

# app.py
import streamlit as st
from pages.home import main as home_page
from pages.summary import main as summary_page
from pages.best_sellers import main as best_sellers_page
from pages.detailed_table import main as detailed_table_page
from pages.settings import main as settings_page
from pages.traffic import main as traffic_page   
from pages.evolution import main as evolution
def main():
    st.set_page_config(page_title="Competitor Price Monitoring", page_icon="📊")

    pages = [
        st.Page(home_page, title="Overview", icon="📊", default=True),
        st.Page(summary_page, title="Breakdown by ASIN", icon="📈", url_path="summary"),
        st.Page(evolution, title="Evolution", icon="🔍", url_path="evolution"),
        st.Page(best_sellers_page, title="Best Sellers Rank", icon="⭐", url_path="best-sellers"),
        st.Page(detailed_table_page, title="Detailed Table", icon="📋", url_path="detailed-table"),
        st.Page(traffic_page, title="Web Traffic", icon="🌐", url_path="traffic"),
        st.Page(settings_page, title="Settings", icon="⚙️", url_path="settings"),
    ]

    nav = st.navigation(pages, position="sidebar")
    nav.run()

if __name__ == "__main__":
    main()

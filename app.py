import streamlit as st
from pages.home import main as home_page
from pages.summary import main as summary_page
from pages.best_sellers import main as best_sellers_page
from pages.detailed_table import main as detailed_table_page
from pages.settings import main as settings_page

def main():
    # Optional: configuration common to all pages
    st.set_page_config(page_title="Competitor Price Monitoring", page_icon="📊")

    # Global widgets / sidebar items (shared across pages)
    # Example: global basket selection or period filter could go here
    # But your existing logic for basket & toggle maybe stays in pages

    # Define pages using st.Page
    pages = [
        st.Page(home_page, title="Home", icon="🏠", default=True),
        st.Page(summary_page, title="Summary", icon="📈"),
        st.Page(best_sellers_page, title="Best Sellers", icon="⭐"),
        st.Page(detailed_table_page, title="Detailed Table", icon="📋"),
        st.Page(settings_page, title="Settings", icon="⚙️"),
    ]

    # Create navigation menu
    pg = st.navigation(pages, position="sidebar")
    pg.run()

if __name__ == "__main__":
    main()

# app.py
import streamlit as st

# ✅ PRIMERA llamada Streamlit (no debe haber nada antes)
st.set_page_config(
    page_title="Competitor Price Monitoring",
    page_icon="📊",
    layout="wide",
)

# ------------------------------------------------------------------
# Page wrappers (IMPORT DIFERIDO: evita ejecutar st.* en imports)
# ------------------------------------------------------------------

def home_page():
    from pages.home import main
    main()

def summary_page():
    from pages.summary import main
    main()

def best_sellers_page():
    from pages.best_sellers import main
    main()

def detailed_table_page():
    from pages.detailed_table import main
    main()

def settings_page():
    from pages.settings import main
    main()

def traffic_page():
    from pages.traffic import main
    main()

def sales_page():
    from pages.sales import main
    main()

def ngram_page():
    from pages.ngram import main
    main()

def campaigns_evolution_page():
    from pages.campaigns_evolution import main
    main()

def rating_evolution_page():
    from pages.rating_evolution import main
    main()

def ranking_evolution_page():
    from pages.ranking_evolution import main
    main()

def placements_page():
    from pages.placements import main
    main()

def parameters_page():
    from pages.parameters import main
    main()

def placements_sb_page():
    from pages.placements_sb import main
    main()

# ------------------------------------------------------------------
# App router
# ------------------------------------------------------------------

def main():
    pages = [
        st.Page(home_page, title="Overview", icon="📊", default=True),
        st.Page(summary_page, title="Pricing Breakdown by ASIN", icon="📈", url_path="summary"),
        st.Page(rating_evolution_page, title="Rating Evolution", icon="⭐", url_path="rating-evolution"),
        st.Page(ranking_evolution_page, title="Ranking Evolution", icon="🏆", url_path="ranking-evolution"),
        st.Page(sales_page, title="Sales Estimate", icon="💰", url_path="sales"),
        st.Page(best_sellers_page, title="Best Sellers Rank", icon="🥇", url_path="best-sellers"),
        st.Page(detailed_table_page, title="Detailed Table", icon="📋", url_path="detailed-table"),
        st.Page(traffic_page, title="Web Traffic", icon="🌐", url_path="traffic"),
        st.Page(ngram_page, title="N-gram", icon="🔤", url_path="ngram"),
        st.Page(campaigns_evolution_page, title="Campaigns Evolution", icon="🔄", url_path="campaigns-evolution"),
        st.Page(placements_page, title="Placements SP", icon="🕵️", url_path="placements"),
        st.Page(placements_sb_page, title="Placements SB", icon="🕵️‍♀️", url_path="placements-sb"),
        st.Page(parameters_page, title="Parameters", icon="🧮", url_path="parameters"),
        st.Page(settings_page, title="Settings", icon="⚙️", url_path="settings"),
    ]

    st.navigation(pages, position="sidebar").run()


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from common import GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH, _raw_url_for, fetch_data


# -------------------------------
# Step 1: Load subcategory data
# -------------------------------
def load_subcategory_data(active_basket_name: str) -> pd.DataFrame:
    """
    Load the sub-category CSV corresponding to the current basket.
    """
    subcategory_url = _raw_url_for(GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH, active_basket_name)
    df_sub = fetch_data(subcategory_url)
    df_sub["date"] = pd.to_datetime(df_sub["date"], errors="coerce")
    return df_sub

# -------------------------------
# Step 2: Get latest date data
# -------------------------------
def get_latest_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Return only the rows corresponding to the latest date in the subcategory data.
    """
    latest_date = df["date"].max()  # Obtener la fecha más reciente
    df_latest = df[df["date"] == latest_date].copy()  # Filtrar datos de esa fecha
    return df_latest, latest_date

# -------------------------------
# Step 3: Top 10 best sellers
# -------------------------------
def top_10_best_sellers(df_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Return the top 10 best-selling products based on the rank column.
    """
    # Ordenar por rank y obtener los 10 primeros
    df_top = df_latest.sort_values("rank").head(10)  
    return df_top

# -------------------------------
# Step 4: Create Plotly bar chart
# -------------------------------
def create_best_sellers_bar(df_top: pd.DataFrame) -> go.Figure:
    """
    Create a horizontal bar chart for the top 10 best sellers.
    """
    # Crear gráfico de barras horizontal
    fig = go.Figure(go.Bar(
        x=df_top["rank"],  # Utilizamos 'rank' para ordenar de menor a mayor
        y=df_top["asin"],  # Usamos 'asin' como etiquetas
        orientation='h',
        text=df_top["rank"],  # Mostrar el ranking en cada barra
        textposition='auto',  # Mostrar texto dentro de las barras
        marker_color='orange'
    ))

    fig.update_layout(
        title="Top 10 Best-sellers by Rank",
        xaxis_title="Rank",
        yaxis_title="ASIN",
        height=500,
        margin=dict(l=150, r=20, t=50, b=50)  # Ajuste de márgenes
    )
    return fig

# -------------------------------
# Step 5: Streamlit section
# -------------------------------
def render_best_sellers_section(active_basket_name: str):
    st.subheader("Top 10 Best-sellers by Rank")
    st.caption("Top 10 products based on the latest available date from the sub-category file.")

    # Cargar datos de subcategorías
    df_sub = load_subcategory_data(active_basket_name)
    df_latest, latest_date = get_latest_data(df_sub)
    df_top10 = top_10_best_sellers(df_latest)

    st.markdown(f"**Latest update:** {latest_date.strftime('%Y-%m-%d')}")

    # Generar gráfico de barras
    best_sellers_fig = create_best_sellers_bar(df_top10)
    st.plotly_chart(best_sellers_fig, use_container_width=True)

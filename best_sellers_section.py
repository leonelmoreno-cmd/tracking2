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
    latest_date = df["date"].max()  # Get the latest date
    df_latest = df[df["date"] == latest_date].copy()  # Filter rows for that date
    return df_latest, latest_date

# -------------------------------
# Step 3: Remove duplicates and get top 10 best sellers
# -------------------------------
def top_10_best_sellers(df_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate ASINs and return the top 10 best-selling products based on the 'rank' column.
    Since rank 1 is the best, we sort in ascending order and get the top 10.
    """
    df_cleaned = df_latest.drop_duplicates(subset=['asin'], keep='first')
    df_top = df_cleaned.sort_values("rank").head(10)
    return df_top

# -------------------------------
# Step 4: Create Vertical Bar Chart with Product Images
# -------------------------------
import plotly.graph_objects as go
import pandas as pd

def create_best_sellers_vertical_bar(df_top: pd.DataFrame, basket_name: str) -> go.Figure:
    """
    Crea un gráfico de barras verticales para los 10 mejores vendedores con imágenes de productos.
    Cada barra representa un ASIN con una imagen asociada en la parte superior.
    """
    fig = go.Figure()

    # Ruta del archivo CSV correspondiente a la cesta activa
    subcategory_csv = f"data/sub-categories2/{basket_name}"

    # Cargar los datos del archivo CSV
    df_sub = pd.read_csv(subcategory_csv)

    # Crear un diccionario para mapear ASIN a URL de imagen
    asin_to_image = dict(zip(df_sub["asin"], df_sub["product_photo"]))

    # Añadir una traza para cada ASIN como una barra vertical
    for _, row in df_top.iterrows():
        asin = row["asin"]
        rank = row["rank"]
        title = row["product_title"]
        price = row["product_price"]
        rating = row["product_star_rating"]
        num_ratings = row["product_num_ratings"]

        # Obtener la URL de la imagen del producto
        image_url = asin_to_image.get(asin, "")

        # Añadir la barra al gráfico
        fig.add_trace(go.Bar(
            x=[asin],
            y=[rank],
            text=[asin],
            textposition='outside',
            marker_color='orange',
            hovertemplate=(
                f'<b>ASIN:</b> {asin}<br>'
                f'<b>Rank:</b> {rank}<br>'
                f'<b>Title:</b> {title}<br>'
                f'<b>Price:</b> ${price:.2f}<br>'
                f'<b>Rating:</b> {rating}<br>'
                f'<b>Reviews:</b> {num_ratings}<br>'
                '<extra></extra>'
            ),
            customdata=[[title, price, rating, num_ratings]],
            name="Best-sellers"
        ))

        # Añadir la imagen sobre la barra
        if image_url:
            fig.add_trace(go.Scatter(
                x=[asin],
                y=[rank + 0.2],  # Posicionar la imagen ligeramente por encima de la barra
                mode="markers+text",
                text=[f"<img src='{image_url}' width='90px' />"],
                textposition="bottom center",
                marker=dict(size=0),
                showlegend=False
            ))

    # Configuración del diseño del gráfico
    fig.update_layout(
        title="Top 10 Best-sellers Rank",
        xaxis_title="ASIN",
        yaxis_title="Rank",
        yaxis_autorange="reversed",
        height=600,
        margin=dict(l=80, r=20, t=50, b=100),
        showlegend=False
    )

    return fig


# -------------------------------
# Step 5: Streamlit sections
# -------------------------------
def render_best_sellers_section_with_table(active_basket_name: str):
    st.subheader("Best-sellers Rank")  # Section header
    st.caption("Top 10 products based on the latest available date from the sub-category file.")

    # Load subcategory data
    df_sub = load_subcategory_data(active_basket_name)
    df_latest, latest_date = get_latest_data(df_sub)  # Get the latest data
    df_top10 = top_10_best_sellers(df_latest)  # Get the top 10 best sellers

    st.markdown(f"**Latest update:** {latest_date.strftime('%Y-%m-%d')}")  # Display the latest date

    # Plot the chart
    best_sellers_fig = create_best_sellers_vertical_bar(df_top10)  # Create the vertical bar chart with images
    st.plotly_chart(best_sellers_fig, use_container_width=True)  # Display the chart in Streamlit

    # Display the data table below the chart
    st.subheader("Top 10 Best-sellers Data")
    st.dataframe(df_top10)  # Show the table with the top 10 ASINs and their ranks

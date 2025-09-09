import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------
# Configuración de la página
# -------------------------------
st.set_page_config(
    page_title="Competitors Price Tracker",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# Paleta/estilos del tema (contraste con fondo #1d293d)
TEXT_COLOR = "#e2e8f0"
LINK_COLOR = "#615fff"
BG_COLOR = "#1d293d"  # para referencia

# -------------------------------
# Data loading
# -------------------------------
@st.cache_data
def fetch_data():
    url = "https://raw.githubusercontent.com/leonelmoreno-cmd/tracking2/main/data/synthethic2.csv"
    df = pd.read_csv(url)
    return df

@st.cache_data
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Asegura date como datetime y crea week (ISO)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['week'] = df['date'].dt.isocalendar().week.astype(int)

    # Etiqueta de "descuento existente"
    df['discount'] = df.apply(
        lambda row: 'Discounted' if pd.notna(row.get('product_original_price')) else 'No Discount',
        axis=1
    )
    # Cambio porcentual por ASIN
    df['price_change'] = df.groupby('asin')['product_price'].pct_change() * 100

    # Asegura columnas que usaremos en títulos
    if 'brand' not in df.columns:
        df['brand'] = df['product_title'].str.split().str[0]
    if 'product_url' not in df.columns:
        df['product_url'] = "https://www.amazon.com/dp/" + df['asin'].astype(str)

    return df

# -------------------------------
# Plot helper
# -------------------------------
def create_price_graph(df: pd.DataFrame) -> go.Figure:
    asins = df['asin'].dropna().unique()
    num_asins = len(asins)

    # Layout en 3 columnas
    cols = 3 if num_asins >= 3 else max(1, num_asins)
    rows = int(np.ceil(num_asins / cols))

    fig = make_subplots(
        rows=rows, cols=cols, shared_xaxes=True,
        vertical_spacing=0.08, horizontal_spacing=0.06,
        subplot_titles=["" for _ in asins]  # luego los sustituimos con anotaciones ricas
    )

    # Para reemplazar títulos luego con brand + enlace + asin
    titles_meta = []

    for i, asin in enumerate(asins):
        asin_data = df[df['asin'] == asin].sort_values('date')
        if asin_data.empty:
            continue

        # Estilo de línea: punteada si en algún día hubo "Discounted"
        dashed = 'dot' if (asin_data['discount'] == 'Discounted').any() else 'solid'

        r = i // cols + 1
        c = i % cols + 1

        # customdata: [price_change, date_str]
        customdata = np.stack([
            asin_data['price_change'].astype(float).fillna(0).values,
            asin_data['date'].dt.strftime('%Y-%m-%d').values
        ], axis=-1)

        fig.add_trace(
            go.Scatter(
                x=asin_data['week'],                 # Semana (eje X)
                y=asin_data['product_price'],        # Precio (eje Y)
                mode='lines+markers',
                name=str(asin),
                line=dict(dash=dashed),
                hovertemplate=(
                    'ASIN: %{text}<br>' +
                    'Week: %{x}<br>' +
                    'Date: %{customdata[1]}<br>' +
                    'Price: $%{y:.2f}<br>' +
                    'Price Change: %{customdata[0]:.2f}%<br>' +
                    '<extra></extra>'
                ),
                text=asin_data['asin'],
                customdata=customdata,
                showlegend=False
            ),
            row=r, col=c
        )

        # Guarda metadata para el título enriquecido (marca linkeable + asin)
        brand = asin_data['brand'].iloc[0] if 'brand' in asin_data.columns else str(asin)
        url = asin_data['product_url'].iloc[0] if 'product_url' in asin_data.columns else f"https://www.amazon.com/dp/{asin}"
        titles_meta.append((r, c, brand, url, asin))

    # Escala uniforme en Y para TODOS los subplots: [0, max_price_global]
    max_price = float(df['product_price'].max())
    fig.update_yaxes(range=[0, max_price])

    # Ejes y estilo
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(
                row=r, col=c,
                title_text="Week #",
                tickmode='auto',
                type='linear'  # semanas como números
            )
            fig.update_yaxes(
                row=r, col=c,
                title_text="Product Price (USD)"
            )

    fig.update_layout(
        height=max(420, 300 * rows),
        margin=dict(l=20, r=20, t=70, b=20),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR)
    )

    # Sustituir los títulos de cada subplot por anotaciones HTML ricas (marca link + asin debajo)
    # Eliminamos anotaciones automáticas (subplot_titles) y añadimos las nuestras
    fig.layout.annotations = tuple(a for a in fig.layout.annotations if a.text != "")

    for (r, c, brand, url, asin) in titles_meta:
        # Título (marca) linkeable + subtítulo (ASIN)
        # Nota: el color del link se fuerza con style; el subtítulo con <span>
        title_html = (
            f"<b><a href='{url}' target='_blank' style='color:{LINK_COLOR};"
            f"text-decoration:none;'>{brand}</a></b>"
            f"<br><span style='color:{TEXT_COLOR}; font-size:12px;'>ASIN: {asin}</span>"
        )

        fig.add_annotation(
            text=title_html,
            xref=f"x{(r-1)*cols+c} domain",
            yref=f"y{(r-1)*cols+c} domain",
            x=0.5, y=1.12, showarrow=False,
            align="center",
            font=dict(size=14, color=TEXT_COLOR),
        )

    return fig

# -------------------------------
# Main UI
# -------------------------------
df = fetch_data()
prepared_df = prepare_data(df)

# Última actualización (fecha máxima del dataset)
last_update = prepared_df['date'].max()
last_update_str = last_update.strftime('%Y-%m-%d') if pd.notna(last_update) else 'N/A'

# Título + Subtítulo centrados
st.markdown(
    f"""
    <div style="text-align:center;">
        <h1 style="font-size: 36px; margin-bottom: 4px; color:{TEXT_COLOR};">
            Competitors Price Tracker
        </h1>
        <h3 style="color:#9ca3af; font-weight:400; margin-top:0;">
            Última actualización: {last_update_str}
        </h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Gráfico
price_graph = create_price_graph(prepared_df)
st.plotly_chart(price_graph, use_container_width=True)

# -------------------------------
# Tabla con filtros
# -------------------------------
st.subheader("Detailed Product Information")

asin_options = ['All'] + prepared_df['asin'].dropna().unique().tolist()
discount_options = ['All', 'Discounted', 'No Discount']

asin_filter = st.selectbox("Filter by ASIN", options=asin_options, index=0)
discount_filter = st.selectbox("Filter by Discount Status", options=discount_options, index=0)

filtered_df = prepared_df.copy()
if asin_filter != 'All':
    filtered_df = filtered_df[filtered_df['asin'] == asin_filter]
if discount_filter != 'All':
    filtered_df = filtered_df[filtered_df['discount'] == discount_filter]

st.dataframe(
    filtered_df[[
        'asin', 'brand', 'product_title', 'product_price', 'product_original_price',
        'product_star_rating', 'product_num_ratings', 'is_amazon_choice',
        'sales_volume', 'discount', 'date', 'week'
    ]]
)

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

    # Tipos
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Semana ISO (número de semana para eje X)
    # Si ya existe la columna 'week', la respetamos; si no, la creamos
    if 'week' not in df.columns:
        df['week'] = df['date'].dt.isocalendar().week

    # Etiqueta de descuento (si hay precio original no nulo)
    df['discount'] = df.apply(
        lambda row: 'Discounted' if pd.notna(row.get('product_original_price')) else 'No Discount',
        axis=1
    )

    # Cambio porcentual por ASIN
    df['price_change'] = df.groupby('asin')['product_price'].pct_change() * 100

    return df

# -------------------------------
# Plot helper
# -------------------------------
def create_price_graph(df: pd.DataFrame) -> go.Figure:
    asins = df['asin'].dropna().unique()
    num_asins = len(asins)

    # Grid en 3 columnas
    cols = 3 if num_asins >= 3 else max(1, num_asins)
    rows = int(np.ceil(num_asins / cols))

    # Prepara títulos "placeholder" (los reemplazaremos con anotaciones personalizadas)
    fig = make_subplots(
        rows=rows, cols=cols, shared_xaxes=True,
        vertical_spacing=0.08, horizontal_spacing=0.06,
        subplot_titles=["" for _ in range(num_asins)]  # vacío; usaremos anotaciones propias
    )

    # Rango Y uniforme
    max_price = float(df['product_price'].max())
    # Rango X por semana
    min_week = int(df['week'].min())
    max_week = int(df['week'].max())

    # Color de títulos/subtítulos (contrastan con #1d293d)
    title_color = "#e2e8f0"

    for i, asin in enumerate(asins):
        asin_data = df[df['asin'] == asin].sort_values('date')
        if asin_data.empty:
            continue

        # Estilo de línea: punteada si en algún día hubo "Discounted"
        dashed = 'dot' if (asin_data['discount'] == 'Discounted').any() else 'solid'

        r = i // cols + 1
        c = i % cols + 1

        # Trazo usando semana como eje X
        fig.add_trace(
            go.Scatter(
                x=asin_data['week'],
                y=asin_data['product_price'],
                mode='lines+markers',
                name=str(asin),
                line=dict(dash=dashed),
                hovertemplate=(
                    'ASIN: %{customdata[0]}<br>' +
                    'Brand: %{customdata[1]}<br>' +
                    'Week: %{x}<br>' +
                    'Date: %{customdata[2]|%Y-%m-%d}<br>' +
                    'Price: $%{y:.2f}<br>' +
                    'Price Change: %{customdata[3]:.2f}%<br>' +
                    'URL: %{customdata[4]}<br>' +
                    '<extra></extra>'
                ),
                customdata=np.stack([
                    asin_data['asin'],
                    asin_data.get('brand', pd.Series(['']*len(asin_data))),
                    asin_data['date'],
                    asin_data['price_change'],
                    asin_data.get('product_url', pd.Series(['']*len(asin_data)))
                ], axis=1),
                showlegend=False
            ),
            row=r, col=c
        )

        # Texto de título para cada subplot: link a product_url con la brand + " — ASIN: ..."
        brand = (asin_data['brand'].dropna().iloc[0]
                 if 'brand' in asin_data.columns and asin_data['brand'].notna().any()
                 else str(asin))
        url = (asin_data['product_url'].dropna().iloc[0]
               if 'product_url' in asin_data.columns and asin_data['product_url'].notna().any()
               else "")

        # Intentamos usar <a href="..."> para que sea cliqueable.
        # Nota: En algunos entornos Plotly, los enlaces en anotaciones pueden no ser clicables;
        # de todas formas se verá claramente y la URL está en el hover también.
        title_text = f'<a href="{url}" target="_blank" style="color:{title_color}; text-decoration:underline;">{brand}</a> — ASIN: {asin}'

        # Agregamos anotación como "título" del subplot
        fig.add_annotation(
            text=title_text,
            xref=f"x{(i+1) if (i+1)>1 else ''} domain",  # dominio del subplot
            yref=f"y{(i+1) if (i+1)>1 else ''} domain",
            x=0.0, y=1.12,  # un poco arriba del área del subplot
            showarrow=False,
            align="left",
            font=dict(color=title_color, size=12)
        )

        # Asegura rangos por subplot (X semanas, Y precio)
        fig.update_xaxes(range=[min_week, max_week], row=r, col=c, title_text="Week")
        fig.update_yaxes(range=[0, max_price], row=r, col=c, title_text="Product Price (USD)")

    # Layout general
    fig.update_layout(
        height=max(420, 300 * rows),
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig

# -------------------------------
# Main UI
# -------------------------------
df = fetch_data()
prepared_df = prepare_data(df)

# Última actualización (máxima fecha del dataset)
last_update = prepared_df['date'].max()
last_update_str = last_update.strftime('%Y-%m-%d') if pd.notna(last_update) else 'N/A'

# Título + Subtítulo centrados (en inglés)
st.markdown(
    f"""
    <div style="text-align:center;">
        <h1 style="font-size: 36px; margin-bottom: 4px; color:#e2e8f0;">Competitors Price Tracker</h1>
        <h3 style="color:#e2e8f0; font-weight:400; margin-top:0;">Last update: {last_update_str}</h3>
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
        'asin', 'product_title', 'product_price', 'product_original_price',
        'product_star_rating', 'product_num_ratings', 'is_amazon_choice',
        'sales_volume', 'discount', 'date', 'week'
    ]]
)

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
    url = "https://raw.githubusercontent.com/leonelmoreno-cmd/tracking2/main/data/synthethic3.csv"
    df = pd.read_csv(url)
    return df

@st.cache_data
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Semana ISO para eje X y hover
    iso = df['date'].dt.isocalendar()
    df['week'] = iso.week  # número de semana ISO
    df['iso_year'] = iso.year

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

    # Creamos subplots SIN títulos automáticos; agregaremos anotaciones personalizadas
    fig = make_subplots(
        rows=rows, cols=cols, shared_xaxes=True,
        vertical_spacing=0.08, horizontal_spacing=0.06
    )

    for i, asin in enumerate(asins):
        asin_data = df[df['asin'] == asin].sort_values('date')
        if asin_data.empty:
            continue

        # Estilo de línea: punteada si en algún día hubo "Discounted"
        dashed = 'dot' if (asin_data['discount'] == 'Discounted').any() else 'solid'

        r = i // cols + 1
        c = i % cols + 1

        # customdata: [price_change, week]
        customdata = np.column_stack([
            asin_data['price_change'].to_numpy(),
            asin_data['week'].to_numpy()
        ])

        fig.add_trace(
            go.Scatter(
                x=asin_data['date'],                    # mantenemos fechas para suavidad
                y=asin_data['product_price'],
                mode='lines+markers',
                name=str(asin),
                line=dict(dash=dashed),
                hovertemplate=(
                    'ASIN: %{text}<br>' +
                    'Price: $%{y:.2f}<br>' +
                    'Date: %{x|%Y-%m-%d}<br>' +
                    'Week: %{customdata[1]}<br>' +
                    'Price Change: %{customdata[0]:.2f}%<br>' +
                    '<extra></extra>'
                ),
                text=asin_data['asin'],
                customdata=customdata,
                showlegend=False
            ),
            row=r, col=c
        )

        # ----- Anotaciones personalizadas por subplot -----
        # Obtenemos brand y url del primer registro disponible
        brand = asin_data['brand'].dropna().iloc[0] if 'brand' in asin_data and not asin_data['brand'].dropna().empty else str(asin)
        url = asin_data['product_url'].dropna().iloc[0] if 'product_url' in asin_data and not asin_data['product_url'].dropna().empty else None

        # Dominio (posición) del subplot en la figura para centrar el título
        subplot_index = i + 1
        xaxis_key = f'xaxis{subplot_index}' if subplot_index > 1 else 'xaxis'
        yaxis_key = f'yaxis{subplot_index}' if subplot_index > 1 else 'yaxis'

        x_domain = fig.layout[xaxis_key].domain
        y_domain = fig.layout[yaxis_key].domain
        x_center = (x_domain[0] + x_domain[1]) / 2.0
        y_top = y_domain[1]

        title_color = "#e2e8f0"   # alto contraste con #1C293C/#1d293d
        subtitle_color = "#cbd5e1" # un poco más suave pero legible

        # Título linkeable (brand)
        if url:
            title_text = f'<a href="{url}" target="_blank" style="text-decoration:none; color:{title_color};">{brand}</a>'
        else:
            title_text = f'<span style="color:{title_color};">{brand}</span>'

        # Subtítulo con ASIN
        subtitle_text = f'<span style="color:{subtitle_color};">ASIN: {asin}</span>'

        # Anotación del título (arriba del subplot)
        fig.add_annotation(
            x=x_center, y=y_top + 0.08, xref='paper', yref='paper',
            text=title_text, showarrow=False,
            xanchor='center', yanchor='bottom',
            font=dict(size=14)
        )
        # Anotación del subtítulo (debajo del título)
        fig.add_annotation(
            x=x_center, y=y_top + 0.04, xref='paper', yref='paper',
            text=subtitle_text, showarrow=False,
            xanchor='center', yanchor='bottom',
            font=dict(size=12)
        )

    # Escala uniforme en Y para TODOS los subplots: [0, max_price_global]
    max_price = float(df['product_price'].max())
    fig.update_yaxes(range=[0, max_price])

    # Mostrar número de semana en el eje X (formato ISO, %V)
    # Manteniendo la serie temporal diaria, solo cambiamos el formato de los ticks del eje X.
    fig.update_xaxes(tickformat="%V", title_text="Week #")

    fig.update_layout(
        height=max(420, 300 * rows),
        xaxis_title="Week #",            # título genérico; cada sub-eje también lo hereda
        yaxis_title="Product Price (USD)",
        margin=dict(l=20, r=20, t=80, b=30),
    )
    return fig

# -------------------------------
# Main UI
# -------------------------------
df = fetch_data()
prepared_df = prepare_data(df)

# Last update (fecha máxima del dataset)
last_update = prepared_df['date'].max()
last_update_str = last_update.strftime('%Y-%m-%d') if pd.notna(last_update) else 'N/A'

# Título + Subtítulo centrados
st.markdown(
    f"""
    <div style="text-align:center;">
        <h1 style="font-size: 36px; margin-bottom: 4px;">Competitors Price Tracker</h1>
        <h3 style="color:#666; font-weight:400; margin-top:0;">Last update: {last_update_str}</h3>
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

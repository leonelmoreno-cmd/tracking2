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

    # Etiqueta de descuento (si hay precio original no nulo)
    df['discount'] = np.where(
        df['product_original_price'].notna(),
        'Discounted', 'No Discount'
    )

    # ---- Agregación semanal ----
    # Definimos el inicio de semana (Lunes). Puedes cambiar a 'W-SUN' si prefieres semanas que terminan domingo.
    df['week_start'] = df['date'].dt.to_period('W-MON').apply(lambda p: p.start_time)

    # Mapeo (asin -> marca/url) tomando el registro más reciente por asin
    latest_meta = (
        df.sort_values('date')
          .groupby('asin', as_index=False)
          .tail(1)[['asin', 'brand', 'product_url']]
    )

    # Agregamos por semana y asin
    weekly = (
        df.groupby(['asin', 'week_start'], as_index=False)
          .agg(
              product_price=('product_price', 'mean'),
              product_original_price=('product_original_price', 'mean'),
              # Cualquier Discounted en la semana -> Discounted
              any_discount=('discount', lambda s: (s == 'Discounted').any()),
          )
    )

    # Recalcular label de descuento semanal coherente
    weekly['discount'] = np.where(
        weekly['product_original_price'].notna() & (weekly['product_original_price'] > 0) &
        (weekly['product_price'] < weekly['product_original_price']),
        'Discounted', 'No Discount'
    )
    # Si alguien marcó any_discount True pero la regla anterior no lo capturó, mantenemos True por consistencia
    weekly.loc[weekly['any_discount'], 'discount'] = 'Discounted'
    weekly = weekly.drop(columns=['any_discount'])

    # Cambio porcentual semana a semana por asin
    weekly['price_change'] = (
        weekly.sort_values(['asin', 'week_start'])
              .groupby('asin')['product_price']
              .pct_change() * 100
    )

    # Añadimos brand y url al weekly
    weekly = weekly.merge(latest_meta, on='asin', how='left')

    return weekly

# -------------------------------
# Plot helper (semanal)
# -------------------------------
def create_price_graph(weekly_df: pd.DataFrame) -> go.Figure:
    asins = weekly_df['asin'].dropna().unique()
    num_asins = len(asins)

    # Layout en 3 columnas
    cols = 3 if num_asins >= 3 else max(1, num_asins)
    rows = int(np.ceil(num_asins / cols))

    # Títulos por subplot: marca como link a la product_url
    # (usamos anotaciones de Plotly que aceptan HTML <a href="...">)
    titles = []
    asin_brand_url = (
        weekly_df[['asin', 'brand', 'product_url']]
        .drop_duplicates('asin')
        .set_index('asin')
        .to_dict(orient='index')
    )
    for asin in asins:
        meta = asin_brand_url.get(asin, {})
        brand = meta.get('brand') or str(asin)
        url = meta.get('product_url') or ''
        # Si no hay URL, dejamos solo el texto
        if url:
            titles.append(f"<a href='{url}' target='_blank'>{brand}</a>")
        else:
            titles.append(brand)

    fig = make_subplots(
        rows=rows, cols=cols, shared_xaxes=True,
        vertical_spacing=0.08, horizontal_spacing=0.06,
        subplot_titles=titles
    )

    # Añadimos trazas
    for i, asin in enumerate(asins):
        asin_data = weekly_df[weekly_df['asin'] == asin].sort_values('week_start')
        if asin_data.empty:
            continue

        dashed = 'dot' if (asin_data['discount'] == 'Discounted').any() else 'solid'
        r = i // cols + 1
        c = i % cols + 1

        fig.add_trace(
            go.Scatter(
                x=asin_data['week_start'],
                y=asin_data['product_price'],
                mode='lines+markers',
                name=str(asin),
                line=dict(dash=dashed),
                hovertemplate=(
                    'ASIN: %{text}<br>' +
                    'Avg Weekly Price: $%{y:.2f}<br>' +
                    'Week Start: %{x|%Y-%m-%d}<br>' +
                    'WoW Δ: %{customdata:.2f}%<br>' +
                    '<extra></extra>'
                ),
                text=asin_data['asin'],
                customdata=asin_data['price_change'],
                showlegend=False
            ),
            row=r, col=c
        )

    # Escala uniforme Y: [0, max_price_global]
    max_price = float(weekly_df['product_price'].max())
    fig.update_yaxes(range=[0, max_price])

    # Formato de eje X semanal
    fig.update_xaxes(
        tickformat="%Y-%m-%d",  # muestra fecha del inicio de semana
        ticklabelmode="period"
    )

    fig.update_layout(
        height=max(400, 280 * rows),
        xaxis_title="Week (start date)",
        yaxis_title="Product Price (USD)",
        margin=dict(l=20, r=20, t=50, b=20)
    )

    # Asegura que las anotaciones (títulos) con <a> se muestren centradas
    # (make_subplots ya crea las annotations; solo nos aseguramos de permitir HTML)
    if 'annotations' in fig.layout:
        for ann in fig.layout.annotations:
            ann.align = 'center'  # centra el texto
            # Plotly permite HTML en annotations; Streamlit respeta en el renderer

    return fig

# -------------------------------
# Main UI
# -------------------------------
df = fetch_data()
weekly_df = prepare_data(df)

# Última actualización (fecha máxima original del dataset)
last_update = pd.to_datetime(df['date'], errors='coerce').max()
last_update_str = last_update.strftime('%Y-%m-%d') if pd.notna(last_update) else 'N/A'

# Título + Subtítulo centrados
st.markdown(
    f"""
    <div style="text-align:center;">
        <h1 style="font-size: 36px; margin-bottom: 4px;">Competitors Price Tracker</h1>
        <h3 style="color:#666; font-weight:400; margin-top:0;">Last Update: {last_update_str}</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Gráfico semanal
price_graph = create_price_graph(weekly_df)
st.plotly_chart(price_graph, use_container_width=True)

# -------------------------------
# Tabla con filtros (semanal)
# -------------------------------
st.subheader("Detailed Product Information (Weekly)")

asin_options = ['All'] + weekly_df['asin'].dropna().unique().tolist()
discount_options = ['All', 'Discounted', 'No Discount']

asin_filter = st.selectbox("Filter by ASIN", options=asin_options, index=0)
discount_filter = st.selectbox("Filter by Discount Status", options=discount_options, index=0)

filtered_df = weekly_df.copy()
if asin_filter != 'All':
    filtered_df = filtered_df[filtered_df['asin'] == asin_filter]
if discount_filter != 'All':
    filtered_df = filtered_df[filtered_df['discount'] == discount_filter]

# Mostramos columnas clave (semanales)
st.dataframe(
    filtered_df[[
        'asin', 'brand', 'product_price', 'product_original_price',
        'discount', 'price_change', 'week_start', 'product_url'
    ]].rename(columns={
        'product_price': 'avg_weekly_price',
        'product_original_price': 'avg_weekly_original_price'
    })
)

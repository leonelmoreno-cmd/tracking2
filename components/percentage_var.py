import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@st.cache_data(show_spinner=False)
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Asegurar columnas numéricas
    df["product_price"] = pd.to_numeric(df.get("product_price"), errors="coerce")
    df["product_star_rating"] = pd.to_numeric(df.get("product_star_rating"), errors="coerce")
    # extraer año + semana para orden (opcional)
    iso = df["date"].dt.isocalendar()
    df["week_number"] = iso.week
    df["year"] = iso.year
    # Ordenar por asin y fecha
    df = df.sort_values(by=["asin", "year", "week_number", "date"])
    # Etiqueta descuento — más estricta si quieres
    if "product_original_price" in df.columns:
        df["product_original_price"] = pd.to_numeric(df["product_original_price"], errors="coerce")
        df["discount"] = df.apply(
            lambda row: "Discounted" if (pd.notna(row.product_original_price) and row.product_price < row.product_original_price)
                        else "No Discount",
            axis=1
        )
    else:
        df["discount"] = "No Discount"
    # Variación de precio (%)
    df["price_change"] = df.groupby("asin")["product_price"].pct_change() * 100
    return df

def plot_all_rating_evolutions(df: pd.DataFrame):
    # Crear un subplot con una fila por ASIN (o hasta N) — si hay muchos, limita
    asins = df["asin"].unique()
    n = len(asins)
    # si muchos ASIN, podrías limitar o paginar (aquí lo hacemos directo)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=[f"ASIN {asin}" for asin in asins])
    for i, asin in enumerate(asins, start=1):
        asin_data = df[df["asin"] == asin].sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=asin_data["date"],
                y=asin_data["product_star_rating"],
                mode="lines+markers",
                name=f"ASIN {asin}"
            ),
            row=i, col=1
        )
    fig.update_layout(height=300 * n, title_text="Evolución del Rating por ASIN")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Star Rating")
    st.plotly_chart(fig, use_container_width=True)
    # tabla resumen dentro de un expander
    with st.expander("Mostrar tabla de ratings por ASIN"):
        st.dataframe(df[["asin", "date", "product_star_rating"]].pivot(index="date", columns="asin"))

def plot_all_price_variations(df: pd.DataFrame):
    asins = df["asin"].unique()
    n = len(asins)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=[f"ASIN {asin}" for asin in asins])
    for i, asin in enumerate(asins, start=1):
        asin_data = df[df["asin"] == asin].sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=asin_data["date"],
                y=asin_data["price_change"],
                mode="lines+markers",
                name=f"ASIN {asin}"
            ),
            row=i, col=1
        )
    fig.update_layout(height=300 * n, title_text="Variación % del Precio por ASIN")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price Change (%)")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Mostrar tabla de variaciones de precio"):
        st.dataframe(df[["asin", "date", "price_change"]].pivot(index="date", columns="asin"))

def plot_all_ranking_evolutions(df: pd.DataFrame):
    # Primero calcular ranking por fecha, para que dentro de cada fecha rankee entre asin
    df2 = df.copy()
    df2["ranking"] = df2.groupby("date")["product_star_rating"].rank(method="first", ascending=False)
    asins = df2["asin"].unique()
    n = len(asins)
    # Ahora queremos un subplot por ASIN mostrando cómo varía su ranking a lo largo del tiempo
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=[f"ASIN {asin}" for asin in asins])
    for i, asin in enumerate(asins, start=1):
        asin_data = df2[df2["asin"] == asin].sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=asin_data["date"],
                y=asin_data["ranking"],
                mode="lines+markers",
                name=f"ASIN {asin}"
            ),
            row=i, col=1
        )
    fig.update_layout(height=300 * n, title_text="Evolución del Ranking por ASIN")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Ranking (1 = mejor)")
    # invertimos el eje y para que 1 aparezca arriba
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Mostrar tabla de rankings"):
        st.dataframe(df2[["asin", "date", "ranking"]].pivot(index="date", columns="asin"))

def main(df: pd.DataFrame):
    # Mostrar los 3 gráficos, cada uno con su expander de tabla
    st.subheader("Evolución del Rating por Producto")
    plot_all_rating_evolutions(df)

    st.subheader("Variación Porcentual de Precio por Producto")
    plot_all_price_variations(df)

    st.subheader("Evolución de Ranking por Producto")
    plot_all_ranking_evolutions(df)

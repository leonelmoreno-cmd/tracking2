# pages/sales.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from components.common import set_page_config, fetch_data, prepare_data
from components.basket_utils import resolve_active_basket
from components.basket_and_toggle_section import render_basket_and_toggle
from components.header import display_header

# =============================================================
# Utilidades de "ventas simuladas"
# =============================================================

def simulate_sales(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    A partir del dataset preparado (con columnas como brand, date, week_number,
    product_price, asin, etc.) genera columnas de ventas ficticias pero plausibles:
      - units_sold: cantidad de unidades vendidas por fila (1..5)
      - sales_amount: ingreso = product_price * units_sold * ruido

    Mantiene las demás columnas para permitir filtrado por brand/fecha.
    """
    if df is None or df.empty:
        return df

    rng = np.random.default_rng(seed)
    work = df.copy()

    # Simular unidades vendidas (1-5) y un pequeño ruido multiplicativo (±20%)
    work["units_sold"] = rng.integers(1, 6, size=len(work))
    noise = rng.uniform(0.8, 1.2, size=len(work))

    # Asegurar precios no nulos/negativos
    price = pd.to_numeric(work["product_price"], errors="coerce").clip(lower=0).fillna(0)
    work["sales_amount"] = (price * work["units_sold"] * noise).astype(float)

    return work


# =============================================================
# Sección de Overview (filtros a la izquierda, highlights a la derecha)
# =============================================================

def render_sales_overview_section(df: pd.DataFrame, period: str):
    """
    Replica la UX de pages/home.py (Overview) pero enfocada a *ventas*.
    - Izquierda: filtros (brands + rango de fechas)
    - Derecha: highlights de ventas sobre el DF filtrado
    Devuelve: df_filtrado, brands_seleccionadas, period
    """
    st.subheader("Sales — Overview")
    st.caption(
        "Filtra a continuación. Las métricas y el gráfico usan ventas simuladas (sales_amount)."
    )

    left_col, right_col = st.columns([0.7, 2.3], gap="large")

    all_brands = sorted(df["brand"].dropna().unique().tolist())
    date_min, date_max = df["date"].dropna().min(), df["date"].dropna().max()

    with left_col:
        st.caption("Selecciona las marcas para filtrar el overview.")
        selected_brands = st.multiselect(
            "Brands to display (sales)",
            options=all_brands,
            default=all_brands,
            help="Marcas a comparar en el gráfico de ventas."
        )

        overview_date_range = None
        if pd.notna(date_min) and pd.notna(date_max):
            overview_date_range = st.date_input(
                "Filter by date (sales)",
                value=(date_min.date(), date_max.date()),
                min_value=date_min.date(),
                max_value=date_max.date(),
                help="Elige un rango de fechas para restringir el overview."
            )

    with right_col:
        df_overview = df.copy()
        if overview_date_range:
            dstart, dend = (
                overview_date_range if isinstance(overview_date_range, tuple) else (overview_date_range, overview_date_range)
            )
            df_overview = df_overview[(df_overview["date"].dt.date >= dstart) & (df_overview["date"].dt.date <= dend)]
        if selected_brands:
            df_overview = df_overview[df_overview["brand"].isin(selected_brands)]

        # ----------------- Highlights de ventas -----------------
        st.markdown("### Highlights (Sales)")
        if df_overview.empty:
            st.info("No hay datos para el filtro actual.")
        else:
            # Ventana "más reciente" para KPIs: última fecha (o semana) disponible dentro del filtro
            if period == "day":
                last_key = df_overview["date"].max()
                df_last = df_overview[df_overview["date"] == last_key]
                prev_key = df_overview["date"][df_overview["date"] < last_key].max()
                df_prev = df_overview[df_overview["date"] == prev_key] if pd.notna(prev_key) else pd.DataFrame(columns=df_overview.columns)
            else:
                last_key = df_overview["week_number"].max()
                df_last = df_overview[df_overview["week_number"] == last_key]
                prev_key = df_overview["week_number"][df_overview["week_number"] < last_key].max()
                df_prev = df_overview[df_overview["week_number"] == prev_key] if pd.notna(prev_key) else pd.DataFrame(columns=df_overview.columns)

            total_sales = float(df_last["sales_amount"].sum())
            prev_sales = float(df_prev["sales_amount"].sum()) if not df_prev.empty else np.nan
            delta_sales = (total_sales - prev_sales) / prev_sales * 100 if prev_sales and not np.isnan(prev_sales) and prev_sales != 0 else np.nan

            transactions = int(df_last.shape[0])
            avg_ticket = float(df_last["sales_amount"].mean()) if not df_last.empty else 0.0

            top_brand_row = (
                df_last.groupby("brand", dropna=False)["sales_amount"].sum().reset_index().sort_values("sales_amount", ascending=False).head(1)
            )
            top_brand = str(top_brand_row.iloc[0]["brand"]) if not top_brand_row.empty else "N/A"
            top_brand_sales = float(top_brand_row.iloc[0]["sales_amount"]) if not top_brand_row.empty else 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Sales (latest)", f"${total_sales:,.0f}", None if np.isnan(delta_sales) else f"{delta_sales:+.1f}% vs prev")
            m2.metric("Avg Ticket (latest)", f"${avg_ticket:,.2f}")
            m3.metric("Transactions (latest)", f"{transactions:,}")
            m4.metric("Top Brand (latest)", top_brand, f"${top_brand_sales:,.0f}")

    return df_overview, selected_brands, period


# =============================================================
# Gráfico de overview de ventas por marca (diario o semanal)
# =============================================================

def create_sales_overview_graph(
    df: pd.DataFrame,
    brands_to_plot=None,
    week_range=None,
    use_markers=False,
    period: str = "week"  # "week" o "day"
) -> go.Figure:
    if brands_to_plot:
        df = df[df["brand"].isin(brands_to_plot)]
    if week_range is not None and period == "week":
        wk_min, wk_max = week_range
        df = df[(df["week_number"] >= wk_min) & (df["week_number"] <= wk_max)]

    group_key = "date" if period == "day" else "week_number"
    x_title = "Date" if period == "day" else "Week Number"

    # Agregar ventas por marca y periodo
    brand_period = (
        df.sort_values("date").groupby(["brand", group_key], as_index=False)["sales_amount"].sum()
    )

    fig = go.Figure()
    trace_mode = "lines+markers" if use_markers else "lines"

    for brand, g in brand_period.groupby("brand"):
        fig.add_trace(
            go.Scatter(
                x=g[group_key],
                y=g["sales_amount"],
                mode=trace_mode,
                name=str(brand),
                hovertemplate=(
                    "Brand: %{text}<br>Sales: $%{y:,.0f}<br>" + ("Date: %{x|%Y-%m-%d}" if period == "day" else "Week: %{x}") + "<extra></extra>"
                ),
                text=g["brand"],
                showlegend=True,
            )
        )

    # Escalas y layout
    y_max = max(1.0, float(pd.to_numeric(brand_period["sales_amount"], errors="coerce").max()))
    fig.update_yaxes(range=[0, y_max * 1.1], title_text="Sales Amount (USD)")

    if period == "week":
        if not df.empty:
            min_week = int(df["week_number"].min())
            max_week = int(df["week_number"].max())
            fig.update_xaxes(range=[min_week, max_week], tickmode="linear", tick0=min_week, dtick=1, title_text=x_title)
    else:
        fig.update_xaxes(title_text=x_title)

    fig.update_layout(
        title="Sales Overview — Total Sales by Brand",
        height=420,
        hovermode="x unified",
        legend_title_text="Brand",
        margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig


# =============================================================
# Desglose (gráfico + tabla) estilo summary.py pero para ventas
# =============================================================

def render_sales_breakdown(df: pd.DataFrame, period: str = "week"):
    st.header("Sales Breakdown")

    # Gráfico de barras apiladas por fecha/semana (ventas por marca)
    if df.empty:
        st.info("No hay datos para mostrar.")
        return

    if period == "day":
        df["x_key"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        x_title = "Date"
    else:
        df["x_key"] = df["week_number"].astype(int)
        x_title = "Week Number"

    pivot_sales = (
        df.pivot_table(index="x_key", columns="brand", values="sales_amount", aggfunc="sum").fillna(0).sort_index()
    )

    # Construir figura de barras apiladas
    stack_fig = go.Figure()
    for brand in pivot_sales.columns:
        stack_fig.add_trace(
            go.Bar(
                x=pivot_sales.index,
                y=pivot_sales[brand],
                name=str(brand),
                hovertemplate="Brand: %{fullData.name}<br>Sales: $%{y:,.0f}<br>" + x_title + ": %{x}<extra></extra>",
            )
        )
    stack_fig.update_layout(
        barmode="stack",
        title="Sales Breakdown — Stacked by Brand",
        xaxis_title=x_title,
        yaxis_title="Sales Amount (USD)",
        margin=dict(l=20, r=20, t=50, b=40),
        hovermode="x unified",
    )

    st.plotly_chart(stack_fig, use_container_width=True)

    # Tabla colapsable (idéntica a summary.py pero con sales_amount)
    with st.expander("Show sales table"):
        tbl = pivot_sales.copy()
        st.dataframe(tbl, use_container_width=True)


# =============================================================
# MAIN PAGE
# =============================================================

def main():
    set_page_config()

    # CSV por defecto
    DEFAULT_BASKET = "synthethic3.csv"
    active_basket_name, active_url, name_to_url = resolve_active_basket(DEFAULT_BASKET)

    # Carga inicial para header (última actualización)
    df = fetch_data(active_url)
    prepared_df = prepare_data(df, basket_name=active_basket_name)
    display_header(prepared_df)

    # Toggle de dataset y período (day/week) como en otras páginas
    period, active_basket_name = render_basket_and_toggle(
        name_to_url, active_basket_name, DEFAULT_BASKET
    )
    active_url = name_to_url.get(active_basket_name, active_url)

    # Recargar y preparar datos (por si cambió el basket) + simular ventas
    df = fetch_data(active_url)
    prepared_df = prepare_data(df, basket_name=active_basket_name)
    sales_df = simulate_sales(prepared_df)

    st.header("Sales Overview")
    if sales_df is None or sales_df.empty:
        st.warning("No data available. Load data first.")
        return

    # 1) Filtros + Highlights (ventas)
    df_overview, selected_brands, period = render_sales_overview_section(sales_df, period=period)

    # 2) Gráfico overview de ventas
    overview_fig = create_sales_overview_graph(df_overview, brands_to_plot=None, period=period)
    st.plotly_chart(overview_fig, use_container_width=True)

    # 3) Desglose (gráfico apilado + tabla)
    render_sales_breakdown(df_overview, period=period)


if __name__ == "__main__":
    main()

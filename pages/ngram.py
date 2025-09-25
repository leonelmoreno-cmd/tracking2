import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Tuple

from components.ngram_utils import (
    extract_asins_from_campaigns,
    build_ngram_table,
    apply_metric_filters,
    DEFAULT_STOPWORDS,
)

def main():
    st.header("N-gram Analysis")

    # --- 1) Carga de archivo ---
    file = st.file_uploader("Upload CSV (Amazon Ads report)", type=["csv"])
    if not file:
        st.info("Sube un CSV con columnas como: Campaign Name, Customer Search Term, Impressions, Clicks, Orders, Spend, 7 Day Total Sales.")
        return

    df = pd.read_csv(file)

    # --- 2) Selección de ASIN ---
    asins = extract_asins_from_campaigns(df.get("Campaign Name", pd.Series(dtype=str)))
    if not asins:
        st.warning("No se detectaron ASINs en 'Campaign Name'. Muestra todo el archivo.")
    asin = st.selectbox("Filtrar por ASIN detectado en Campaign Name", options=["(All)"] + asins, index=0)

    # --- 3) Parámetros de n-gram y limpieza ---
    st.subheader("N-gram Settings")
    cols = st.columns(3)
    with cols[0]:
        n_values = st.multiselect("N-gram size", [1, 2, 3], default=[1, 2, 3])
        if not n_values:
            st.stop()
    with cols[1]:
        min_char_len = st.number_input("Min chars per n-gram", min_value=1, max_value=10, value=2, step=1)
    with cols[2]:
        extra_exclude = st.text_input("Exclude n-grams containing...", value="")

    exclude_contains = [s.strip().lower() for s in extra_exclude.split(",") if s.strip()]

    # --- 4) Construir tabla base de n-grams ---
    base_table = build_ngram_table(
        df=df,
        asin=None if asin == "(All)" else asin,
        n_values=tuple(n_values),
        stopwords=DEFAULT_STOPWORDS,
        min_char_len=min_char_len,
        exclude_contains=exclude_contains,
    )
    if base_table.empty:
        st.warning("No se generaron n-grams con los parámetros actuales.")
        return

    # --- 5) Filtros de métricas ---
    st.subheader("Filters")
    c1, c2, c3 = st.columns(3)
    with c1:
        impressions_min = st.number_input("Impressions ≥", 0, 1_000_000, 100)
        clicks_min = st.number_input("Clicks ≥", 0, 1_000_000, 5)
        orders_min = st.number_input("Orders ≥", 0, 1_000_000, 0)
    with c2:
        cvr_range: Tuple[float, float] = st.slider("CVR %", 0.0, 100.0, (0.0, 100.0))
        rpc_range: Tuple[float, float] = st.slider("RPC", 0.0, float(base_table["RPC"].max() or 1.0), (0.0, float(base_table["RPC"].max() or 1.0)))
    with c3:
        cpc_range: Tuple[float, float] = st.slider("CPC", 0.0, float(base_table["CPC"].max() or 1.0), (0.0, float(base_table["CPC"].max() or 1.0)))
        acos_max_default = float(max(100.0, (base_table["ACOS"].dropna().max() or 0.0)))
        acos_range: Tuple[float, float] = st.slider("ACOS %", 0.0, acos_max_default, (0.0, acos_max_default))

    filtered = apply_metric_filters(
        base_table,
        impressions_min=impressions_min,
        clicks_min=clicks_min,
        orders_min=orders_min,
        cvr_range=cvr_range,
        rpc_range=rpc_range,
        cpc_range=cpc_range,
        acos_range=acos_range,
        n_values=tuple(n_values),
    )

    # --- 6) Scatter configurable ---
    st.subheader("N-gram Scatter")
    metric_options = ["Impressions", "Clicks", "Orders", "Spend", "Sales", "CTR", "CVR", "CPC", "RPC", "ACOS"]
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        x_metric = st.selectbox("X axis", options=metric_options, index=metric_options.index("Clicks"))
    with sc2:
        y_metric = st.selectbox("Y axis", options=metric_options, index=metric_options.index("CVR"))
    with sc3:
        size_metric = st.selectbox("Bubble size", options=metric_options, index=metric_options.index("Impressions"))

    if filtered.empty:
        st.warning("No hay filas tras aplicar los filtros. Ajusta los rangos.")
        return

    fig = px.scatter(
        filtered,
        x=x_metric, y=y_metric, size=size_metric, hover_name="ngram",
        color="n",  # colorea por tamaño del n-gram (1, 2, 3)
        size_max=40,
        title="N-gram performance",
        labels={"n": "N size"},
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=40))
    st.plotly_chart(fig, use_container_width=True)

    # --- 7) Tabla (en expander) ---
    with st.expander("Show n-gram table"):
        # Orden más “accionable”
        show_cols = ["ngram", "n", "Impressions", "Clicks", "Orders", "Spend", "Sales", "CTR", "CVR", "CPC", "RPC", "ACOS"]
        to_show = filtered.loc[:, [c for c in show_cols if c in filtered.columns]].copy()
        # Redondeos amigables
        for c in ["CTR", "CVR", "ACOS", "CPC", "RPC"]:
            if c in to_show.columns:
                to_show[c] = to_show[c].round(2)
        st.dataframe(to_show, use_container_width=True)

if __name__ == "__main__":
    main()

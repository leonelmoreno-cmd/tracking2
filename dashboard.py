import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests  # NEW: para consultar la GitHub API
from typing import Dict, List

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Competitor Price Monitoring",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# -------------------------------
# Constantes del repositorio (ajusta si cambias)
# -------------------------------
GITHUB_OWNER = "leonelmoreno-cmd"
GITHUB_REPO = "tracking2"
GITHUB_PATH = "data"      # carpeta donde están los CSV
GITHUB_BRANCH = "main"    # rama por defecto
DEFAULT_BASKET = "synthethic3.csv"  # basket inicial por defecto

# -------------------------------
# Utilidades: listar CSVs del repo (cacheado)
# -------------------------------
@st.cache_data(show_spinner=False)
def list_repo_csvs(owner: str, repo: str, path: str, branch: str = "main") -> List[dict]:
    """
    Devuelve una lista de dicts con {name, download_url, path} para archivos .csv
    en la carpeta 'path' de un repo público (o privado con token).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    headers = {"Accept": "application/vnd.github+json"}
    # Si configuras un token opcional en st.secrets["GITHUB_TOKEN"], se usará.
    token = st.secrets.get("GITHUB_TOKEN", None)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    items = resp.json()
    csvs = []
    for it in items:
        if it.get("type") == "file" and it.get("name", "").lower().endswith(".csv"):
            # La GitHub API entrega download_url apuntando al raw.
            csvs.append({
                "name": it["name"],
                "download_url": it["download_url"],
                "path": it.get("path", "")
            })
    # Orden alfabético por nombre de archivo
    csvs = sorted(csvs, key=lambda x: x["name"])
    return csvs

def _raw_url_for(owner: str, repo: str, branch: str, path: str, fname: str) -> str:
    """Construye una raw URL directa como fallback si hiciera falta."""
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}/{fname}"

# -------------------------------
# Data loading
# -------------------------------
@st.cache_data(show_spinner=False)
def fetch_data(url: str) -> pd.DataFrame:
    # pandas puede leer directamente desde URL (incluye http/https)
    df = pd.read_csv(url)
    return df

@st.cache_data(show_spinner=False)
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["week_number"] = df["date"].dt.isocalendar().week
    df = df.sort_values(by=["asin", "week_number"])
    df["discount"] = df.apply(
        lambda row: "Discounted" if pd.notna(row.get("product_original_price")) else "No Discount",
        axis=1
    )
    df["price_change"] = df.groupby("asin")["product_price"].pct_change() * 100
    return df

# -------------------------------
# Overview chart (by brand)
# -------------------------------
def create_overview_graph(
    df: pd.DataFrame,
    brands_to_plot=None,
    week_range=None,
    use_markers=False
) -> go.Figure:
    # Brand filter
    if brands_to_plot is not None and len(brands_to_plot) > 0:
        df = df[df["brand"].isin(brands_to_plot)]

    # Week range filter (inclusive) - ignored if None
    if week_range is not None:
        wk_min, wk_max = week_range
        df = df[(df["week_number"] >= wk_min) & (df["week_number"] <= wk_max)]

    # Ranges for axes
    max_price = float(np.nanmax([df["product_price"].max(), df["product_original_price"].max()]))
    min_week = int(df["week_number"].min())
    max_week = int(df["week_number"].max())

    fig = go.Figure()
    trace_mode = "lines+markers" if use_markers else "lines"

    # One line per brand (avg weekly price if multiple ASINs per brand)
    brand_week = (
        df.sort_values("date")
          .groupby(["brand", "week_number"], as_index=False)["product_price"].mean()
    )
    for brand, g in brand_week.groupby("brand"):
        fig.add_trace(go.Scatter(
            x=g["week_number"],
            y=g["product_price"],
            mode=trace_mode,
            name=str(brand),
            hovertemplate=(
                "Brand: %{text}<br>" +
                "Price: $%{y:.2f}<br>" +
                "Week: %{x}<extra></extra>"
            ),
            text=g["brand"],
            showlegend=True
        ))

    # Axes and layout
    fig.update_yaxes(range=[0, max_price], title_text="Product Price (USD)")
    fig.update_xaxes(
        range=[min_week, max_week],
        tickmode="linear", tick0=min_week, dtick=1,   # integer ISO weeks
        title_text="Week Number"
    )
    # Title with selected range (global if no filter)
    fig.update_layout(
        title=f"Overview — Weekly Price by Brand (Weeks {min_week}–{max_week})",
        height=420,
        hovermode="x unified",
        legend_title_text="Brand",
        margin=dict(l=20, r=20, t=50, b=40)
    )
    return fig

# -------------------------------
# Subplots per ASIN
# -------------------------------
def create_price_graph(df: pd.DataFrame) -> go.Figure:
    asins = df["asin"].dropna().unique()
    num_asins = len(asins)

    # Grid: 3 columns
    cols = 3 if num_asins >= 3 else max(1, num_asins)
    rows = int(np.ceil(num_asins / cols))

    fig = make_subplots(
        rows=rows, cols=cols, shared_xaxes=True,
        vertical_spacing=0.15, horizontal_spacing=0.06,
        subplot_titles=[
            f"<a href='{df[df['asin']==asin]['product_url'].iloc[0]}' target='_blank' "
            f"style='color:#FFFBFE; text-decoration:none;'>{df[df['asin']==asin]['brand'].iloc[0]} - {asin}</a>"
            for asin in asins
        ]
    )

    max_price = float(np.nanmax([df["product_price"].max(), df["product_original_price"].max()]))
    min_week = int(df["week_number"].min())
    max_week = int(df["week_number"].max())

    # Ensure x tick labels are visible on all subplots
    fig.for_each_xaxis(lambda ax: ax.update(showticklabels=True))

    for i, asin in enumerate(asins):
        asin_data = df[df["asin"] == asin].sort_values("date")
        if asin_data.empty:
            continue

        dashed = "dot" if (asin_data["discount"] == "Discounted").any() else "solid"
        r = i // cols + 1
        c = i % cols + 1

        fig.add_trace(
            go.Scatter(
                x=asin_data["week_number"],
                y=asin_data["product_price"],
                mode="lines+markers",
                name=str(asin),
                line=dict(dash=dashed),
                hovertemplate=(
                    "ASIN: %{text}<br>" +
                    "Price: $%{y:.2f}<br>" +
                    "Week: %{x}<br>" +
                    "Price Change: %{customdata:.2f}%<br>" +
                    "<extra></extra>"
                ),
                text=asin_data["asin"],
                customdata=asin_data["price_change"],
                showlegend=False
            ),
            row=r, col=c
        )

    # Uniform scales and integer week ticks
    fig.update_yaxes(range=[0, max_price])
    fig.update_xaxes(range=[min_week, max_week])
    fig.for_each_xaxis(lambda ax: ax.update(tickmode="linear", tick0=min_week, dtick=1))

    fig.update_layout(
        height=max(400, 280 * rows),
        xaxis_title="Week Number",
        yaxis_title="Product Price (USD)",
        margin=dict(l=20, r=20, t=50, b=50)
    )
    return fig

# -------------------------------
# Resolución de selección de basket (URL activa)
# -------------------------------
# 1) Obtenemos lista de CSVs del repo (cacheado)
csv_items = list_repo_csvs(GITHUB_OWNER, GITHUB_REPO, GITHUB_PATH, GITHUB_BRANCH)
name_to_url: Dict[str, str] = {it["name"]: it["download_url"] for it in csv_items}

# 2) Leer 'basket' desde query params si existe; si no, usar session_state o default
#    (st.query_params es dict-like; .get devuelve lista o str según versión)
qp = st.query_params.to_dict() if hasattr(st, "query_params") else {}
qp_basket = qp.get("basket")
if isinstance(qp_basket, list):  # por si viene como lista
    qp_basket = qp_basket[0] if qp_basket else None

if "basket" not in st.session_state:
    st.session_state["basket"] = qp_basket if qp_basket else DEFAULT_BASKET

# 3) Determinar URL activa a partir del nombre elegido (fallback a raw si no aparece)
active_basket_name = st.session_state["basket"]
active_url = name_to_url.get(
    active_basket_name,
    _raw_url_for(GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH, active_basket_name)
)

# -------------------------------
# Main UI - carga de datos según basket activa
# -------------------------------
df = fetch_data(active_url)
prepared_df = prepare_data(df)

# Last update
last_update = prepared_df["date"].max()
last_update_str = last_update.strftime("%Y-%m-%d") if pd.notna(last_update) else "N/A"

# Title + Subtitle
st.markdown(
    f"""
    <div style="text-align:center;">
        <h1 style="font-size: 36px; margin-bottom: 4px;">Competitor Price Monitoring</h1>
        <h3 style="color:#666; font-weight:400; margin-top:0;">Last update: {last_update_str} - Developed by Economist Leonel Moreno </h3>    
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Botón/Popover: Seleccionar basket (GitHub) — justo debajo de Last update
# -------------------------------
with st.popover("🧺 Seleccionar basket (GitHub)"):
    st.caption("Elige un CSV de la carpeta del repositorio y confirma para aplicar.")
    options = list(name_to_url.keys()) if name_to_url else [DEFAULT_BASKET]
    # indice seguro
    try:
        idx = options.index(active_basket_name)
    except ValueError:
        idx = 0
    sel = st.selectbox("Archivo (CSV) en el repo", options=options, index=idx, key="basket_select")
    apply = st.button("Usar esta basket", type="primary")
    if apply:
        st.session_state["basket"] = sel
        # reflejar en la URL para compartir/reproducir
        if hasattr(st, "query_params"):
            st.query_params["basket"] = sel
        else:
            # fallback legacy: experimental_set_query_params (para versiones antiguas)
            try:
                st.experimental_set_query_params(basket=sel)
            except Exception:
                pass
        st.rerun()

# Info rápida de la fuente activa (debajo del popover)
with st.container(border=True):
    st.markdown(
        f"**Basket activa:** `{active_basket_name}`  \n"
        f"**Filas x Columnas:** {prepared_df.shape[0]} x {prepared_df.shape[1]}  \n"
        f"**Fuente:** {active_url}"
    )

# -------- Overview (by brand) --------
st.subheader("Overview — All Brands")
st.caption("Use the controls below to filter the overview. The metrics summarize the latest ISO week across selected brands.")

# Two columns: left (selectors) and right (metrics). Right is wider, with extra gap for breathing room.
left_col, right_col = st.columns([0.7, 2.3], gap="large")  # narrower selector + more inter-column space

# Available brands and week bounds
all_brands = sorted(prepared_df["brand"].dropna().unique().tolist())
wk_min_glob = int(prepared_df["week_number"].min())
wk_max_glob = int(prepared_df["week_number"].max())

# LEFT: selectors inside a bordered container
with left_col:
    with st.container(border=True):
        st.caption("Select the brands to filter the overview chart.")
        selected_brands = st.multiselect(
            "Brands to display (overview)",
            options=all_brands,
            default=all_brands,
            help="Select the brands you want to compare in the overview chart."
        )
        # Small internal spacer
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# RIGHT: metrics — latest week only, filtered by selected brands
with right_col:
    # ✅ NEW: bordered metrics container
    with st.container(border=True):
        last_week = int(prepared_df["week_number"].max())
        df_week = prepared_df[
            (prepared_df["week_number"] == last_week) &
            (prepared_df["brand"].isin(selected_brands))
        ].copy()

        # Discount % (only when original price present and > 0)
        df_week["discount_pct"] = np.where(
            df_week["product_original_price"].notna() & (df_week["product_original_price"] != 0),
            (df_week["product_original_price"] - df_week["product_price"]) / df_week["product_original_price"] * 100.0,
            np.nan
        )

        # Highest / Lowest discount (last week)
        row_max_disc = df_week.loc[df_week["discount_pct"].idxmax()] if df_week["discount_pct"].notna().any() else None
        row_min_disc = df_week.loc[df_week["discount_pct"].idxmin()] if df_week["discount_pct"].notna().any() else None

        # Highest / Lowest price (last week)
        row_max_price = df_week.loc[df_week["product_price"].idxmax()] if not df_week["product_price"].isna().all() else None
        row_min_price = df_week.loc[df_week["product_price"].idxmin()] if not df_week["product_price"].isna().all() else None

        # Largest / Lowest price change on the last update of the last week:
        if not df_week.empty:
            latest_by_brand = df_week.loc[df_week.groupby("brand")["date"].idxmax()].copy()
        else:
            latest_by_brand = pd.DataFrame()

        row_max_change = (
            latest_by_brand.loc[latest_by_brand["price_change"].idxmax()]
            if not latest_by_brand.empty and latest_by_brand["price_change"].notna().any()
            else None
        )
        row_min_change = (
            latest_by_brand.loc[latest_by_brand["price_change"].idxmin()]
            if not latest_by_brand.empty and latest_by_brand["price_change"].notna().any()
            else None
        )

        st.markdown("### Last week highlights")

        # 3 columns: Discounts | Prices | Price changes
        dcol, pcol, ccol = st.columns(3)

        # Discounts
        with dcol:
            if row_max_disc is not None:
                st.metric(f"🏷️ Highest discount — week {last_week} — {row_max_disc['brand']}", f"{row_max_disc['discount_pct']:.1f}%")
            else:
                st.metric(f"🏷️ Highest discount — week {last_week}", "N/A")

            if row_min_disc is not None:
                st.metric(f"🏷️ Lowest discount — week {last_week} — {row_min_disc['brand']}", f"{row_min_disc['discount_pct']:.1f}%")
            else:
                st.metric(f"🏷️ Lowest discount — week {last_week}", "N/A")

        # Prices
        with pcol:
            if row_max_price is not None:
                st.metric(f"💲 Highest price — week {last_week} — {row_max_price['brand']}", f"${row_max_price['product_price']:.2f}")
            else:
                st.metric(f"💲 Highest price — week {last_week}", "N/A")

            if row_min_price is not None:
                st.metric(f"💲 Lowest price — week {last_week} — {row_min_price['brand']}", f"${row_min_price['product_price']:.2f}")
            else:
                st.metric(f"💲 Lowest price — week {last_week}", "N/A")

        # Price changes (last update)
        with ccol:
            if row_max_change is not None:
                st.metric(
                    f"🔺 Largest price change (last update) — week {last_week} — {row_max_change['brand']}",
                    f"{row_max_change['price_change']:+.1f}%"
                )
            else:
                st.metric(f"🔺 Largest price change (last update) — week {last_week}", "N/A")

            if row_min_change is not None:
                st.metric(
                    f"🔻 Lowest price change (last update) — week {last_week} — {row_min_change['brand']}",
                    f"{row_min_change['price_change']:+.1f}%"
                )
            else:
                st.metric(f"🔻 Lowest price change (last update) — week {last_week}", "N/A")

# Overview chart
overview_fig = create_overview_graph(
    prepared_df,
    brands_to_plot=selected_brands,
    use_markers=False
)
st.plotly_chart(overview_fig, use_container_width=True)

# -------- Subplots by brand/ASIN --------
st.subheader("By Brand — Individual ASINs")
st.caption("Each small chart tracks a single ASIN. Subplot titles link to the product pages.")
price_graph = create_price_graph(prepared_df)
st.plotly_chart(price_graph, use_container_width=True)

# -------------------------------
# Table + filters
# -------------------------------
st.subheader("Detailed Product Information")
st.caption("Filter the table and download the filtered data as CSV.")

asin_options = ["All"] + sorted(prepared_df["asin"].dropna().unique().tolist())
discount_options = ["All", "Discounted", "No Discount"]

# We still keep a week filter for the table only
table_week_range = st.slider(
    "Filter by week (range)",
    min_value=wk_min_glob,
    max_value=wk_max_glob,
    value=(wk_min_glob, wk_max_glob),
    help="Pick an ISO week range to filter the table."
)

asin_filter = st.selectbox("Filter by ASIN", options=asin_options, index=0, help="Narrow the table to a single ASIN.")
discount_filter = st.selectbox("Filter by Discount Status", options=discount_options, index=0, help="Show only discounted or non-discounted items.")

filtered_df = prepared_df.copy()
if asin_filter != "All":
    filtered_df = filtered_df[filtered_df["asin"] == asin_filter]
if discount_filter != "All":
    filtered_df = filtered_df[filtered_df["discount"] == discount_filter]

filtered_df = filtered_df[
    (filtered_df["week_number"] >= table_week_range[0]) &
    (filtered_df["week_number"] <= table_week_range[1])
]

st.dataframe(filtered_df)

# Download CSV
csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered table as CSV",
    data=csv_data,
    file_name=f"product_details_weeks_{table_week_range[0]}_{table_week_range[1]}.csv",
    mime="text/csv",
    help="Click to download the current filtered table."
)

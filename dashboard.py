import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List  # Importar Dict y List desde typing
from common import GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH, _raw_url_for, fetch_data, list_repo_csvs
from best_sellers_section import render_best_sellers_section_with_table


# -------------------------------
# Page config 
# -------------------------------
st.set_page_config(
    page_title="Competitor Price Monitoring - JC",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# -------------------------------
# Repo constants (adjust if needed)
# -------------------------------
DEFAULT_BASKET = "synthethic3.csv"

# -------------------------------
# Data loading
# -------------------------------
@st.cache_data(show_spinner=False)
def fetch_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

@st.cache_data(show_spinner=False)
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # ISO week number (recommended by pandas; old .week is deprecated)
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
    week_range=None,     # kept for compatibility (ignored for daily)
    use_markers=False,
    period: str = "week"  # <<< NEW: "week" or "day"
) -> go.Figure:
    if brands_to_plot is not None and len(brands_to_plot) > 0:
        df = df[df["brand"].isin(brands_to_plot)]

    if week_range is not None and period == "week":
        wk_min, wk_max = week_range
        df = df[(df["week_number"] >= wk_min) & (df["week_number"] <= wk_max)]

    max_price = float(np.nanmax([
        df["product_price"].max(),
        df["product_original_price"].max()
    ]))

    # Group key and labels depend on period
    if period == "day":
        group_key = "date"
        x_title = "Date"
        title_label = "Daily"
    else:
        group_key = "week_number"
        x_title = "Week Number"
        title_label = "Weekly"

    fig = go.Figure()
    trace_mode = "lines+markers" if use_markers else "lines"

    brand_period = (
        df.sort_values("date")
          .groupby(["brand", group_key], as_index=False)["product_price"].mean()
    )

    # Human-friendly hover label
    hover_x = "Date: %{x|%Y-%m-%d}" if period == "day" else "Week: %{x}"

    for brand, g in brand_period.groupby("brand"):
        fig.add_trace(go.Scatter(
            x=g[group_key],
            y=g["product_price"],
            mode=trace_mode,
            name=str(brand),
            hovertemplate=(
                "Brand: %{text}<br>" +
                "Price: $%{y:.2f}<br>" +
                f"{hover_x}<extra></extra>"
            ),
            text=g["brand"],
            showlegend=True
        ))

    # Axes
    fig.update_yaxes(range=[0, max_price], title_text="Product Price (USD)")
    if period == "week":
        min_week = int(df["week_number"].min())
        max_week = int(df["week_number"].max())
        fig.update_xaxes(
            range=[min_week, max_week],
            tickmode="linear", tick0=min_week, dtick=1,
            title_text=x_title
        )
        title_suffix = f"(Weeks {min_week}–{max_week})"
    else:
        fig.update_xaxes(title_text=x_title)
        title_suffix = ""

    fig.update_layout(
        title=f"Overview — {title_label} Price by Brand {title_suffix}",
        height=420,
        hovermode="x unified",
        legend_title_text="Brand",
        margin=dict(l=20, r=20, t=50, b=40)
    )
    return fig

# -------------------------------
# Subplots per ASIN (legend explained below the caption)
# -------------------------------
def create_price_graph(df: pd.DataFrame, period: str = "week") -> go.Figure:  # <<< NEW
    asins = df["asin"].dropna().unique()
    num_asins = len(asins)
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

    max_price = float(np.nanmax([
        df["product_price"].max(),
        df["product_original_price"].max()
    ]))

    # Precompute for week axis when needed
    if period == "week":
        min_week = int(df["week_number"].min())
        max_week = int(df["week_number"].max())

    fig.for_each_xaxis(lambda ax: ax.update(showticklabels=True))

    for i, asin in enumerate(asins):
        asin_data = df[df["asin"] == asin].sort_values("date")
        if asin_data.empty:
            continue

        dashed = "dot" if (asin_data["discount"] == "Discounted").any() else "solid"
        r = i // cols + 1
        c = i % cols + 1

        x_vals = asin_data["date"] if period == "day" else asin_data["week_number"]
        hover_x = "Date: %{x|%Y-%m-%d}" if period == "day" else "Week: %{x}"
        x_title = "Date" if period == "day" else "Week Number"

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=asin_data["product_price"],
                mode="lines+markers",
                name=str(asin),
                line=dict(dash=dashed),
                hovertemplate=(
                    "ASIN: %{text}<br>" +
                    "Price: $%{y:.2f}<br>" +
                    f"{hover_x}<br>" +
                    "Price Change: %{customdata:.2f}%<br>" +
                    "<extra></extra>"
                ),
                text=asin_data["asin"],
                customdata=asin_data["price_change"],
                showlegend=False
            ),
            row=r, col=c
        )

    fig.update_yaxes(range=[0, max_price])
    if period == "week":
        fig.update_xaxes(range=[min_week, max_week])
        fig.for_each_xaxis(lambda ax: ax.update(tickmode="linear", tick0=min_week, dtick=1))
        x_title = "Week Number"
    else:
        x_title = "Date"

    fig.update_layout(
        height=max(400, 280 * rows),
        xaxis_title=x_title,
        yaxis_title="Product Price (USD)",
        margin=dict(l=20, r=20, t=50, b=50),
        showlegend=False
    )
    return fig

# -------------------------------
# Resolve active basket (URL & session)
# -------------------------------
csv_items = list_repo_csvs(GITHUB_OWNER, GITHUB_REPO, GITHUB_PATH, GITHUB_BRANCH)
name_to_url: Dict[str, str] = {it["name"]: it["download_url"] for it in csv_items}

# Obtén el parámetro de la cesta desde la URL o la sesión
qp = st.query_params.to_dict() if hasattr(st, "query_params") else {}
qp_basket = qp.get("basket")
if isinstance(qp_basket, list):
    qp_basket = qp_basket[0] if qp_basket else None

if "basket" not in st.session_state:
    st.session_state["basket"] = qp_basket if qp_basket else DEFAULT_BASKET

# El nombre de la cesta activa seleccionada
active_basket_name = st.session_state["basket"]

# Obtener la URL del archivo de la cesta activa directamente, sin buscar una subcategoría automáticamente
active_url = name_to_url.get(
    active_basket_name,
    _raw_url_for(GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH, active_basket_name)
)
# -------------------------------
# Main UI - load data
# -------------------------------
df = fetch_data(active_url)
prepared_df = prepare_data(df)

last_update = prepared_df["date"].max()
last_update_str = last_update.strftime("%Y-%m-%d") if pd.notna(last_update) else "N/A"

# Title + Subtitle
st.markdown(
    f"""
    <div style="text-align:center;">
        <h1 style="font-size: 36px; margin-bottom:-15px;">Competitor Price Monitoring</h1>
        <h6 style="color:#666; font-weight:200; margin-top:0;">Last update: {last_update_str} - Developed by JC Team</h6>
    </div>
    """,
    unsafe_allow_html=True
)

# ===============================
# Centered container (Current + Change basket + Global toggle)
# ===============================
col1, col2, col3 = st.columns([3, 2, 2], gap="small")


with col1:
    st.markdown(
        f"<div style='text-align:left; margin:4px 0;'>"
        f"<span style='color:#16a34a; font-weight:600;'>Current basket:</span> "
        f"<code style='color:#16a34a;'>{active_basket_name}</code>"
        f"</div>",
        unsafe_allow_html=True
    )

with col2:
    with st.popover("🧺 Change basket"):
        st.caption("Pick a CSV from the list and click Apply.")
        options = list(name_to_url.keys()) if name_to_url else [DEFAULT_BASKET]
        try:
            idx = options.index(active_basket_name)
        except ValueError:
            idx = 0
        sel = st.selectbox("File (CSV) in repo", options=options, index=idx, key="basket_select")
        apply = st.button("Apply", type="primary")
        if apply:
            st.session_state["basket"] = sel
            if hasattr(st, "query_params"):
                st.query_params["basket"] = sel
            else:
                try:
                    st.experimental_set_query_params(basket=sel)
                except Exception:
                    pass
            st.rerun()

with col3:
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    aggregate_daily = st.toggle(
        "Aggregate by day (instead of week)",
        value=False,
        help="When ON, all charts use daily prices; when OFF, weekly averages."
    )
    period = "day" if aggregate_daily else "week"
# -------- Overview (by brand) --------
st.subheader("Overview — All Brands")
st.caption("Use the controls below to filter the overview. The metrics summarize the latest period across selected brands.")

left_col, right_col = st.columns([0.7, 2.3], gap="large")

all_brands = sorted(prepared_df["brand"].dropna().unique().tolist())
wk_min_glob = int(prepared_df["week_number"].min())
wk_max_glob = int(prepared_df["week_number"].max())

# Global date range for overview
date_min = prepared_df["date"].dropna().min()
date_max = prepared_df["date"].dropna().max()

with left_col:
    with st.container(border=True):
        st.caption("Select the brands to filter the overview chart.")
        selected_brands = st.multiselect(
            "Brands to display (overview)",
            options=all_brands,
            default=all_brands,
            help="Select the brands you want to compare in the overview chart."
        )
        # Date range picker for overview
        overview_date_range = None
        if pd.notna(date_min) and pd.notna(date_max):
            overview_date_range = st.date_input(
                "Filter by date (overview)",
                value=(date_min.date(), date_max.date()),
                min_value=date_min.date(),
                max_value=date_max.date(),
                help="Choose a start/end date to constrain the overview."
            )
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

with right_col:
    with st.container(border=True):
        # Overview dataframe with date/brand filters
        df_overview = prepared_df.copy()
        if overview_date_range:
            if isinstance(overview_date_range, tuple):
                dstart, dend = overview_date_range
            else:
                dstart = dend = overview_date_range
            df_overview = df_overview[
                (df_overview["date"].dt.date >= dstart) & (df_overview["date"].dt.date <= dend)
            ]

        if selected_brands:
            df_overview = df_overview[df_overview["brand"].isin(selected_brands)]

        # Last period subset for KPIs
        if not df_overview.empty:
            if period == "week":
                last_period = int(df_overview["week_number"].max())
                df_period = df_overview[df_overview["week_number"] == last_period].copy()
                label = f"week {last_period}"
            else:
                last_day_ts = df_overview["date"].max()
                last_day = last_day_ts.date()
                df_period = df_overview[df_overview["date"].dt.date == last_day].copy()
                label = last_day.strftime("%Y-%m-%d")
        else:
            last_period = None
            df_period = pd.DataFrame()
            label = "N/A"

        # KPI computations
        df_period["discount_pct"] = np.where(
            df_period["product_original_price"].notna() & (df_period["product_original_price"] != 0),
            (df_period["product_original_price"] - df_period["product_price"]) / df_period["product_original_price"] * 100.0,
            np.nan
        )

        row_max_disc = df_period.loc[df_period["discount_pct"].idxmax()] if df_period["discount_pct"].notna().any() else None
        row_min_disc = df_period.loc[df_period["discount_pct"].idxmin()] if df_period["discount_pct"].notna().any() else None

        row_max_price = df_period.loc[df_period["product_price"].idxmax()] if not df_period["product_price"].isna().all() and not df_period.empty else None
        row_min_price = df_period.loc[df_period["product_price"].idxmin()] if not df_period["product_price"].isna().all() and not df_period.empty else None

        if not df_period.empty:
            latest_by_brand = df_period.loc[df_period.groupby("brand")["date"].idxmax()].copy()
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

        st.markdown("### Last period highlights")

        dcol, pcol, ccol = st.columns(3)

        with dcol:
            if row_max_disc is not None:
                st.metric(f"🏷️ Highest discount — {label} — {row_max_disc['brand']}", f"{row_max_disc['discount_pct']:.1f}%")
            else:
                st.metric(f"🏷️ Highest discount — {label}", "N/A")

            if row_min_disc is not None:
                st.metric(f"🏷️ Lowest discount — {label} — {row_min_disc['brand']}", f"{row_min_disc['discount_pct']:.1f}%")
            else:
                st.metric(f"🏷️ Lowest discount — {label}", "N/A")

        with pcol:
            if row_max_price is not None:
                st.metric(f"💲 Highest price — {label} — {row_max_price['brand']}", f"${row_max_price['product_price']:.2f}")
            else:
                st.metric(f"💲 Highest price — {label}", "N/A")

            if row_min_price is not None:
                st.metric(f"💲 Lowest price — {label} — {row_min_price['brand']}", f"${row_min_price['product_price']:.2f}")
            else:
                st.metric(f"💲 Lowest price — {label}", "N/A")

        with ccol:
            if row_max_change is not None:
                st.metric(
                    f"🔺 Largest price change (last update) — {label} — {row_max_change['brand']}",
                    f"{row_max_change['price_change']:+.1f}%"
                )
            else:
                st.metric(f"🔺 Largest price change (last update) — {label}", "N/A")

            if row_min_change is not None:
                st.metric(
                    f"🔻 Lowest price change (last update) — {label} — {row_min_change['brand']}",
                    f"{row_min_change['price_change']:+.1f}%"
                )
            else:
                st.metric(f"🔻 Lowest price change (last update) — {label}", "N/A")

# Overview chart with the selected period
overview_fig = create_overview_graph(
    df_overview if 'df_overview' in locals() else prepared_df,
    brands_to_plot=selected_brands,
    use_markers=False,
    period=period  # <<< NEW
)
st.plotly_chart(overview_fig, use_container_width=True)

# Best seller leonel
render_best_sellers_section_with_table(active_basket_name: str)

# -------- Subplots by brand/ASIN --------
st.subheader("By Brand — Individual ASINs")
st.caption("Each small chart tracks a single ASIN. Subplot titles link to the product pages.")

st.markdown(
    """
    <div style="margin-top:-4px; margin-bottom:8px; font-size:0.95rem;">
        <strong>Legend:</strong>
        <span style="border-bottom:3px solid currentColor; padding-bottom:2px;">&nbsp;&nbsp;&nbsp;&nbsp;</span>
        No discount &nbsp; | &nbsp;
        <span style="border-bottom:3px dotted currentColor; padding-bottom:2px;">&nbsp;&nbsp;&nbsp;&nbsp;</span>
        Discounted
    </div>
    """,
    unsafe_allow_html=True
)

price_graph = create_price_graph(prepared_df, period=period)  # <<< NEW
st.plotly_chart(price_graph, use_container_width=True)

# -------------------------------
# Table + filters (unchanged)
# -------------------------------
st.subheader("Detailed Product Information")
st.caption("Filter the table and download the filtered data as CSV.")

brand_options = ["All"] + sorted(prepared_df["brand"].dropna().unique().tolist())
discount_options = ["All", "Discounted", "No Discount"]

wk_min_glob = int(prepared_df["week_number"].min())
wk_max_glob = int(prepared_df["week_number"].max())

date_min_tbl = prepared_df["date"].dropna().min()
date_max_tbl = prepared_df["date"].dropna().max()
table_date_range = None
if pd.notna(date_min_tbl) and pd.notna(date_max_tbl):
    table_date_range = st.date_input(
        "Filter by date (range)",
        value=(date_min_tbl.date(), date_max_tbl.date()),
        min_value=date_min_tbl.date(),
        max_value=date_max_tbl.date(),
        help="Pick a start and end date to filter the table."
    )

brand_filter = st.selectbox(
    "Filter by Brand",
    options=brand_options,
    index=0,
    help="Narrow the table to a single brand."
)

discount_filter = st.selectbox(
    "Filter by Discount Status",
    options=discount_options,
    index=0,
    help="Show only discounted or non-discounted items."
)

filtered_df = prepared_df.copy()

if brand_filter != "All":
    filtered_df = filtered_df[filtered_df["brand"] == brand_filter]

if discount_filter != "All":
    filtered_df = filtered_df[filtered_df["discount"] == discount_filter]

if table_date_range:
    if isinstance(table_date_range, tuple):
        start_date, end_date = table_date_range
    else:
        start_date = end_date = table_date_range
    filtered_df = filtered_df[
        (filtered_df["date"].dt.date >= start_date) &
        (filtered_df["date"].dt.date <= end_date)
    ]

st.dataframe(filtered_df, use_container_width=True)


csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered table as CSV",
    data=csv_data,
    file_name="product_details_filtered.csv",
    mime="text/csv",
    help="Click to download the current filtered table."
)

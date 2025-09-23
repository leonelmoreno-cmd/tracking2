import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# -------------------------------
# Overview chart (by brand)
# -------------------------------
st.write("DEBUG period:", period, type(period))

def create_overview_graph(
    df: pd.DataFrame,
    brands_to_plot=None,
    week_range=None,  # ignored for daily
    use_markers=False,
    period: str = "week"  # "week" or "day"
) -> go.Figure:
    if brands_to_plot is not None and len(brands_to_plot) > 0:
        df = df[df["brand"].isin(brands_to_plot)]

    if week_range is not None and period == "week":
        wk_min, wk_max = week_range
        df = df[(df["week_number"] >= wk_min) & (df["week_number"] <= wk_max)]

    max_price = float(np.nanmax([df["product_price"].max(), df["product_original_price"].max()]))

    group_key = "date" if period == "day" else "week_number"
    x_title = "Date" if period == "day" else "Week Number"
    title_label = "Daily" if period == "day" else "Weekly"

    fig = go.Figure()
    trace_mode = "lines+markers" if use_markers else "lines"

    brand_period = df.sort_values("date").groupby(["brand", group_key], as_index=False)["product_price"].mean()

    hover_x = "Date: %%{x|%Y-%m-%d}" if period == "day" else "Week: %%{x}"

    for brand, g in brand_period.groupby("brand"):
        fig.add_trace(
            go.Scatter(
                x=g[group_key],
                y=g["product_price"],
                mode=trace_mode,
                name=str(brand),
                hovertemplate=f"Brand: %%{{text}}<br>Price: $%%{{y:.2f}}<br>{hover_x}<extra></extra>",
                text=g["brand"],
                showlegend=True
            )
        )

    fig.update_yaxes(range=[0, max_price], title_text="Product Price (USD)")

    if period == "week":
        min_week = int(df["week_number"].min())
        max_week = int(df["week_number"].max())
        fig.update_xaxes(range=[min_week, max_week], tickmode="linear", tick0=min_week, dtick=1, title_text=x_title)
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
# Subplots per ASIN
# -------------------------------
def create_price_graph(df: pd.DataFrame, period: str = "week") -> go.Figure:
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

    max_price = float(np.nanmax([df["product_price"].max(), df["product_original_price"].max()]))

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
        hover_x = "Date: %%{x|%Y-%m-%d}" if period == "day" else "Week: %%{x}"

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=asin_data["product_price"],
                mode="lines+markers",
                name=str(asin),
                line=dict(dash=dashed),
                hovertemplate=f"ASIN: %%{{text}}<br>Price: $%%{{y:.2f}}<br>{hover_x}<br>Price Change: %%{{customdata:.2f}}%<extra></extra>",
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

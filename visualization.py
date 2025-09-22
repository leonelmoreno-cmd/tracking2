# visualization.py
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def create_overview_graph(df: pd.DataFrame, brands_to_plot=None, week_range=None, use_markers=False, period="week") -> go.Figure:
    if brands_to_plot:
        df = df[df["brand"].isin(brands_to_plot)]

    if week_range and period == "week":
        wk_min, wk_max = week_range
        df = df[(df["week_number"] >= wk_min) & (df["week_number"] <= wk_max)]

    max_price = float(np.nanmax([df["product_price"].max(), df["product_original_price"].max()]))

    group_key = "date" if period == "day" else "week_number"
    x_title = "Date" if period == "day" else "Week Number"
    title_label = "Daily" if period == "day" else "Weekly"

    fig = go.Figure()
    trace_mode = "lines+markers" if use_markers else "lines"

    brand_period = df.sort_values("date").groupby(["brand", group_key], as_index=False)["product_price"].mean()

    hover_x = "Date: %{x|%Y-%m-%d}" if period == "day" else "Week: %{x}"

    for brand, g in brand_period.groupby("brand"):
        fig.add_trace(go.Scatter(
            x=g[group_key],
            y=g["product_price"],
            mode=trace_mode,
            name=str(brand),
            hovertemplate=f"Brand: %{text}<br>Price: $%{{y:.2f}}<br>{hover_x}<extra></extra>",
            text=g["brand"]
        ))

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

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ------------------------------------------------------------
# Price % Variation — overall ASINs in one chart
# ------------------------------------------------------------

def plot_price_variation_by_asin(df: pd.DataFrame, period: str = "day") -> None:
    """
    Creates an interactive figure for price percentage variation across all ASINs.
    The chart displays the ASINs in the y-axis and price percentage variation in the x-axis.
    """
    # Ensure that necessary columns exist
    required_columns = ["asin", "brand", "product_price", "product_original_price", "date"]
    if not all(col in df.columns for col in required_columns):
        st.error("Missing necessary columns in the DataFrame")
        return

    # Calculate price percentage change
    df['price_change'] = df.groupby('asin')['product_price'].pct_change() * 100
    
    # Handle missing values in price_change column
    df = df.dropna(subset=["price_change"])

    # Get the available dates for filtering
    available_dates = sorted(df["date"].unique())

    # Add a date selector for the user, with the last available date as default
    selected_date = st.date_input(
        "Select Date",
        min_value=min(available_dates),
        max_value=max(available_dates),
        value=max(available_dates)  # Default to the latest date
    )

    # Filter the dataframe by the selected date
    df_latest = df[df["date"] == pd.to_datetime(selected_date)]

    if df_latest.empty:
        st.info(f"No data available for the selected date: {selected_date}.")
        return

    # Create a list of ASINs and brands
    asin_with_brand = df_latest.apply(lambda row: f"{row['brand']} — {row['asin']}", axis=1)

    # Create a horizontal bar chart for all ASINs
    fig = go.Figure()

    # Add bars for price changes
    fig.add_trace(go.Bar(
        y=asin_with_brand,  # y-axis: ASIN and brand
        x=df_latest["price_change"],  # x-axis: price change percentage
        orientation="h",
        marker=dict(color=np.where(df_latest["price_change"] >= 0, "green", "red")),  # Green for positive, Red for negative
        hovertemplate=(
            "<b>ASIN</b>: %{customdata[0]}<br>"
            "<b>Price % change</b>: %{x:.2f}%<br>"
            "<b>Date</b>: %{customdata[1]}<br>"
            "<extra></extra>"
        ),
        customdata=np.stack([df_latest['asin'], df_latest['date'].dt.strftime('%Y-%m-%d')], axis=1),
    ))

    # Apply a common layout
    fig.update_layout(
        template="plotly_white",
        title=f"Price Percentage Variation (by ASIN) - {period.capitalize()}",
        xaxis_title="Price Change (%)",
        yaxis_title="ASIN — Brand",
        margin=dict(t=60, l=70, r=30, b=50),
        height=600,
        hovermode="x unified",
    )

    # Show the chart
    st.plotly_chart(fig, use_container_width=True)

    # Display the selected date above the chart
    st.write(f"**Data for selected date: {selected_date}**")

    # Collapsed table with average price change per ASIN
    with st.expander("Show price % variation table"):
        tbl = (
            df_latest.pivot_table(
                index="asin",
                columns="brand",
                values="price_change",
                aggfunc="mean"
            ).sort_index()
        )
        st.dataframe(tbl)

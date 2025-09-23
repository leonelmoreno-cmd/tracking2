import streamlit as st
import pandas as pd
from typing import Tuple

def render_detailed_table(prepared_df: pd.DataFrame) -> pd.DataFrame:
    """
    Render a detailed product table with date, brand, and discount filters.
    Returns the filtered dataframe.
    """
    st.subheader("Detailed Product Information")
    st.caption("Filter the table and download the filtered data as CSV.")

    all_brands = sorted(prepared_df["brand"].dropna().unique().tolist())
    brand_options = ["All"] + all_brands
    discount_options = ["All", "Discounted", "No Discount"]

    table_date_range = st.date_input(
        "Filter by date (range)",
        value=(prepared_df["date"].min().date(), prepared_df["date"].max().date()),
        min_value=prepared_df["date"].min().date(),
        max_value=prepared_df["date"].max().date(),
        help="Pick a start and end date to filter the table."
    )

    brand_filter = st.selectbox("Filter by Brand", options=brand_options, index=0)
    discount_filter = st.selectbox("Filter by Discount Status", options=discount_options, index=0)

    filtered_df = prepared_df.copy()
    if brand_filter != "All":
        filtered_df = filtered_df[filtered_df["brand"] == brand_filter]
    if discount_filter != "All":
        filtered_df = filtered_df[filtered_df["discount"] == discount_filter]

    start_date, end_date = table_date_range if isinstance(table_date_range, tuple) else (table_date_range, table_date_range)
    filtered_df = filtered_df[(filtered_df["date"].dt.date >= start_date) & (filtered_df["date"].dt.date <= end_date)]

    st.dataframe(filtered_df, use_container_width=True)
    st.download_button(
        "Download filtered table as CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="product_details_filtered.csv",
        mime="text/csv",
        help="Click to download the current filtered table."
    )

    return filtered_df

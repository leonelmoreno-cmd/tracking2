# overview_section.py
import streamlit as st
import pandas as pd
from components.highlights_section import render_highlights

def render_overview_section(df: pd.DataFrame, period: str):
    """
    Render the Overview section with brand selection, date filters, and highlights.
    Returns the filtered DataFrame and selected brands.
    """
    st.subheader("Overview — All Brands")
    st.caption("Use the controls below to filter the overview. The metrics summarize the latest period across selected brands.")

    left_col, right_col = st.columns([0.7, 2.3], gap="large")
    all_brands = sorted(df["brand"].dropna().unique().tolist())
    date_min, date_max = df["date"].dropna().min(), df["date"].dropna().max()

    with left_col:
        st.caption("Select the brands to filter the overview chart.")
        selected_brands = st.multiselect(
            "Brands to display (overview)",
            options=all_brands,
            default=all_brands,
            help="Select the brands you want to compare in the overview chart."
        )

        overview_date_range = None
        if pd.notna(date_min) and pd.notna(date_max):
            overview_date_range = st.date_input(
                "Filter by date (overview)",
                value=(date_min.date(), date_max.date()),
                min_value=date_min.date(),
                max_value=date_max.date(),
                help="Choose a start/end date to constrain the overview."
            )

    with right_col:
        df_overview = df.copy()
        if overview_date_range:
            dstart, dend = overview_date_range if isinstance(overview_date_range, tuple) else (overview_date_range, overview_date_range)
            df_overview = df_overview[(df_overview["date"].dt.date >= dstart) & (df_overview["date"].dt.date <= dend)]
        if selected_brands:
            df_overview = df_overview[df_overview["brand"].isin(selected_brands)]

        # Render highlights using the modular component
        render_highlights(df_overview, period=period)

    return df_overview, selected_brands

# pages/placements.py
# -*- coding: utf-8 -*-

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# If you prefer speed you can switch to rapidfuzz:
# from rapidfuzz import fuzz, process
from fuzzywuzzy import fuzz, process


# --------------------------- Helpers ---------------------------
def clean_money(series: pd.Series) -> pd.Series:
    """Remove $ and commas and cast to float."""
    return series.replace({r"\$": "", ",": ""}, regex=True).astype(float)

def parse_date_maybe(series: pd.Series) -> pd.Series:
    """Safe to_datetime conversion."""
    return pd.to_datetime(series, errors="coerce")

def left_part(text: str) -> str:
    """Extract left side before ' - '."""
    if pd.isna(text):
        return ""
    parts = str(text).split(" - ")
    return parts[0].strip() if parts else str(text).strip()

def right_part(text: str) -> str:
    """Extract right side after the first ' - '."""
    if pd.isna(text):
        return ""
    parts = str(text).split(" - ", 1)
    return parts[1].strip() if len(parts) > 1 else ""


# --------------------------- Page entrypoint ---------------------------
def main():
    # DO NOT call st.set_page_config here; it's already set in app.py

    # --------------------------- UI: top ---------------------------
    st.title("Placements Analysis")
    st.caption("Upload your placements Excel, select/enter portfolio, and run the analysis.")

    uploaded_file = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        # User input: right side of the portfolio name (suffix after the hyphen)
        user_suffix = st.text_input(
            "Portfolio name (right side)",
            value="",
            placeholder="e.g., JungleClick",
            key="user_suffix"
        )

    with col2:
        # Placeholder kept for layout symmetry (not strictly needed)
        prefix_placeholder = st.empty()

    with col3:
        run_clicked = st.button("Run", type="primary", key="run_btn")

    st.divider()

    # --------------------------- Main flow ---------------------------
    if run_clicked:
        # 1) Basic validations
        if uploaded_file is None:
            st.error("Please upload an Excel file (.xlsx).")
            st.stop()

        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read the Excel file: {e}")
            st.stop()

        # 2) Minimal cleaning
        df.columns = df.columns.str.strip()
        if "Date" in df.columns:
            df["Date"] = parse_date_maybe(df["Date"])

        # Required columns
        required_cols = ["Portfolio name", "Placement", "Clicks", "Spend", "7 Day Total Orders (#)"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required columns in the Excel: {missing}")
            st.stop()

        # Types
        df["Clicks"] = pd.to_numeric(df["Clicks"], errors="coerce")
        try:
            df["Spend"] = clean_money(df["Spend"])
        except Exception:
            df["Spend"] = pd.to_numeric(df["Spend"], errors="coerce")

        if "7 Day Total Sales" in df.columns:
            try:
                df["7 Day Total Sales"] = clean_money(df["7 Day Total Sales"])
            except Exception:
                df["7 Day Total Sales"] = pd.to_numeric(df["7 Day Total Sales"], errors="coerce")

        # 3) Detect portfolio prefixes (left side of 'Portfolio name')
        df["_prefix"] = df["Portfolio name"].apply(left_part)
        prefixes = df["_prefix"].dropna().unique().tolist()

        # Default prefix: the most frequent
        if len(prefixes) == 0:
            st.warning("No values detected in 'Portfolio name'. Continuing without portfolio filter…")
            selected_prefix = ""
        else:
            prefix_counts = df["_prefix"].value_counts()
            with col2:
                selected_prefix = st.selectbox(
                    "Portfolio prefix (left side)",
                    options=prefix_counts.index.tolist(),
                    index=0,
                    help="Taken from the 'Portfolio name' column before the hyphen '-'"
                )

        # 4) Fuzzy match for the RIGHT side (suffix)
        df["_suffix"] = df["Portfolio name"].apply(right_part)

        full_portfolio = None
        if user_suffix.strip() and selected_prefix:
            # Candidates: all suffixes present for this prefix
            candidates = (
                df.loc[df["_prefix"] == selected_prefix, "_suffix"]
                .dropna()
                .unique()
                .tolist()
            )
            if len(candidates) == 0:
                st.warning(f"No suffixes found for prefix '{selected_prefix}'. Continuing without portfolio filter.")
            else:
                best = process.extractOne(user_suffix.strip(), candidates, scorer=fuzz.token_sort_ratio)
                best_suffix = best[0] if best else user_suffix.strip()
                full_portfolio = f"{selected_prefix} - {best_suffix}"
                st.info(f"Estimated portfolio by fuzzy match: **{full_portfolio}**")

        # 5) Filter (if we have full_portfolio)
        if full_portfolio:
            df = df[df["Portfolio name"] == full_portfolio]
            if df.empty:
                st.warning("The portfolio filter returned no rows. Please check the name/suffix.")
        else:
            st.caption("No portfolio filter applied (missing prefix/suffix or not provided).")

        # Persist final DataFrame for later interactions (e.g., campaign fuzzy search)
        st.session_state["df"] = df

        # 6) Quick EDA
        with st.expander("Peek data and schema"):
            st.dataframe(df.head(50), use_container_width=True)
            # Capture DataFrame.info() into a string
            sio = io.StringIO()
            df.info(verbose=True, memory_usage="deep", buf=sio)
            st.text(sio.getvalue())

        # 7) Charts

        # --- Clicks by Placement ---
        clicks_by_placement = (
            df.groupby("Placement", dropna=False)["Clicks"].sum().reset_index()
            .sort_values("Clicks", ascending=False)
        )
        fig_clicks = px.bar(
            clicks_by_placement, x="Placement", y="Clicks",
            title="Click Distribution by Placement",
            labels={"Placement": "Placement", "Clicks": "Total Clicks"},
            template="plotly_dark"
        )
        fig_clicks.update_layout(xaxis_tickangle=-45, xaxis_title="Placement", yaxis_title="Total Clicks")
        st.plotly_chart(fig_clicks, use_container_width=True)

        # --- Orders by Placement (7 Day Total Orders (#)) ---
        orders_by_placement = (
            df.groupby("Placement", dropna=False)["7 Day Total Orders (#)"].sum().reset_index()
            .sort_values("7 Day Total Orders (#)", ascending=False)
        )
        fig_orders = px.bar(
            orders_by_placement, x="Placement", y="7 Day Total Orders (#)",
            title="Orders Distribution by Placement",
            labels={"Placement": "Placement", "7 Day Total Orders (#)": "Total Orders (7 Days)"},
            template="plotly_dark"
        )
        fig_orders.update_layout(xaxis_tickangle=-45, xaxis_title="Placement", yaxis_title="Total Orders (7 Days)")
        st.plotly_chart(fig_orders, use_container_width=True)

        # --- % Clicks vs % Orders by Placement ---
        placement_summary = df.groupby("Placement").agg(
            Clicks=("Clicks", "sum"),
            Orders=("7 Day Total Orders (#)", "sum")
        ).reset_index()

        total_clicks = placement_summary["Clicks"].sum()
        total_orders = placement_summary["Orders"].sum()

        placement_summary["Clicks Percentage"] = (placement_summary["Clicks"] / total_clicks * 100) if total_clicks > 0 else 0
        placement_summary["Orders Percentage"] = (placement_summary["Orders"] / total_orders * 100) if total_orders > 0 else 0
        placement_summary = placement_summary.sort_values("Clicks Percentage", ascending=False)

        fig_comp = go.Figure(data=[
            go.Bar(name="Clicks Percentage", x=placement_summary["Placement"], y=placement_summary["Clicks Percentage"]),
            go.Bar(name="Orders Percentage", x=placement_summary["Placement"], y=placement_summary["Orders Percentage"]),
        ])
        fig_comp.update_layout(
            barmode="group",
            xaxis_tickangle=-45,
            title="Clicks vs Orders Percentage Distribution by Placement",
            xaxis_title="Placement",
            yaxis_title="Percentage (%)",
            template="plotly_dark",
            legend_title="Metric"
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # --- Orders by Date and Placement ---
        if "Date" in df.columns:
            orders_by_day_placement = df.groupby(["Date", "Placement"])["7 Day Total Orders (#)"].sum().reset_index()
            fig_orders_by_day = px.line(
                orders_by_day_placement, x="Date", y="7 Day Total Orders (#)", color="Placement",
                title="Orders Evolution by Placement Over Time",
                labels={"Date": "Date", "7 Day Total Orders (#)": "Total Orders (7 Days)", "Placement": "Placement"},
                template="plotly_dark"
            )
            fig_orders_by_day.update_layout(xaxis_title="Date", yaxis_title="Total Orders (7 Days)")
            st.plotly_chart(fig_orders_by_day, use_container_width=True)

        # --- Spend % by Date and Placement (stacked) ---
        if "Date" in df.columns:
            spend_by_day_placement = df.groupby(["Date", "Placement"])["Spend"].sum().reset_index()
            total_daily_spend = (
                spend_by_day_placement.groupby("Date")["Spend"]
                .sum()
                .reset_index()
                .rename(columns={"Spend": "Total Daily Spend"})
            )
            spend_by_day_placement = spend_by_day_placement.merge(total_daily_spend, on="Date", how="left")
            spend_by_day_placement["Spend Percentage"] = np.where(
                spend_by_day_placement["Total Daily Spend"] > 0,
                spend_by_day_placement["Spend"] / spend_by_day_placement["Total Daily Spend"] * 100,
                0
            )
            fig_spend_pct = px.bar(
                spend_by_day_placement, x="Date", y="Spend Percentage", color="Placement",
                title="Evolution of Spend Percentage by Placement Over Time",
                labels={"Date": "Date", "Spend Percentage": "Spend Percentage (%)", "Placement": "Placement"},
                template="plotly_dark"
            )
            fig_spend_pct.update_layout(barmode="stack", xaxis_title="Date", yaxis_title="Spend Percentage (%)")
            st.plotly_chart(fig_spend_pct, use_container_width=True)

        # --- Product Pages: campaigns with highest Clicks ---
        product_pages_data = df[df["Placement"].str.contains("Product Pages", case=False, na=False)].copy()
        product_pages_data = product_pages_data[product_pages_data["Clicks"] > 12]
        if not product_pages_data.empty:
            clicks_by_campaign_pp = (
                product_pages_data.groupby("Campaign Name")["Clicks"].sum().reset_index()
                .sort_values("Clicks", ascending=False)
            )
            fig_campaigns_clicks = px.bar(
                clicks_by_campaign_pp, y="Campaign Name", x="Clicks",
                title="Campaigns with Highest Clicks in Product Pages",
                labels={"Campaign Name": "Campaign Name", "Clicks": "Total Clicks"},
                color="Clicks", color_continuous_scale="Viridis", template="plotly_dark"
            )
            fig_campaigns_clicks.update_layout(yaxis_title="Campaign Name", xaxis_title="Total Clicks", showlegend=False)
            st.plotly_chart(fig_campaigns_clicks, use_container_width=True)

        # --- Product Pages: campaigns with highest Spend (> $10) ---
        product_pages_spend = df[df["Placement"].str.contains("Product Pages", case=False, na=False)].copy()
        product_pages_spend = product_pages_spend[product_pages_spend["Spend"] > 10]
        if not product_pages_spend.empty:
            spend_by_campaign = (
                product_pages_spend.groupby("Campaign Name")["Spend"].sum().reset_index()
                .sort_values("Spend", ascending=False)
            )
            st.markdown("**Top Spend in Product Pages (> $10)**")
            st.dataframe(spend_by_campaign, use_container_width=True)

            fig_campaigns_spend = px.bar(
                spend_by_campaign, y="Campaign Name", x="Spend",
                title="Campaigns with Highest Spend in Product Pages (> $10)",
                labels={"Campaign Name": "Campaign Name", "Spend": "Total Spend"},
                color="Spend", color_continuous_scale="Viridis", template="plotly_dark"
            )
            fig_campaigns_spend.update_layout(yaxis_title="Campaign Name", xaxis_title="Total Spend")
            st.plotly_chart(fig_campaigns_spend, use_container_width=True)

        # --- Spend by Date and Placement ---
        if "Date" in df.columns:
            spend_by_day_placement2 = df.groupby(["Date", "Placement"])["Spend"].sum().reset_index()
            fig_spend_by_day = px.line(
                spend_by_day_placement2, x="Date", y="Spend", color="Placement",
                title="Spend Evolution by Placement Over Time",
                labels={"Date": "Date", "Spend": "Total Spend", "Placement": "Placement"},
                template="plotly_dark"
            )
            fig_spend_by_day.update_layout(xaxis_title="Date", yaxis_title="Total Spend")
            st.plotly_chart(fig_spend_by_day, use_container_width=True)

    else:
        st.info("Upload an Excel file, select/enter portfolio, and click **Run**.")

    # --------------------------- Campaign analysis (fuzzy) ---------------------------
    # Always visible using the persisted df in session_state
    df_state = st.session_state.get("df")

    if df_state is None or df_state.empty:
        st.info("Upload an Excel file and click **Run** before filtering by campaign.")
    else:
        with st.expander("Campaign analysis (fuzzy match)"):
            camp_query = st.text_input(
                "Type part of the 'Campaign Name' to fuzzy match",
                value="",
                key="camp_query"
            )

            if camp_query.strip():
                candidates = df_state["Campaign Name"].dropna().unique().tolist()
                if candidates:
                    best = process.extractOne(camp_query.strip(), candidates, scorer=fuzz.token_sort_ratio)
                    best_campaign = best[0] if best else camp_query.strip()
                    st.info(f"Estimated campaign by fuzzy match: **{best_campaign}**")

                    selected_campaign_data = df_state[df_state["Campaign Name"] == best_campaign].copy()

                    if not selected_campaign_data.empty and "Date" in selected_campaign_data.columns:
                        # Clicks by day
                        clicks_by_day = (
                            selected_campaign_data.groupby(["Date", "Placement"])["Clicks"]
                            .sum()
                            .reset_index()
                        )
                        st.plotly_chart(
                            px.line(
                                clicks_by_day, x="Date", y="Clicks", color="Placement",
                                title=f"Clicks Evolution by Placement — Campaign: {best_campaign}",
                                labels={"Date": "Date", "Clicks": "Total Clicks", "Placement": "Placement"},
                                template="plotly_dark"
                            ),
                            use_container_width=True
                        )

                        # Orders by day
                        orders_by_day = (
                            selected_campaign_data.groupby(["Date", "Placement"])["7 Day Total Orders (#)"]
                            .sum()
                            .reset_index()
                        )
                        st.plotly_chart(
                            px.line(
                                orders_by_day, x="Date", y="7 Day Total Orders (#)", color="Placement",
                                title=f"Orders Evolution by Placement — Campaign: {best_campaign}",
                                labels={"Date": "Date", "7 Day Total Orders (#)": "Total Orders (7 Days)", "Placement": "Placement"},
                                template="plotly_dark"
                            ),
                            use_container_width=True
                        )

                        # Spend by day
                        spend_by_day = (
                            selected_campaign_data.groupby(["Date", "Placement"])["Spend"]
                            .sum()
                            .reset_index()
                        )
                        st.plotly_chart(
                            px.line(
                                spend_by_day, x="Date", y="Spend", color="Placement",
                                title=f"Spend Evolution by Placement — Campaign: {best_campaign}",
                                labels={"Date": "Date", "Spend": "Total Spend", "Placement": "Placement"},
                                template="plotly_dark"
                            ),
                            use_container_width=True
                        )
                    else:
                        st.warning("No rows for that campaign or the 'Date' column is missing.")
            else:
                st.caption("Type a term to search the campaign using fuzzy match.")

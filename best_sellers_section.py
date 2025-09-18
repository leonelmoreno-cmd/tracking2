import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from common import GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH, _raw_url_for, fetch_data


# -------------------------------
# Step 1: Load subcategory data
# -------------------------------
def load_subcategory_data(active_basket_name: str) -> pd.DataFrame:
    """
    Load the sub-category CSV corresponding to the current basket.
    """
    subcategory_url = _raw_url_for(GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH, active_basket_name)
    df_sub = fetch_data(subcategory_url)
    df_sub["date"] = pd.to_datetime(df_sub["date"], errors="coerce")
    return df_sub

# -------------------------------
# Step 2: Get latest date data
# -------------------------------
def get_latest_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Return only the rows corresponding to the latest date in the subcategory data.
    """
    latest_date = df["date"].max()
    df_latest = df[df["date"] == latest_date].copy()
    return df_latest, latest_date

# -------------------------------
# Step 3: Top 10 best sellers
# -------------------------------
def top_10_best_sellers(df_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Return the top 10 best-selling products based on the rank column.
    """
    df_top = df_latest.sort_values("rank").head(10)
    return df_top

# -------------------------------
# Step 4: Create Plotly bar chart
# -------------------------------
def create_best_sellers_bar(df_top: pd.DataFrame) -> go.Figure:
    """
    Create a horizontal bar chart for the top 10 best sellers.
    """
    fig = go.Figure(go.Bar(
        x=df_top["rank"][::-1],  # invert for horizontal bar
        y=df_top["asin"][::-1],  # Use ASIN as the label on the Y-axis
        orientation='h',
        text=df_top["rank"][::-1],
        textposition='auto',
        marker_color='orange'
    ))

    fig.update_layout(
        title="Top 10 Best-sellers Rank",
        xaxis_title="Rank",
        yaxis_title="ASIN",
        height=500,
        margin=dict(l=150, r=20, t=50, b=50)
    )
    return fig

# -------------------------------
# Step 5: Streamlit section
# -------------------------------
def render_best_sellers_section(active_basket_name: str):
    st.subheader("Best-sellers Rank")
    st.caption("Top 10 products based on the latest available date from the sub-category file.")

    # Load subcategory data
    df_sub = load_subcategory_data(active_basket_name)
    df_latest, latest_date = get_latest_data(df_sub)
    df_top10 = top_10_best_sellers(df_latest)

    st.markdown(f"**Latest update:** {latest_date.strftime('%Y-%m-%d')}")

    # Plot the chart
    best_sellers_fig = create_best_sellers_bar(df_top10)
    st.plotly_chart(best_sellers_fig, use_container_width=True)

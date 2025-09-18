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
    latest_date = df["date"].max()  # Get the latest date
    df_latest = df[df["date"] == latest_date].copy()  # Filter rows for that date
    return df_latest, latest_date

# -------------------------------
# Step 3: Top 10 best sellers
# -------------------------------
def top_10_best_sellers(df_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Return the top 10 best-selling products based on the 'rank' column.
    Since rank 1 is the best, we sort in ascending order and get the top 10.
    """
    df_top = df_latest.sort_values("rank").head(10)  # Sort by rank (ascending) and get top 10
    return df_top

# -------------------------------
# Step 4: Create Plotly column chart
# -------------------------------
def create_best_sellers_column(df_top: pd.DataFrame) -> go.Figure:
    """
    Create a horizontal bar chart for the top 10 best sellers.
    Each ASIN gets its own bar, ranked from 1 to 10.
    """
    fig = go.Figure(go.Bar(
        y=df_top["asin"],                # ASIN on y-axis (horizontal bars)
        x=df_top["rank"],                # Rank on x-axis (lower rank means better)
        text=df_top["asin"],             # Show ASIN as text on the bars
        textposition='inside',           # Place ASIN text inside the bars
        marker_color='orange',           # Bar color
        orientation='h'                  # Horizontal bars
    ))

    fig.update_layout(
        title="Top 10 Best-sellers Rank",  # Chart title
        xaxis_title="Rank",                 # X-axis label (Rank)
        yaxis_title="ASIN",                 # Y-axis label (ASIN)
        yaxis_autorange="reversed",         # Reverse Y-axis so rank 1 is at the top
        height=500,
        margin=dict(l=80, r=20, t=50, b=100),  # Adjust margins
        showlegend=False                   # Hide the legend (it's not needed here)
    )
    return fig


# -------------------------------
# Step 5: Streamlit section
# -------------------------------
def render_best_sellers_section(active_basket_name: str):
    st.subheader("Best-sellers Rank")  # Section header
    st.caption("Top 10 products based on the latest available date from the sub-category file.")

    # Load subcategory data
    df_sub = load_subcategory_data(active_basket_name)
    df_latest, latest_date = get_latest_data(df_sub)  # Get the latest data
    df_top10 = top_10_best_sellers(df_latest)  # Get the top 10 best sellers

    st.markdown(f"**Latest update:** {latest_date.strftime('%Y-%m-%d')}")  # Display the latest date

    # Plot the chart
    best_sellers_fig = create_best_sellers_column(df_top10)  # Create the column chart
    st.plotly_chart(best_sellers_fig, use_container_width=True)  # Display the chart in Streamlit

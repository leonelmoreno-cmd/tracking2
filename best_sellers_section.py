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
# Step 3: Remove duplicates and get top 10 best sellers
# -------------------------------
def top_10_best_sellers(df_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate ASINs and return the top 10 best-selling products based on the 'rank' column.
    Since rank 1 is the best, we sort in ascending order and get the top 10.
    """
    # Remove duplicates based on 'asin' and 'rank', keeping only the first occurrence
    df_cleaned = df_latest.drop_duplicates(subset=['asin'], keep='first')

    # Sort by rank (ascending) and get top 10
    df_top = df_cleaned.sort_values("rank").head(10)
    return df_top

# -------------------------------
# Step 4: Create Horizontal Stacked Bar Chart
# -------------------------------
def create_best_sellers_stacked_bar(df_top: pd.DataFrame) -> go.Figure:
    """
    Create a horizontal stacked bar chart for the top 10 best sellers.
    Each bar represents an ASIN, stacked according to its rank.
    """
    fig = go.Figure()

    # Create a stacked bar chart, where each ASIN gets its own "stack"
    fig.add_trace(go.Bar(
        y=df_top["asin"],                    # ASIN on the y-axis
        x=df_top["rank"],                    # Rank on the x-axis (lower rank means better)
        text=df_top["asin"],                 # ASIN as text on bars
        textposition='inside',               # Text inside the bars
        marker_color='orange',               # Bar color
        orientation='h',                     # Horizontal bars
        name="Best-sellers",                 # Trace name for legend
        hovertemplate=(
            '<b>ASIN:</b> %{y}<br>'            # Display ASIN
            '<b>Rank:</b> %{x}<br>'            # Display Rank
            '<extra></extra>'                  # Hide the trace name in the hover label
        )
    ))

    fig.update_layout(
        title="Top 10 Best-sellers Rank",      # Chart title
        xaxis_title="Rank",                    # X-axis label (Rank)
        yaxis_title="ASIN",                    # Y-axis label (ASIN)
        yaxis_autorange="reversed",            # Reverse Y-axis so rank 1 is at the top
        height=500,
        margin=dict(l=80, r=20, t=50, b=100),  # Adjust margins
        showlegend=False                       # Hide the legend
    )

    return fig

# -------------------------------
# Step 5: Streamlit section
# -------------------------------
def render_best_sellers_section_with_table(active_basket_name: str):
    st.subheader("Best-sellers Rank")  # Section header
    st.caption("Top 10 products based on the latest available date from the sub-category file.")

    # Load subcategory data
    df_sub = load_subcategory_data(active_basket_name)
    df_latest, latest_date = get_latest_data(df_sub)  # Get the latest data
    df_top10 = top_10_best_sellers(df_latest)  # Get the top 10 best sellers

    st.markdown(f"**Latest update:** {latest_date.strftime('%Y-%m-%d')}")  # Display the latest date

    # Plot the chart
    best_sellers_fig = create_best_sellers_stacked_bar(df_top10)  # Create the stacked bar chart
    st.plotly_chart(best_sellers_fig, use_container_width=True)  # Display the chart in Streamlit

    # Display the data table below the chart
    st.subheader("Top 10 Best-sellers Data")
    st.dataframe(df_top10)  # Show the table with the top 10 ASINs and their ranks

    # Display the product image when hovering over an ASIN
    hover_data = st.empty()  # Placeholder for displaying the image

    # JavaScript to handle hover event
    hover_js = """
    <script>
    var plot = document.getElementsByClassName('js-plotly-plot')[0];
    plot.on('plotly_hover', function(event) {
        var asin = event.points[0].y;
        var image_url = 'https://images-na.ssl-images-amazon.com/images/I/' + asin + '.jpg';
        var img = '<img src="' + image_url + '" width="200">';
        document.getElementById('hover-image').innerHTML = img;
    });
    </script>
    """
    st.markdown(hover_js, unsafe_allow_html=True)

    # Container to display the image
    st.markdown('<div id="hover-image"></div>', unsafe_allow_html=True)

    # Generate unique CSS for each image
    for i, row in df_top10.iterrows():
        image_url = f"https://images-na.ssl-images-amazon.com/images/I/{row['asin']}.jpg"
        hover_class = f"hoverable_{i}"
        tooltip_class = f"tooltip_{i}"
        image_popup_class = f"image-popup_{i}"

        hover_css = f"""
        <style>
        .{hover_class} {{
            position: relative;
            display: inline-block;
            cursor: pointer;
        }}
        .{hover_class} .{tooltip_class} {{
            opacity: 0;
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            transition: opacity 0.5s;
            background-color: rgba(0, 0, 0, 0.8);
            color: #fff;
            padding: 4px;
            border-radius: 4px;
            text-align: center;
            white-space: nowrap;
        }}
        .{hover_class}:hover .{tooltip_class} {{
            opacity: 1;
        }}
        .{image_popup_class} {{
            position: absolute;
            display: none;
            background-image: url({image_url});
            background-size: cover;
            width: 200px;
            height: 200px;
            top: -220px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 999;
        }}
        .{hover_class}:hover .{image_popup_class} {{
            display: block;
        }}
        </style>
        """
        st.markdown(hover_css, unsafe_allow_html=True)

        # Add hoverable div for each ASIN
        hover_html = f"""
        <div class="{hover_class}">
            <div class="{tooltip_class}">Image {i}</div>
            <div class="{image_popup_class}"></div>
        </div>
        """
        st.markdown(hover_html, unsafe_allow_html=True)

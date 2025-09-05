import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="UR - Competitors Price Tracker",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# Centrar el título usando HTML
st.markdown(
    """
    <h1 style="text-align: center; font-size: 36px;">Ultimatum Roach - Competitors Price Tracker</h1>
    """, 
    unsafe_allow_html=True
)

# Function to fetch CSV from GitHub
@st.cache_data
def fetch_data():
    url = "https://raw.githubusercontent.com/leonelmoreno-cmd/tracking2/main/data/competitors_history.csv"
    df = pd.read_csv(url)
    return df

# Function to clean and prepare the data
@st.cache_data
def prepare_data(df):
    # Convert 'date' to datetime type for easier plotting
    df['date'] = pd.to_datetime(df['date'])
    
    # Creating a new column for discount status
    df['discount'] = df.apply(lambda row: 'Discounted' if pd.notna(row['product_original_price']) else 'No Discount', axis=1)
    
    # Calculate the price change and mark where there is a significant change
    df['price_change'] = df.groupby('asin')['product_price'].pct_change() * 100  # Calculate percentage change
    
    return df

# Main Streamlit UI
cols = st.columns([1, 3])

# Load and prepare the data
df = fetch_data()
prepared_df = prepare_data(df)

# Calculate required values
max_discount = prepared_df[prepared_df['discount'] == 'Discounted']['product_price'].max()
min_discount = prepared_df[prepared_df['discount'] == 'Discounted']['product_price'].min()

# Find the ASIN with the largest price change
max_price_change_asin = prepared_df.loc[prepared_df['price_change'].idxmax()]['asin']
min_price_change_asin = prepared_df.loc[prepared_df['price_change'].idxmin()]['asin']

# Top left cell for discount and price change information
top_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

with top_left_cell:
    st.subheader("Key Metrics")
    st.write(f"**Biggest Discount**: ${max_discount:.2f}")
    st.write(f"**Smallest Discount**: ${min_discount:.2f}")
    st.write(f"**ASIN with Largest Price Change**: {max_price_change_asin}")
    st.write(f"**ASIN with Smallest Price Change**: {min_price_change_asin}")

# Right cell for plotting all ASINs
right_cell = cols[1].container(
    border=True, height="stretch", vertical_alignment="center"
)

with right_cell:
    # Plot all ASINs over time (normalized)
    all_asins = prepared_df.pivot(index='date', columns='asin', values='product_price')
    normalized_all_asins = all_asins.div(all_asins.iloc[0])

    st.subheader("All ASINs Over Time (Normalized)")
    fig_all_asins = go.Figure()

    for asin in normalized_all_asins.columns:
        fig_all_asins.add_trace(go.Scatter(x=normalized_all_asins.index, y=normalized_all_asins[asin], mode='lines', name=asin))

    fig_all_asins.update_layout(
        title="Normalized Price History for All ASINs",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        showlegend=True,
        height=400
    )
    st.plotly_chart(fig_all_asins)

# Minigraphs section
st.subheader("Price History for Individual ASINs")

# Create subplots for individual ASINs
def create_price_graph(df):
    asins = df['asin'].unique()  # Get unique ASINs
    num_asins = len(asins)  # Number of subplots we need to create
    
    # Find the maximum price across all ASINs
    max_price = df['product_price'].max()
    
    # Create a subplot layout with 3 columns and the number of rows determined by the number of ASINs
    rows = (num_asins // 3) + (1 if num_asins % 3 != 0 else 0)  # Determine how many rows are needed
    
    fig = make_subplots(
        rows=rows, cols=3, shared_xaxes=True, 
        vertical_spacing=0.1,  # Space between subplots
        horizontal_spacing=0.1,  # Space between subplots
        subplot_titles=[f"ASIN: {asin}" for asin in asins]  # Title with ASIN
    )
    
    # For each ASIN, create a graph in the respective subplot
    for i, asin in enumerate(asins):  # Iterate over each ASIN
        # Filter data for this ASIN
        asin_data = df[df['asin'] == asin]
        
        # Create a scatter plot for this ASIN
        fig.add_trace(
            go.Scatter(
                x=asin_data['date'], 
                y=asin_data['product_price'], 
                mode='lines',
                name=asin,
                line=dict(dash='dot' if asin_data['discount'].iloc[0] == 'Discounted' else 'solid'),
                hovertemplate=(
                    'ASIN: %{text}<br>' +
                    'Price: $%{y:.2f}<br>' +
                    'Date: %{x}<br>' +
                    'Price Change: %{customdata:.2f}%<br>' +
                    '<extra></extra>'
                ),
                text=asin_data['asin'],  # Tooltip with ASIN
                customdata=asin_data['price_change'],  # Custom data for price change percentage
                showlegend=False
            ),
            row=(i // 3) + 1, col=(i % 3) + 1  # Place in correct row and column
        )

    # Update the layout of the plot to set the same Y-axis scale
    fig.update_layout(
        height=400 * rows,  # Set the height for the total grid
        showlegend=True,
        legend_title="ASIN",
        xaxis_title="Date",
        yaxis_title="Product Price",
        yaxis=dict(range=[0, max_price]),  # Set the Y-axis range from 0 to max price
    )
    
    return fig

# Generate and display minigraphs
price_graph = create_price_graph(prepared_df)
st.plotly_chart(price_graph)

# Filters for the Detailed Product Information table
st.subheader("Detailed Product Information")

# Create filters
asin_filter = st.selectbox("Filter by ASIN", options=['All'] + prepared_df['asin'].unique().tolist())
discount_filter = st.selectbox("Filter by Discount Status", options=['All', 'Discounted', 'No Discount'])

# Filter the dataframe based on user input
filtered_df = prepared_df.copy()

if asin_filter != 'All':
    filtered_df = filtered_df[filtered_df['asin'] == asin_filter]

if discount_filter != 'All':
    filtered_df = filtered_df[filtered_df['discount'] == discount_filter]

# Show the filtered table
st.dataframe(filtered_df[['asin', 'product_title', 'product_price', 'product_original_price', 'product_star_rating', 'product_num_ratings', 'is_amazon_choice', 'sales_volume', 'discount', 'date']])

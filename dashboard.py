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
    
    # Create a 'price_group' to divide the data into price ranges
    price_bins = [0, 10, 20, 30, 40, 50, float('inf')]
    price_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50+']
    df['price_group'] = pd.cut(df['product_price'], bins=price_bins, labels=price_labels)
    
    # Calculate the price change and mark where there is a significant change
    df['price_change'] = df.groupby('asin')['product_price'].pct_change() * 100  # Calculate percentage change
    
    return df

# Function to create multiple subplots, each showing one group of prices
def create_price_graph(df):
    price_groups = df['price_group'].unique()  # Get unique price groups
    num_groups = len(price_groups)  # Number of subplots we need to create
    
    # Create a subplot layout with a number of rows based on the price groups
    fig = make_subplots(
        rows=num_groups, cols=1, shared_xaxes=True, 
        vertical_spacing=0.1,  # Space between subplots
        subplot_titles=[f"Price Range: {group}" for group in price_groups]
    )
    
    # For each price group, create a graph
    for i, price_group in enumerate(price_groups):
        # Filter data for this price group
        group_data = df[df['price_group'] == price_group]
        
        # Create a scatter plot for this group
        for asin in group_data['asin'].unique():
            asin_data = group_data[group_data['asin'] == asin]
            
            # Highlight points where the price change is significant (e.g., > 5% change)
            highlight = asin_data[asin_data['price_change'].abs() > 5]  # You can adjust the threshold as needed
            
            fig.add_trace(
                go.Scatter(
                    x=asin_data['date'], 
                    y=asin_data['product_price'], 
                    mode='lines',
                    name=asin,
                    line=dict(dash='dot' if asin_data['discount'].iloc[0] == 'Discounted' else 'solid'),
                    legendgroup=asin,  # Group traces by ASIN
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
                row=i+1, col=1  # Add trace to the correct subplot (row=i+1, col=1)
            )
            
            # Add points where there's a significant price change (highlighted)
            fig.add_trace(
                go.Scatter(
                    x=highlight['date'],
                    y=highlight['product_price'],
                    mode='markers',
                    name=f"Significant Change {asin}",
                    marker=dict(size=10, color='red', symbol='star'),
                    hovertemplate=(
                        'ASIN: %{text}<br>' +
                        'Price: $%{y:.2f}<br>' +
                        'Date: %{x}<br>' +
                        'Price Change: %{customdata:.2f}%<br>' +
                        '<extra></extra>'
                    ),
                    text=highlight['asin'],  # Tooltip with ASIN
                    customdata=highlight['price_change'],  # Custom data for price change percentage
                    showlegend=False
                ),
                row=i+1, col=1
            )

    # Update the layout of the plot
    fig.update_layout(
        height=300 * num_groups,  # Set the height based on the number of subplots
        title="Competitors Price History by Price Range",
        title_x=0.5,  # Center the title
        showlegend=True,
        legend_title="ASIN",
        xaxis_title="Date",
        yaxis_title="Product Price"
    )
    
    return fig

# Main Streamlit UI
st.title("Ultimatum Roach - Competitors Price Tracker")

# Load and prepare the data
df = fetch_data()
prepared_df = prepare_data(df)

# Create the price graph with subplots
price_graph = create_price_graph(prepared_df)

# Show the plot
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

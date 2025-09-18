import streamlit as st
import pandas as pd
import plotly.graph_objects as go

@st.cache_data(show_spinner=False)
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the dataset by processing the input DataFrame.
    
    Does it properly convert the 'date' column to datetime? 
    Is the 'week_number' calculated correctly using the ISO calendar? 
    Does it add the correct 'discount' label? 
    Does it compute the percentage change in 'product_price' correctly for each ASIN?
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["week_number"] = df["date"].dt.isocalendar().week
    df = df.sort_values(by=["asin", "week_number"])
    
    # Assigning 'discount' column based on product's original price.
    df["discount"] = df.apply(
        lambda row: "Discounted" if pd.notna(row.get("product_original_price")) else "No Discount",
        axis=1
    )
    
    # Calculating price change for each ASIN (Product).
    df["price_change"] = df.groupby("asin")["product_price"].pct_change() * 100
    return df

def plot_rating_evolution(df: pd.DataFrame):
    """
    Plots the evolution of product ratings (star ratings) over time (or weeks).
    
    Does the 'asin' column contain unique ASINs for each product? 
    Are the ratings being displayed correctly over time? 
    Are you able to compare the rating changes for different ASINs?
    """
    fig = go.Figure()
    for asin in df['asin'].unique():
        asin_data = df[df['asin'] == asin]
        fig.add_trace(go.Scatter(
            x=asin_data['date'],
            y=asin_data['product_star_rating'],
            mode='lines+markers',
            name=f'ASIN {asin}'
        ))
    
    fig.update_layout(
        title="Rating Evolution",
        xaxis_title="Date",
        yaxis_title="Product Star Rating"
    )
    st.plotly_chart(fig)

def plot_price_variation(df: pd.DataFrame):
    """
    Plots the percentage variation in the product price over time (or weeks).
    
    Is the price change percentage being calculated correctly for each product (ASIN)? 
    Are the price variations shown clearly for comparison?
    """
    df['price_change'] = df.groupby('asin')['product_price'].pct_change() * 100
    fig = go.Figure()
    for asin in df['asin'].unique():
        asin_data = df[df['asin'] == asin]
        fig.add_trace(go.Scatter(
            x=asin_data['date'],
            y=asin_data['price_change'],
            mode='lines+markers',
            name=f'ASIN {asin}'
        ))
    
    fig.update_layout(
        title="Price Percentage Variation",
        xaxis_title="Date",
        yaxis_title="Price Variation (%)"
    )
    st.plotly_chart(fig)

def plot_ranking_evolution(df: pd.DataFrame):
    """
    Plots the ranking of products based on their ratings over time.
    
    Are the products (ASINs) being ranked correctly for each date? 
    Is the ranking evolution based on star ratings visualized properly?
    """
    df['ranking'] = df.groupby('date')['product_star_rating'].rank(method='first')
    fig = go.Figure()
    for date in df['date'].unique():
        date_data = df[df['date'] == date]
        fig.add_trace(go.Scatter(
            x=date_data['asin'],
            y=date_data['ranking'],
            mode='lines+markers',
            name=f'Date {date.date()}'
        ))
    
    fig.update_layout(
        title="Ranking Evolution",
        xaxis_title="ASIN",
        yaxis_title="Ranking"
    )
    st.plotly_chart(fig)

def main(df):
    """
    Main function to choose which chart to display.
    
    Does the chart option work correctly based on user input? 
    Are the charts generated accurately for 'Rating Evolution', 'Price Variation', and 'Ranking Evolution'?
    """
    chart_option = st.selectbox(
        "Select Chart",
        ("Rating Evolution", "Price Percentage Variation", "Ranking Evolution")
    )
    
    if chart_option == "Rating Evolution":
        plot_rating_evolution(df)
    elif chart_option == "Price Percentage Variation":
        plot_price_variation(df)
    elif chart_option == "Ranking Evolution":
        plot_ranking_evolution(df)

if __name__ == "__main__":
    main(df)  # Make sure 'df' is being passed correctly when you call the main function.

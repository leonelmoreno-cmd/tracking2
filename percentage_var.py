import streamlit as st
import pandas as pd
import plotly.graph_objects as go

@st.cache_data(show_spinner=False)
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["week_number"] = df["date"].dt.isocalendar().week
    df = df.sort_values(by=["asin", "week_number"])
    df["discount"] = df.apply(
        lambda row: "Discounted" if pd.notna(row.get("product_original_price")) else "No Discount",
        axis=1
    )
    df["price_change"] = df.groupby("asin")["product_price"].pct_change() * 100
    return df

def plot_rating_evolution(df: pd.DataFrame):
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

def main():
    df = prepare_data(st.session_state['df'])
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
    main()

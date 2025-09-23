import streamlit as st
from components.common import set_page_config

def main():
    set_page_config()

    st.header("Settings")

    st.write("In this section you can adjust configuration options.")

    # Example: allow changing default basket selection
    # This assumes you will persist the choice somehow (session_state, config file, etc.)
    default_basket = st.text_input(
        "Default Basket Filename", value="synthethic3.csv", help="Enter the CSV file name of the default basket"
    )

    # Example: changing display options
    show_raw_dates = st.checkbox("Show raw dates in tables", value=False)
    st.write(f"Show raw dates: {show_raw_dates}")

    # You can add more settings here: theme toggles, filters defaults, etc.

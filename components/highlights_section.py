# highlights_section.py
import streamlit as st
import pandas as pd
import numpy as np

def render_highlights(df_overview: pd.DataFrame, period: str = "week"):
    """
    Render the last period highlights as 3 columns of metrics.
    Should be called inside the right_col of your layout.
    """
    st.markdown("### Last period highlights")
    highlights = compute_highlights(df_overview, period=period)
    label = highlights.get("label", "N/A")

    dcol, pcol, ccol = st.columns(3)

    with dcol:
        if highlights.get("row_max_disc") is not None:
            st.metric(f"🏷️ Highest discount — {label} — {highlights['row_max_disc']['brand']}", 
                      f"{highlights['row_max_disc']['discount_pct']:.1f}%")
        else:
            st.metric(f"🏷️ Highest discount — {label}", "N/A")

        if highlights.get("row_min_disc") is not None:
            st.metric(f"🏷️ Lowest discount — {label} — {highlights['row_min_disc']['brand']}", 
                      f"{highlights['row_min_disc']['discount_pct']:.1f}%")
        else:
            st.metric(f"🏷️ Lowest discount — {label}", "N/A")

    with pcol:
        if highlights.get("row_max_price") is not None:
            st.metric(f"💲 Highest price — {label} — {highlights['row_max_price']['brand']}", 
                      f"${highlights['row_max_price']['product_price']:.2f}")
        else:
            st.metric(f"💲 Highest price — {label}", "N/A")

        if highlights.get("row_min_price") is not None:
            st.metric(f"💲 Lowest price — {label} — {highlights['row_min_price']['brand']}", 
                      f"${highlights['row_min_price']['product_price']:.2f}")
        else:
            st.metric(f"💲 Lowest price — {label}", "N/A")

    with ccol:
        if highlights.get("row_max_change") is not None:
            st.metric(f"🔺 Largest price change (last update) — {label} — {highlights['row_max_change']['brand']}", 
                      f"{highlights['row_max_change']['price_change']:+.1f}%")
        else:
            st.metric(f"🔺 Largest price change (last update) — {label}", "N/A")

        if highlights.get("row_min_change") is not None:
            st.metric(f"🔻 Lowest price change (last update) — {label} — {highlights['row_min_change']['brand']}", 
                      f"{highlights['row_min_change']['price_change']:+.1f}%")
        else:
            st.metric(f"🔻 Lowest price change (last update) — {label}", "N/A")

import streamlit as st
from typing import Dict

def render_basket_and_toggle(name_to_url: Dict[str, str], active_basket_name: str, default_basket: str):
    """
    Render the top-centered container for:
    - Current basket display
    - Basket selection (popover)
    - Global toggle for daily/weekly aggregation

    Returns:
        period (str): "day" or "week"
        active_basket_name (str): updated basket name if changed
    """
    col1, col2, col3 = st.columns([3, 2, 2], gap="small")

    with col1:
        st.markdown(
            f"<div style='text-align:left; margin:4px 0;'>"
            f"<span style='color:#16a34a; font-weight:600;'>Current basket:</span> "
            f"<code style='color:#16a34a;'>{active_basket_name}</code>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col2:
        with st.popover("🧺 Change basket"):
            st.caption("Pick a CSV from the list and click Apply.")
            options = list(name_to_url.keys()) if name_to_url else [default_basket]
            idx = options.index(active_basket_name) if active_basket_name in options else 0
            sel = st.selectbox("File (CSV) in repo", options=options, index=idx, key="basket_select")
            if st.button("Apply", type="primary"):
                st.session_state["basket"] = sel
                if hasattr(st, "query_params"):
                    st.query_params["basket"] = sel
                else:
                    try:
                        st.experimental_set_query_params(basket=sel)
                    except Exception:
                        pass
                st.experimental_rerun()  # rerun seguro

    with col3:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        aggregate_daily = st.toggle(
            "Aggregate by day (instead of week)",
            value=False,
            help="When ON, all charts use daily prices; when OFF, weekly averages."
        )
        period = "day" if aggregate_daily else "week"

    return period, st.session_state["basket"]

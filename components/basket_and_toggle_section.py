import streamlit as st
from typing import Dict

def render_basket_and_toggle(
    name_to_url: Dict[str, str],
    active_basket_name: str,
    default_basket: str
):
    """
    Render the top-centered container for:
    - Current basket display
    - Basket selection (popover)
    - Global toggle for daily/weekly aggregation

    Returns:
        period (str): "day" or "week"
        active_basket_name (str): updated basket name if changed
    """

    # Asegura valor inicial para el basket en session_state
    st.session_state.setdefault("basket", active_basket_name or default_basket)

    col1, col2, col3 = st.columns([3, 2, 2], gap="small")

    # Columna 1: indicador de basket actual
    with col1:
        st.markdown(
            (
                "<div style='text-align:left; margin:4px 0;'>"
                "<span style='color:#16a34a; font-weight:600;'>Current basket:</span> "
                f"<code style='color:#16a34a;'>{st.session_state['basket']}</code>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    # Columna 2: selector de basket
    with col2:
        with st.popover("🧺 Change basket"):
            st.caption("Pick a CSV from the list and click Apply.")
            options = list(name_to_url.keys()) if name_to_url else [default_basket]

            # usar el valor actual como índice por defecto si existe
            idx = options.index(st.session_state["basket"]) if st.session_state["basket"] in options else 0
            sel = st.selectbox("File (CSV) in repo", options=options, index=idx, key="basket_select")

            if st.button("Apply", type="primary"):
                if sel != st.session_state["basket"]:
                    st.session_state["basket"] = sel
                    # Escribe el parámetro en la URL con la API moderna (1.30+)
                    try:
                        st.query_params["basket"] = sel  # setter directo tipo dict
                    except Exception:
                        # Fallback legacy si fuera necesario
                        try:
                            st.experimental_set_query_params(basket=sel)
                        except Exception:
                            pass
                    # Re-ejecuta la app con la API moderna (1.27+)
                    st.rerun()

    # Columna 3: toggle de agregación
    with col3:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        aggregate_daily = st.toggle(
            "By day",
            value=False,
            help="When ON, all charts use daily prices; when OFF, weekly averages.",
        )
        period = "day" if aggregate_daily else "week"

    return period, st.session_state["basket"]

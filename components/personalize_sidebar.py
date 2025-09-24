import streamlit as st

def inject_sidebar_nav_css(font_px: int = 18, row_gap_px: int = 6):
    st.markdown(
        f"""
        <style>
          /* Aumenta el tamaño de fuente de los links del nav */
          section[data-testid="stSidebarNav"] a {{
            font-size: {font_px}px !important;
            line-height: 1.3 !important;
          }}
          /* Espaciado entre items del nav (funciona en la mayoría de versiones) */
          section[data-testid="stSidebarNav"] li {{
            margin-bottom: {row_gap_px}px !important;
          }}
        </style>
        """,
        unsafe_allow_html=True
    )

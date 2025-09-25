import streamlit as st

def display_header(df):
    """
    Muestra el encabezado con la última fecha de actualización.
    :param df: El DataFrame con la columna 'date'.
    """
    # Obtener la última fecha de actualización
    last_update = df["date"].max()
    last_update_str = last_update.strftime('%Y-%m-%d') if pd.notna(last_update) else "No data available"

    # Mostrar el encabezado con la última actualización
    st.markdown(
        f"""
        <div style="text-align:center;">
            <h6 style="color:#666; font-weight:200; margin-top:0;">
                Last update: {last_update_str} - Made with ❤️ from 🇻🇪
            </h6>
        </div>
        """,
        unsafe_allow_html=True,
    )

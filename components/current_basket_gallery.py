import re
import math
import pandas as pd
import streamlit as st

def _fix_photo_url(url: str) -> str:
    """Fix common typos like '.jpgm' -> '.jpg'."""
    if not isinstance(url, str):
        return ""
    url = re.sub(r"(\.jpg)m($|\b)", r"\1", url, flags=re.IGNORECASE)
    url = re.sub(r"(\.png)m($|\b)", r"\1", url, flags=re.IGNORECASE)
    return url

@st.cache_data(show_spinner=False)
def _latest_snapshot_by_asin(df: pd.DataFrame) -> pd.DataFrame:
    """Return 1 latest row per ASIN (by date)."""
    if df.empty or "asin" not in df.columns or "date" not in df.columns:
        return pd.DataFrame()
    # Garantizar datetime y ordenar por fecha
    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce")
    dfx = dfx.sort_values(["asin", "date"])
    # Tomar la última fila por asin
    idx = dfx.groupby("asin")["date"].idxmax()
    snap = dfx.loc[idx].copy().sort_values("brand", na_position="last")
    return snap

def _short(s: str, n: int = 80) -> str:
    """Shorten the string to a given length."""
    if not isinstance(s, str):
        return ""
    return s if len(s) <= n else s[: n - 1] + "…"

def render_current_basket_gallery(df: pd.DataFrame, columns: int = 5) -> None:
    """
    Render a grid of small product cards (Current basket):
    - Photo (st.image)
    - Brand — ASIN
    - Title (truncated)
    - Link to product
    - Popover with detailed info
    """
    st.subheader("Current basket")

    # Filtrar para mostrar solo las filas con la fecha más reciente
    df_filtered = df[df['date'] == df['date'].max()]

    # Filtrar para asegurarnos de que solo mostramos productos con 'product_photo' disponible
    df_filtered = df_filtered[df_filtered['product_photo'].notna()]

    snap = _latest_snapshot_by_asin(df_filtered)
    if snap.empty:
        st.info("No products to display in the current basket.")
        return

    # Ordenar por brand y luego asin para una grilla estable
    if "brand" in snap.columns:
        snap = snap.sort_values(["brand", "asin"])
    else:
        snap = snap.sort_values(["asin"])

    # Render en grid
    cols_per_row = max(2, min(8, columns))
    rows = math.ceil(len(snap) / cols_per_row)

    # Campos que intentaremos mostrar en Details si existen
    detail_fields = [
        ("Brand", "brand"),
        ("ASIN", "asin"),
        ("Title", "product_title"),
        ("Price", "product_price"),
        ("Original price", "product_original_price"),
        ("Star rating", "product_star_rating"),
        ("# Ratings", "product_num_ratings"),
        ("Discount", "discount"),
        ("Sales volume", "sales_volume"),
        ("Unit price", "unit_price"),
        ("Rank", "rank"),
        ("URL", "product_url"),
    ]

    for r in range(rows):
        row = snap.iloc[r * cols_per_row : (r + 1) * cols_per_row]
        cols = st.columns(cols_per_row, gap="large")

        for col, (_, item) in zip(cols, row.iterrows()):
            with col:
                brand = item.get("brand", "")
                asin = item.get("asin", "")
                title = item.get("product_title", "")
                product_url = item.get("product_url", "")
                photo_url = item.get("product_photo", "")  # Usar product_photo directamente

                # Imagen
                if photo_url:
                    photo_url = _fix_photo_url(photo_url)  # Corregir la URL si tiene errores
                    st.image(photo_url, caption=None, width="stretch")
                else:
                    st.write("📷 No image")
                # Encabezado corto
                st.markdown(f"**{brand} — {asin}**" if brand else f"**{asin}**")
                if title:
                    st.caption(_short(title, 90))

                # Link al producto (si existe)
                if isinstance(product_url, str) and product_url:
                    st.markdown(f"[Open product page]({product_url})")

                # Popover con detalles
                with st.popover("Details"):
                    # Render limpio de campos disponibles
                    for label, colname in detail_fields:
                        if colname in item and pd.notna(item[colname]):
                            val = item[colname]
                            # Formateos sencillos
                            if label in ("Price", "Original price") and isinstance(val, (int, float)):
                                st.write(f"**{label}:** ${val:,.2f}")
                            else:
                                st.write(f"**{label}:** {val}")

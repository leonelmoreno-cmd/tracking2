# components/campaigns_evolution_components.py

import pandas as pd
import logging
from typing import List, Dict, Tuple
from rapidfuzz import process, fuzz
import plotly.graph_objects as go
from fpdf import FPDF
import tempfile
import os
import plotly.io as pio
import streamlit as st
logging.basicConfig(level=logging.INFO)

def load_weekly_file(file, sheet_name: str = "Sponsored Products Campaigns") -> pd.DataFrame:
    """Load a weekly Excel/CSV file and extract the campaigns and status columns."""
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file, sheet_name=sheet_name)

        # Verificar las primeras filas y las columnas del DataFrame cargado
        st.write("DataFrame cargado antes de cualquier transformación:")
        st.dataframe(df.head())  # Muestra las primeras filas
        st.write(f"Columnas en DataFrame cargado: {df.columns}\n")  # Muestra las columnas

        # Rename relevant columns
        df = df.rename(
            columns={
                "Campaign Name (Informational only)": "campaign",
                "Status": "status",
                "Entity": "entity",
                "State": "state",  
                "Campaign State (Informational only)": "campaign_state", 
                "Ad Group State (Informational only)": "ad_group_state",
                "Keyword Text": "keyword_text",
                "Product Targeting Expression": "product_targeting_expression"
            }
        )

        # Filtro: solo conservar Keywords y Product Targeting
        df = df[df["entity"].isin(["Keyword", "Product Targeting"])]

        # Filtro: solo campañas habilitadas
        df = df[df["state"].isin(["enabled"])]

        # Filtro adicional: asegurarse de que las columnas necesarias estén habilitadas
        df = df[df["campaign_state"].isin(["enabled"])]
        df = df[df["ad_group_state"].isin(["enabled"])]

        # Si "Keyword Text" está vacío, tomamos "Product Targeting Expression"
        df["keyword_text"] = df["keyword_text"].fillna(df["product_targeting_expression"])

        # Clean up
        df["status"] = df["status"].fillna("White").astype(str).str.strip()
        df["campaign"] = df["campaign"].astype(str).str.strip()

        return df[["campaign", "status", "keyword_text", "product_targeting_expression"]]
    
    except Exception as e:
        logging.error(f"Error loading file {file.name}: {e}")
        return pd.DataFrame(columns=["campaign", "status", "keyword_text", "product_targeting_expression"])

# ---------- Fuzzy Matching ----------
def unify_campaign_names(weekly_dfs: List[pd.DataFrame], threshold: int = 90) -> List[pd.DataFrame]:
    """Unify campaign names across weeks using fuzzy matching."""
    master_names = set(weekly_dfs[0]["campaign"].unique())
    mapping: Dict[str, str] = {}

    for df in weekly_dfs[1:]:
        for name in df["campaign"].unique():
            match, score, _ = process.extractOne(name, master_names, scorer=fuzz.ratio)
            if score >= threshold:
                mapping[name] = match
            else:
                mapping[name] = name
                master_names.add(name)

    # Apply mapping to all dataframes
    unified_dfs = []
    for df in weekly_dfs:
        df = df.copy()
        df["campaign"] = df["campaign"].apply(lambda x: mapping.get(x, x))
        unified_dfs.append(df)
    return unified_dfs


# ---------- Transformations ----------
def build_transitions(weekly_dfs: List[pd.DataFrame]) -> Tuple[List[str], List[int], List[int], List[int]]:
    """Build Sankey nodes and links from weekly dataframes."""
    weeks = ["W1", "W2", "W3"]
    nodes = []
    node_map = {}
    index = 0

    for i, df in enumerate(weekly_dfs):
        for status in df["status"].unique():
            label = f"{weeks[i]}-{status}"
            if label not in node_map:
                node_map[label] = index
                nodes.append(label)
                index += 1

    sources, targets, values = [], [], []
    for i in range(len(weekly_dfs) - 1):
        merged = pd.merge(
            weekly_dfs[i], weekly_dfs[i + 1],
            on="campaign", how="outer", suffixes=(f"_{i}", f"_{i+1}")
        )
        for _, row in merged.iterrows():
            s1 = f"{weeks[i]}-{row.get(f'status_{i}', 'not_present')}"
            s2 = f"{weeks[i+1]}-{row.get(f'status_{i+1}', 'not_present')}"
            if s1 not in node_map:
                node_map[s1] = index
                nodes.append(s1)
                index += 1
            if s2 not in node_map:
                node_map[s2] = index
                nodes.append(s2)
                index += 1
            sources.append(node_map[s1])
            targets.append(node_map[s2])
            values.append(1)
    return nodes, sources, targets, values


# ---------- Visualization ----------
def create_sankey(nodes: List[str], sources: List[int], targets: List[int], values: List[int]) -> go.Figure:
    """Create a Plotly Sankey diagram."""
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(label=nodes, pad=20, thickness=20),
            link=dict(source=sources, target=targets, value=values)
        )
    )
    fig.update_layout(title_text="Campaign Status Evolution", font_size=10)
    return fig


# ---------- PDF Export ----------
def export_pdf(fig: go.Figure, filtered_df: pd.DataFrame) -> str:
    """Generate a PDF with Sankey image, filtered campaigns, and full evolution table."""
    tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")

    # Guardar figura como PNG (requiere kaleido en requirements.txt)
    fig.write_image(tmp_img.name, format="png")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Título principal
    pdf.cell(200, 10, "Campaigns Evolution Overview", ln=True, align="C")
    pdf.image(tmp_img.name, x=10, y=20, w=180)

    # Espacio después de la imagen
    pdf.ln(110)

    # Sección 1: campañas críticas en W3
    pdf.cell(200, 10, "Filtered Campaigns (W3: Purple/White)", ln=True, align="L")
    for _, row in filtered_df.iterrows():
        pdf.cell(200, 10, f"- {row['campaign']} (Final: {row['W3_status']})", ln=True, align="L")

    pdf.ln(10)

    # Sección 2: evolución completa
    pdf.cell(200, 10, "Full Evolution of Campaigns", ln=True, align="L")
    for _, row in filtered_df.iterrows():
        w1 = row.get("W1_status", "not_present")
        w2 = row.get("W2_status", "not_present")
        w3 = row.get("W3_status", "not_present")
        pdf.cell(200, 10, f"- {row['campaign']} : {w1} -> {w2} -> {w3}", ln=True, align="L")

    # Guardar PDF temporal
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_pdf.name)

    # Eliminar imagen temporal
    os.unlink(tmp_img.name)

    return tmp_pdf.name

# ---------- Evolution Table ----------
def build_evolution_table(weekly_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Return a dataframe with campaigns and their status across W1, W2, W3.
       Only keep campaigns that appear in all weeks (inner join)."""

    # Verificar el contenido de cada DataFrame antes del merge
    for i, df in enumerate(weekly_dfs, start=1):
        st.write(f"DataFrame {i} antes del merge:")
        st.dataframe(df.head())  # Muestra las primeras filas del DataFrame en Streamlit
        st.write(f"Columnas en DataFrame {i}: {df.columns}\n")  # Muestra las columnas

    # Primer DataFrame (W1), sin renombrar las columnas
    combined = weekly_dfs[0]  # No renombramos columnas

    # Merge W1 con W2 sin renombrar las columnas
    combined = combined.merge(
        weekly_dfs[1],  # Merge con el segundo DataFrame
        on=["campaign", "keyword_text"], how="inner"  # Usamos las mismas columnas para hacer el merge
    )

    # Merge con W3
    combined = combined.merge(
        weekly_dfs[2],  # Merge con el tercer DataFrame
        on=["campaign", "keyword_text"], how="inner"  # Usamos las mismas columnas para hacer el merge
    )

    return combined

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

        return df[["campaign", "status", "keyword_text"]]
    
    except Exception as e:
        logging.error(f"Error loading file {file.name}: {e}")
        return pd.DataFrame(columns=["campaign", "status", "keyword_text"])

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
def build_transitions(weekly_dfs: List[pd.DataFrame]) -> Tuple[List[str], List[int], List[int], List[int], pd.DataFrame]:
    weeks = ["W1", "W2", "W3"]
    nodes = []
    node_map = {}
    index = 0
    transitions = []

    # Crear nodos para cada combinación de semana y estado
    for i, df in enumerate(weekly_dfs):
        for status in df["status"].unique():
            label = f"{weeks[i]}-{status}"
            if label not in node_map:
                node_map[label] = index
                nodes.append(label)
                index += 1

    sources, targets, values = [], [], []
    # Generar transiciones entre semanas
    for i in range(len(weekly_dfs)):
        for j in range(i + 1, len(weekly_dfs)):
            merged = pd.merge(
                weekly_dfs[i], weekly_dfs[j],
                on="campaign", how="inner", suffixes=(f"_{i}", f"_{j}")
            )
            for _, row in merged.iterrows():
                s1 = f"{weeks[i]}-{row.get(f'status_{i}', 'not_present')}"
                s2 = f"{weeks[j]}-{row.get(f'status_{j}', 'not_present')}"
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
                transitions.append([row["campaign"], s1, s2])

    transitions_df = pd.DataFrame(transitions, columns=["Campaign", "From", "To"])
    return nodes, sources, targets, values, transitions_df

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
        pdf.cell(200, 10, f"- {row['campaign']} (Final: {row['status']})", ln=True, align="L")

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
def build_evolution_table(weekly_dfs: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construye dos tablas:
       1. Evolución completa de campañas a través de W1, W2 y W3 (solo campañas presentes en todas las semanas).
       2. Evolución filtrada por W3 con estado 'Purple' o 'White'."""

    # Realizar merge 'inner' entre las tres semanas
    combined = weekly_dfs[0]
    for i in range(1, len(weekly_dfs)):
        combined = pd.merge(
            combined, weekly_dfs[i],
            on=["campaign", "keyword_text"], how="inner"
        )

        # Renombrar las columnas de estado para que sean más claras
    combined = combined.rename(columns={
        'status': 'W1',        # Renombramos 'status' (W1) a 'W1'
        'status_x': 'W2',      # Renombramos 'status_x' (W2) a 'W2'
        'status_y': 'W3'       # Renombramos 'status_y' (W3) a 'W3'
    })
    
    # Filtrar campañas en W3 con estado 'Purple' o 'White'
    filtered_w3 = combined[combined["status"] == "Purple"]
    filtered_w3 = pd.concat([filtered_w3, combined[combined["status"] == "White"]])

    return combined, filtered_w3

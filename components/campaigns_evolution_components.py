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

# Asignamos un color para cada estado
state_colors = {
    "Purple": "purple",
    "White": "white",
    "Green": "green",
    "Red": "red",
    "Orange": "orange"
}

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
    node_colors = []  # Lista para los colores de los nodos
    index = 0
    transitions = []

    # Crear nodos para cada combinación de semana y estado
    for i, df in enumerate(weekly_dfs):
        for status in df["status"].unique():
            label = f"{weeks[i]}-{status}"
            if label not in node_map:
                node_map[label] = index
                nodes.append(label)
                node_colors.append(state_colors.get(status, "gray"))  # Asigna color basado en el estado
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
                    node_colors.append(state_colors.get(row.get(f'status_{i}', 'not_present'), "gray"))  # Color del nodo
                    index += 1
                if s2 not in node_map:
                    node_map[s2] = index
                    nodes.append(s2)
                    node_colors.append(state_colors.get(row.get(f'status_{j}', 'not_present'), "gray"))  # Color del nodo
                    index += 1

                sources.append(node_map[s1])
                targets.append(node_map[s2])
                values.append(1)
                transitions.append([row["campaign"], s1, s2])

    transitions_df = pd.DataFrame(transitions, columns=["Campaign", "From", "To"])
    return nodes, sources, targets, values, node_colors, transitions_df

# ---------- Visualization ----------
def create_sankey(nodes: List[str], sources: List[int], targets: List[int], values: List[int], node_colors: List[str], status_filter: str) -> go.Figure:
    """Create a Plotly Sankey diagram for a specific status (e.g. Purple, White, etc.)"""
    
    # Filtrar los nodos y enlaces basados en el estado
    filtered_sources = [i for i, source in enumerate(sources) if node_colors[source] == status_filter]
    filtered_targets = [i for i, target in enumerate(targets) if node_colors[target] == status_filter]
    filtered_values = [values[i] for i in filtered_sources]

    # Crear diagrama Sankey para el estado filtrado
    fig = go.Figure(
        go.Sankey(
            arrangement="freeform",  # Freeform arrangement for better node placement
            node=dict(
                label=[nodes[i] for i in filtered_sources],
                pad=50,  # Increased pad to provide more space between nodes
                thickness=20,
                line=dict(color="black", width=0.5),
                color=[node_colors[i] for i in filtered_sources],
                hovertemplate="Node: %{label}<br>Value: %{value}<extra></extra>"  # Hover info for nodes
            ),
            link=dict(
                source=filtered_sources,
                target=filtered_targets,
                value=filtered_values,
                color=[node_colors[i] for i in filtered_sources],  # Color based on source node
                hovertemplate="Source: %{source.label}<br>Target: %{target.label}<br>Value: %{value}<extra></extra>"  # Hover info for links
            )
        )
    )
    fig.update_layout(
        title_text=f"Campaign Status Evolution - {status_filter}",
        font_size=10,
        hovermode="x unified"  # Enable hover for both source and target
    )
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

    # Sección 1: campañas filtradas (sin el W3 en el nombre de la sección)
    pdf.cell(200, 10, "Filtered Campaigns (Purple/White)", ln=True, align="L")
    for _, row in filtered_df.iterrows():
        # Usar las columnas W1, W2 y W3 para mostrar los estados en cada semana
        pdf.cell(200, 10, f"- {row['campaign']} (Final: {row['W3']})", ln=True, align="L")

    pdf.ln(10)

    # Sección 2: evolución completa de las campañas
    pdf.cell(200, 10, "Full Evolution of Campaigns", ln=True, align="L")
    for _, row in filtered_df.iterrows():
        w1 = row.get("W1", "not_present")
        w2 = row.get("W2", "not_present")
        w3 = row.get("W3", "not_present")
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
    combined = weekly_dfs[0]  # Empezamos con los datos de la primera semana (W1)
    
    # Hacemos un merge entre W1, W2 y W3
    for i in range(1, len(weekly_dfs)):
        combined = pd.merge(
            combined, weekly_dfs[i],  # Fusionamos el DataFrame combinado con el siguiente
            on=["campaign", "keyword_text"],  # Usamos "campaign" y "keyword_text" como claves para la fusión
            how="inner"  # "inner" significa que solo se mantendrán las campañas presentes en todas las semanas
        )

    # Renombrar las columnas de estado para que sean más claras
    combined = combined.rename(columns={
        'status': 'W1',        # Renombramos 'status' (W1) a 'W1'
        'status_x': 'W2',      # Renombramos 'status_x' (W2) a 'W2'
        'status_y': 'W3'       # Renombramos 'status_y' (W3) a 'W3'
    })

    # Reordenar las columnas para que queden en el orden deseado
    combined = combined[["campaign","keyword_text", "W1", "W2", "W3"]]
    
    # Filtrar solo las campañas que tienen 'Purple' o 'White' en W1, W2 y W3
    filtered_w3 = combined[
        (combined["W1"].isin(["Purple", "White"])) &
        (combined["W2"].isin(["Purple", "White"])) &
        (combined["W3"].isin(["Purple", "White"]))
    ]

    return combined, filtered_w3

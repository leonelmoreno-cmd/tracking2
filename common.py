import pandas as pd
import requests
from typing import Dict, List  # Asegúrate de que Dict y List estén correctamente importados
import streamlit as st  # Importa streamlit

# -------------------------------
# Repo constants
# -------------------------------
GITHUB_OWNER = "leonelmoreno-cmd"
GITHUB_REPO = "tracking2"
GITHUB_PATH = "data"
GITHUB_BRANCH = "main"

# -------------------------------
# Helper functions
# -------------------------------

def _raw_url_for(owner: str, repo: str, branch: str, path: str, fname: str) -> str:
    """
    Construye una URL de GitHub para acceder a un archivo en el repositorio especificado.
    Si el archivo tiene una subcategoría asociada, ajusta la ruta para incluir la subcarpeta correspondiente.
    Si no, devuelve el archivo principal directamente.
    """
    COMPETITOR_TO_SUBCATEGORY_MAP = {
        "competitors_history - BL.csv": "sub-categories2/sub_BL.csv",
        "competitors_history - GS.csv": "sub-categories2/sub_GS.csv",
        "competitors_history - IC.csv": "sub-categories2/sub_IC.csv",
        "competitors_history - LGM.csv": "sub-categories2/sub_LGM.csv",
        "competitors_history - QC.csv": "sub-categories2/sub_QC.csv",
        "competitors_history - RIO.csv": "sub-categories2/sub_RIO.csv",
        "competitors_history - UR.csv": "sub-categories2/sub_UR.csv",
        "synthethic3.csv": "sub-categories2/sub_SYN.csv"
    }

    subcategory_file = COMPETITOR_TO_SUBCATEGORY_MAP.get(fname)
    if subcategory_file:
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}/{subcategory_file}"
    else:
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}/{fname}"

    # Agregar un print o st.write para verificar la URL
    print(f"Generated URL: {url}")
    return url

def fetch_data(url: str) -> pd.DataFrame:
    """
    Fetch CSV data from a URL and return it as a pandas DataFrame.
    """
    return pd.read_csv(url)

@st.cache_data(show_spinner=False)
def list_repo_csvs(owner: str, repo: str, path: str, branch: str = "main") -> List[dict]:
    """
    Returns a list of the main CSV files (not sub-categories) from a GitHub repository.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    headers = {"Accept": "application/vnd.github+json"}
    token = st.secrets.get("GITHUB_TOKEN", None)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    items = resp.json()

    main_files = set({
        "competitors_history - BL.csv", "competitors_history - GS.csv", "competitors_history - IC.csv", 
        "competitors_history - LGM.csv", "competitors_history - QC.csv", "competitors_history - RIO.csv", 
        "competitors_history - UR.csv", "synthethic3.csv"
    })
    
    csvs = [
        {
            "name": it["name"],
            "download_url": it["download_url"],
            "path": it.get("path", "")
        }
        for it in items
        if it.get("type") == "file" and it.get("name", "").lower().endswith(".csv") and it["name"] in main_files
    ]
    
    csvs.sort(key=lambda x: x["name"])
    
    return csvs

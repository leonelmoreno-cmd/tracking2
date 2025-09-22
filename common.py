import os
import pandas as pd
import requests
from typing import Dict, List
import streamlit as st

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
    Construye la URL raw del CSV (usa subcarpetas si corresponde).
    """
    COMPETITOR_TO_SUBCATEGORY_MAP = {
        "competitors_history - BL.csv": "sub-categories2/sub_BL.csv",
        "competitors_history - GS.csv": "sub-categories2/sub_GS.csv",
        "competitors_history - IC.csv": "sub-categories2/sub_IC.csv",
        "competitors_history - LGM.csv": "sub-categories2/sub_LGM.csv",
        "competitors_history - QC.csv": "sub-categories2/sub_QC.csv",
        "competitors_history - RIO.csv": "sub-categories2/sub_RIO.csv",
        "competitors_history - UR.csv": "sub-categories2/sub_UR.csv",
        "synthethic3.csv": "sub-categories2/sub_SYN.csv",
    }

    subcategory_file = COMPETITOR_TO_SUBCATEGORY_MAP.get(fname)
    if subcategory_file:
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}/{subcategory_file}"
    else:
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}/{fname}"

    print(f"Generated URL: {url}")
    return url


def fetch_data(url: str) -> pd.DataFrame:
    """Descarga un CSV y lo devuelve como DataFrame."""
    return pd.read_csv(url)


def _github_headers() -> Dict[str, str]:
    """
    Construye headers para la API de GitHub con token si está disponible.
    - Lee el token de st.secrets['GITHUB_TOKEN'] o variable de entorno GITHUB_TOKEN.
    - Añade User-Agent y versión de API recomendados.
    """
    token = st.secrets.get("GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "streamlit-tracking-app",  # recomendado por GitHub
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@st.cache_data(show_spinner=False)
def list_repo_csvs(owner: str, repo: str, path: str, branch: str = "main") -> List[dict]:
    """
    Devuelve la lista de CSVs 'principales' desde la carpeta /data del repo.
    Requiere autenticación si quieres evitar límites bajos o 401/403.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": branch}

    resp = requests.get(url, headers=_github_headers(), params=params, timeout=15)

    # Si hay fallo de auth, levanta un error claro (útil para debug en UI)
    if resp.status_code == 401:
        raise RuntimeError(
            "GitHub 401 Unauthorized: revisa tu GITHUB_TOKEN en .streamlit/secrets.toml "
            "o en la variable de entorno. Asegúrate de pegar el PAT COMPLETO y con permiso 'Contents: Read'."
        )

    resp.raise_for_status()
    items = resp.json()

    main_files = {
        "competitors_history - BL.csv", "competitors_history - GS.csv", "competitors_history - IC.csv",
        "competitors_history - LGM.csv", "competitors_history - QC.csv", "competitors_history - RIO.csv",
        "competitors_history - UR.csv", "synthethic3.csv",
    }

    csvs = [
        {"name": it["name"], "download_url": it["download_url"], "path": it.get("path", "")}
        for it in items
        if it.get("type") == "file"
        and it.get("name", "").lower().endswith(".csv")
        and it["name"] in main_files
    ]
    csvs.sort(key=lambda x: x["name"])
    return csvs

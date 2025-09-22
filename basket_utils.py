import streamlit as st
from typing import Dict
from common import GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH, _raw_url_for, list_repo_csvs

def resolve_active_basket(default_basket: str) -> (str, str, Dict[str, str]):
    """
    Resuelve el basket activo (nombre y URL) basado en session state y query params.
    Devuelve:
        active_basket_name: str
        active_url: str
        name_to_url: Dict[str, str] -> todos los CSV disponibles con sus URLs
    """
    # Listar CSV en repo
    csv_items = list_repo_csvs(GITHUB_OWNER, GITHUB_REPO, GITHUB_PATH, GITHUB_BRANCH)
    name_to_url: Dict[str, str] = {it["name"]: it["download_url"] for it in csv_items}

    # Leer query param si existe
    qp = st.query_params.to_dict() if hasattr(st, "query_params") else {}
    qp_basket = qp.get("basket")
    if isinstance(qp_basket, list):
        qp_basket = qp_basket[0] if qp_basket else None

    # Inicializar session state
    if "basket" not in st.session_state:
        st.session_state["basket"] = qp_basket if qp_basket else default_basket

    active_basket_name = st.session_state["basket"]
    active_url = name_to_url.get(
        active_basket_name,
        _raw_url_for(GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_PATH, active_basket_name)
    )

    return active_basket_name, active_url, name_to_url

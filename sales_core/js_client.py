from __future__ import annotations
import time
import typing as t
import requests
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from .config import JS_BASE_URL

@dataclass
class JSToken:
    key_name: str
    api_key: str

    @classmethod
    def from_secrets(cls) -> "JSToken":
        """
        Prefer Streamlit secrets (when running the UI). Fallback to environment
        variables for headless runs (GitHub Actions).
        """
        key_name = None
        api_key = None

        # 1) Try Streamlit secrets
        try:
            key_name = st.secrets.get("JS_KEY_NAME")
            api_key  = st.secrets.get("JS_API_KEY")
        except Exception:
            # st.secrets not available (e.g., CLI)
            pass

        # 2) Fallback to env vars
        if not key_name:
            key_name = os.getenv("JS_KEY_NAME")
        if not api_key:
            api_key  = os.getenv("JS_API_KEY")

        if not key_name or not api_key:
            raise RuntimeError(
                "Missing Jungle Scout credentials. Provide JS_KEY_NAME and JS_API_KEY "
                "via Streamlit secrets or environment variables."
            )

        return cls(key_name, api_key)

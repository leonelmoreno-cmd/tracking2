@dataclass
class JSToken:
    key_name: str
    api_key: str

    @classmethod
    def from_secrets(cls) -> "JSToken":
        """
        Load Jungle Scout credentials from environment variables (preferred for GitHub Actions)
        or from Streamlit secrets when running locally.
        """
        import os
        import streamlit as st

        key_name = None
        api_key = None

        # 1️⃣ Try Streamlit secrets (for local dev)
        try:
            key_name = st.secrets.get("JS_KEY_NAME")
            api_key  = st.secrets.get("JS_API_KEY")
        except Exception:
            pass  # st.secrets may not exist in GitHub Actions

        # 2️⃣ Fallback: environment variables (for GitHub Actions)
        if not key_name:
            key_name = os.getenv("JS_KEY_NAME")
        if not api_key:
            api_key = os.getenv("JS_API_KEY")

        # 3️⃣ Validate
        if not key_name or not api_key:
            raise RuntimeError(
                "❌ Missing Jungle Scout credentials. "
                "Set JS_KEY_NAME and JS_API_KEY as GitHub Secrets or environment variables."
            )

        return cls(key_name, api_key)

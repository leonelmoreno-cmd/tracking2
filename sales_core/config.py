from __future__ import annotations
import os

# —— Jungle Scout ——
JS_BASE_URL = "https://developer.junglescout.com/api/sales_estimates_query"
JS_KEY_NAME = None  # loaded from st.secrets at runtime
JS_API_KEY = None   # loaded from st.secrets at runtime

# —— Repo paths (respect your current repo) ——
# Your repo already has "Asins" (capital A); we keep that.
# —— Repo paths ——
ASINS_DIR = "Asins/JS"               # TXT files: Asins/JS/<basket_basename>.txt
SALES_DIR = "sales_core/sales"       # (also covers point 3 below)
GITHUB_OWNER = "leonelmoreno-cmd"
GITHUB_REPO  = "tracking2"
GITHUB_BRANCH = "main"

# NOTE: competitor CSV download_url comes from your existing components/common.py logic.
# We don't duplicate it here.

# —— Misc ——
AMAZON_DP_FMT = "https://www.amazon.com/dp/{asin}"
DEFAULT_BASKET = "synthethic3.csv"    # UI default
TIMEZONE = "America/Caracas"

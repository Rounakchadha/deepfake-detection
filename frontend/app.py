"""
app.py — Main Entry Point for the Deepfake Detection Frontend
Multi-page Streamlit application with professional dark UI, sidebar navigation,
API health indicator, and routing to all 5 educational pages.

Run with:
    streamlit run frontend/app.py

Or from the root deepfake/ directory:
    cd /Users/rounakchadha/Desktop/deepfake
    source venv/bin/activate
    streamlit run frontend/app.py
"""

import sys
import os

# ── Ensure project root is on the path so all imports resolve ──
# This is needed when running `streamlit run frontend/app.py` from root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

# ── Page configuration (must be first Streamlit call) ──────────
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com",
        "Report a bug": None,
        "About": "Production-grade Deepfake Detection System with Explainable AI (Grad-CAM)",
    }
)

# ── Import page modules ─────────────────────────────────────────
from frontend.styles import inject_css
from frontend.utils import get_api_status
from frontend.pages import home, detect, how_it_works, metrics, about

# ── Module-level API status cache (TTL = 10s) ──────────────────
# We intentionally use a plain Python dict + time.time() instead of
# @st.cache_data because st.cache_data is session-scoped and accesses
# SessionInfo during the very first pre-render hydration frame —
# which is what causes "Bad message format / Tried to use SessionInfo
# before it was initialized". A plain module-level cache is
# completely outside Streamlit's session system → no popup.
import time as _time
_api_status_cache: dict = {"result": None, "ts": 0.0}

def _cached_api_status() -> dict:
    now = _time.time()
    if now - _api_status_cache["ts"] > 10 or _api_status_cache["result"] is None:
        try:
            _api_status_cache["result"] = get_api_status()
        except Exception:
            _api_status_cache["result"] = {"online": False}
        _api_status_cache["ts"] = now
    return _api_status_cache["result"]

# ── Navigation labels ───────────────────────────────────────────
PAGES = {
    "🏠 Home":            home,
    "🔍 Detect":          detect,
    "📖 How It Works":    how_it_works,
    "📊 Results & Metrics": metrics,
    "ℹ️ About":           about,
}


def main():
    # ── Inject global CSS ─────────────────────────────────────────
    inject_css()

    # ── Sidebar branding ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:1.5rem 0 1rem;">
            <div style="font-size:2.5rem;">🛡️</div>
            <div style="font-weight:800; font-size:1.1rem; color:#E8E8F0; margin-top:0.4rem;">
                Deepfake<br>Detection System
            </div>
            <div style="color:#8B8BA7; font-size:0.78rem; margin-top:0.3rem;">
                EfficientNet-B0 + Grad-CAM XAI
            </div>
        </div>
        <hr style="border-color:rgba(124,58,237,0.25); margin:0.5rem 0 1.25rem;">
        """, unsafe_allow_html=True)

        # ── Navigation ─────────────────────────────────────────────
        st.markdown('<div style="font-size:0.78rem; color:#8B8BA7; letter-spacing:1.2px; text-transform:uppercase; font-weight:700; margin-bottom:0.6rem;">Navigation</div>', unsafe_allow_html=True)

        # Let session state drive page (CTA button on home page sets this)
        if "page" not in st.session_state:
            st.session_state["page"] = "🏠 Home"

        selected = st.radio(
            "Navigate",
            options=list(PAGES.keys()),
            index=list(PAGES.keys()).index(st.session_state["page"]),
            label_visibility="collapsed",
            key="nav_radio",
        )
        # Sync radio selection back to session state
        st.session_state["page"] = selected

        st.markdown("<hr style='border-color:rgba(124,58,237,0.25); margin:1rem 0;'>", unsafe_allow_html=True)

        # ── API Status Indicator ───────────────────────────────────
        st.markdown('<div style="font-size:0.78rem; color:#8B8BA7; letter-spacing:1.2px; text-transform:uppercase; font-weight:700; margin-bottom:0.6rem;">System Status</div>', unsafe_allow_html=True)
        api = _cached_api_status()
        if api["online"]:
            st.markdown("""
            <div class="success-box" style="padding:0.6rem 0.75rem;">
                <span class="status-online"></span>
                <strong style="font-size:0.85rem;">Backend Connected</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="danger-box" style="padding:0.6rem 0.75rem;">
                <span class="status-offline"></span>
                <strong style="font-size:0.85rem;">Backend Offline</strong><br>
                <small style="color:#8B8BA7;">uvicorn backend.api:app --port 8000</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:rgba(124,58,237,0.25); margin:1rem 0;'>", unsafe_allow_html=True)

        # ── Model Info ─────────────────────────────────────────────
        st.markdown("""
        <div class="glass-card" style="padding:0.9rem;">
            <div style="font-size:0.78rem; color:#8B8BA7; letter-spacing:1.2px; text-transform:uppercase; font-weight:700; margin-bottom:0.6rem;">Model Info</div>
            <table style="width:100%; font-size:0.8rem; color:#C4C4D4; border-collapse:collapse;">
                <tr><td style="color:#8B8BA7; padding:0.18rem 0;">Architecture</td><td>EfficientNet-B0</td></tr>
                <tr><td style="color:#8B8BA7; padding:0.18rem 0;">XAI Method</td><td>Grad-CAM</td></tr>
                <tr><td style="color:#8B8BA7; padding:0.18rem 0;">Parameters</td><td>~5.3M</td></tr>
                <tr><td style="color:#8B8BA7; padding:0.18rem 0;">Inference</td><td>~0.3s (CPU)</td></tr>
                <tr><td style="color:#8B8BA7; padding:0.18rem 0;">Platform</td><td>Mac M2 / CPU</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # ── Version Info ───────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center; margin-top:1rem; color:#4B5563; font-size:0.75rem;">
            v1.0 · IEEE Research-Grade<br>
            Built with Streamlit + FastAPI
        </div>
        """, unsafe_allow_html=True)

    # ── Route to Selected Page ──────────────────────────────────
    PAGES[selected].render()


if __name__ == "__main__":
    main()

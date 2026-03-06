"""
styles.py — Global CSS Injection Module
Injects a premium dark-mode UI with violet/indigo accent palette,
glassmorphism panels, custom metric cards, step indicators,
animated badges, and smooth transitions.
"""

GLOBAL_CSS = """
<style>
/* ─────────────────────────────────────────────────────────
   IMPORT FONTS
───────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─────────────────────────────────────────────────────────
   ROOT VARIABLES
───────────────────────────────────────────────────────── */
:root {
    --bg-primary:    #0F0F1A;
    --bg-secondary:  #1A1A2E;
    --bg-glass:      rgba(26, 26, 46, 0.7);
    --accent-1:      #7C3AED;
    --accent-2:      #4F46E5;
    --accent-cyan:   #06B6D4;
    --accent-green:  #10B981;
    --accent-red:    #EF4444;
    --accent-amber:  #F59E0B;
    --text-primary:  #E8E8F0;
    --text-muted:    #8B8BA7;
    --border:        rgba(124, 58, 237, 0.25);
    --radius:        12px;
    --radius-lg:     20px;
}

/* ─────────────────────────────────────────────────────────
   GLOBAL RESET & BODY
───────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* ─────────────────────────────────────────────────────────
   MAIN CONTENT AREA
───────────────────────────────────────────────────────── */
.main .block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px !important;
}

/* ─────────────────────────────────────────────────────────
   HERO SECTION
───────────────────────────────────────────────────────── */
.hero-container {
    text-align: center;
    padding: 4rem 2rem 3rem;
    position: relative;
}

.hero-badge {
    display: inline-block;
    background: rgba(124, 58, 237, 0.15);
    border: 1px solid rgba(124, 58, 237, 0.4);
    color: #A78BFA;
    padding: 6px 16px;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 900;
    line-height: 1.1;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #E8E8F0 0%, #A78BFA 50%, #06B6D4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: var(--text-muted);
    max-width: 700px;
    margin: 0 auto 2.5rem;
    line-height: 1.7;
    font-weight: 400;
}

/* ─────────────────────────────────────────────────────────
   GLASS CARDS
───────────────────────────────────────────────────────── */
.glass-card {
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.75rem;
    margin-bottom: 1.5rem;
    transition: border-color 0.3s ease, transform 0.2s ease;
}

.glass-card:hover {
    border-color: rgba(124, 58, 237, 0.5);
    transform: translateY(-2px);
}

.glass-card h3 {
    color: #A78BFA;
    font-size: 1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 0.75rem;
}

/* ─────────────────────────────────────────────────────────
   STEP INDICATORS
───────────────────────────────────────────────────────── */
.step-indicator {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.5rem;
    background: var(--bg-secondary);
    border-left: 3px solid var(--accent-1);
    border-radius: 0 var(--radius) var(--radius) 0;
    margin-bottom: 0.75rem;
}

.step-number {
    width: 36px;
    height: 36px;
    min-width: 36px;
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.85rem;
    color: white;
}

.step-content strong {
    display: block;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
}

.step-content span {
    font-size: 0.82rem;
    color: var(--text-muted);
}

/* ─────────────────────────────────────────────────────────
   PREDICTION BADGES
───────────────────────────────────────────────────────── */
.prediction-card {
    text-align: center;
    padding: 2rem;
    border-radius: var(--radius-lg);
    margin: 1rem 0;
}

.prediction-fake {
    background: linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(239,68,68,0.05) 100%);
    border: 2px solid rgba(239, 68, 68, 0.5);
}

.prediction-real {
    background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(16,185,129,0.05) 100%);
    border: 2px solid rgba(16, 185, 129, 0.5);
}

.prediction-label {
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: 4px;
}

.prediction-label.fake { color: var(--accent-red); }
.prediction-label.real { color: var(--accent-green); }

.confidence-label {
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 0.5rem;
}

/* ─────────────────────────────────────────────────────────
   METRIC CARDS
───────────────────────────────────────────────────────── */
.metric-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: var(--accent-1);
    box-shadow: 0 0 20px rgba(124, 58, 237, 0.15);
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #A78BFA;
    font-family: 'JetBrains Mono', monospace;
}

.metric-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.25rem;
    font-weight: 600;
}

/* ─────────────────────────────────────────────────────────
   INFO BOXES
───────────────────────────────────────────────────────── */
.info-box {
    background: rgba(6, 182, 212, 0.08);
    border: 1px solid rgba(6, 182, 212, 0.3);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}

.warning-box {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}

.danger-box {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}

.success-box {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}

/* ─────────────────────────────────────────────────────────
   SECTION HEADERS
───────────────────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2rem 0 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
}

.section-header h2 {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0;
}

.section-pill {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 50px;
    background: rgba(124, 58, 237, 0.2);
    color: #A78BFA;
    border: 1px solid rgba(124, 58, 237, 0.4);
}

/* ─────────────────────────────────────────────────────────
   SIDEBAR NAV
───────────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    padding: 0.4rem 0 !important;
    cursor: pointer !important;
}

[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.95rem !important;
}

/* Status badge */
.status-online {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--accent-green);
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}

.status-offline {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--accent-red);
    border-radius: 50%;
    margin-right: 6px;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ─────────────────────────────────────────────────────────
   DIVIDERS
───────────────────────────────────────────────────────── */
.gradient-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-1), var(--accent-cyan), transparent);
    border: none;
    margin: 2rem 0;
    border-radius: 2px;
}

/* ─────────────────────────────────────────────────────────
   TABLE STYLING
───────────────────────────────────────────────────────── */
.research-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 1rem 0;
}

.research-table th {
    background: rgba(124, 58, 237, 0.2);
    color: #A78BFA;
    padding: 0.75rem 1rem;
    text-align: left;
    font-weight: 700;
    letter-spacing: 0.5px;
    border-bottom: 2px solid var(--border);
}

.research-table td {
    padding: 0.65rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: var(--text-primary);
}

.research-table tr:hover td {
    background: rgba(124, 58, 237, 0.06);
}

/* ─────────────────────────────────────────────────────────
   BUTTONS
───────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    padding: 0.6rem 1.5rem !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(124, 58, 237, 0.4) !important;
}

/* ─────────────────────────────────────────────────────────
   HEATMAP TOGGLE CONTAINER
───────────────────────────────────────────────────────── */
.heatmap-toggle-container {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    margin: 1rem 0;
}

/* ─────────────────────────────────────────────────────────
   TIMELINE / CHART AREA
───────────────────────────────────────────────────────── */
.timeline-container {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin: 1rem 0;
}

/* ─────────────────────────────────────────────────────────
   SCROLLBAR
───────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent-1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-2); }
</style>
"""


def inject_css():
    """Call this at the top of every page to inject global styles."""
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def section_header(icon: str, title: str, pill: str = None):
    """Render a styled section header with optional pill badge."""
    import streamlit as st
    pill_html = f'<span class="section-pill">{pill}</span>' if pill else ""
    st.markdown(
        f'<div class="section-header"><span style="font-size:1.4rem">{icon}</span>'
        f'<h2>{title}</h2>{pill_html}</div>',
        unsafe_allow_html=True
    )


def gradient_divider():
    import streamlit as st
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


def metric_card(value: str, label: str):
    return (
        f'<div class="metric-card">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>'
    )

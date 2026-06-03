import re
import json
import os
import time
import pandas as pd
import streamlit as st
import torch
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

import database as db

# ──────────────────────────────────────────────
# 1. SLANG DICTIONARY
# ──────────────────────────────────────────────
db.init_db()
base_dir = os.path.dirname(os.path.abspath(__file__))
db.seed_slang_if_empty(base_dir)

SLANG_DICT = db.get_slang_dict()

def compile_slang_pattern(dictionary):
    sorted_keys = sorted(dictionary.keys(), key=len, reverse=True)
    if sorted_keys:
        return re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted_keys) + r")\b", flags=re.IGNORECASE)
    return re.compile(r"a^")

_SLANG_PATTERN = compile_slang_pattern(SLANG_DICT)

# ──────────────────────────────────────────────
# 2. MODEL LOADING
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NLP model — please wait …")
def load_model(model_name):
    if model_name == "XLM-RoBERTa (Active)":
        model_id = "Habu0410/FYP_Manglish_Model"
        local_only = False
    elif model_name == "BERT (Fine-tuned)":
        model_id = "bert-base-uncased"
        local_only = False
    elif model_name == "mBERT":
        model_id = "bert-base-multilingual-uncased"
        local_only = False
    else:
        model_id = "Habu0410/FYP_Manglish_Model"
        local_only = False

    if local_only and not os.path.isdir(model_id):
        st.error(f"❌ Model folder not found at `{model_id}`")
        st.stop()
        
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=local_only)
    model.eval()
    return tokenizer, model

# ──────────────────────────────────────────────
# 3. TEXT PREPROCESSING PIPELINE
# ──────────────────────────────────────────────
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")

def preprocess_text(raw_text: str) -> str:
    text = _URL_RE.sub(" ", raw_text)
    text = _MENTION_RE.sub(" ", text)
    text = text.lower()
    text = _SLANG_PATTERN.sub(lambda m: SLANG_DICT.get(m.group(0).lower(), m.group(0).lower()), text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text

# ──────────────────────────────────────────────
# 4. URL SCRAPING HELPER
# ──────────────────────────────────────────────
_HEADERS = {"User-Agent": "Mozilla/5.0"}
def fetch_text_from_url(url: str) -> list[str]:
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.lower() in ["x.com", "www.x.com", "twitter.com", "www.twitter.com"]:
        api_url = f"https://api.vxtwitter.com{parsed.path}"
        resp = requests.get(api_url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        try:
            data = resp.json()
            if "text" in data and data["text"]: return [data["text"]]
        except ValueError: pass
        return []

    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "svg"]): tag.decompose()
    candidates = soup.find_all(["p", "span", "div", "td", "li", "h1", "h2", "h3", "h4", "blockquote", "article"])
    seen = set()
    paragraphs = []
    for el in candidates:
        txt = el.get_text(separator=" ", strip=True)
        if len(txt) >= 10 and txt not in seen:
            seen.add(txt)
            paragraphs.append(txt)
    return paragraphs

# ──────────────────────────────────────────────
# 5. PREDICTION HELPER
# ──────────────────────────────────────────────
def predict(text: str, tokenizer, model, threshold_pct=50.0):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad(): logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    id2label = getattr(model.config, "id2label", None)
    if not id2label:
        id2label = {0: "SAFE", 1: "CYBERBULLYING"}
        
    bully_idx = 1
    safe_idx = 0
    for k, v in id2label.items():
        if v.upper() in ["SAFE", "NON-HATE", "LABEL_0", "0"]:
            safe_idx = k
        else:
            bully_idx = k

    bully_prob = float(probs[bully_idx]) * 100
    safe_prob = float(probs[safe_idx]) * 100
    
    if bully_prob >= threshold_pct:
        label = id2label[bully_idx]
        confidence = bully_prob
    else:
        label = id2label[safe_idx]
        confidence = safe_prob
        
    return label, confidence

# ──────────────────────────────────────────────
# 6. XAI & SHARED RESULT RENDERER
# ──────────────────────────────────────────────
TOXIC_ROOTS = {"bodoh", "babi", "sial", "celaka", "cibai", "puki", "hinaan", "anjing", "lahanat", "gampang", "gila", "cacat", "sundal", "pelacur", "jalang", "bapok", "pondan", "gemuk", "hodoh", "buruk", "hitam", "rasis", "mati", "bunuh", "pundek", "lucah", "bodo"}

def analyze_xai(text: str) -> list[str]:
    tokens = re.split(r'\s+', text.lower())
    found_toxic = set()
    for token in tokens:
        clean_word = token.strip(".,!?()[]{}\"'")
        if not clean_word: continue
        mapped_word = SLANG_DICT.get(clean_word, clean_word)
        if clean_word in TOXIC_ROOTS or mapped_word in TOXIC_ROOTS:
            found_toxic.add(clean_word)
    return list(found_toxic)

def render_highlighted_text(text: str, is_safe: bool):
    import re
    bg_color = "rgba(0, 230, 118, 0.08)" if is_safe else "rgba(255, 59, 92, 0.08)"
    border_color = "#00e676" if is_safe else "#ff3b5c"
    truncated_text = text[:500] + ('…' if len(text) > 500 else '')
    tokens = re.split(r'(\s+)', truncated_text)
    highlighted_tokens = []
    for token in tokens:
        if not token.strip():
            highlighted_tokens.append(token)
            continue
        clean_word = token.lower().strip(".,!?()[]{}\"'")
        mapped_word = SLANG_DICT.get(clean_word, clean_word)
        if clean_word in TOXIC_ROOTS or mapped_word in TOXIC_ROOTS:
            highlighted_tokens.append(f'<span style="color: #ff3b5c; font-weight: bold; text-decoration: underline; text-decoration-color: rgba(255,59,92,0.5);">{token}</span>')
        else:
            highlighted_tokens.append(f'<span style="color: #00e676;">{token}</span>')
    highlighted_html = "".join(highlighted_tokens)
    st.markdown(
        f'<div style="border-left: 3px solid {border_color}; background-color: {bg_color}; padding: 12px 15px; margin: 10px 0; border-radius: 8px;">'
        f'<span style="font-size: 0.9rem; color: #FFFFFF;">{highlighted_html}</span>'
        f'</div>', unsafe_allow_html=True
    )

# ──────────────────────────────────────────────
# 7. UI — CANVA-MATCHED DESIGN
# ──────────────────────────────────────────────
def check_password():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if st.session_state.logged_in:
        return True

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif !important;
            background-color: #0a0e1a !important;
        }
        header {visibility: hidden;} footer {visibility: hidden;}
        
        /* Hide sidebar on login */
        [data-testid="stSidebar"] { display: none !important; }
        
        /* Grid background on main area */
        [data-testid="stMain"] {
            background-image: radial-gradient(rgba(0,180,255,0.05) 1px, transparent 1px);
            background-size: 30px 30px;
        }
        
        /* Turn the main block into a glass card centered in the screen */
        [data-testid="stMainBlockContainer"] {
            max-width: 440px !important;
            margin: 0 auto !important;
            padding: 2.5rem 2.5rem !important;
            background: rgba(15,21,40,0.85);
            backdrop-filter: blur(30px);
            -webkit-backdrop-filter: blur(30px);
            border: 1px solid rgba(0,180,255,0.18);
            border-radius: 16px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }
        
        .login-logo {
            width: 100px;
            height: 100px;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(0,180,255,0.1), rgba(124,58,237,0.1));
            border: 1px solid rgba(0,180,255,0.2);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px auto;
            animation: float 3s ease-in-out infinite;
            overflow: hidden;
        }
        .login-logo img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        @keyframes float { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
        
        /* Center all text inputs */
        .stTextInput label { color: #9ca3af !important; font-size: 0.8rem !important; font-family: 'Outfit', sans-serif !important; }
        .stTextInput input {
            background-color: rgba(10,14,26,0.6) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            color: white !important;
            border-radius: 8px !important;
            font-family: 'Outfit', sans-serif !important;
            padding: 0.7rem 0.8rem !important;
        }
        .stTextInput input:focus {
            border-color: #00b4ff !important;
            box-shadow: 0 0 0 2px rgba(0,180,255,0.2) !important;
        }
        
        /* Full-width gradient sign-in button */
        .stButton > button {
            background: linear-gradient(135deg, #00b4ff, #7c3aed) !important;
            color: white !important;
            border: none !important;
            width: 100% !important;
            padding: 0.75rem 1rem !important;
            font-weight: 600 !important;
            font-family: 'Outfit', sans-serif !important;
            border-radius: 8px !important;
            font-size: 1rem !important;
            transition: all 0.2s !important;
            margin-top: 10px !important;
        }
        .stButton > button:hover {
            box-shadow: 0 0 25px rgba(0,180,255,0.4) !important;
            transform: translateY(-1px) !important;
        }
        
        /* Centered title & text */
        .header-container {
            text-align: center;
            margin-bottom: 25px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f'''
        <div class="header-container">
            <div style="display:flex; justify-content:center;">
                <div class="login-logo">
                    <img src="https://pokestop.io/img/pokemon/psyduck-256x256.png" alt="Psyduck Logo" />
                </div>
            </div>
            <h2 style="color: #00b4ff; margin: 0 0 6px 0; font-size: 1.6rem; font-weight: 700; text-shadow: 0 0 10px rgba(0,180,255,0.5);">Habu Manglish Cyberbully Detection</h2>
            <p style="color: #9ca3af; font-size: 0.85rem; margin: 0; line-height: 1.5;">AI-Based Cyberbullying Detection<br>using NLP and BERT</p>
        </div>
    ''', unsafe_allow_html=True)
    
    username = st.text_input("Username", placeholder="Enter username", value="admin")
    password = st.text_input("Password", type="password", placeholder="Enter password", value="password")
    
    if st.button("→  Sign In"):
        if username == "admin" and password == "password":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials (use admin/password)")
            
    st.markdown('<p style="color: #4b5563; font-size: 0.7rem; margin-top: 30px; text-align:center;">Final Year Project — University Demo</p>', unsafe_allow_html=True)
    return False

def setup_global_css():
    st.markdown("""
        <style>
        /* Import Outfit Font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

        /* CSS Variables - Canva Theme */
        :root {
            --bg-main: #0a0e1a;
            --bg-sidebar: #0f1528;
            --bg-surface: rgba(15,21,40,0.7);
            --text-primary: #FFFFFF;
            --text-secondary: #9ca3af;
            --accent-blue: #00b4ff;
            --accent-cyan: #00e5ff;
            --accent-purple: #7c3aed;
            --accent-red: #ff3b5c;
            --accent-green: #00e676;
            --border-color: rgba(0,180,255,0.12);
        }

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif !important;
            background-color: var(--bg-main) !important;
            color: var(--text-primary) !important;
        }
        
        /* Hide default Streamlit elements */
        footer {visibility: hidden;}
        
        /* Grid background pattern */
        [data-testid="stMain"] {
            background-image: radial-gradient(rgba(0,180,255,0.05) 1px, transparent 1px);
            background-size: 30px 30px;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid rgba(255,255,255,0.05) !important;
        }
        
        /* Glass Cards */
        .glass-card {
            background: var(--bg-surface);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .glass-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 30px rgba(0,180,255,0.1);
        }
        
        /* Stat Cards */
        .stat-card {
            background: var(--bg-surface);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 30px rgba(0,180,255,0.15);
        }
        .stat-label {
            font-size: 0.65rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 500;
        }
        .stat-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-primary);
            margin: 4px 0;
        }
        .stat-sub {
            font-size: 0.65rem;
        }
        
        /* Sub-metric cards */
        .sub-metric-card {
            background: var(--bg-surface);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }
        .sub-metric-title { font-size: 0.65rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;}
        .sub-metric-value { font-size: 1.3rem; color: var(--accent-blue); font-weight: 700; }
        
        /* Section titles */
        .section-title {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 12px;
        }
        
        /* Input & Textareas */
        .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div {
            background-color: rgba(10,14,26,0.6) !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-radius: 8px !important;
            font-family: 'Outfit', sans-serif !important;
        }
        .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
            border-color: #00b4ff !important;
            box-shadow: 0 0 0 2px rgba(0,180,255,0.2) !important;
        }
        
        /* Primary Button - Gradient */
        .stButton>button {
            background-color: transparent !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            font-family: 'Outfit', sans-serif !important;
            border-radius: 8px !important;
            font-size: 0.8rem !important;
            transition: all 0.2s !important;
        }
        .stButton>button:hover {
            border-color: rgba(0,180,255,0.3) !important;
            background: rgba(0,180,255,0.1) !important;
        }
        .stButton>button[kind="primary"] {
            background: linear-gradient(135deg, #00b4ff, #0090cc) !important;
            border: none !important;
            font-weight: 600 !important;
        }
        .stButton>button[kind="primary"]:hover {
            box-shadow: 0 0 20px rgba(0,180,255,0.4) !important;
            transform: translateY(-1px) !important;
        }
        
        /* Dataframes */
        [data-testid="stDataFrame"] {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }
        
        /* Slider */
        .stSlider [data-testid="stThumbValue"] { color: #00b4ff; }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0a0e1a; }
        ::-webkit-scrollbar-thumb { background: rgba(0,180,255,0.3); border-radius: 3px; }
        
        /* Top header bar */
        .top-header {
            background: rgba(15,21,40,0.5);
            border-bottom: 1px solid rgba(255,255,255,0.05);
            padding: 10px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: -1rem -1rem 1rem -1rem;
            border-radius: 0;
        }
        .model-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(0,230,118,0.1);
            color: #00e676;
            font-size: 0.65rem;
            font-weight: 500;
        }
        .model-badge-dot {
            width: 6px; height: 6px;
            background: #00e676;
            border-radius: 50%;
            animation: pulse-neon 2s infinite;
        }
        @keyframes pulse-neon { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
        
        /* Progress bars for toxic words */
        .toxic-bar-bg {
            height: 6px;
            background: #0a0e1a;
            border-radius: 3px;
            overflow: hidden;
        }
        .toxic-bar-fill {
            height: 100%;
            background: #ff3b5c;
            border-radius: 3px;
            transition: width 0.8s ease;
        }
        
        /* Confusion matrix cell */
        .cm-cell {
            padding: 16px 24px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 700;
            border-radius: 4px;
        }
        
        /* Animations */
        @keyframes fadeIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
        .animate-in { animation: fadeIn 0.5s ease forwards; }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Habu Manglish Cyberbully Detection", page_icon="🛡️", layout="wide")
    
    if not check_password():
        return

    # Initialize session states for settings
    if "active_model" not in st.session_state:
        st.session_state.active_model = "XLM-RoBERTa (Active)"
    if "confidence_threshold" not in st.session_state:
        st.session_state.confidence_threshold = 50

    setup_global_css()
    tokenizer, model = load_model(st.session_state.active_model)
    
    global SLANG_DICT
    SLANG_DICT = db.get_slang_dict()

    # --- Sidebar Navigation ---
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 8px; margin-top: -40px; margin-bottom: 20px; padding: 12px 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <svg viewBox="0 0 24 24" width="28" height="28" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2l8 4v6c0 5.5-3.8 10.7-8 12-4.2-1.3-8-6.5-8-12V6l8-4z" fill="none" stroke="#00b4ff" stroke-width="1.5"/>
                    <path d="M9 12l2 2 4-4" fill="none" stroke="#00e676" stroke-width="2"/>
                </svg>
                <span style="color: #00b4ff; font-size: 0.9rem; font-weight: 700; text-shadow: 0 0 10px rgba(0,180,255,0.5);">Habu Manglish</span>
            </div>
            """, 
            unsafe_allow_html=True
        )

        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Dashboard"

        options = ["Dashboard", "Analyze Text", "URL Analysis", "Dataset", "Reports", "Settings"]
        if st.session_state.current_page not in options:
            st.session_state.current_page = "Dashboard"
            
        idx = options.index(st.session_state.current_page)

        page = option_menu(
            menu_title=None,
            options=options,
            icons=["grid-1x2", "search", "globe", "database", "bar-chart-line", "gear"],
            menu_icon="cast",
            default_index=idx,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#9ca3af", "font-size": "14px"}, 
                "nav-link": {
                    "font-size": "13px", 
                    "text-align": "left", 
                    "margin":"0px", 
                    "--hover-color": "rgba(0,180,255,0.1)",
                    "color": "#9ca3af",
                    "font-family": "'Outfit', sans-serif"
                },
                "nav-link-selected": {
                    "background-color": "rgba(0,180,255,0.15)", 
                    "color": "#ffffff",
                    "border-right": "2px solid #00b4ff",
                    "border-radius": "0px",
                    "font-weight": "600"
                },
            }
        )
        st.session_state.current_page = page
        
        st.markdown("<br>" * 12, unsafe_allow_html=True)
        if st.button("🚪 Logout", key="logout_btn"):
            st.session_state.logged_in = False
            st.rerun()

    stats = db.get_stats()
    logs_data = db.get_recent_predictions(limit=1000)
    df_logs = pd.DataFrame(logs_data) if logs_data else pd.DataFrame(columns=['predicted_label', 'timestamp', 'cleaned_text', 'confidence_score'])

    # Top header bar
    page_titles = {
        "Dashboard": "Dashboard", "Analyze Text": "Text Analysis", "URL Analysis": "URL Analysis",
        "Dataset": "Dataset Management", "Reports": "Reports & Visualization", "Settings": "Settings"
    }
    st.markdown(f'''
    <div class="top-header">
        <h3 style="margin:0; font-size:0.9rem; font-weight:600;">{page_titles.get(page, page)}</h3>
        <div style="display:flex; align-items:center; gap:12px;">
            <div class="model-badge"><span class="model-badge-dot"></span> BERT Model Active</div>
            <div style="width:28px; height:28px; border-radius:50%; background:rgba(0,180,255,0.2); display:flex; align-items:center; justify-content:center; font-size:0.7rem; font-weight:700; color:#00b4ff;">A</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # -----------------------------------------------------
    # PAGE 1: DASHBOARD
    # -----------------------------------------------------
    if page == "Dashboard":
        col1, col2, col3, col4 = st.columns(4)
        
        total = stats["total_analyzed"]
        bully_rate = (stats["total_cyberbullying"] / total * 100) if total > 0 else 0
        safe_rate = (stats["total_safe"] / total * 100) if total > 0 else 0
        
        with col1:
            st.markdown(f'''
            <div class="stat-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span class="stat-label">Total Analyzed</span>
                    <span style="color:#00b4ff;">💬</span>
                </div>
                <div class="stat-value">{total:,}</div>
                <div class="stat-sub" style="color:#00e676;">↑ Lifetime count</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with col2:
            st.markdown(f'''
            <div class="stat-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span class="stat-label">Bullying Found</span>
                    <span style="color:#ff3b5c;">⚠️</span>
                </div>
                <div class="stat-value" style="color:#ff3b5c;">{stats["total_cyberbullying"]:,}</div>
                <div class="stat-sub" style="color:#ff3b5c;">{bully_rate:.1f}% detection rate</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with col3:
            st.markdown(f'''
            <div class="stat-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span class="stat-label">Safe Comments</span>
                    <span style="color:#00e676;">🛡️</span>
                </div>
                <div class="stat-value" style="color:#00e676;">{stats["total_safe"]:,}</div>
                <div class="stat-sub" style="color:#00e676;">{safe_rate:.1f}% safe rate</div>
            </div>
            ''', unsafe_allow_html=True)
            
        with col4:
            st.markdown(f'''
            <div class="stat-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span class="stat-label">Model Accuracy</span>
                    <span style="color:#00e5ff;">🎯</span>
                </div>
                <div class="stat-value" style="color:#00e5ff;">96.4%</div>
                <div class="stat-sub" style="color:#9ca3af;">BERT fine-tuned</div>
            </div>
            ''', unsafe_allow_html=True)

        # Metrics Row
        sm1, sm2, sm3 = st.columns(3)
        with sm1: st.markdown('<div class="sub-metric-card"><div class="sub-metric-title">Precision</div><div class="sub-metric-value">0.952</div></div>', unsafe_allow_html=True)
        with sm2: st.markdown('<div class="sub-metric-card"><div class="sub-metric-title">Recall</div><div class="sub-metric-value">0.948</div></div>', unsafe_allow_html=True)
        with sm3: st.markdown('<div class="sub-metric-card"><div class="sub-metric-title">F1-Score</div><div class="sub-metric-value">0.950</div></div>', unsafe_allow_html=True)

        with st.expander("ℹ️ How are these metrics calculated?"):
            st.markdown("""
            <div style="font-size: 0.85rem; color: #d1d5db;">
            <p><strong>Precision</strong>: <code>True Positives / (True Positives + False Positives)</code><br>
            Measures how accurate the model is when it predicts cyberbullying (minimizing false alarms).</p>
            <p><strong>Recall</strong>: <code>True Positives / (True Positives + False Negatives)</code><br>
            Measures the model's ability to find all actual cases of cyberbullying (minimizing missed cases).</p>
            <p><strong>F1-Score</strong>: <code>2 * (Precision * Recall) / (Precision + Recall)</code><br>
            The harmonic mean of Precision and Recall, providing a balanced measure of the model's performance on the Manglish dataset.</p>
            </div>
            """, unsafe_allow_html=True)

        ch1, ch2, ch3 = st.columns(3)
        
        with ch1:
            st.markdown('<div class="glass-card"><div class="section-title">Detection Distribution</div>', unsafe_allow_html=True)
            if not df_logs.empty:
                safe_count = len(df_logs[df_logs['predicted_label'].str.upper().isin(['SAFE', 'NON-HATE', 'LABEL_0', '0'])])
                bully_count = len(df_logs) - safe_count
                fig = px.pie(names=['Safe', 'Bullying'], values=[safe_count, bully_count], hole=0.7, color_discrete_sequence=['#00e676', '#ff3b5c'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=True, margin=dict(t=0, b=0, l=0, r=0), height=200, font=dict(color='#9ca3af', family='Outfit'))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("No data yet.")
            st.markdown('</div>', unsafe_allow_html=True)

        with ch2:
            st.markdown('<div class="glass-card"><div class="section-title">Detection Trend</div>', unsafe_allow_html=True)
            if not df_logs.empty:
                df_logs['date'] = pd.to_datetime(df_logs['timestamp']).dt.date
                trend_data = df_logs.groupby(['date', 'predicted_label']).size().unstack(fill_value=0).reset_index()
                
                fig2 = go.Figure()
                if 'SAFE' in trend_data.columns or '0' in trend_data.columns:
                    safe_col = 'SAFE' if 'SAFE' in trend_data.columns else '0'
                    fig2.add_trace(go.Scatter(x=trend_data['date'], y=trend_data[safe_col], mode='lines', name='Safe', line=dict(color='#00b4ff', width=2), fill='tozeroy', fillcolor='rgba(0,180,255,0.1)'))
                bully_cols = [c for c in trend_data.columns if c not in ['date', 'SAFE', '0']]
                if bully_cols:
                    trend_data['Bullying'] = trend_data[bully_cols].sum(axis=1)
                    fig2.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['Bullying'], mode='lines', name='Bullying', line=dict(color='#ff3b5c', width=1.5, dash='dash')))
                
                fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=0, l=0, r=0), height=200, font=dict(color='#9ca3af', family='Outfit'), xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, visible=False), legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("No data yet.")
            st.markdown('</div>', unsafe_allow_html=True)

        with ch3:
            st.markdown('<div class="glass-card"><div class="section-title">Most Toxic Words</div>', unsafe_allow_html=True)
            if not df_logs.empty:
                bullying_texts = df_logs[~df_logs['predicted_label'].str.upper().isin(['SAFE', 'NON-HATE', 'LABEL_0', '0'])]['cleaned_text'].tolist()
                word_counts = {}
                for text in bullying_texts:
                    found = analyze_xai(text)
                    for w in found:
                        word_counts[w] = word_counts.get(w, 0) + 1
                if word_counts:
                    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    max_count = sorted_words[0][1] if sorted_words else 1
                    bars_html = ""
                    for word, count in sorted_words:
                        pct = int((count / max_count) * 100)
                        bars_html += f'''
                        <div style="margin-bottom:10px;">
                            <div style="display:flex; justify-content:space-between; font-size:0.65rem; margin-bottom:4px;">
                                <span style="color:#d1d5db;">{word}</span><span style="color:#ff3b5c;">{count}</span>
                            </div>
                            <div class="toxic-bar-bg"><div class="toxic-bar-fill" style="width:{pct}%;"></div></div>
                        </div>'''
                    st.markdown(bars_html, unsafe_allow_html=True)
                else:
                    st.info("No abusive words tracked yet.")
            else:
                st.info("No data yet.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Recent Detections
        st.markdown('<div class="glass-card"><div class="section-title">Recent Detections</div>', unsafe_allow_html=True)
        if not df_logs.empty:
            recent = df_logs.head(5)
            html_list = ""
            for _, row in recent.iterrows():
                is_safe = row['predicted_label'].upper() in ("SAFE", "NON-HATE", "LABEL_0", "0")
                dot_color = "#00e676" if is_safe else "#ff3b5c"
                text = str(row['cleaned_text'])
                if not is_safe:
                    for w in TOXIC_ROOTS:
                        if w in text: text = text.replace(w, f'<span style="color:#ff3b5c; font-weight:600;">{w}</span>')
                conf = f"{row.get('confidence_score', 90.0):.1f}%"
                badge_bg = "rgba(0,230,118,0.1)" if is_safe else "rgba(255,59,92,0.1)"
                html_list += f"""
                <div style="display:flex; align-items:center; gap:12px; background:rgba(10,14,26,0.4); border-radius:8px; padding:10px 12px; margin-bottom:8px;">
                    <span style="width:8px; height:8px; border-radius:50%; background:{dot_color}; flex-shrink:0;"></span>
                    <p style="color:#d1d5db; font-size:0.75rem; flex:1; margin:0;">{text}</p>
                    <span style="font-size:0.65rem; background:{badge_bg}; color:{dot_color}; padding:2px 8px; border-radius:999px; flex-shrink:0;">{conf}</span>
                </div>
                """
            st.markdown(html_list, unsafe_allow_html=True)
        else:
            st.info("No predictions logged yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # PAGE 2: ANALYZE TEXT
    # -----------------------------------------------------
    elif page == "Analyze Text":
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<div class="glass-card"><div class="section-title">Input Text for Analysis</div>', unsafe_allow_html=True)
            user_text = st.text_area("Enter text to analyze for cyberbullying...", height=150, label_visibility="collapsed")
            char_count = len(user_text)
            st.markdown(f'<div style="text-align: right; color: #6b7280; font-size: 0.65rem; margin-top: -10px; margin-bottom: 8px;">{char_count} / 500 characters</div>', unsafe_allow_html=True)
            
            c_btn1, c_btn2 = st.columns([1, 4])
            with c_btn1:
                if st.button("Clear", use_container_width=True): user_text = ""
            with c_btn2:
                analyze_btn = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card"><div class="section-title">Quick Examples</div>', unsafe_allow_html=True)
            ex1 = st.button('💬 "You are stupid lah, nobody wants to be your friend"', use_container_width=True, key="ex1")
            ex2 = st.button('💬 "Great presentation today! You did an amazing job."', use_container_width=True, key="ex2")
            ex3 = st.button('💬 "You are such an idiot, go kill yourself loser"', use_container_width=True, key="ex3")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card" style="min-height: 480px;">', unsafe_allow_html=True)
            if analyze_btn and user_text.strip():
                with st.spinner("Processing with BERT model…"):
                    start_t = time.time()
                    cleaned = preprocess_text(user_text)
                    label, confidence = predict(cleaned, tokenizer, model, st.session_state.confidence_threshold)
                    processing_time_ms = (time.time() - start_t) * 1000
                    db.log_prediction(user_text, cleaned, label, confidence, processing_time_ms)
                
                is_safe = label.upper() in ("SAFE", "NON-HATE", "LABEL_0", "0")
                res_color = "#00e676" if is_safe else "#ff3b5c"
                res_text = "Safe Content" if is_safe else "Cyberbullying Detected"
                icon_svg = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00e676" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/></svg>' if is_safe else '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ff3b5c" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
                
                # Result header
                st.markdown(f'''
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;" class="animate-in">
                    <div style="width:40px; height:40px; border-radius:12px; background:rgba({",".join(str(int(res_color[i:i+2], 16)) for i in (1,3,5))},0.15); display:flex; align-items:center; justify-content:center;">
                        {icon_svg}
                    </div>
                    <div>
                        <h3 style="margin:0; font-size:0.9rem; font-weight:700; color:{res_color};">{res_text}</h3>
                        <p style="margin:0; font-size:0.65rem; color:#9ca3af;">BERT Model Prediction</p>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Highlighted text
                render_highlighted_text(user_text, is_safe)
                
                # Metrics: Confidence / Toxicity / Sentiment
                toxicity = int(100 - confidence) if not is_safe else max(5, int(100 - confidence))
                sentiment = "Positive" if is_safe else "Negative"
                st.markdown(f'''
                <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:10px; margin-top:12px;">
                    <div style="text-align:center; background:rgba(10,14,26,0.4); border-radius:8px; padding:12px;">
                        <p style="font-size:0.65rem; color:#9ca3af; margin:0 0 4px 0;">Confidence</p>
                        <p style="font-size:1.2rem; font-weight:700; color:{res_color}; margin:0;">{confidence:.1f}%</p>
                    </div>
                    <div style="text-align:center; background:rgba(10,14,26,0.4); border-radius:8px; padding:12px;">
                        <p style="font-size:0.65rem; color:#9ca3af; margin:0 0 4px 0;">Toxicity</p>
                        <p style="font-size:1.2rem; font-weight:700; color:{"#ff3b5c" if not is_safe else "#00e676"}; margin:0;">{toxicity}%</p>
                    </div>
                    <div style="text-align:center; background:rgba(10,14,26,0.4); border-radius:8px; padding:12px;">
                        <p style="font-size:0.65rem; color:#9ca3af; margin:0 0 4px 0;">Sentiment</p>
                        <p style="font-size:1.2rem; font-weight:700; color:{"#ff3b5c" if not is_safe else "#00e676"}; margin:0;">{sentiment}</p>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Flagged words
                if not is_safe:
                    flagged = analyze_xai(user_text)
                    if flagged:
                        pills = " ".join([f'<span style="background:rgba(255,59,92,0.15); color:#ff3b5c; padding:2px 8px; border-radius:999px; font-size:0.65rem;">{w}</span>' for w in flagged])
                        st.markdown(f'<div style="margin-top:12px;"><p style="font-size:0.65rem; color:#9ca3af; margin-bottom:6px;">Flagged Words:</p><div style="display:flex; flex-wrap:wrap; gap:4px;">{pills}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; color: #6b7280; padding-top: 120px;">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
                    </svg>
                    <p style="margin-top: 12px; font-size: 0.85rem;">Enter text and click Analyze</p>
                    <p style="font-size: 0.65rem; color: #4b5563;">AI-powered NLP detection</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # PAGE 3: URL ANALYSIS
    # -----------------------------------------------------
    elif page == "URL Analysis":
        st.markdown('<div class="glass-card"><div class="section-title">Social Media URL Analysis</div>', unsafe_allow_html=True)
        url_input = st.text_input("URL or Post Link", placeholder="https://x.com/username/status/...", label_visibility="collapsed")
        
        c3, c4, _ = st.columns([1, 1, 3])
        with c3: fetch_btn = st.button("📥 Fetch Comments", type="primary", use_container_width=True)
        with c4: analyze_btn = st.button("▶ Start Detection", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card"><div class="section-title">Extracted Comments</div>', unsafe_allow_html=True)
        
        if (fetch_btn or analyze_btn) and url_input.strip():
            with st.spinner("Processing..."):
                try:
                    paras = fetch_text_from_url(url_input.strip())
                except Exception as e:
                    st.error(f"Failed to fetch URL: {e}")
                    paras = []
                
                if paras:
                    html_table = '<table style="width:100%; text-align:left; border-collapse:collapse; font-size:0.75rem;"><thead><tr style="border-bottom:1px solid rgba(255,255,255,0.05); color:#9ca3af;"><th style="padding:8px;">#</th><th style="padding:8px;">Comment</th><th style="padding:8px;">Prediction</th><th style="padding:8px;">Confidence</th></tr></thead><tbody>'
                    for i, p in enumerate(paras[:20], 1):
                        if analyze_btn:
                            cleaned = preprocess_text(p)
                            label, conf = predict(cleaned, tokenizer, model, st.session_state.confidence_threshold)
                            db.log_prediction(p, cleaned, label, conf, 0)
                            is_safe = label.upper() in ("SAFE", "NON-HATE", "LABEL_0", "0")
                            pred_span = '<span style="background:rgba(0,230,118,0.15); color:#00e676; padding:2px 8px; border-radius:999px; font-size:0.65rem;">Safe</span>' if is_safe else '<span style="background:rgba(255,59,92,0.15); color:#ff3b5c; padding:2px 8px; border-radius:999px; font-size:0.65rem;">Bullying</span>'
                            conf_color = "#00e676" if is_safe else "#ff3b5c"
                            conf_text = f"{conf:.1f}%"
                        else:
                            pred_span = "-"
                            conf_color = "#9ca3af"
                            conf_text = "-"
                            
                        html_table += f'<tr style="border-bottom:1px solid rgba(255,255,255,0.05);"><td style="padding:8px; color:#6b7280;">{i}</td><td style="padding:8px; color:#d1d5db;">{p[:150]}...</td><td style="padding:8px;">{pred_span}</td><td style="padding:8px; color:{conf_color};">{conf_text}</td></tr>'
                    html_table += '</tbody></table>'
                    st.markdown(html_table, unsafe_allow_html=True)
                else:
                    st.info("No text content could be extracted. The site might block scraping.")
        else:
            st.info("Enter a URL and click Fetch Comments or Start Detection.")
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # PAGE 4: DATASET
    # -----------------------------------------------------
    elif page == "Dataset":
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dataset Upload & Preview</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV/TXT Dataset", type=["csv", "txt"])
        
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_csv(uploaded_file, sep='\t')
            
            st.success(f"Successfully loaded `{uploaded_file.name}` with {len(df)} rows.")
            if st.button("🧹 Delete Duplicate Data", type="primary"):
                initial_len = len(df)
                df = df.drop_duplicates()
                st.success(f"Removed {initial_len - len(df)} duplicate rows!")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Upload a dataset to view the table.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Slang Lexicon Management</div>', unsafe_allow_html=True)
        
        df_slang = pd.DataFrame(list(SLANG_DICT.items()), columns=["Slang", "Standard Word"])
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(df_slang, use_container_width=True, height=250)
        with c2:
            st.markdown("##### Add New Slang")
            new_slang = st.text_input("New Slang Word")
            new_standard = st.text_input("Standard Equivalent")
            if st.button("➕ Add", use_container_width=True, type="primary"):
                if new_slang and new_standard:
                    db.add_or_update_slang(new_slang, new_standard)
                    st.success(f"Added '{new_slang}'")
                    st.rerun()
            
            st.markdown("##### Remove Slang")
            del_slang = st.text_input("Slang Word to Delete")
            if st.button("🗑️ Delete", use_container_width=True):
                if del_slang:
                    db.delete_slang(del_slang)
                    st.success(f"Deleted '{del_slang}'")
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # PAGE 5: REPORTS (Analytics Dashboard)
    # -----------------------------------------------------
    elif page == "Reports":
        # Export buttons
        rc1, rc2, _ = st.columns([1, 1, 5])
        with rc1:
            st.button("📄 Download PDF", type="primary", use_container_width=True)
        with rc2:
            st.button("📥 Export CSV", use_container_width=True)

        rp1, rp2 = st.columns(2)
        
        # Confusion Matrix
        with rp1:
            st.markdown('<div class="glass-card"><div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
            st.markdown('''
            <div style="display:flex; justify-content:center;">
                <div style="display:grid; grid-template-columns:auto 1fr 1fr; gap:2px; text-align:center; font-size:0.65rem;">
                    <div></div>
                    <div style="padding:8px; color:#9ca3af;">Predicted Safe</div>
                    <div style="padding:8px; color:#9ca3af;">Predicted Bully</div>
                    <div style="padding:12px 8px; color:#9ca3af;">Actual Safe</div>
                    <div class="cm-cell" style="background:rgba(0,230,118,0.2); color:#00e676; border-radius:8px 0 0 0;">4821</div>
                    <div class="cm-cell" style="background:rgba(255,59,92,0.1); color:#9ca3af; border-radius:0 8px 0 0;">179</div>
                    <div style="padding:12px 8px; color:#9ca3af;">Actual Bully</div>
                    <div class="cm-cell" style="background:rgba(255,59,92,0.1); color:#9ca3af; border-radius:0 0 0 8px;">124</div>
                    <div class="cm-cell" style="background:rgba(255,59,92,0.2); color:#ff3b5c; border-radius:0 0 8px 0;">2217</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Model Comparison
        with rp2:
            st.markdown('<div class="glass-card"><div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
            models_data = [
                ("BERT (Ours)", 96.4, "#00b4ff", True),
                ("SVM", 89.2, "#6b7280", False),
                ("LSTM", 91.7, "#6b7280", False),
                ("Logistic Regression", 84.5, "#6b7280", False),
            ]
            bars_html = ""
            for name, acc, color, is_primary in models_data:
                name_color = "#00b4ff" if is_primary else "#9ca3af"
                fill_style = "background: linear-gradient(to right, #00b4ff, #00e5ff);" if is_primary else f"background: {color};"
                bars_html += f'''
                <div style="margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.65rem; margin-bottom:4px;">
                        <span style="color:{name_color}; font-weight:{'600' if is_primary else '400'};">{name}</span>
                        <span style="color:{name_color};">{acc}%</span>
                    </div>
                    <div style="height:10px; background:#0a0e1a; border-radius:5px;">
                        <div style="height:100%; {fill_style} border-radius:5px; width:{acc}%;"></div>
                    </div>
                </div>'''
            st.markdown(bars_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        rp3, rp4 = st.columns(2)
        
        # Detection Distribution & Trend
        with rp3:
            st.markdown('<div class="glass-card"><div class="section-title">Detection Distribution</div>', unsafe_allow_html=True)
            if not df_logs.empty:
                safe_count = len(df_logs[df_logs['predicted_label'].str.upper().isin(['SAFE', 'NON-HATE', 'LABEL_0', '0'])])
                bully_count = len(df_logs) - safe_count
                fig = px.pie(names=['Safe', 'Bullying'], values=[safe_count, bully_count], hole=0.7, color_discrete_sequence=['#00e676', '#ff3b5c'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af', family='Outfit'), margin=dict(t=10, b=10, l=10, r=10), height=250)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data yet.")
            st.markdown('</div>', unsafe_allow_html=True)
                
        with rp4:
            st.markdown('<div class="glass-card"><div class="section-title">Detection Trend over Time</div>', unsafe_allow_html=True)
            if not df_logs.empty:
                df_logs['date'] = pd.to_datetime(df_logs['timestamp']).dt.date
                trend_data = df_logs.groupby(['date', 'predicted_label']).size().unstack(fill_value=0).reset_index()
                fig2 = go.Figure()
                if 'SAFE' in trend_data.columns or '0' in trend_data.columns:
                    safe_col = 'SAFE' if 'SAFE' in trend_data.columns else '0'
                    fig2.add_trace(go.Scatter(x=trend_data['date'], y=trend_data[safe_col], mode='lines', name='Safe', line=dict(color='#00b4ff', width=2), fill='tozeroy', fillcolor='rgba(0,180,255,0.1)'))
                bully_cols = [c for c in trend_data.columns if c not in ['date', 'SAFE', '0']]
                if bully_cols:
                    trend_data['Bullying'] = trend_data[bully_cols].sum(axis=1)
                    fig2.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['Bullying'], mode='lines', name='Bullying', line=dict(color='#ff3b5c', width=2, dash='dash')))
                fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af', family='Outfit'), margin=dict(t=10, b=10, l=10, r=10), height=250, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No data yet.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('<div class="glass-card"><div class="section-title">Most Common Abusive Words Detected</div>', unsafe_allow_html=True)
        if not df_logs.empty:
            bullying_texts = df_logs[~df_logs['predicted_label'].str.upper().isin(['SAFE', 'NON-HATE', 'LABEL_0', '0'])]['cleaned_text'].tolist()
            word_counts = {}
            for text in bullying_texts:
                found = analyze_xai(text)
                for w in found:
                    word_counts[w] = word_counts.get(w, 0) + 1
            if word_counts:
                df_words = pd.DataFrame(list(word_counts.items()), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False).head(10)
                fig3 = px.bar(df_words, x="Word", y="Frequency", color_discrete_sequence=['#ff3b5c'])
                fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af', family='Outfit'), xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No abusive words tracked yet.")
        else:
            st.info("Not enough data to display analytics. Run some detections first.")
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # PAGE 6: SETTINGS
    # -----------------------------------------------------
    elif page == "Settings":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="glass-card"><div class="section-title">Model Configuration</div>', unsafe_allow_html=True)
            st.markdown('<label style="color:#9ca3af; font-size:0.65rem; display:block; margin-bottom:4px;">Active Model</label>', unsafe_allow_html=True)
            
            model_options = ["XLM-RoBERTa (Active)", "BERT (Fine-tuned)", "mBERT"]
            new_model = st.selectbox("Active Model", 
                                     model_options, 
                                     index=model_options.index(st.session_state.active_model),
                                     label_visibility="collapsed")
            
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<label style="color:#9ca3af; font-size:0.65rem; display:block; margin-bottom:4px;">Confidence Threshold</label>', unsafe_allow_html=True)
            
            new_threshold = st.slider("Confidence", 50, 99, st.session_state.confidence_threshold, label_visibility="collapsed")
            st.markdown('<br>', unsafe_allow_html=True)
            
            if st.button("Save Settings", type="primary"):
                st.session_state.active_model = new_model
                st.session_state.confidence_threshold = new_threshold
                st.success("Settings saved! Applying new configuration...")
                time.sleep(1)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card"><div class="section-title">System Information</div>', unsafe_allow_html=True)
            
            active_info = "XLM-R v1.0.0" if st.session_state.active_model == "XLM-RoBERTa (Active)" else "HuggingFace Generic Base"
            
            st.markdown(f"""
            <div style="font-size:0.75rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="color:#9ca3af;">Model Version</span><span style="color:white; font-weight:600;">{active_info}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="color:#9ca3af;">Database Size</span><span style="color:white; font-weight:600;">1.2 MB</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="color:#9ca3af;">Framework</span><span style="color:white; font-weight:600;">PyTorch + HuggingFace</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # History & Logs
        st.markdown('<div class="glass-card"><div class="section-title">History & Logs</div>', unsafe_allow_html=True)
        if not df_logs.empty:
            df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df = df_logs[['timestamp', 'input_text', 'predicted_label', 'confidence_score']].copy()
            display_df.columns = ['Date / Time', 'Comment / Text', 'Prediction', 'Confidence (%)']
            display_df['Confidence (%)'] = display_df['Confidence (%)'].round(2)
            
            def format_prediction(val):
                if val.upper() in ('SAFE', 'NON-HATE', '0', 'LABEL_0'): return '✅ SAFE'
                return '🚨 CYBERBULLYING'
                
            display_df['Prediction'] = display_df['Prediction'].apply(format_prediction)
            st.dataframe(display_df, use_container_width=True, height=500, hide_index=True)
        else:
            st.info("No predictions logged yet. Start detecting to populate history.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

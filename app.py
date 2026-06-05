import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
import base64
from io import BytesIO

try:
    from rdkit.Chem import Draw
    DRAW_AVAILABLE = True
except ImportError:
    DRAW_AVAILABLE = False

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="NeuroBarrier-AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700;900&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #080e1a !important;
    color: #e2eaf5 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background-image:
        radial-gradient(ellipse 70% 50% at 50% -10%, rgba(99,179,237,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 40% 30% at 90% 90%, rgba(129,140,248,0.07) 0%, transparent 60%),
        linear-gradient(180deg, #080e1a 0%, #0b1423 100%);
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

.block-container {
    max-width: 1080px !important;
    padding: 1.5rem 2.5rem 4rem !important;
    margin: 0 auto;
}

/* ── HERO ── */
.hero-wrap {
    text-align: center;
    padding: 2.5rem 0 1.8rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    color: #63b3ed;
    border: 1px solid rgba(99,179,237,0.35);
    padding: 0.35rem 1.1rem;
    border-radius: 999px;
    margin-bottom: 1.4rem;
    background: rgba(99,179,237,0.07);
    text-transform: uppercase;
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-weight: 900;
    font-size: 3.2rem;
    letter-spacing: 0.04em;
    line-height: 1.05;
    margin: 0 0 0.6rem;
    color: #ffffff;
    text-shadow: 0 0 40px rgba(99,179,237,0.35), 0 0 80px rgba(129,140,248,0.2);
}
.hero-title span { color: #63b3ed; }
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #4a6d8c;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── DIVIDER ── */
.mol-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(99,179,237,0.3) 30%, rgba(129,140,248,0.25) 70%, transparent 100%);
    margin: 1.8rem 0;
}

/* ── SECTION LABEL ── */
.sec-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    color: #63b3ed;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(99,179,237,0.15);
}

/* ── WIDGET OVERRIDES ── */
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #8ab4d4 !important;
    letter-spacing: 0.03em !important;
}

[data-testid="stSelectbox"] > div > div {
    background: rgba(99,179,237,0.05) !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    border-radius: 10px !important;
    color: #e2eaf5 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.83rem !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(99,179,237,0.45) !important;
}

[data-testid="stTextInput"] > div > div > input {
    background: #f0f6ff !important;
    border: 1px solid rgba(99,179,237,0.35) !important;
    border-radius: 10px !important;
    color: #0f1f33 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.83rem !important;
    padding: 0.6rem 1rem !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: rgba(99,179,237,0.55) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.1) !important;
}
[data-testid="stTextInput"] > div > div > input::placeholder {
    color: #8aabcc !important;
}

/* ── PREDICT BUTTON ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.12em !important;
    cursor: pointer !important;
    margin-top: 0.8rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 28px rgba(99,102,241,0.45) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── MOLECULE IMAGE ── */
.mol-img-wrap {
    display: flex;
    justify-content: center;
    margin: 0.5rem 0 0.8rem;
}
.mol-img-wrap img {
    border-radius: 14px;
    border: 1px solid rgba(99,179,237,0.2);
    background: rgba(255,255,255,0.85);
    padding: 8px;
    width: 100%;
    max-width: 320px;
}
.smiles-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #63b3ed;
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 8px;
    padding: 0.5rem 0.9rem;
    margin-top: 0.4rem;
    word-break: break-all;
    line-height: 1.5;
}
.mol-placeholder {
    height: 190px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: rgba(99,179,237,0.03);
    border: 1px dashed rgba(99,179,237,0.12);
    border-radius: 14px;
    gap: 0.5rem;
}
.mol-placeholder-icon { font-size: 2rem; opacity: 0.25; }
.mol-placeholder-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #1e3a52;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* ── RESULT CARDS ── */
.result-section-title {
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.08em;
    color: #e2eaf5;
    margin: 0.5rem 0 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.result-section-title::before {
    content: '';
    display: inline-block;
    width: 4px;
    height: 1.1em;
    background: linear-gradient(180deg, #63b3ed, #6366f1);
    border-radius: 4px;
}

.result-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.1rem;
    margin-bottom: 1.2rem;
}

.rcard {
    border-radius: 16px;
    padding: 1.5rem 1.6rem 1.3rem;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.rcard-glow {
    position: absolute;
    top: -30px; right: -30px;
    width: 100px; height: 100px;
    border-radius: 50%;
    filter: blur(35px);
    pointer-events: none;
    opacity: 0.3;
}
.rcard.bbb-yes {
    background: rgba(16,185,129,0.08);
    border-color: rgba(16,185,129,0.35);
}
.rcard.bbb-yes .rcard-glow { background: #10b981; }
.rcard.bbb-no {
    background: rgba(245,101,60,0.08);
    border-color: rgba(245,101,60,0.35);
}
.rcard.bbb-no .rcard-glow { background: #f5653c; }
.rcard.sol {
    background: rgba(99,179,237,0.07);
    border-color: rgba(99,179,237,0.3);
}
.rcard.sol .rcard-glow { background: #63b3ed; }

.rcard-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #8ab4d4;
    margin-bottom: 0.7rem;
}
.rcard-icon { font-size: 2rem; margin-bottom: 0.4rem; line-height: 1; }
.rcard-value {
    font-family: 'Orbitron', monospace;
    font-weight: 900;
    font-size: 1.75rem;
    letter-spacing: 0.04em;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.rcard-value.green  { color: #4ade80; }
.rcard-value.orange { color: #fb923c; }
.rcard-value.blue   { color: #63b3ed; }
.rcard-value.teal   { color: #2dd4bf; }
.rcard-value.amber  { color: #fbbf24; }
.rcard-value.red    { color: #f87171; }

.rcard-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    font-weight: 400;
    color: #6a8fad;
    margin-top: 0.1rem;
}

/* Confidence bar */
.conf-bar-outer {
    margin-top: 0.9rem;
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    height: 6px;
    overflow: hidden;
}
.conf-bar-inner {
    height: 100%;
    border-radius: 99px;
}

/* ── STAT ROW ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
    margin-top: 0.2rem;
}
.stat-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    text-align: center;
}
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    color: #4a6d8c;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.stat-value {
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    font-size: 1.05rem;
    color: #e2eaf5;
    letter-spacing: 0.06em;
}

/* ── STREAMLIT METRIC HIDE DEFAULT ── */
[data-testid="stMetric"] { display: none; }

/* ── ALERTS ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}

/* ── FOOTER ── */
.mol-footer {
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: #1e3a52;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding-top: 0.5rem;
    line-height: 2;
}
</style>
""", unsafe_allow_html=True)

# ── Load Models ──
@st.cache_resource
def load_models():
    bbb = joblib.load("bbb_model.pkl")
    esol = joblib.load("esol_model.pkl")
    return bbb, esol

bbb_model, esol_model = load_models()

# ── Sample SMILES ──
sample_smiles_list = [
    "CCOCCN", "C1=CC=CC=C1", "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CCN(CC)CC", "CCOC(=O)C1=CC=CC=C1", "CNC", "CC(C)O",
    "C1CCCCC1", "CC(C)CC1=CC=CC=C1", "COC1=CC=CC=C1",
    "CC(C)C(=O)O", "CCC(=O)O", "CCN", "C1=CN=CN1", "CCO",
    "C=CCO", "CC(C)OCC", "CCOC", "CNC(=O)C", "CC(C)N"
]

# ── Feature helpers ──
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((2048,))
    ConvertToNumpyArray(fp, arr)
    return arr, mol

def mol_to_b64(mol, size=(320, 200)):
    if not DRAW_AVAILABLE:
        return None
    img = Draw.MolToImage(mol, size=size, kekulize=True)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ════════════════════════════
#  HERO
# ════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">🧠 ADMET &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; RDKit &nbsp;·&nbsp; v2.0</div>
    <div class="hero-title">🧠 <span>NeuroBarrier</span>-AI</div>
    <div class="hero-sub">Blood–Brain Barrier &amp; Solubility Prediction Engine</div>
</div>
<hr class="mol-divider">
""", unsafe_allow_html=True)

# ════════════════════════════
#  TWO-COLUMN LAYOUT
# ════════════════════════════
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown('<div class="sec-label">Input Molecule</div>', unsafe_allow_html=True)

    option = st.selectbox(
        "Select from sample library",
        ["— Choose a molecule —"] + sample_smiles_list
    )
    smiles_input = st.text_input(
        "Or type a custom SMILES",
        placeholder="e.g.  CC(=O)OC1=CC=CC=C1C(=O)O"
    )

    final_smiles = smiles_input.strip() if smiles_input.strip() else (
        option if option != "— Choose a molecule —" else None
    )

    predict_btn = st.button("⚡  RUN PREDICTION")

with right:
    st.markdown('<div class="sec-label">Molecule Preview</div>', unsafe_allow_html=True)
    if final_smiles:
        mol_chk = Chem.MolFromSmiles(final_smiles)
        if mol_chk:
            b64 = mol_to_b64(mol_chk)
            if b64:
                st.markdown(
                    f'<div class="mol-img-wrap"><img src="data:image/png;base64,{b64}"/></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown('<div class="mol-placeholder"><div class="mol-placeholder-icon">🔬</div><div class="mol-placeholder-text">Molecule loaded (preview unavailable)</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="smiles-pill">📌 {final_smiles}</div>', unsafe_allow_html=True)
        else:
            st.error("⚠️ Invalid SMILES string — cannot render molecule.")
    else:
        st.markdown("""
        <div class="mol-placeholder">
            <div class="mol-placeholder-icon">🔬</div>
            <div class="mol-placeholder-text">Awaiting SMILES input</div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════
#  RESULTS
# ════════════════════════════
st.markdown('<hr class="mol-divider">', unsafe_allow_html=True)

if predict_btn:
    if not final_smiles:
        st.warning("⚠️ Please select or enter a SMILES string before running.")
    else:
        with st.spinner("Generating molecular fingerprint & running models…"):
            fp, mol = smiles_to_fp(final_smiles)

        if fp is None:
            st.error("❌ Invalid SMILES — could not parse molecule. Please check your input.")
        else:
            fp_arr = np.array(fp).reshape(1, -1)

            bbb_pred  = bbb_model.predict(fp_arr)[0]
            bbb_prob  = bbb_model.predict_proba(fp_arr)[0][1]
            sol_pred  = esol_model.predict(fp_arr)[0]

            bbb_pct   = int(bbb_prob * 100)
            bbb_yes   = (bbb_pred == 1)

            # Solubility tier
            if sol_pred > -1:
                sol_label = "Highly Soluble";       sol_cls = "teal"
            elif sol_pred > -3:
                sol_label = "Moderately Soluble";   sol_cls = "blue"
            elif sol_pred > -5:
                sol_label = "Slightly Soluble";     sol_cls = "amber"
            else:
                sol_label = "Poorly Soluble";       sol_cls = "red"

            bbb_card_cls   = "bbb-yes" if bbb_yes else "bbb-no"
            bbb_val_cls    = "green"   if bbb_yes else "orange"
            bbb_text       = "PENETRATES BBB" if bbb_yes else "BLOCKED BY BBB"
            bbb_icon       = "✅" if bbb_yes else "🚫"
            bar_color      = "#4ade80" if bbb_yes else "#fb923c"

            st.markdown('<div class="result-section-title">Prediction Results</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-row">

              <div class="rcard {bbb_card_cls}">
                <div class="rcard-glow"></div>
                <div class="rcard-tag">Blood–Brain Barrier</div>
                <div class="rcard-icon">{bbb_icon}</div>
                <div class="rcard-value {bbb_val_cls}">{bbb_text}</div>
                <div style="margin:0.55rem 0 0.35rem;">
                  <span style="display:inline-block; padding:0.28rem 1.1rem; border-radius:999px; font-family:Orbitron,monospace; font-weight:900; font-size:1.05rem; letter-spacing:0.18em; background:{'rgba(74,222,128,0.15)' if bbb_yes else 'rgba(251,146,60,0.15)'}; color:{'#4ade80' if bbb_yes else '#fb923c'}; border:1.5px solid {'rgba(74,222,128,0.5)' if bbb_yes else 'rgba(251,146,60,0.5)'}; box-shadow:0 0 12px {'rgba(74,222,128,0.2)' if bbb_yes else 'rgba(251,146,60,0.2)'};">{'YES' if bbb_yes else 'NO'}</span>
                </div>
                <div class="rcard-sub">Confidence &nbsp;<strong style="color:#e2eaf5;">{bbb_pct}%</strong></div>
                <div class="conf-bar-outer">
                  <div class="conf-bar-inner" style="width:{bbb_pct}%; background:{bar_color};"></div>
                </div>
              </div>

              <div class="rcard sol">
                <div class="rcard-glow"></div>
                <div class="rcard-tag">Aqueous Solubility (ESOL)</div>
                <div class="rcard-icon">💧</div>
                <div class="rcard-value {sol_cls}">{sol_pred:.3f}</div>
                <div class="rcard-sub">log(mol/L) &nbsp;·&nbsp; <strong style="color:#e2eaf5;">{sol_label}</strong></div>
              </div>

            </div>

            <div class="stat-row">
              <div class="stat-box">
                <div class="stat-label">BBB Raw Probability</div>
                <div class="stat-value">{bbb_prob:.4f}</div>
              </div>
              <div class="stat-box">
                <div class="stat-label">Solubility (log)</div>
                <div class="stat-value">{sol_pred:.4f}</div>
              </div>
              <div class="stat-box">
                <div class="stat-label">Model</div>
                <div class="stat-value" style="font-size:0.78rem;">XGBoost</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════
#  FOOTER
# ════════════════════════════
st.markdown('<hr class="mol-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="mol-footer">
    🧠 NeuroBarrier-AI &nbsp;·&nbsp; ADMET Prediction System &nbsp;·&nbsp; Powered by XGBoost + RDKit
</div>
""", unsafe_allow_html=True)

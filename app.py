import streamlit as st
import joblib
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

# -------------------------------
# Load Models
# -------------------------------

bbb_model = joblib.load("bbb_model.pkl")
esol_model = joblib.load("esol_model.pkl")

# -------------------------------
# Sample SMILES for Demo
# -------------------------------
sample_smiles_list = [
    "CCOCCN",
    "C1=CC=CC=C1",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CCN(CC)CC",
    "CCOC(=O)C1=CC=CC=C1",
    "CNC",
    "CC(C)O",
    "C1CCCCC1",
    "CC(C)CC1=CC=CC=C1",
    "COC1=CC=CC=C1",
    "CC(C)C(=O)O",
    "CCC(=O)O",
    "CCN",
    "C1=CN=CN1",
    "CCO",
    "C=CCO",
    "CC(C)OCC",
    "CCOC",
    "CNC(=O)C",
    "CC(C)N"
]

# -------------------------------
# Feature Function
# -------------------------------
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    
    arr = np.zeros((2048,))
    ConvertToNumpyArray(fp, arr)
    
    return arr


# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="NeuroMol AI", layout="centered")

st.title("NeuroMol AI")
st.subheader("ADMET Prediction System")
st.caption("Powered by XGBoost + RDKit")

st.markdown("---")

# -------------------------------
# Input Section
# -------------------------------
st.subheader("Input Molecule")

option = st.selectbox(
    "Choose a sample SMILES (or type your own below):",
    ["-- Select Sample --"] + sample_smiles_list
)

smiles_input = st.text_input("Or enter custom SMILES:")

# Priority: custom input > dropdown
if smiles_input:
    final_smiles = smiles_input
elif option != "-- Select Sample --":
    final_smiles = option
else:
    final_smiles = None

st.markdown("---")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    
    if final_smiles is None:
        st.warning("Please enter or select a SMILES string")
    
    else:
        with st.spinner("Processing molecule..."):
            fp = smiles_to_fp(final_smiles)
            
            if fp is None:
                st.error("Invalid SMILES string")
            
            else:
                fp = np.array(fp).reshape(1, -1)
                
                # BBB Prediction
                bbb_pred = bbb_model.predict(fp)[0]
                bbb_prob = bbb_model.predict_proba(fp)[0][1]
                
                # ESOL Prediction
                sol_pred = esol_model.predict(fp)[0]
                
                st.subheader("Results")
                
                if bbb_pred == 1:
                    st.success(f"BBB Penetration: YES ({bbb_prob:.2f})")
                else:
                    st.warning(f"BBB Penetration: NO ({bbb_prob:.2f})")
                
                st.info(f"Solubility (log mol/L): {sol_pred:.2f}")

st.markdown("---")
st.caption("Tip: Use sample inputs for quick demo")
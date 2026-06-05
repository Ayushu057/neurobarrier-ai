# 🚀 NeuroBarrier-AI

🔗 **Live App:** https://neurobarrier-ai.streamlit.app/

---

## 🧩 Problem It Solves

Drug discovery is expensive and time-consuming, with heavy reliance on laboratory testing.

Before moving to lab experiments, researchers need early insights into:
- Whether a compound can cross the **blood-brain barrier (BBBP)**
- Whether it is **soluble in water (ESOL)**

NeuroBarrier-AI provides instant predictions for these properties, helping reduce early-stage cost and accelerating compound screening.

---

## ⚙️ How It Works

The system uses machine learning models trained on curated datasets of molecular (ADMET-related) features.

### Workflow:
1. User inputs molecular descriptors/features  
2. The trained model processes the input  
3. Predictions are generated instantly via the UI  

### 🧠 Models & Tech

- **BBBP Model** → Classification model  
- **ESOL Model** → Regression model  

**Built with:**
- Python  
- scikit-learn  
- NumPy  
- Pandas  
- Joblib (model serialization)  
- Streamlit (UI & deployment)  

---

## 👥 Who Would Use It

- Pharmaceutical researchers for early-stage drug screening  
- Biotech startups building drug discovery pipelines  
- Students and researchers exploring cheminformatics and ML  

---

## 🎯 Features

- 🧪 BBBP Prediction – Predicts blood-brain barrier permeability  
- 💧 ESOL Prediction – Estimates aqueous solubility  
- ⚡ Real-time Predictions – Instant results through UI  
- 🌐 Web App – Accessible via browser  

---

## 📊 How It Works (App Flow)

1. Select prediction type (BBBP or ESOL)  
2. Input molecular features  
3. Model processes input  
4. Prediction displayed instantly  

---

## 🌍 Deployment

This app is deployed using Streamlit Cloud, enabling fast and free deployment of ML applications directly from GitHub.

📌 Future Improvements
SMILES input support using RDKit
Model performance visualization
Deep learning-based models
API integration for scalability

## 📬 Contact

Abhinav Mishra
GitHub: https://github.com/ninjaabhinav

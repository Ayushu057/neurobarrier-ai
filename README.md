# 🧠 NeuroBarrier-AI

### AI-Powered Drug Discovery Platform for BBBP & ESOL Prediction

🔗 **Live Application:** https://neurobarrierai.streamlit.app/

👨‍💻 **Developer:** Ayush Upadhyay

🌐 **GitHub:** https://github.com/Ayushu057

---

## 📌 Overview

NeuroBarrier-AI is a machine learning-powered web application designed to accelerate early-stage drug discovery by predicting critical molecular properties directly from SMILES representations.

The platform enables researchers, students, and pharmaceutical professionals to evaluate compounds before costly laboratory experiments, reducing both time and development costs.

---

## 🎯 Problem Statement

Drug development is an expensive and time-intensive process. Thousands of candidate molecules are screened before identifying a viable drug.

Before moving to wet-lab testing, researchers need to understand:

* Can the compound cross the Blood-Brain Barrier (BBBP)?
* Is the compound sufficiently water-soluble (ESOL)?
* Is the molecule suitable for further development?

Traditional laboratory evaluation requires significant resources and time.

NeuroBarrier-AI provides instant AI-driven predictions that help prioritize promising compounds for further investigation.

---

## 🚀 Key Features

### 🧪 Blood-Brain Barrier Prediction (BBBP)

Predicts whether a molecule is likely to penetrate the blood-brain barrier.

### 💧 Aqueous Solubility Prediction (ESOL)

Estimates water solubility, an important factor affecting drug absorption and bioavailability.

### ⚡ Real-Time Predictions

Instant predictions through an interactive web interface.

### 🧬 SMILES-Based Input

Accepts molecular structures in SMILES format for flexible compound screening.

### 🌐 Web-Based Interface

Accessible directly from any browser without installation.

---

## ⚙️ How It Works

### Step 1: Input Molecule

The user enters a SMILES representation of a compound.

### Step 2: Molecular Fingerprint Generation

RDKit converts the molecule into Morgan Fingerprints (2048-bit molecular descriptors).

### Step 3: Machine Learning Inference

Two trained XGBoost models process the molecular fingerprints:

* BBBP Classification Model
* ESOL Regression Model

### Step 4: Prediction Generation

The system returns:

* BBB Permeability Prediction
* Prediction Confidence Score
* ESOL Solubility Estimate

---

## 🧠 Machine Learning Pipeline

### Data Processing

* Molecular Structure Parsing
* Morgan Fingerprint Generation
* Feature Engineering

### Models

#### BBBP Prediction

* Task: Binary Classification
* Algorithm: XGBoost Classifier
* Output: BBB Penetrating / Non-Penetrating

#### ESOL Prediction

* Task: Regression
* Algorithm: XGBoost Regressor
* Output: Solubility (log mol/L)

---

## 🛠️ Technology Stack

### Programming

* Python

### Machine Learning

* XGBoost
* Scikit-Learn

### Cheminformatics

* RDKit

### Data Processing

* NumPy
* Pandas

### Deployment

* Streamlit
* Streamlit Community Cloud

---

## 📊 Application Workflow

User Input (SMILES)
↓
RDKit Processing
↓
Morgan Fingerprint Generation
↓
BBBP Model Prediction
↓
ESOL Model Prediction
↓
Results & Confidence Scores

---

## 🎓 Learning Outcomes

This project demonstrates practical experience in:

* Machine Learning Model Development
* Cheminformatics
* Molecular Feature Engineering
* Drug Discovery Analytics
* Model Deployment
* Interactive Web Application Development

---

## 🌍 Live Demo

https://neurobarrierai.streamlit.app/

---

## 🔮 Future Enhancements

* ADMET Property Expansion
* Toxicity Prediction Models
* Drug-Likeness Scoring
* Molecular Visualization Dashboard
* Batch Compound Screening
* REST API Integration
* Deep Learning-Based Molecular Models

---

## 📬 Contact

### Ayush Upadhyay

📧 Email: [ayushup345@gmail.com](mailto:ayushup345@gmail.com)

🔗 GitHub: https://github.com/Ayushu057

---

⭐ If you found this project useful, consider giving the repository a star.

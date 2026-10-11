# 🛡️ PhishGuard — Phishing URL Detection System

A machine learning web application that detects phishing URLs using Random Forest, Decision Tree, and Artificial Neural Network models.

## 🚀 Run the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open your browser at `http://localhost:8501`

## 📁 Project Structure

```
PhishGuard/
├── app.py               # Streamlit web application
├── pjt-2.ipynb          # Jupyter notebook (training + EDA)
├── project.csv          # Dataset
├── requirements.txt     # Python dependencies
├── ann_model.keras      # ANN (Keras) model
├── rf_model.joblib      # Random Forest model
├── dt_model.joblib      # Decision Tree model
├── scaler.joblib        # StandardScaler
└── README.md
```

## 🤖 Models

| Model | Type | Role |
|-------|------|------|
| Random Forest | Ensemble | Primary classifier |
| Decision Tree | Single tree | Reference model |
| ANN | Neural Network | Secondary classifier |

> **Verdict rule**: RF + ANN must both agree → PHISHING (reduces false positives)

## 📊 Features

- **51 features** extracted from URL structure, domain info, and page content
- Single URL scan with confidence scores
- Batch scan via CSV upload
- Model comparison dashboard

# Customer Churn Prediction - ML Project

A complete, production-ready machine learning project for predicting customer churn using classification algorithms. This project demonstrates the full ML workflow from data exploration to model deployment.

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Technologies](#technologies)
5. [Project Structure](#project-structure)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Results](#results)
9. [Model Evaluation](#model-evaluation)
10. [Limitations](#limitations)

---

## 🎯 Project Overview

This project builds a **binary classification system** that predicts whether a customer will churn based on demographics, account info, services, and billing data.

**Key Objectives:**
- Build and compare multiple ML models
- Handle class imbalance using SMOTE
- Perform hyperparameter tuning
- Explain predictions using feature importance and SHAP
- Deploy a ready-to-use prediction pipeline

---

## 🔍 Problem Statement

**Business Problem:**
- Customer churn is critical for telecom companies
- Retaining customers is cheaper than acquiring new ones
- Proactively identify at-risk customers for retention

**ML Problem:**
Build a binary classification model: Churn (Yes/No)

---

## 📊 Dataset

**Source:** IBM Telco Customer Churn (Kaggle)  
**Size:** 7,043 customers × 21 features  
**Target:** Churn (26.5% churn rate - imbalanced)

**Key Features:**
- Tenure (months), Contract Type, Monthly/Total Charges
- Internet Service, Phone Service
- Online Security, Backup, Device Protection, Tech Support
- Streaming Services, Payment Method

---

## 🛠️ Technologies

**ML Libraries:**
- Scikit-learn, XGBoost, Imbalanced-learn
- NumPy, Pandas

**Visualization:**
- Matplotlib, Seaborn, SHAP

**Persistence:**
- Joblib

---

## 📁 Project Structure

```
customer-churn-prediction/
├── data/dataset.csv
├── notebooks/customer_churn_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── models/churn_model.pkl
├── reports/
├── requirements.txt
└── README.md
```

---

## 📥 Installation

### 1. Download Dataset
```bash
# From Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Save to: data/dataset.csv
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train Model (Jupyter Notebook)
```bash
jupyter notebook notebooks/customer_churn_analysis.ipynb
```

### Make Predictions
```python
from src.predict import ChurnPredictor
import pandas as pd

predictor = ChurnPredictor('models/churn_model.pkl')

# Single prediction
customer = {
    'gender': 'Male',
    'tenure': 2,
    'MonthlyCharges': 29.85,
    'Contract': 'Month-to-month',
    # ... other features
}
pred, prob = predictor.predict_single(customer)
print(f"Prediction: {pred}, Probability: {prob:.4f}")

# Batch prediction
df = pd.read_csv('data/customers.csv')
results = predictor.predict_batch(df)
results.to_csv('predictions.csv')
```

---

## 📊 Results

**Best Model:** XGBoost

| Metric | Score |
|--------|-------|
| Accuracy | 0.85 |
| Precision | 0.76 |
| Recall | 0.68 |
| F1-Score | 0.72 |
| ROC-AUC | 0.90 |

**Top Features:** Tenure, Contract Type, Monthly Charges

---

## 🎯 Model Evaluation

**Metrics Explained:**
- **Accuracy:** Overall correctness (use carefully on imbalanced data)
- **Precision:** Of predicted churners, how many actually churned?
- **Recall:** Of actual churners, how many did we find?
- **F1-Score:** Balance between precision and recall
- **ROC-AUC:** Best metric for imbalanced classification (0.5-1.0)

**Trade-off:**
- High Precision: Fewer false alarms (save money)
- High Recall: Catch most churners (better retention)

---

## ⚠️ Limitations

1. Dataset may be outdated
2. Class imbalance (handled but still present)
3. No external data integration
4. No temporal patterns captured
5. Model may have business context gaps
6. Potential bias against certain segments
7. Concept drift over time

---

## 🚀 Future Improvements

- Add external data (macro indicators, competitors)
- Include temporal patterns
- Ensemble methods and deep learning
- REST API deployment
- Model monitoring and retraining pipeline
- Fairness and bias audits
- Business-level KPI tracking

---

## 📚 Key Learnings Demonstrated

✓ Data preprocessing & cleaning  
✓ Exploratory data analysis  
✓ Feature engineering & encoding  
✓ Class imbalance handling (SMOTE)  
✓ Model training & comparison  
✓ Hyperparameter tuning  
✓ Model evaluation & metrics  
✓ Feature importance analysis  
✓ Model interpretability (SHAP)  
✓ Model serialization & deployment  

---

## 📝 File Descriptions

| File | Purpose |
|------|---------|
| `data_preprocessing.py` | Clean, validate, and prepare data |
| `feature_engineering.py` | Create features, encode, scale |
| `train_model.py` | Train, tune, and compare models |
| `evaluate_model.py` | Evaluate, visualize, interpret |
| `predict.py` | Make predictions on new data |
| `customer_churn_analysis.ipynb` | Complete workflow in Jupyter |

---

**For questions, refer to docstrings and inline comments in source code.**

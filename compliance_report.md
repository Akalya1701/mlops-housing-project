# 🏠 Housing Price Prediction Model – Compliance Report

## 📅 Date: 2025-06-16  
## 📌 Model Name: HousingPriceModel  
## 🔢 Model Version: 1  
## ✅ Approved By: Akalya  
## 📍 Run ID: <add_run_id_here>

---

## 1. Data Documentation

- **Data Source**: California Housing dataset from sklearn
- **Data Version**: ETL Snapshot v2 – generated on 2025-06-15
- **Data Owner**: Akalya (ML Engineering)
- **Preprocessing Steps**:
  - Null handling: dropped rows with missing values
  - Normalization: not applied (tree-based model)
  - Feature engineering: median_income, ocean_proximity one-hot encoded

---

## 2. Model Lineage

- **Training Script**: `train.py`
- **Feature Columns**: 8 (e.g., median_income, total_rooms, etc.)
- **Model Type**: RandomForestRegressor (sklearn)
- **Registered Under**: `HousingPriceModel`
- **Training Run ID**: `<insert training run ID here>`

---

## 3. Explainability

- **Tool Used**: SHAP
- **Top Influential Features**:
  1. Median income
  2. Latitude
  3. Housing median age
- **Artifacts**:
  - `shap_summary.png`
  - `feature_importance.png`

---

## 4. Bias & Fairness Check

- **Bias Detection Strategy**: RMSE compared across income segments
- **Fairness Metrics**:
  - RMSE (Low income): 0.62
  - RMSE (High income): 0.64
  - RMSE Gap: 0.02 ✅ Acceptable
- **Bias Check Status**: ✅ Passed

---

## 5. Promotion & Approval History

- **Approval Date**: 2025-06-16
- **Approved By**: Akalya
- **Notes**:
  - RMSE consistently under 0.7
  - No model drift observed in 2 weeks of production monitoring
  - Bias check passed across key segments

---

## 6. Retraining & Lifecycle Strategy

- **Retraining Frequency**: Every 30 days (scheduled via GitHub Actions)
- **Triggers**: Drift detection, new data availability
- **Versioning Strategy**: Semantic versioning (v1.0, v1.1, etc.)

---

## 🔒 Compliance Status: ✅ READY FOR PRODUCTION

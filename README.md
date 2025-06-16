# 🏠 Housing Price Prediction - MLOps Project

This project demonstrates an end-to-end MLOps workflow using the California Housing dataset. It includes model training, serving, monitoring, and automated deployment.

## 🚀 Features

- Train a Random Forest model
- Log and track models with MLflow
- Serve predictions via Flask API
- Monitor model performance (RMSE)
- Automate retraining using GitHub Actions
- Promote models to Production
- Explain model predictions using SHAP
- Check fairness based on features

## 🗂️ Project Structure

mlops-housing-project/
├── data/ # Dataset files
├── src/ # Python scripts
│ ├── train.py # Model training script
│ ├── flask_app.py # Flask API for serving
│ ├── monitor.py # Monitoring script
│ ├── promote_model.py # Promotion logic
│ ├── explain_model.py # SHAP explanations
├── test/ # Test scripts
├── .github/workflows/ # CI/CD workflows
├── requirements.txt
└── README.md


## 🔧 How to Run

1. Install dependencies:
2. Train the model:
3. Serve using Flask:
4. Test API:


## 📦 CI/CD

GitHub Actions is used to:
- Train and monitor models
- Retrain and promote to production if performance is good

## 📊 Monitoring

- Performance (e.g., RMSE)
- Drift detection (basic)
- Model versioning with MLflow

---



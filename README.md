# 🏠 House Price Prediction  
## Diagnostic-Driven Regression Analysis (End-to-End ML Pipeline)

---

## 📌 Project Overview

This project builds a complete regression pipeline on the **California Housing dataset**, focusing not only on predictive performance but also on:

- Model stability  
- Regularization effects  
- Multicollinearity detection  
- Assumption validation  
- Bias–variance behavior  

The objective was to move beyond basic model training and perform structured regression diagnostics aligned with real-world ML engineering practices.

---

## 📊 Dataset

**Dataset:** California Housing Dataset (Scikit-learn)

**Target Variable:**
- Median House Value

**Key Features Include:**
- Median income  
- House age  
- Average rooms  
- Average bedrooms  
- Population  
- Latitude  
- Longitude  

---

## 🛠 Tech Stack

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Statsmodels (for VIF analysis)  
- Joblib  

---

## 🔄 End-to-End Workflow

### 1️⃣ Data Preparation
- Loaded dataset using Scikit-learn
- Converted to Pandas DataFrame
- Checked for missing values
- Verified feature distributions

---

### 2️⃣ Exploratory Data Analysis (EDA)
- Correlation heatmap
- Feature-target relationship inspection
- Distribution analysis

---

### 3️⃣ Train-Test Split
- 80/20 split
- Random state fixed for reproducibility

---

### 4️⃣ Feature Scaling
- StandardScaler applied to ensure:
  - Stable gradient-based optimization
  - Comparable feature magnitudes
  - Reduced condition number

---

### 5️⃣ Model Training

Implemented and compared:

- **Linear Regression (OLS)** — Baseline  
- **Ridge Regression (L2 Regularization)** — Variance control  
- **Lasso Regression (L1 Regularization)** — Sparsity and feature shrinkage  

---

### 6️⃣ Cross-Validation

- 5-Fold Cross Validation
- Evaluated mean R² across folds
- Reduced dependency on single train-test split
- Validated model stability

---

### 7️⃣ Multicollinearity Detection

- Computed **Variance Inflation Factor (VIF)**
- Analyzed feature-level inflation
- Discussed coefficient instability risks
- Evaluated impact of regularization on multicollinearity

---

### 8️⃣ Residual Diagnostics

Performed assumption checks:

- Residuals vs Predicted plot
- Residual distribution analysis
- Homoscedasticity inspection

This validates core linear regression assumptions.

---

### 9️⃣ Learning Behavior (Bias–Variance Insight)

- Generated learning curves
- Compared training vs validation performance
- Identified underfitting/overfitting patterns

---

### 🔟 Model Comparison

| Model | Test R² | Cross-Val Mean R² | Observations |
|-------|----------|------------------|--------------|
| OLS | ~0.57 | ~0.61 | Strong baseline performance |
| Ridge | ~0.57 | ~0.61 | Improved coefficient stability |
| Lasso | ~0.00 (alpha=1.0) | ~0.00 | Underfitting due to strong regularization |

> Lasso performance indicates that aggressive L1 regularization removes predictive information in this dataset.

---

## 📈 Key Insights

- Regularization improves numerical stability.
- Ridge reduces coefficient variance without major performance change.
- Lasso may severely underfit if alpha is too large.
- Cross-validation provides more reliable generalization estimates.
- Multicollinearity impacts coefficient interpretability more than raw performance.

---

## 🧠 Engineering Considerations

- Scaling improves gradient convergence.
- Condition number analysis performed for matrix stability.
- Residual diagnostics ensure assumption validation.
- Model comparison includes both predictive and stability evaluation.

---

## 💾 Model Persistence

Model saved using Joblib:

```python
joblib.dump(model_scaled, "house_price_model.pkl")
```

---

## 🚀 Future Enhancements

- Hyperparameter tuning using GridSearchCV  
- Non-linear model comparison (e.g., Random Forest)  
- Feature interaction engineering  
- Deployment pipeline (FastAPI / Streamlit)  
- Monitoring and retraining workflow  

---

## 👩‍💻 Author

**Dharvi Sharma**  
B.Tech CSE (AI/ML)  
Focused on building diagnostic-driven ML systems.

# 🏠 House Price Prediction using Linear Regression

## 📌 Project Overview

This project builds a machine learning model to predict house prices using the California Housing dataset.

The goal is to implement an end-to-end regression pipeline including:
- Data loading
- Exploratory Data Analysis (EDA)
- Feature scaling
- Model training
- Model comparison
- Evaluation
- Model persistence

---

## 📊 Dataset

Dataset used: California Housing Dataset  
Source: Scikit-learn built-in dataset

Features include:
- Median income
- House age
- Average rooms
- Population
- Latitude & longitude
- And more

Target variable:
- Median house value (Price)

---

## 🛠️ Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## ⚙️ Workflow

1. Data Loading
2. Exploratory Data Analysis
3. Correlation Analysis
4. Train-Test Split (80/20)
5. Feature Scaling using StandardScaler
6. Linear Regression Model
7. Ridge Regression Model
8. Model Evaluation
9. Model Saving

---

## 📈 Model Performance

| Model | R² Score | MSE |
|-------|----------|------|
| Linear Regression | 0.575 | 0.556 |
| Ridge Regression | ~0.57 | ~0.55 |

R² ≈ 0.57 indicates the model explains approximately 57% of the variance in housing prices.

---

## 📊 Visualization

Actual vs Predicted price comparison was plotted to evaluate prediction distribution and residual spread.

---

## 💾 Model Saving

The trained model was saved using Joblib:

```python
joblib.dump(model_scaled, "house_price_model.pkl")
```

---

## 🚀 How to Run

Clone the repository:

```bash
git clone https://github.com/yourusername/house_price_prediction.git
cd house_price_prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook or execute the training script.

---

## 📌 Future Improvements

- Hyperparameter tuning
- Cross-validation
- Feature engineering
- Polynomial regression
- Deployment using Streamlit
- MLOps pipeline integration

---

## 👩‍💻 Author

Dharvi Sharma  
B.Tech CSE (AI/ML)

---


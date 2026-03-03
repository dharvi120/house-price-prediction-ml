🏠 House Price Prediction — Diagnostic-Driven Regression Analysis
 Project Overview
This project builds a regression pipeline on the California Housing dataset with emphasis not only on prediction, but also on:

Regularization

Multicollinearity detection

Model stability validation

Assumption diagnostics

The goal was to move beyond basic model training and evaluate regression behavior under realistic constraints.

Dataset
Dataset: California Housing (Scikit-learn)

Target:

Median House Value

Features:

Median income

House age

Average rooms

Population

Latitude & longitude

etc.

Tech Stack
Python

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

Statsmodels (for VIF)

Joblib

 Workflow
Data Cleaning & EDA

Correlation Analysis

Multicollinearity Detection using VIF

Train-Test Split (80/20)

Feature Scaling (StandardScaler)

Model Training:

OLS (Linear Regression)

Ridge Regression (L2)

Lasso Regression (L1)

5-Fold Cross-Validation

Residual Diagnostics

Model Evaluation

Model Persistence

 Model Comparison
Model	Test R²	Cross-Val Mean R²	Notes
OLS	0.575 0.6115	Baseline
Ridge	0.57 0.6115	Reduced variance
Lasso	  	-0.0003	Feature shrinkage


Diagnostic Analysis
Multicollinearity
Computed VIF for each feature.

Identified features with high inflation factors.

Compared stability under Ridge regularization.

Residual Analysis
Residual vs Predicted plot examined.

Checked homoscedasticity assumption.

Residual distribution analyzed.

 Key Observations
Regularization improves coefficient stability.

Cross-validation provides more reliable performance estimates.

L1 regularization introduces sparsity.

Multicollinearity affects variance of estimates.

 Model Saving
joblib.dump(model_scaled, "house_price_model.pkl")

🚀 Future Extensions
Hyperparameter tuning using GridSearchCV

Learning curve analysis

Feature interaction exploration

Deployment pipeline

👩‍💻 Author
Dharvi Sharma
B.Tech CSE (AI/ML)

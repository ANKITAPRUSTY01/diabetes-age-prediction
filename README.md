# diabetes-age-prediction
ML regression model predicting diabetes age of onset using XGBoost, achieving R² of 0.82
# Diabetes Age Onset Prediction

A machine learning regression project predicting the age of diabetes onset based on health, lifestyle, and demographic factors, using a benchmarked and hyperparameter-tuned XGBoost model.

## Overview

This project applies supervised regression techniques to a health dataset to predict **age of diabetes onset**, using features including BMI, physical activity level, smoking habits, alcohol consumption, family history, stress level, and sleep pattern.

## Dataset

- 300 records, 11 features (numerical + categorical)
- Features include: Height, Weight, BMI, Physical Activity Level, Smoking Habits, Alcohol Consumption, Family Diabetes History, Stress Level, Sleep Pattern
- Target variable: `Age_of_Onset`

## Approach

1. **Exploratory Data Analysis** — correlation heatmaps, distribution histograms, and boxplots to identify outliers and relationships between features
2. **Preprocessing Pipeline** — missing value imputation, categorical encoding, and feature scaling using `ColumnTransformer` and `Pipeline`
3. **Model Benchmarking** — compared 5 regression models:
   - Linear Regression
   - Decision Tree
   - Random Forest
   - Support Vector Regression
   - XGBoost
4. **Hyperparameter Tuning** — RandomizedSearchCV (50 iterations, 5-fold CV) followed by GridSearchCV to fine-tune the best-performing model
5. **Model Evaluation** — 10-fold cross-validation, feature importance analysis
6. **Deployment Prep** — final model serialized using `joblib` for reuse

## Results

| Metric | Baseline XGBoost | Tuned XGBoost |
|---|---|---|
| R² Score | 0.63 | **0.82** |
| MAE | — | 3.68 |
| RMSE | — | 4.30 |
| Cross-Validated R² (10-fold) | — | **0.85** |

**Key finding:** BMI emerged as the strongest predictor of diabetes age of onset, based on feature-importance analysis.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, joblib

## Files

- `Diabetes_age_prediction.ipynb` — full notebook (EDA, preprocessing, model training, tuning, evaluation)
- `diabetes_age_prediction_model.pkl` — serialized final trained model
- `strong_diabetes_dataset_300_utf8.csv` — dataset used for training and evaluation

## Author

Ankita Prusty — [LinkedIn](https://linkedin.com/in/ankita-prusty-57461b307) | [GitHub](https://github.com/ANKITAPRUSTY01)

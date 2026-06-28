# Auto-Insurance Claim Prediction

## Overview
A machine learning pipeline to predict whether a customer will file an auto-insurance 
claim, based on demographic, vehicle, financial, and driving-record variables.

**Dataset:** 10,302 customer records · 26 predictor features · binary target (26.6% claimants)  
**Source:** https://www.kaggle.com/datasets/xiaomengsun/car-insurance-claim-data

## Approach

### Data Preparation
- Removed data leakage variables (`CLM_AMT`, `OLDCLAIM`) after discovering they caused 
near-perfect accuracy in early runs — a key lesson in realistic pipeline design
- Imputed missing values (median for numeric, mode for categorical)
- Applied SMOTE inside the pipeline to prevent leakage across cross-validation folds
- 80/20 stratified train-test split to preserve class proportions

### Models Compared
| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.72 | 0.810 | 0.49 | 0.74 | — |
| Random Forest | 0.79 | 0.815 | 0.62 | 0.50 | — |
| Gradient Boosting | **0.80** | **0.827** | **0.63** | **0.57** | **0.60** |

### Key Findings
- Gradient Boosting achieved the strongest AUC and most balanced precision-recall tradeoff
- Past claim frequency, driving-record points, and urbanicity were the top predictors
- Gradient Boosting uniquely captured weaker signals (vehicle type, occupation) that other 
models deprioritised， explaining its stronger performance on borderline cases
- Each model suits a different business priority: Logistic Regression for high recall, 
Random Forest for high precision, Gradient Boosting for balanced operational use

## Tools
- Python (pandas, scikit-learn, imbalanced-learn, matplotlib)

## Files
- `car_insurance.ipynb` — full pipeline: data cleaning, modelling, and evaluation

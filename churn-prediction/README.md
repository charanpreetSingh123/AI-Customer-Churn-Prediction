# AI-Based Customer Churn Prediction Model

Project-II (AIML, 7th Semester) — School of Engineering & Technology, CGC University, Mohali
Department of Artificial Intelligence & Data Science · Academic Year 2026–2027

**Team:** Vishesh Kumar (2330666), Ankita Kumari (2330604), Aarohan (2330607), Charanpreet Singh (2330613)
**Guide:** Mr. Gourav (Assistant Professor)

A machine-learning system that predicts whether a telecom customer is likely to churn, gives a churn
probability with a **Low / Medium / High** risk category, and explains each prediction with **SHAP**.
The core classifier is a tuned **XGBoost** model, benchmarked against Logistic Regression, Decision Tree
and Random Forest using imbalance-aware metrics (precision, recall, F1, ROC-AUC).

## Pipeline

```
Customer Data → Data Cleaning & Preprocessing → Feature Engineering → XGBoost Churn Model → Churn / Retain Prediction
```

## Dataset

[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)
(also on Kaggle as *Telco Customer Churn*), stored in `data/Telco-Customer-Churn.csv`.

| Fact (verified in this implementation) | Value |
|---|---|
| Records | 7,043 |
| Columns | 21 (20 features + `Churn` target) |
| Churned / Retained | 1,869 / 5,174 |
| Churn rate | 26.54 % |
| Duplicate customer IDs | 0 |
| Blank `TotalCharges` (tenure = 0) | 11 → filled with 0 |
| Train / Test split | 5,634 / 1,409 (stratified 80/20, seed 42) |

## Methodology

1. **Preprocessing** (`src/preprocess.py`) — convert `TotalCharges` to numeric, fill blanks, remove duplicates,
   one-hot encode categorical columns, standard-scale numeric columns (scikit-learn `ColumnTransformer`).
2. **Feature engineering** — `TenureGroup`, `AvgMonthlyCharge`, `NumServices`, `HasFamily`,
   `IsMonthToMonth`, `AutoPayment`.
3. **Class imbalance** — `class_weight="balanced"` for baselines; `scale_pos_weight` tuned for XGBoost.
4. **Model development** (`src/train.py`) — baselines + XGBoost tuned with `RandomizedSearchCV`
   (40 candidates, 5-fold stratified CV on training data) over `max_depth`, `learning_rate`,
   `n_estimators`, `scale_pos_weight`, `subsample`, `colsample_bytree`, `min_child_weight`; refit on F1.
5. **Evaluation** — confusion matrix, precision, recall, F1, ROC-AUC on the untouched test set.
6. **Explainability** — XGBoost feature importance and SHAP (global summary + per-customer factors).

## Results (test set, threshold 0.5)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.732 | 0.498 | **0.786** | 0.609 | 0.842 |
| Decision Tree | 0.737 | 0.503 | 0.781 | 0.612 | 0.824 |
| Random Forest | 0.761 | 0.534 | 0.778 | **0.633** | 0.840 |
| **XGBoost (tuned)** | **0.773** | **0.554** | 0.735 | 0.632 | **0.844** |

XGBoost gives the best ROC-AUC, accuracy and precision; Random Forest is level on F1 and the
linear baseline catches slightly more churners at a lower precision. The class-weighted models
all beat the naive "no churn" baseline (73.5 % accuracy, 0 recall) by a wide margin on recall.

**Top churn drivers (mean |SHAP|):** month-to-month contract, contract type, tenure, online security,
internet service (fiber optic), tech support, monthly charges, payment method (electronic check).

Figures are saved in `reports/figures/` (ROC curves, confusion matrices, model comparison,
feature importance, SHAP summary, EDA charts).

## Project structure

```
churn-prediction/
├── app.py                     # Streamlit dashboard
├── requirements.txt
├── data/
│   ├── Telco-Customer-Churn.csv
│   └── sample_customers.csv   # 25 rows for testing batch upload
├── models/churn_model.joblib  # trained pipeline (preprocessing + XGBoost)
├── reports/
│   ├── dataset_summary.json
│   ├── metrics.json
│   ├── model_comparison.csv
│   └── figures/*.png
└── src/
    ├── preprocess.py          # cleaning, feature engineering, encoders
    ├── eda.py                 # exploratory analysis + charts
    ├── train.py               # training, tuning, evaluation, SHAP
    └── predict.py             # scoring, risk category, per-customer explanation
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/eda.py                  # optional: regenerate EDA charts
python src/train.py                # optional: retrain (model is already included)
streamlit run app.py               # open http://localhost:8501
```

> **macOS:** XGBoost needs the OpenMP runtime — run `brew install libomp` once if you see a `libomp.dylib` error.

## Dashboard

- **Predict a customer** — enter details, get churn probability, risk category and top SHAP factors.
- **Batch prediction (CSV)** — upload many customers, download ranked predictions.
- **Model performance** — comparison table, ROC curves, confusion matrices, feature importance, SHAP.
- **Data insights** — dataset facts and EDA charts.

## Tools & technologies

Python · Pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn · SHAP · Streamlit · Git/GitHub

## References

1. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD 2016.
2. A. Dal Pozzolo et al., "Calibrating Probability with Undersampling for Unbalanced Classification," IEEE SSCI 2015.
3. L. Breiman, "Random Forests," *Machine Learning*, 45(1), 2001.
4. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, 12, 2011.
5. S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017.

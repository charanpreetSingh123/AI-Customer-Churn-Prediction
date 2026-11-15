from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import shap

from preprocess import ROOT, original_feature_of, prepare

MODEL_PATH = ROOT / "models" / "churn_model.joblib"
LOW_MAX, MEDIUM_MAX = 0.30, 0.60


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python src/train.py` first.")
    pipe = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    return pipe, explainer


def risk_category(p):
    if p < LOW_MAX:
        return "Low"
    if p < MEDIUM_MAX:
        return "Medium"
    return "High"


def predict(df):
    pipe, _ = load_model()
    X = prepare(df.drop(columns=["Churn"], errors="ignore"))
    proba = pipe.predict_proba(X)[:, 1]
    return pd.DataFrame({
        "ChurnProbability": proba.round(4),
        "Prediction": np.where(proba >= 0.5, "Churn", "Retain"),
        "RiskCategory": [risk_category(p) for p in proba],
    }, index=df.index)


def explain(df_row, top_n=6):
    pipe, explainer = load_model()
    prep = pipe.named_steps["prep"]
    X = prepare(df_row.drop(columns=["Churn"], errors="ignore"))
    Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out())
    sv = pd.Series(explainer.shap_values(Xt)[0], index=Xt.columns)
    grouped = sv.groupby(original_feature_of).sum()
    grouped = grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(top_n)
    values = X.iloc[0]
    shown = [values.get(f, "") for f in grouped.index]
    return pd.DataFrame({
        "Feature": grouped.index,
        "Value": [str(round(v, 2) if isinstance(v, float) else v) for v in shown],
        "Impact": grouped.values.round(4),
        "Effect": np.where(grouped.values > 0, "Raises risk", "Lowers risk"),
    })

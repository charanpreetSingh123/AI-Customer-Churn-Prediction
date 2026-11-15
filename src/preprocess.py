from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
TARGET = "Churn"

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

RAW_CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
RAW_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

ENG_CATEGORICAL = ["TenureGroup"]
ENG_NUMERIC = ["AvgMonthlyCharge", "NumServices", "HasFamily", "IsMonthToMonth", "AutoPayment"]

CATEGORICAL_FEATURES = RAW_CATEGORICAL + ENG_CATEGORICAL
NUMERIC_FEATURES = RAW_NUMERIC + ENG_NUMERIC


def load_raw(path=DATA_PATH):
    return pd.read_csv(path)


def clean(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    if "customerID" in df.columns:
        df = df.drop_duplicates(subset="customerID")
    df = df.drop_duplicates()
    if TARGET in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = (df[TARGET].str.strip() == "Yes").astype(int)
    return df


def engineer_features(df):
    df = df.copy()
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 60, np.inf],
        labels=["0-12", "13-24", "25-48", "49-60", "61+"],
    ).astype(str)
    df["AvgMonthlyCharge"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"].replace(0, 1), df["MonthlyCharges"]
    )
    df["NumServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df.loc[df["InternetService"] != "No", "NumServices"] += 1
    df["HasFamily"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
    df["AutoPayment"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    return df


def prepare(df):
    df = engineer_features(clean(df))
    cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    if TARGET in df.columns:
        cols = cols + [TARGET]
    return df[cols]


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def original_feature_of(transformed_name):
    name = transformed_name.split("__", 1)[-1]
    for col in sorted(CATEGORICAL_FEATURES + NUMERIC_FEATURES, key=len, reverse=True):
        if name == col or name.startswith(col + "_"):
            return col
    return name

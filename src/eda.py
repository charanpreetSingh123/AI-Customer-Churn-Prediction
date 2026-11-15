"""Exploratory data analysis: prints dataset facts and saves charts to reports/figures/."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import ROOT, TARGET, clean, engineer_features, load_raw

FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def main():
    raw = load_raw()
    blank_total = int((raw["TotalCharges"].str.strip() == "").sum())
    df = engineer_features(clean(raw))

    summary = {
        "records_raw": int(len(raw)),
        "columns_raw": int(raw.shape[1]),
        "records_after_cleaning": int(len(df)),
        "duplicate_customer_ids": int(raw["customerID"].duplicated().sum()),
        "blank_TotalCharges_filled_with_0": blank_total,
        "churned_customers": int(df[TARGET].sum()),
        "retained_customers": int((df[TARGET] == 0).sum()),
        "churn_rate_percent": round(100 * df[TARGET].mean(), 2),
    }
    print(json.dumps(summary, indent=2))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    label = df[TARGET].map({0: "Retained", 1: "Churned"})

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=label, ax=ax, palette=["#4C72B0", "#DD8452"], hue=label, legend=False)
    ax.set_title("Class distribution (imbalanced)")
    ax.set_xlabel("")
    fig.tight_layout(); fig.savefig(FIG_DIR / "class_distribution.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["Contract", "InternetService", "PaymentMethod"]):
        rate = df.groupby(col)[TARGET].mean().sort_values(ascending=False) * 100
        sns.barplot(x=rate.values, y=rate.index, ax=ax, color="#DD8452")
        ax.set_title(f"Churn rate by {col}"); ax.set_xlabel("Churn rate (%)"); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(FIG_DIR / "churn_by_category.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        sns.kdeplot(data=df, x=col, hue=label, fill=True, common_norm=False, ax=ax)
        ax.set_title(f"{col} by churn status")
    fig.tight_layout(); fig.savefig(FIG_DIR / "numeric_distributions.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    rate = df.groupby("TenureGroup")[TARGET].mean() * 100
    rate = rate.reindex(["0-12", "13-24", "25-48", "49-60", "61+"])
    sns.barplot(x=rate.index, y=rate.values, ax=ax, color="#4C72B0")
    ax.set_title("Churn rate by tenure group (months)"); ax.set_ylabel("Churn rate (%)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "churn_by_tenure_group.png", dpi=120); plt.close(fig)

    num = df[["tenure", "MonthlyCharges", "TotalCharges", "NumServices", "AvgMonthlyCharge", TARGET]]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation of numeric features")
    fig.tight_layout(); fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=120); plt.close(fig)

    print(f"Saved EDA figures to {FIG_DIR}")


if __name__ == "__main__":
    main()

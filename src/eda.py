import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import style
from preprocess import ROOT, TARGET, clean, engineer_features, load_raw

FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
style.apply()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=130)
    plt.close(fig)


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
    (ROOT / "reports" / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    status = df[TARGET].map({0: "Retained", 1: "Churned"})

    fig, ax = plt.subplots(figsize=(5, 4))
    counts = status.value_counts().reindex(["Retained", "Churned"])
    bars = ax.bar(counts.index, counts.values, color=[style.RETAINED, style.CHURNED], width=0.55)
    ax.bar_label(bars, labels=[f"{v:,}\n({v / counts.sum():.1%})" for v in counts.values], padding=4)
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_title("Class distribution")
    ax.grid(axis="x", visible=False)
    save(fig, "class_distribution.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["Contract", "InternetService", "PaymentMethod"]):
        rate = df.groupby(col)[TARGET].mean().sort_values() * 100
        bars = ax.barh(rate.index, rate.values, color=style.ORANGE, height=0.6)
        ax.bar_label(bars, fmt="%.1f%%", padding=4)
        ax.set_xlim(0, rate.max() * 1.25)
        ax.set_title(f"Churn rate by {col}")
        ax.set_xlabel("Churn rate (%)")
        ax.grid(axis="y", visible=False)
    save(fig, "churn_by_category.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        sns.kdeplot(data=df.assign(Status=status), x=col, hue="Status", fill=True, alpha=0.35,
                    common_norm=False, palette=style.STATUS_PALETTE, ax=ax, linewidth=1.5)
        ax.set_title(f"{col} by churn status")
    save(fig, "numeric_distributions.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["0-12", "13-24", "25-48", "49-60", "61+"]
    rate = (df.groupby("TenureGroup")[TARGET].mean() * 100).reindex(order)
    bars = ax.bar(rate.index, rate.values, color=style.TEAL, width=0.6)
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_ylim(0, rate.max() * 1.2)
    ax.set_title("Churn rate by tenure group (months)")
    ax.set_ylabel("Churn rate (%)")
    ax.grid(axis="x", visible=False)
    save(fig, "churn_by_tenure_group.png")

    num = df[["tenure", "MonthlyCharges", "TotalCharges", "NumServices", "AvgMonthlyCharge", TARGET]]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap=style.DIV_CMAP, vmin=-1, vmax=1,
                linewidths=0.5, linecolor=style.BG, ax=ax)
    ax.set_title("Correlation of numeric features")
    ax.grid(False)
    save(fig, "correlation_heatmap.png")

    print(f"Saved EDA figures to {FIG_DIR}")


if __name__ == "__main__":
    main()

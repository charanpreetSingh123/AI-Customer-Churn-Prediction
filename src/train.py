import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import style
from preprocess import ROOT, TARGET, build_preprocessor, load_raw, original_feature_of, prepare

SEED = 42
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
for d in (MODEL_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
style.apply()


def evaluate(name, y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    data = prepare(load_raw())
    X, y = data.drop(columns=TARGET), data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Train: {len(X_train)}  Test: {len(X_test)}  scale_pos_weight={spw:.2f}")

    baselines = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=SEED
        ),
    }

    results, probas = [], {}
    for name, clf in baselines.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)]).fit(X_train, y_train)
        probas[name] = pipe.predict_proba(X_test)[:, 1]
        results.append(evaluate(name, y_test, probas[name]))
        print(f"{name:20s} done")

    xgb_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1)),
    ])
    param_dist = {
        "clf__n_estimators": [200, 300, 400, 600],
        "clf__max_depth": [2, 3, 4, 5, 6],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5],
        "clf__scale_pos_weight": [1.0, spw * 0.75, spw],
    }
    search = RandomizedSearchCV(
        xgb_pipe, param_dist, n_iter=40, scoring={"roc_auc": "roc_auc", "f1": "f1"}, refit="f1",
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), random_state=SEED, n_jobs=-1,
    ).fit(X_train, y_train)
    best = search.best_estimator_
    best_params = {
        k.replace("clf__", ""): (round(float(v), 4) if isinstance(v, (np.floating, float)) else v)
        for k, v in search.best_params_.items()
    }
    cv_auc = search.cv_results_["mean_test_roc_auc"][search.best_index_]
    print("Best XGBoost params:", best_params, f"CV F1={search.best_score_:.4f} CV ROC-AUC={cv_auc:.4f}")
    probas["XGBoost (tuned)"] = best.predict_proba(X_test)[:, 1]
    results.append(evaluate("XGBoost (tuned)", y_test, probas["XGBoost (tuned)"]))

    comparison = pd.DataFrame(results).set_index("model").round(4)
    print("\nModel comparison (test set, threshold 0.5)")
    print(comparison.to_string())
    comparison.to_csv(REPORT_DIR / "model_comparison.csv")

    xgb_pred = (probas["XGBoost (tuned)"] >= 0.5).astype(int)
    print("\nXGBoost classification report")
    print(classification_report(y_test, xgb_pred, target_names=["Retained", "Churned"]))

    labels = ["Retained", "Churned"]
    fig, axes = plt.subplots(1, len(probas), figsize=(4.4 * len(probas), 4))
    for ax, (name, p) in zip(axes, probas.items()):
        cm = confusion_matrix(y_test, (p >= 0.5).astype(int))
        sns.heatmap(cm, annot=True, fmt="d", cmap=style.SEQ_CMAP, cbar=False, ax=ax,
                    linewidths=2, linecolor=style.BG, annot_kws={"size": 13, "weight": "bold"},
                    xticklabels=labels, yticklabels=labels)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.grid(False)
    save(fig, "confusion_matrices.png")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        lw = 2.6 if name == "XGBoost (tuned)" else 1.6
        ax.plot(fpr, tpr, lw=lw, color=style.MODEL_COLORS[name],
                label=f"{name} (AUC = {roc_auc_score(y_test, p):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=style.MUTED)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, "roc_curves.png")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    metrics = ["precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(comparison))
    width = 0.2
    for i, (m, c) in enumerate(zip(metrics, style.METRIC_COLORS)):
        ax.bar(x + (i - 1.5) * width, comparison[m].values, width, label=m.replace("_", "-").upper(), color=c)
    ax.set_xticks(x, comparison.index)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on test set")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax.grid(axis="x", visible=False)
    save(fig, "model_comparison.png")

    prep, xgb = best.named_steps["prep"], best.named_steps["clf"]
    feat_names = prep.get_feature_names_out()
    imp = pd.Series(xgb.feature_importances_, index=feat_names)
    imp_grouped = imp.groupby(original_feature_of).sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    top = imp_grouped.head(15)[::-1]
    ax.barh(top.index, top.values, color=style.TEAL, height=0.65)
    ax.set_title("XGBoost feature importance")
    ax.set_xlabel("Importance (gain, grouped by feature)")
    ax.grid(axis="y", visible=False)
    save(fig, "feature_importance.png")

    X_test_t = pd.DataFrame(prep.transform(X_test), columns=feat_names)
    shap_values = shap.TreeExplainer(xgb).shap_values(X_test_t)
    shap_grouped = (pd.DataFrame(np.abs(shap_values), columns=feat_names)
                    .T.groupby(original_feature_of).sum().T.mean().sort_values(ascending=False))
    display_names = [n.split("__", 1)[-1] for n in feat_names]
    plt.figure()
    shap.summary_plot(shap_values, X_test_t, feature_names=display_names, show=False,
                      max_display=15, cmap=style.SHAP_CMAP)
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP summary (XGBoost)")
    style.recolor_text(fig)
    save(fig, "shap_summary.png")

    joblib.dump(best, MODEL_DIR / "churn_model.joblib")
    metrics_out = {
        "train_size": int(len(X_train)), "test_size": int(len(X_test)),
        "test_churn_cases": int(y_test.sum()),
        "best_params": best_params,
        "cv_f1": round(float(search.best_score_), 4), "cv_roc_auc": round(float(cv_auc), 4),
        "comparison": comparison.reset_index().to_dict(orient="records"),
        "xgb_confusion_matrix": confusion_matrix(y_test, xgb_pred).tolist(),
        "top_features_shap": shap_grouped.head(10).round(4).to_dict(),
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    print("\nTop churn drivers (mean |SHAP|):")
    print(shap_grouped.head(10).round(4).to_string())
    print(f"\nSaved model to {MODEL_DIR / 'churn_model.joblib'}")


if __name__ == "__main__":
    main()

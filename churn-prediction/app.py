"""Streamlit dashboard for the AI-Based Customer Churn Prediction Model.

Run with:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predict import LOW_MAX, MEDIUM_MAX, explain, load_model, predict  # noqa: E402

FIG = ROOT / "reports" / "figures"
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

RISK_COLOR = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c62828"}

try:
    load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.sidebar.title("📉 Churn Prediction")
page = st.sidebar.radio(
    "Go to", ["Predict a customer", "Batch prediction (CSV)", "Model performance", "Data insights"]
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Risk categories: **Low** < {LOW_MAX:.0%} ≤ **Medium** < {MEDIUM_MAX:.0%} ≤ **High**"
)
st.sidebar.caption("Model: tuned XGBoost · Dataset: IBM Telco Customer Churn")


# ------------------------------------------------------------------ single prediction
if page == "Predict a customer":
    st.title("Predict churn for a single customer")
    st.write("Enter the customer's account, service and billing details.")

    with st.form("customer"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            st.subheader("Account")
            tenure = st.slider("Tenure (months)", 0, 72, 5)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox("Payment method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])
        with c2:
            st.subheader("Services")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multi = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            no_int = internet == "No"
            opts = ["No internet service"] if no_int else ["No", "Yes"]
            security = st.selectbox("Online security", opts)
            backup = st.selectbox("Online backup", opts)
            device = st.selectbox("Device protection", opts)
        with c3:
            st.subheader(" ")
            tech = st.selectbox("Tech support", opts)
            tv = st.selectbox("Streaming TV", opts)
            movies = st.selectbox("Streaming movies", opts)
            st.subheader("Billing")
            monthly = st.number_input("Monthly charges", 0.0, 200.0, 85.0, step=0.5)
            total = st.number_input("Total charges", 0.0, 10000.0, float(monthly * tenure), step=1.0)
        submitted = st.form_submit_button("Predict churn", type="primary")

    if submitted:
        if phone == "No":
            multi = "No phone service"
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": device,
            "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }])
        res = predict(row).iloc[0]
        risk = res["RiskCategory"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{res['ChurnProbability']:.1%}")
        m2.metric("Prediction", res["Prediction"])
        m3.markdown(
            f"<div style='padding-top:8px'>Risk category<br>"
            f"<span style='font-size:2rem;font-weight:700;color:{RISK_COLOR[risk]}'>{risk}</span></div>",
            unsafe_allow_html=True,
        )
        st.progress(float(res["ChurnProbability"]))

        st.subheader("Top contributing factors (SHAP)")
        exp = explain(row)
        st.dataframe(exp, hide_index=True, use_container_width=True)
        st.bar_chart(exp.set_index("Feature")["Impact"], horizontal=True)
        st.caption("Positive impact pushes the prediction towards churn; negative pushes towards retention.")


# ------------------------------------------------------------------ batch prediction
elif page == "Batch prediction (CSV)":
    st.title("Batch prediction from a CSV file")
    st.write(
        "Upload a CSV with the same columns as the Telco dataset "
        "(`customerID` optional; a `Churn` column is ignored if present)."
    )
    sample = ROOT / "data" / "sample_customers.csv"
    if sample.exists():
        st.download_button("Download a sample CSV", sample.read_bytes(), "sample_customers.csv")
    file = st.file_uploader("Customer CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        try:
            res = predict(df)
        except KeyError as e:
            st.error(f"Missing column in the uploaded file: {e}")
            st.stop()
        out = pd.concat([df[["customerID"]] if "customerID" in df else pd.DataFrame(index=df.index), res], axis=1)
        out = out.sort_values("ChurnProbability", ascending=False)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers", len(out))
        for col, cat in zip((c2, c3, c4), ("High", "Medium", "Low")):
            col.metric(f"{cat} risk", int((out["RiskCategory"] == cat).sum()))
        st.dataframe(out, hide_index=True, use_container_width=True)
        st.download_button("Download predictions", out.to_csv(index=False).encode(), "churn_predictions.csv")


# ------------------------------------------------------------------ model performance
elif page == "Model performance":
    st.title("Model evaluation")
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    st.write(
        f"Stratified split: **{metrics['train_size']}** training / **{metrics['test_size']}** test "
        f"customers ({metrics['test_churn_cases']} churned in test). "
        f"5-fold CV on training set — F1 {metrics['cv_f1']}, ROC-AUC {metrics['cv_roc_auc']}."
    )
    comp = pd.DataFrame(metrics["comparison"]).set_index("model")
    st.subheader("XGBoost vs baselines (test set)")
    st.dataframe(comp.style.highlight_max(axis=0, props="background-color:#2e7d32;color:white;font-weight:bold").format("{:.3f}"), use_container_width=True)
    st.subheader("Best XGBoost hyperparameters")
    st.json(metrics["best_params"])
    c1, c2 = st.columns(2)
    c1.image(str(FIG / "roc_curves.png"), caption="ROC curves")
    c2.image(str(FIG / "model_comparison.png"), caption="Precision / Recall / F1 / ROC-AUC")
    st.image(str(FIG / "confusion_matrices.png"), caption="Confusion matrices (threshold 0.5)")
    c1, c2 = st.columns(2)
    c1.image(str(FIG / "feature_importance.png"), caption="XGBoost feature importance")
    c2.image(str(FIG / "shap_summary.png"), caption="SHAP summary")


# ------------------------------------------------------------------ data insights
else:
    st.title("Dataset insights")
    summary = json.loads((ROOT / "reports" / "dataset_summary.json").read_text())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", summary["records_after_cleaning"])
    c2.metric("Raw columns", summary["columns_raw"])
    c3.metric("Churned", summary["churned_customers"])
    c4.metric("Churn rate", f"{summary['churn_rate_percent']}%")
    st.caption(f"{summary['blank_TotalCharges_filled_with_0']} blank TotalCharges values (tenure 0) were filled with 0.")
    c1, c2 = st.columns(2)
    c1.image(str(FIG / "class_distribution.png"))
    c2.image(str(FIG / "churn_by_tenure_group.png"))
    st.image(str(FIG / "churn_by_category.png"))
    st.image(str(FIG / "numeric_distributions.png"))
    st.image(str(FIG / "correlation_heatmap.png"), width=600)

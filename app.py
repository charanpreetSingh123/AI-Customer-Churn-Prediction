import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predict import LOW_MAX, MEDIUM_MAX, explain, load_model, predict

FIG = ROOT / "reports" / "figures"
RISK_COLOR = {"Low": "#34D399", "Medium": "#FBBF24", "High": "#F87171"}
RAISES, LOWERS = "#F97316", "#2DD4BF"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 2.2rem; max-width: 1280px;}
[data-testid="stSidebar"] {background: #0B1222; border-right: 1px solid #1E293B;}
[data-testid="stMetric"] {background: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 14px 18px;}
[data-testid="stMetricLabel"] p {color: #94A3B8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;}
[data-testid="stForm"] {background: #131D33; border: 1px solid #334155; border-radius: 14px; padding: 1.4rem 1.6rem;}
[data-testid="stImage"] img {border-radius: 10px; border: 1px solid #1E293B;}
h3 {color: #F1F5F9;}
.hero {padding-bottom: 18px; border-bottom: 1px solid #1E293B; margin-bottom: 24px;}
.hero .tag {display: inline-block; background: rgba(45,212,191,0.12); color: #2DD4BF; border: 1px solid rgba(45,212,191,0.35); border-radius: 999px; padding: 2px 12px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;}
.hero .title {font-size: 2.1rem; font-weight: 800; color: #F8FAFC; margin: 10px 0 4px 0; line-height: 1.2;}
.hero .sub {color: #94A3B8; font-size: 1rem;}
.card {background: #1E293B; border: 1px solid #334155; border-radius: 14px; padding: 22px 24px;}
.card .label {color: #94A3B8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em;}
.card .big {font-size: 3rem; font-weight: 800; line-height: 1.1; margin-top: 4px;}
.card .bar {height: 10px; background: #0F172A; border-radius: 999px; overflow: hidden; margin-top: 14px;}
.card .bar div {height: 100%; border-radius: 999px;}
.card .row {display: flex; justify-content: space-between; align-items: flex-end; margin-top: 22px;}
.card .verdict {font-size: 1.25rem; font-weight: 600; color: #F1F5F9; margin-top: 4px;}
.pill {display: inline-block; padding: 4px 16px; border-radius: 999px; font-weight: 700; margin-top: 6px;}
.legend div {color: #CBD5E1; font-size: 0.85rem; margin: 6px 0;}
.legend span {display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px;}
.brand {font-size: 1.25rem; font-weight: 800; color: #F8FAFC;}
.brand span {color: #2DD4BF;}
</style>
""",
    unsafe_allow_html=True,
)


def hero(tag, title, subtitle):
    st.markdown(
        f"<div class='hero'><span class='tag'>{tag}</span>"
        f"<div class='title'>{title}</div><div class='sub'>{subtitle}</div></div>",
        unsafe_allow_html=True,
    )


try:
    load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.markdown("<div class='brand'>Churn<span>Predictor</span></div>", unsafe_allow_html=True)
    st.caption("Customer retention analytics")
    st.write("")
    page = st.radio(
        "Navigate",
        ["Single customer", "Batch scoring", "Model performance", "Data insights"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        "<div class='legend'>"
        f"<div><span style='background:{RISK_COLOR['Low']}'></span>Low risk: below {LOW_MAX:.0%}</div>"
        f"<div><span style='background:{RISK_COLOR['Medium']}'></span>Medium risk: {LOW_MAX:.0%} to {MEDIUM_MAX:.0%}</div>"
        f"<div><span style='background:{RISK_COLOR['High']}'></span>High risk: {MEDIUM_MAX:.0%} and above</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Model: tuned XGBoost")
    st.caption("Data: IBM Telco Customer Churn")


if page == "Single customer":
    hero("Prediction", "Will this customer churn?",
         "Enter the account, service and billing details to get a churn risk score and the reasons behind it.")

    with st.form("customer"):
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.markdown("### Profile")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            st.markdown("### Account")
            tenure = st.slider("Tenure (months)", 0, 72, 5)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox("Payment method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])
        with c2:
            st.markdown("### Services")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multi = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            opts = ["No internet service"] if internet == "No" else ["No", "Yes"]
            security = st.selectbox("Online security", opts)
            backup = st.selectbox("Online backup", opts)
            device = st.selectbox("Device protection", opts)
        with c3:
            st.markdown("### Add-ons")
            tech = st.selectbox("Tech support", opts)
            tv = st.selectbox("Streaming TV", opts)
            movies = st.selectbox("Streaming movies", opts)
            st.markdown("### Billing")
            monthly = st.number_input("Monthly charges", 0.0, 200.0, 85.0, step=0.5)
            total = st.number_input("Total charges", 0.0, 10000.0, float(monthly * tenure), step=1.0)
        submitted = st.form_submit_button("Predict churn risk", type="primary", use_container_width=True)

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
        p = float(res["ChurnProbability"])
        risk = res["RiskCategory"]
        color = RISK_COLOR[risk]
        verdict = "Likely to churn" if res["Prediction"] == "Churn" else "Likely to stay"

        st.write("")
        left, right = st.columns([1, 1.5], gap="large")
        with left:
            st.markdown(
                f"<div class='card'><div class='label'>Churn probability</div>"
                f"<div class='big' style='color:{color}'>{p:.1%}</div>"
                f"<div class='bar'><div style='width:{p * 100:.1f}%;background:{color}'></div></div>"
                f"<div class='row'><div><div class='label'>Prediction</div><div class='verdict'>{verdict}</div></div>"
                f"<div style='text-align:right'><div class='label'>Risk level</div>"
                f"<span class='pill' style='background:{color}22;color:{color};border:1px solid {color}66'>{risk}</span>"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )
        with right:
            st.markdown("### What drives this prediction")
            exp = explain(row)
            exp["Factor"] = exp["Feature"] + " = " + exp["Value"]
            chart = (
                alt.Chart(exp)
                .mark_bar(cornerRadiusEnd=4, height=22)
                .encode(
                    x=alt.X("Impact:Q", title="Impact on churn risk (SHAP)"),
                    y=alt.Y("Factor:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
                    color=alt.Color(
                        "Effect:N",
                        scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[RAISES, LOWERS]),
                        legend=alt.Legend(orient="bottom", title=None),
                    ),
                    tooltip=["Feature", "Value", "Impact", "Effect"],
                )
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)


elif page == "Batch scoring":
    hero("Batch scoring", "Score a whole customer list",
         "Upload a CSV with the Telco dataset columns. customerID is optional and any Churn column is ignored.")
    sample = ROOT / "data" / "sample_customers.csv"
    if sample.exists():
        st.download_button("Download sample CSV", sample.read_bytes(), "sample_customers.csv")
    file = st.file_uploader("Customer CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        try:
            res = predict(df)
        except KeyError as e:
            st.error(f"Missing column in the uploaded file: {e}")
            st.stop()
        ids = df[["customerID"]] if "customerID" in df else pd.DataFrame(index=df.index)
        out = pd.concat([ids, res], axis=1).sort_values("ChurnProbability", ascending=False)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers scored", len(out))
        for col, cat in zip((c2, c3, c4), ("High", "Medium", "Low")):
            col.metric(f"{cat} risk", int((out["RiskCategory"] == cat).sum()))
        st.write("")
        styled = out.style.map(lambda v: f"color:{RISK_COLOR[v]};font-weight:700", subset=["RiskCategory"])
        st.dataframe(
            styled, hide_index=True, use_container_width=True, height=460,
            column_config={
                "customerID": st.column_config.TextColumn("Customer ID"),
                "ChurnProbability": st.column_config.ProgressColumn(
                    "Churn probability", min_value=0.0, max_value=1.0, format="percent"),
                "Prediction": st.column_config.TextColumn("Prediction"),
                "RiskCategory": st.column_config.TextColumn("Risk"),
            },
        )
        st.download_button("Download predictions", out.to_csv(index=False).encode(), "churn_predictions.csv",
                           type="primary")


elif page == "Model performance":
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    hero("Evaluation", "Model performance",
         f"Stratified split of {metrics['train_size']:,} training and {metrics['test_size']:,} test customers "
         f"({metrics['test_churn_cases']} churned in the test set).")
    comp = pd.DataFrame(metrics["comparison"]).set_index("model")
    xgb = comp.loc["XGBoost (tuned)"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{xgb['roc_auc']:.3f}")
    c2.metric("Recall (churn)", f"{xgb['recall']:.1%}")
    c3.metric("Precision (churn)", f"{xgb['precision']:.1%}")
    c4.metric("F1 score", f"{xgb['f1']:.3f}")

    st.markdown("### XGBoost vs baseline models")
    table = comp.rename(columns=str.upper).rename(columns={"ROC_AUC": "ROC-AUC"})
    table.index.name = "Model"
    st.dataframe(
        table.style.format("{:.3f}").highlight_max(
            axis=0, props="background-color: rgba(45,212,191,0.22); color: #5EEAD4; font-weight: 700"),
        use_container_width=True,
    )
    with st.expander("Best XGBoost hyperparameters"):
        st.json(metrics["best_params"])
        st.caption(f"5-fold cross-validation on the training set: F1 {metrics['cv_f1']}, ROC-AUC {metrics['cv_roc_auc']}")

    c1, c2 = st.columns(2, gap="large")
    c1.image(str(FIG / "roc_curves.png"))
    c2.image(str(FIG / "model_comparison.png"))
    st.image(str(FIG / "confusion_matrices.png"))
    c1, c2 = st.columns(2, gap="large")
    c1.image(str(FIG / "feature_importance.png"))
    c2.image(str(FIG / "shap_summary.png"))


else:
    summary = json.loads((ROOT / "reports" / "dataset_summary.json").read_text())
    hero("Exploration", "Dataset insights",
         f"IBM Telco Customer Churn. {summary['blank_TotalCharges_filled_with_0']} blank TotalCharges values "
         "for brand-new customers were filled with 0.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{summary['records_after_cleaning']:,}")
    c2.metric("Columns", summary["columns_raw"])
    c3.metric("Churned", f"{summary['churned_customers']:,}")
    c4.metric("Churn rate", f"{summary['churn_rate_percent']}%")
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    c1.image(str(FIG / "class_distribution.png"))
    c2.image(str(FIG / "churn_by_tenure_group.png"))
    st.image(str(FIG / "churn_by_category.png"))
    st.image(str(FIG / "numeric_distributions.png"))
    c1, c2 = st.columns([1, 1])
    c1.image(str(FIG / "correlation_heatmap.png"))

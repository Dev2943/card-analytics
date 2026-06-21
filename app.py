"""
streamlit_app.py — Credit Card Portfolio Analytics
Live demo: ETL → segmentation → default model → portfolio KPIs,
on the UCI Default of Credit Card Clients dataset (30,000 cardholders).
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import data_pipeline as dp
import segmentation as seg
import default_model as dm
import portfolio_kpis as pk

st.set_page_config(page_title="Credit Card Portfolio Analytics", page_icon="💳", layout="wide")

st.title("💳 Credit Card Portfolio Analytics")
st.caption("Consumer-credit analytics: segmentation, default prediction, and risk-adjusted portfolio KPIs · UCI dataset (30,000 cardholders)")


@st.cache_data(show_spinner=True)
def run_pipeline():
    """Run the full analytics pipeline once and cache the result."""
    # 1. ETL + feature engineering
    df = dp.load_raw()
    df = dp.clean(df)
    df = dp.engineer_features(df)

    # 2. Segmentation
    X_scaled = seg.scale_features(df)
    k_values, inertias, silhouettes = seg.choose_k(X_scaled)
    best_k = k_values[int(np.argmax(silhouettes))]
    df = seg.fit_segments(df, X_scaled, best_k)
    profile = seg.profile_segments(df)
    names = seg.suggest_names(profile)
    df["segment_name"] = df["segment"].map(names)

    # 3. Default model
    X_train, X_test, y_train, y_test = dm.split_data(df)
    logreg, gb, scaler = dm.train_models(X_train, y_train)
    prob_lr = logreg.predict_proba(scaler.transform(X_test))[:, 1]
    prob_gb = gb.predict_proba(X_test)[:, 1]
    res_lr = dm.evaluate("LogReg", y_test, prob_lr)
    res_gb = dm.evaluate("GradBoost", y_test, prob_gb)
    auc_lr, auc_gb = res_lr["roc_auc"], res_gb["roc_auc"]

    # Score the full population with logreg for KPIs
    X_all = scaler.transform(df[dm.MODEL_FEATURES].values)
    df["pred_prob_default"] = logreg.predict_proba(X_all)[:, 1]

    # 4. Portfolio KPIs
    df = pk.build_churn_proxy(df)
    df = pk.compute_expected_loss(df)
    df = pk.compute_revenue(df)
    kpis = pk.segment_kpis(df)

    return df, profile, names, kpis, (k_values, inertias, silhouettes), (auc_lr, auc_gb)


with st.spinner("Running the full analytics pipeline (fetch → clean → segment → model → KPIs)…"):
    try:
        df, profile, names, kpis, k_data, aucs = run_pipeline()
        ok = True
    except Exception as e:
        ok = False
        st.error(f"Pipeline error: {e}")
        st.info("This demo fetches the UCI dataset live. If it fails, it may be a temporary UCI server issue — try refreshing.")

if ok:
    # ── Top KPIs ──
    total_customers = len(df)
    default_rate = df["default"].mean()
    total_el = df["expected_loss"].sum()
    total_rev = df["monthly_revenue"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cardholders", f"{total_customers:,}")
    c2.metric("Default Rate", f"{default_rate:.1%}")
    c3.metric("Portfolio Expected Loss", f"NT${total_el:,.0f}")
    c4.metric("Monthly Revenue", f"NT${total_rev:,.0f}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 Segmentation", "🎯 Default Model", "💰 Portfolio KPIs"])

    with tab1:
        st.subheader("Behavioral Customer Segments (K-Means)")
        st.write("Customers clustered on utilization, payment ratio, credit limit, delinquency, and spend — then named by behavior.")
        prof_display = profile.copy()
        prof_display.index = [f"{i} — {names.get(i, '')}" for i in prof_display.index]
        st.dataframe(prof_display.style.format({
            "default_rate": "{:.1%}", "avg_utilization": "{:.2f}",
            "avg_payment_ratio": "{:.2f}", "avg_limit": "{:,.0f}",
            "avg_months_delinquent": "{:.1f}", "avg_bill": "{:,.0f}",
        }), use_container_width=True)

        k_values, inertias, silhouettes = k_data
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        ax1.plot(k_values, inertias, "o-", color="#1a73e8")
        ax1.set_title("Elbow Method"); ax1.set_xlabel("k"); ax1.set_ylabel("Inertia"); ax1.grid(alpha=0.3)
        ax2.plot(k_values, silhouettes, "o-", color="#2a9d8f")
        ax2.set_title("Silhouette Score"); ax2.set_xlabel("k"); ax2.grid(alpha=0.3)
        st.pyplot(fig)

    with tab2:
        st.subheader("Default Prediction — Logistic Regression vs Gradient Boosting")
        auc_lr, auc_gb = aucs
        m1, m2 = st.columns(2)
        m1.metric("Logistic Regression ROC-AUC", f"{auc_lr:.3f}")
        m2.metric("Gradient Boosting ROC-AUC", f"{auc_gb:.3f}")
        st.markdown("""
        **Why two models?** Logistic regression is the *interpretable, regulator-friendly* baseline used for actual credit decisions (ECOA compliance). Gradient boosting benchmarks the performance ceiling.

        The model **excludes protected attributes** (sex, marriage) and is checked for **disparate impact** using the four-fifths rule — standard fair-lending practice.
        """)
        st.caption(f"Imbalanced problem: only {default_rate:.0%} default, so ROC-AUC / PR-AUC are used instead of accuracy.")

    with tab3:
        st.subheader("Risk-Adjusted Portfolio KPIs by Segment")
        st.write("Expected Loss = PD × EAD × LGD. Risk-adjusted contribution = revenue − expected loss.")
        st.dataframe(kpis.style.format({
            "actual_default_rate": "{:.1%}", "avg_pd": "{:.1%}",
            "total_expected_loss": "NT${:,.0f}", "total_monthly_revenue": "NT${:,.0f}",
            "total_risk_adj_contribution": "NT${:,.0f}", "churn_risk_rate": "{:.1%}",
        }), use_container_width=True)
        st.caption("This table is the strategic dashboard's data layer — it turns predictions into portfolio decisions.")

    st.divider()
    with st.expander("📖 About this project"):
        st.markdown("""
        Built for a consumer-credit analytics role (JPMorgan Chase Quant Analytics style).
        Full pipeline: **ETL & feature engineering → K-means segmentation → default prediction with fair-lending check → Expected Loss & risk-adjusted return KPIs.**

        Dataset: *Default of Credit Card Clients* (Yeh & Lien, 2009), UCI ML Repository — 30,000 Taiwanese cardholders.
        Built with pandas, scikit-learn, and Streamlit.
        """)

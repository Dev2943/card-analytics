"""
app.py — Credit Card Portfolio Analytics (fast-loading demo)
Reads precomputed results.json (built at Docker build time) so the page
loads instantly — no live data fetch, no model training at runtime.
"""
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Card Portfolio Analytics", page_icon="💳", layout="wide")

st.title("💳 Credit Card Portfolio Analytics")
st.caption("Consumer-credit analytics: segmentation, default prediction, and risk-adjusted portfolio KPIs · UCI dataset (30,000 cardholders)")


@st.cache_data
def load_results():
    with open("results.json") as f:
        return json.load(f)


try:
    R = load_results()
except Exception:
    st.error("Precomputed results not found. The build step should generate results.json.")
    st.stop()

# ── Top KPIs ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cardholders", f"{R['total_customers']:,}")
c2.metric("Default Rate", f"{R['default_rate']:.1%}")
c3.metric("Portfolio Expected Loss", f"NT${R['total_expected_loss']:,.0f}")
c4.metric("Monthly Revenue", f"NT${R['total_revenue']:,.0f}")

st.divider()
tab1, tab2, tab3 = st.tabs(["📊 Segmentation", "🎯 Default Model", "💰 Portfolio KPIs"])

with tab1:
    st.subheader("Behavioral Customer Segments (K-Means)")
    st.write("Customers clustered on utilization, payment ratio, credit limit, delinquency, and spend — then named by behavior.")
    prof = pd.DataFrame(R["profile"])
    st.dataframe(prof, use_container_width=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.plot(R["k_values"], R["inertias"], "o-", color="#1a73e8")
    ax1.set_title("Elbow Method"); ax1.set_xlabel("k"); ax1.set_ylabel("Inertia"); ax1.grid(alpha=0.3)
    ax2.plot(R["k_values"], R["silhouettes"], "o-", color="#2a9d8f")
    ax2.set_title("Silhouette Score"); ax2.set_xlabel("k"); ax2.grid(alpha=0.3)
    st.pyplot(fig)

with tab2:
    st.subheader("Default Prediction — Logistic Regression vs Gradient Boosting")
    m1, m2 = st.columns(2)
    m1.metric("Logistic Regression ROC-AUC", f"{R['auc_lr']:.3f}")
    m2.metric("Gradient Boosting ROC-AUC", f"{R['auc_gb']:.3f}")
    st.markdown("""
    **Why two models?** Logistic regression is the *interpretable, regulator-friendly* baseline used for actual credit decisions (ECOA compliance). Gradient boosting benchmarks the performance ceiling.

    The model **excludes protected attributes** (sex, marriage) and is checked for **disparate impact** using the four-fifths rule — standard fair-lending practice.
    """)
    st.caption(f"Imbalanced problem: only {R['default_rate']:.0%} default, so ROC-AUC / PR-AUC are used instead of accuracy.")

with tab3:
    st.subheader("Risk-Adjusted Portfolio KPIs by Segment")
    st.write("Expected Loss = PD × EAD × LGD. Risk-adjusted contribution = revenue − expected loss.")
    kpis = pd.DataFrame(R["kpis"])
    st.dataframe(kpis, use_container_width=True)
    st.caption("This table is the strategic dashboard's data layer — it turns predictions into portfolio decisions.")

st.divider()
with st.expander("📖 About this project"):
    st.markdown("""
    Built for a consumer-credit analytics role (JPMorgan Chase Quant Analytics style).
    Full pipeline: **ETL & feature engineering → K-means segmentation → default prediction with fair-lending check → Expected Loss & risk-adjusted return KPIs.**

    Dataset: *Default of Credit Card Clients* (Yeh & Lien, 2009), UCI ML Repository — 30,000 Taiwanese cardholders.
    Results are precomputed at build time for instant loading. Built with pandas, scikit-learn, and Streamlit.
    """)

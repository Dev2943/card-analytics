"""
Day 4: Churn Risk & Portfolio KPIs.

Builds a dormancy-based churn-risk proxy, computes Expected Loss
(PD x EAD x LGD), a revenue proxy, and risk-adjusted return, then rolls
everything into a segment-level KPI summary for the dashboard.


Run with: python3 portfolio_kpis.py
Produces: data/cards_kpis.parquet, segment_kpis.csv
"""

import numpy as np
import pandas as pd


# Loss Given Default assumption for unsecured credit card debt (~75% not recovered)
LGD = 0.75


def load_data():
    """Load the scored test set from Day 3 (has pred_prob_default)."""
    df = pd.read_parquet("data/cards_scored.parquet")
    print(f"  Loaded {len(df):,} scored cardholders")
    return df


# ----------------------------- CHURN PROXY -----------------------------
def build_churn_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Flag accounts at risk of attrition using dormancy signals.

    Dormancy = very low recent activity: near-zero recent bills AND near-zero
    recent payments. These accounts generate no interchange and no interest —
    economically churned, prime attrition-risk candidates.
    """
    df = df.copy()

    # Recent activity signals (BILL_AMT1 = most recent month)
    recent_bill = df["BILL_AMT1"].abs()
    recent_payment = df["PAY_AMT1"].abs()

    # Build churn_risk flag (1 = at risk of attrition, 0 = active).
    # Define dormancy as: recent bill very low AND recent payment very low.
    # Use a small threshold (e.g., 500 NT dollars) for "very low".
    
    dormant = (recent_bill < 500) & (recent_payment < 500)
    df["churn_risk"] = dormant.astype(int)
    
    return df


# ----------------------------- EXPECTED LOSS -----------------------------
def compute_expected_loss(df: pd.DataFrame) -> pd.DataFrame:
    """Expected Loss = PD x EAD x LGD per customer."""
    df = df.copy()

    # PD = predicted probability of default (from Day 3 model)
    pd_default = df["pred_prob_default"]

    # EAD = Exposure at Default. Proxy with the most recent bill (current balance),
    # floored at 0 (can't have negative exposure).
    ead = df["BILL_AMT1"].clip(lower=0)


    # Expected Loss = PD * EAD * LGD.
    df["expected_loss"] = pd_default * ead * LGD
    df["ead"] = ead
    return df


# ----------------------------- REVENUE PROXY -----------------------------
def compute_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Simple revenue proxy: interchange on spend + interest on carried balance.

    - Interchange: ~2% of spend. We proxy monthly spend with the payment amount
      (what they paid ~ what they charged for a transactor).
    - Interest: ~18% APR on the carried balance (monthly ~1.5%), only on the
      portion of the bill not paid off (the revolving balance).
    """
    df = df.copy()

    monthly_spend = df["PAY_AMT1"].clip(lower=0)
    carried_balance = (df["BILL_AMT1"] - df["PAY_AMT1"]).clip(lower=0)

    interchange = 0.02 * monthly_spend
    monthly_interest = 0.015 * carried_balance

    # Monthly revenue proxy = interchange + interest.
    df["monthly_revenue"] = interchange + monthly_interest
    
    # Risk-adjusted monthly contribution: revenue minus (expected loss spread
    # over 12 months, since EL is an annual-ish figure and revenue is monthly).
    df["risk_adj_contribution"] = df["monthly_revenue"] - df["expected_loss"] / 12.0

    return df


# ----------------------------- SEGMENT KPI ROLL-UP -----------------------------
def segment_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Roll up to a segment-level KPI table for the dashboard."""

    # Group by segment_name and aggregate the KPIs.
    # For each segment compute: size, actual default rate, mean predicted PD,
    # total expected loss, total monthly revenue, total risk-adjusted contribution,
    # and churn-risk rate.
    
    kpis = df.groupby("segment_name").agg(
        customers=("default", "size"),
        actual_default_rate=("default", "mean"),
        avg_pd=("pred_prob_default", "mean"),
        total_expected_loss=("expected_loss", "sum"),
        total_monthly_revenue=("monthly_revenue", "sum"),
        total_risk_adj_contribution=("risk_adj_contribution", "sum"),
        churn_risk_rate=("churn_risk", "mean"),
    )
    return kpis


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    print("Loading scored data...")
    df = load_data()

    print("\nBuilding churn proxy...")
    df = build_churn_proxy(df)
    print(f"  Churn-risk (dormant) accounts: {df['churn_risk'].sum():,} "
          f"({df['churn_risk'].mean():.1%})")

    print("\nComputing expected loss...")
    df = compute_expected_loss(df)
    print(f"  Total portfolio expected loss: "
          f"{df['expected_loss'].sum():,.0f} NT$")
    print(f"  Average EL per customer: {df['expected_loss'].mean():,.0f} NT$")

    print("\nComputing revenue & risk-adjusted return...")
    df = compute_revenue(df)
    print(f"  Total monthly revenue: {df['monthly_revenue'].sum():,.0f} NT$")
    print(f"  Total risk-adjusted monthly contribution: "
          f"{df['risk_adj_contribution'].sum():,.0f} NT$")

    print("\n" + "=" * 90)
    print("SEGMENT-LEVEL KPI SUMMARY")
    print("=" * 90)
    kpis = segment_kpis(df)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(kpis.to_string())

    kpis.to_csv("segment_kpis.csv")
    df.to_parquet("data/cards_kpis.parquet")
    print("\nSaved segment_kpis.csv and data/cards_kpis.parquet")

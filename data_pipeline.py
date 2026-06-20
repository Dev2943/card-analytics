"""
Day 1: Data Pipeline — ETL & Feature Engineering.

Loads the UCI Default of Credit Card Clients dataset (30,000 cardholders),
cleans the undocumented category codes with documented decisions, and engineers
the behavioral features that drive segmentation (Day 2), default prediction
(Day 3), and churn/portfolio analysis (Day 4).


Run with: python3 data_pipeline.py
Produces: data/cards_clean.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------- LOAD -----------------------------
def load_raw() -> pd.DataFrame:
    """Load the UCI dataset via the ucimlrepo package (no manual download)."""
    from ucimlrepo import fetch_ucirepo

    print("  Fetching dataset from UCI (id=350)...")
    ds = fetch_ucirepo(id=350)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()

    # This version returns columns as X1..X23 — map to descriptive names.
    if "X1" in X.columns:
        name_map = {
            "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION",
            "X4": "MARRIAGE", "X5": "AGE",
            "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3",
            "X9": "PAY_4", "X10": "PAY_5", "X11": "PAY_6",
            "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3",
            "X15": "BILL_AMT4", "X16": "BILL_AMT5", "X17": "BILL_AMT6",
            "X18": "PAY_AMT1", "X19": "PAY_AMT2", "X20": "PAY_AMT3",
            "X21": "PAY_AMT4", "X22": "PAY_AMT5", "X23": "PAY_AMT6",
        }
        X = X.rename(columns=name_map)

    df = X.copy()
    target_col = y.columns[0]
    df["default"] = y[target_col].values

    return df


# ----------------------------- CLEAN -----------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean undocumented category codes with documented decisions.

    Decisions (each is defensible and should be documented):
      - EDUCATION: codes 0, 5, 6 are undocumented -> collapse into 4 ("others")
      - MARRIAGE:  code 0 is undocumented        -> collapse into 3 ("others")
      - PAY_*:     codes -2, -1, 0 all mean "not delinquent"; 1+ means months late.
                   We keep the raw values but will derive clean delinquency features.
    """
    df = df.copy()

    # EDUCATION: collapse 0, 5, 6 -> 4
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE: collapse 0 -> 3
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # Sanity: AGE should be positive and reasonable
    df = df[(df["AGE"] > 0) & (df["AGE"] < 120)]

    return df


# ----------------------------- FEATURE ENGINEERING -----------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer behavioral features that capture how a customer uses their card."""
    df = df.copy()

    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    pay_status_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    # Average bill and average payment over the 6 months
    df["avg_bill"] = df[bill_cols].mean(axis=1)
    df["avg_payment"] = df[pay_amt_cols].mean(axis=1)

    # Credit utilization = average bill / credit limit.
    # The single most predictive consumer-credit feature.
    # Guard against division by zero (some LIMIT_BAL could be 0 — use a small floor).
   
    df["utilization"] = df["avg_bill"] / df["LIMIT_BAL"].clip(lower=1)
    df["utilization"] = df["utilization"].clip(lower=0, upper=5)


    # Payment ratio = average payment / average bill.
    # How much of their bill do they actually pay each month?
    # A ratio >= ~1 means they pay in full (transactor); near 0 means minimum payer (revolver).
    # Guard against division by zero (zero bills).
    
    df["payment_ratio"] = df["avg_payment"] / df["avg_bill"].clip(lower=1)
    df["payment_ratio"] = df["payment_ratio"].clip(lower=0, upper=2)

    # Delinquency features from the PAY_* status columns.
    # Recall: -2,-1,0 = not delinquent; 1,2,3... = months late.
    pay_status = df[pay_status_cols]

    # Max delinquency across the 6 months (worst lateness)
    df["max_delinquency"] = pay_status.max(axis=1).clip(lower=0)

    # Most recent delinquency (PAY_0 is the latest month, September)
    df["recent_delinquency"] = df["PAY_0"].clip(lower=0)

    # Months_delinquent = count of months where PAY_* status >= 1.
    # Captures chronic vs. one-off lateness.
    
    df["months_delinquent"] = (pay_status >= 1).sum(axis=1)


    # Bill trend: is the balance growing? Compare recent 3 months vs. older 3 months.
    # BILL_AMT1 is most recent (Sept), BILL_AMT6 is oldest (April).
    recent_bills = df[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3"]].mean(axis=1)
    older_bills = df[["BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]].mean(axis=1)
    df["bill_trend"] = recent_bills - older_bills  # positive = growing balance

    # Transactor vs. revolver flag.
    # Transactor: pays most of bill (payment_ratio high) AND carries low balance.
    # We define a simple rule: payment_ratio >= 0.9 -> transactor (pays in full).
    df["is_transactor"] = (df["payment_ratio"] >= 0.9).astype(int)

    # Has any balance at all (active account)
    df["has_balance"] = (df["avg_bill"] > 0).astype(int)

    return df


# ----------------------------- SUMMARY -----------------------------
def summarize(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"Total cardholders: {len(df):,}")
    print(f"Overall default rate: {df['default'].mean():.1%}")

    print("\nDefault rate by credit utilization band:")
    df["util_band"] = pd.cut(df["utilization"],
                             bins=[-0.01, 0.2, 0.5, 0.8, 5],
                             labels=["Low (<20%)", "Med (20-50%)",
                                     "High (50-80%)", "Very High (>80%)"])
    print(df.groupby("util_band", observed=True)["default"].agg(["mean", "count"]).round(3).to_string())

    print("\nDefault rate: transactors vs. revolvers:")
    summary = df.groupby("is_transactor")["default"].agg(["mean", "count"])
    summary.index = ["Revolver", "Transactor"]
    print(summary.round(3).to_string())

    print("\nDefault rate by recent delinquency (PAY_0 status):")
    print(df.groupby("recent_delinquency")["default"].agg(["mean", "count"]).round(3).to_string())

    print("\nEngineered feature ranges:")
    for col in ["utilization", "payment_ratio", "max_delinquency",
                "months_delinquent", "bill_trend"]:
        print(f"  {col:20s}: min={df[col].min():>10.2f}  "
              f"median={df[col].median():>10.2f}  max={df[col].max():>12.2f}")


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    print("Loading raw data...")
    raw = load_raw()
    print(f"  Loaded {len(raw):,} rows, {len(raw.columns)} columns")

    print("\nCleaning...")
    cleaned = clean(raw)
    print(f"  {len(cleaned):,} rows after cleaning")

    print("\nEngineering features...")
    featured = engineer_features(cleaned)

    summarize(featured)

    featured.to_parquet(out_dir / "cards_clean.parquet")
    print(f"\nSaved to data/cards_clean.parquet")

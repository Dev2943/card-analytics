"""
Day 2: Customer Segmentation via K-means.

Clusters the 30,000 cardholders into behavioral segments, chooses k via the
elbow method and silhouette score, profiles each segment, and names them for
portfolio strategy.


Run with: python3 segmentation.py
Produces: data/cards_segmented.parquet, segment_profile.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Features we cluster on — behavioral, NOT the default outcome
CLUSTER_FEATURES = [
    "utilization",
    "payment_ratio",
    "LIMIT_BAL",
    "months_delinquent",
    "avg_bill",
]


# ----------------------------- LOAD -----------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_parquet("data/cards_clean.parquet")
    print(f"  Loaded {len(df):,} cardholders")
    return df


# ----------------------------- SCALE -----------------------------
def scale_features(df: pd.DataFrame) -> np.ndarray:
    """Standardize the clustering features (z-score)."""
    X = df[CLUSTER_FEATURES].values

    # Standardize X using StandardScaler.
    # K-means uses Euclidean distance, so features must be on comparable scales.
    # Without this, LIMIT_BAL (hundreds of thousands) dominates utilization (0-5).
   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  

    return X_scaled


# ----------------------------- CHOOSE K -----------------------------
def choose_k(X_scaled: np.ndarray, k_range=range(2, 9)):
    """Compute inertia (elbow) and silhouette score across candidate k values."""
    inertias = []
    silhouettes = []

    for k in k_range:
        # Fit KMeans for this k, record inertia and silhouette score.
        # Use random_state=42 and n_init=10 for reproducibility.
        
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        

    return list(k_range), inertias, silhouettes


def plot_k_selection(k_values, inertias, silhouettes, out_path="segment_k_selection.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(k_values, inertias, "o-", color="#2a4d8f", linewidth=2)
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia (within-cluster sum of squares)")
    ax1.set_title("Elbow Method")
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_values, silhouettes, "o-", color="#c44e52", linewidth=2)
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette Analysis (higher = better separation)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ----------------------------- FIT FINAL -----------------------------
def fit_segments(df: pd.DataFrame, X_scaled: np.ndarray, k: int) -> pd.DataFrame:
    """Fit final K-means with chosen k and attach segment labels."""
    df = df.copy()

    # Fit final KMeans with the chosen k and assign labels to df["segment"].
    
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["segment"] = km.fit_predict(X_scaled)

    return df


# ----------------------------- PROFILE -----------------------------
def profile_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Build a profile table: per-segment averages, size, and default rate."""

    #  Group by segment and compute the profile.
    # For each segment, compute the mean of the cluster features, the count,
    # and the mean default rate.
    
    profile = df.groupby("segment").agg(

           size=("default", "size"),
           default_rate=("default", "mean"),
           avg_utilization=("utilization", "mean"),
           avg_payment_ratio=("payment_ratio", "mean"),
           avg_limit=("LIMIT_BAL", "mean"),
           avg_months_delinquent=("months_delinquent", "mean"),
           avg_bill=("avg_bill", "mean"),
    )
    

    return profile


def suggest_names(profile: pd.DataFrame) -> dict:
    """Heuristic naming based on segment profiles. Adjust after seeing real numbers."""
    names = {}
    for seg, row in profile.iterrows():
        util = row["avg_utilization"]
        pay = row["avg_payment_ratio"]
        delinq = row["avg_months_delinquent"]
        limit = row["avg_limit"]

        if pay >= 0.7 and util < 0.3:
            name = "Transactors (pay-in-full)"
        elif delinq >= 1.0 or row["default_rate"] > 0.30:
            name = "Stressed / At-Risk"
        elif util >= 0.6:
            name = "High-Utilization Revolvers"
        elif util < 0.15 and row["avg_bill"] < profile["avg_bill"].median():
            name = "Dormant / Low-Engagement"
        else:
            name = "Healthy Revolvers"
        names[seg] = name
    return names


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("\nScaling features...")
    X_scaled = scale_features(df)

    print("\nChoosing k (elbow + silhouette)...")
    k_values, inertias, silhouettes = choose_k(X_scaled)
    for k, inr, sil in zip(k_values, inertias, silhouettes):
        print(f"  k={k}: inertia={inr:>12.0f}  silhouette={sil:.4f}")
    plot_k_selection(k_values, inertias, silhouettes)

    # Choose k: highest silhouette is a good default starting point.
    best_k = k_values[int(np.argmax(silhouettes))]
    print(f"\n  Best silhouette at k={best_k}")
    # You may override best_k for business interpretability — try 5 if close.
    CHOSEN_K = best_k

    print(f"\nFitting final segmentation with k={CHOSEN_K}...")
    segmented = fit_segments(df, X_scaled, CHOSEN_K)

    print("\nSegment profiles:")
    profile = profile_segments(segmented)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(profile.round(3).to_string())

    names = suggest_names(profile)
    print("\nSuggested segment names:")
    for seg, name in names.items():
        size = int(profile.loc[seg, "size"])
        dr = profile.loc[seg, "default_rate"]
        print(f"  Segment {seg}: {name:32s}  (n={size:,}, default={dr:.1%})")

    segmented["segment_name"] = segmented["segment"].map(names)
    segmented.to_parquet("data/cards_segmented.parquet")
    print("\nSaved data/cards_segmented.parquet")

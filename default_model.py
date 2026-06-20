"""
Day 3: Default Prediction Model.

Trains an interpretable logistic regression baseline and a gradient-boosting
challenger, evaluates both for imbalanced data (ROC-AUC, PR-AUC, calibration,
confusion matrix), checks feature importance, and runs a fair-lending
disparate-impact test.


Run with: python3 default_model.py
Produces: data/cards_scored.parquet, model_evaluation.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report,
)
from sklearn.calibration import calibration_curve


# Features to train on — behavioral + engineered.
# DELIBERATELY EXCLUDE protected attributes (SEX, MARRIAGE) and the outcome (default).
MODEL_FEATURES = [
    "LIMIT_BAL", "AGE", "EDUCATION",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "utilization", "payment_ratio", "max_delinquency",
    "recent_delinquency", "months_delinquent", "bill_trend",
    "avg_bill", "avg_payment",
]


def load_data():
    df = pd.read_parquet("data/cards_segmented.parquet")
    print(f"  Loaded {len(df):,} cardholders, default rate {df['default'].mean():.1%}")
    return df


# ----------------------------- SPLIT -----------------------------
def split_data(df):
    """Train/test split, stratified on the (imbalanced) target."""
    X = df[MODEL_FEATURES].values
    y = df["default"].values

    # Stratified train/test split (75/25), random_state=42.
    # Stratify on y so both sets keep the ~22% default rate.
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
   

    return X_train, X_test, y_train, y_test


# ----------------------------- TRAIN -----------------------------
def train_models(X_train, y_train):
    """Train logistic regression (scaled) and gradient boosting."""
    # Logistic regression needs scaled features; tree models don't.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train logistic regression and gradient boosting.
    # Logistic: use class_weight="balanced" to handle imbalance, max_iter=1000.
    # Boosting: GradientBoostingClassifier with random_state=42.
    
    logreg = LogisticRegression(class_weight="balanced", max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)   # trees don't need scaling
    
    return logreg, gb, scaler


# ----------------------------- EVALUATE -----------------------------
def evaluate(name, y_test, y_prob, threshold=0.5):
    """Compute imbalanced-classification metrics from predicted probabilities."""
    # Compute ROC-AUC and PR-AUC (average precision) from y_prob.
   
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  {name}")
    print(f"    ROC-AUC: {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}")
    print(f"    Confusion matrix (threshold={threshold}):")
    print(f"      TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"      FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")
    tpr = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0
    print(f"    Recall (caught defaulters): {tpr:.1%}")

    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "cm": cm}


# ----------------------------- FAIR LENDING -----------------------------
def disparate_impact(df_test, y_prob, threshold=0.5):
    """Four-fifths rule check: does the model flag one sex group more than another?

    'Favorable outcome' = NOT flagged as default risk (i.e., would get credit).
    """
    flagged = (y_prob >= threshold).astype(int)
    favorable = 1 - flagged  # favorable = not flagged

    out = df_test.copy()
    out["favorable"] = favorable

    # SEX: 1 = male, 2 = female
    rate_by_sex = out.groupby("SEX")["favorable"].mean()

    # Compute the disparate impact ratio = min(rate)/max(rate).
    # The four-fifths rule: ratio < 0.80 is a red flag.

    di_ratio = rate_by_sex.min() / rate_by_sex.max()
    

    print("\n  FAIR-LENDING DISPARATE-IMPACT CHECK")
    print(f"    Favorable (credit-granted) rate by sex:")
    for sex_code, rate in rate_by_sex.items():
        label = "Male" if sex_code == 1 else "Female"
        print(f"      {label}: {rate:.1%}")
    print(f"    Disparate impact ratio: {di_ratio:.3f}")
    if di_ratio >= 0.80:
        print(f"    PASS — ratio >= 0.80 (four-fifths rule satisfied)")
    else:
        print(f"    FLAG — ratio < 0.80, warrants investigation")

    return di_ratio


# ----------------------------- FEATURE IMPORTANCE -----------------------------
def feature_importance(logreg, gb, scaler):
    """Report which features drive default for both models."""
    print("\n  FEATURE IMPORTANCE")

    #  Build a comparison of feature importance.
    # Logistic: absolute value of coefficients (on scaled features).
    # Boosting: gb.feature_importances_.
    # Print the top features for each, sorted descending.

    logreg_imp = pd.Series(np.abs(logreg.coef_[0]), index=MODEL_FEATURES)
    gb_imp = pd.Series(gb.feature_importances_, index=MODEL_FEATURES)
    print("    Top 6 by logistic |coef|:")
    print(logreg_imp.sort_values(ascending=False).head(6).round(3).to_string())
    print("    Top 6 by gradient boosting importance:")
    print(gb_imp.sort_values(ascending=False).head(6).round(3).to_string())
    


def plot_evaluation(y_test, prob_lr, prob_gb, out_path="model_evaluation.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ROC curves
    for name, prob, color in [("Logistic", prob_lr, "#2a4d8f"),
                               ("Gradient Boosting", prob_gb, "#c44e52")]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # PR curves
    for name, prob, color in [("Logistic", prob_lr, "#2a4d8f"),
                               ("Gradient Boosting", prob_gb, "#c44e52")]:
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, linewidth=2)
    axes[1].axhline(y_test.mean(), color="gray", linestyle="--", alpha=0.5,
                    label=f"Base rate ({y_test.mean():.2f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend(); axes[1].grid(alpha=0.3)

    # Calibration
    for name, prob, color in [("Logistic", prob_lr, "#2a4d8f"),
                               ("Gradient Boosting", prob_gb, "#c44e52")]:
        frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)
        axes[2].plot(mean_pred, frac_pos, "o-", label=name, color=color, linewidth=2)
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    axes[2].set_xlabel("Mean predicted probability"); axes[2].set_ylabel("Observed frequency")
    axes[2].set_title("Calibration Curve"); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved {out_path}")


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("\nSplitting...")
    # We need the test-set rows (with SEX) for the fairness check, so split indices too.
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.25, random_state=42, stratify=df["default"])
    df_train, df_test = df.loc[train_idx], df.loc[test_idx]

    X_train = df_train[MODEL_FEATURES].values
    X_test = df_test[MODEL_FEATURES].values
    y_train = df_train["default"].values
    y_test = df_test["default"].values

    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    print("\nTraining models...")
    logreg, gb, scaler = train_models(X_train, y_train)

    # Predicted probabilities on the test set
    prob_lr = logreg.predict_proba(scaler.transform(X_test))[:, 1]
    prob_gb = gb.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    evaluate("Logistic Regression", y_test, prob_lr)
    evaluate("Gradient Boosting", y_test, prob_gb)

    feature_importance(logreg, gb, scaler)

    # Fairness check on the better-ranking model (gradient boosting)
    disparate_impact(df_test, prob_gb)

    plot_evaluation(y_test, prob_lr, prob_gb)

    # Save scored test set
    df_test = df_test.copy()
    df_test["pred_prob_default"] = prob_gb
    df_test.to_parquet("data/cards_scored.parquet")
    print("\nSaved data/cards_scored.parquet")

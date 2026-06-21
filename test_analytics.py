"""
Test suite for the Credit Card Portfolio Analytics Suite.

Covers data integrity, feature engineering, segmentation sanity,
model evaluation, and KPI math (expected loss, risk-adjusted return).

Run with: pytest test_analytics.py -v
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# ----------------------------- FIXTURES -----------------------------
@pytest.fixture(scope="module")
def clean_df():
    path = Path("data/cards_clean.parquet")
    if not path.exists():
        pytest.skip("Run data_pipeline.py first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def segmented_df():
    path = Path("data/cards_segmented.parquet")
    if not path.exists():
        pytest.skip("Run segmentation.py first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def kpi_df():
    path = Path("data/cards_kpis.parquet")
    if not path.exists():
        pytest.skip("Run portfolio_kpis.py first")
    return pd.read_parquet(path)


# ----------------------------- DATA INTEGRITY -----------------------------
def test_row_count(clean_df):
    """Dataset should have ~30,000 cardholders."""
    assert 29000 <= len(clean_df) <= 30000


def test_default_rate_reasonable(clean_df):
    """Overall default rate should be ~22% (known for this dataset)."""
    rate = clean_df["default"].mean()
    assert 0.20 <= rate <= 0.24, f"Default rate {rate} outside expected range"


def test_education_codes_collapsed(clean_df):
    """EDUCATION should only contain documented codes 1-4 after cleaning."""
    assert set(clean_df["EDUCATION"].unique()).issubset({1, 2, 3, 4})


def test_marriage_codes_collapsed(clean_df):
    """MARRIAGE should only contain documented codes 1-3 after cleaning."""
    assert set(clean_df["MARRIAGE"].unique()).issubset({1, 2, 3})


# ----------------------------- FEATURE ENGINEERING -----------------------------
def test_utilization_bounds(clean_df):
    """Utilization should be clipped to [0, 5]."""
    assert clean_df["utilization"].min() >= 0
    assert clean_df["utilization"].max() <= 5.0


def test_payment_ratio_bounds(clean_df):
    """Payment ratio should be clipped to [0, 2]."""
    assert clean_df["payment_ratio"].min() >= 0
    assert clean_df["payment_ratio"].max() <= 2.0


def test_months_delinquent_range(clean_df):
    """months_delinquent counts over 6 months, so 0-6."""
    assert clean_df["months_delinquent"].min() >= 0
    assert clean_df["months_delinquent"].max() <= 6


def test_high_utilization_defaults_more(clean_df):
    """Economic sanity: high-utilization customers default more than low."""
    low = clean_df[clean_df["utilization"] < 0.2]["default"].mean()
    high = clean_df[clean_df["utilization"] > 0.8]["default"].mean()
    assert high > low, "High utilization should have higher default rate"


def test_recent_delinquency_predicts_default(clean_df):
    """Economic sanity: recently delinquent customers default far more."""
    current = clean_df[clean_df["recent_delinquency"] == 0]["default"].mean()
    late = clean_df[clean_df["recent_delinquency"] >= 2]["default"].mean()
    assert late > 2 * current, "2+ months late should default >2x the current group"


# ----------------------------- SEGMENTATION -----------------------------
def test_segments_exist(segmented_df):
    """Segmentation should produce multiple named segments."""
    assert "segment_name" in segmented_df.columns
    assert segmented_df["segment_name"].nunique() >= 3


def test_segments_have_distinct_risk(segmented_df):
    """Segments should differ meaningfully in default rate (validation of clustering)."""
    rates = segmented_df.groupby("segment_name")["default"].mean()
    assert rates.max() - rates.min() > 0.20, "Segments should have distinct risk profiles"


def test_stressed_segment_high_default(segmented_df):
    """There should be a high-risk segment with notably elevated default rate."""
    rates = segmented_df.groupby("segment_name")["default"].mean()
    assert rates.max() > 0.40, "Expected a stressed segment with >40% default"


# ----------------------------- KPI MATH -----------------------------
def test_expected_loss_nonnegative(kpi_df):
    """Expected loss = PD x EAD x LGD, all nonnegative, so EL >= 0."""
    assert (kpi_df["expected_loss"] >= 0).all()


def test_expected_loss_formula(kpi_df):
    """Spot-check EL = PD x EAD x LGD on a sample row."""
    LGD = 0.75
    row = kpi_df.iloc[0]
    expected = row["pred_prob_default"] * row["ead"] * LGD
    assert abs(row["expected_loss"] - expected) < 1.0


def test_churn_flag_binary(kpi_df):
    """Churn risk flag should be 0/1."""
    assert set(kpi_df["churn_risk"].unique()).issubset({0, 1})


def test_risk_adjusted_below_revenue(kpi_df):
    """Risk-adjusted contribution must be <= revenue (we subtract expected loss)."""
    assert (kpi_df["risk_adj_contribution"] <= kpi_df["monthly_revenue"] + 1e-6).all()


def test_pd_is_probability(kpi_df):
    """Predicted default probabilities must be in [0, 1]."""
    assert kpi_df["pred_prob_default"].min() >= 0
    assert kpi_df["pred_prob_default"].max() <= 1

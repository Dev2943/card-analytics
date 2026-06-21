# Strategic Recommendations: Credit Card Portfolio
### Prepared for Senior Management — Portfolio Analytics

**Scope:** Behavioral segmentation, default-risk modeling, and risk-adjusted profitability analysis across a 7,500-account held-out portfolio (UCI Default of Credit Card Clients, Taiwan, 2005). Default probabilities are model-scored; expected loss follows the PD × EAD × LGD framework with a 75% LGD assumption for unsecured card debt.

---

## Executive summary

The portfolio nets approximately **NT$982K in risk-adjusted monthly contribution** — but this figure conceals a sharp divergence across segments. A single segment, **Stressed / At-Risk** (11% of accounts), destroys **NT$891K of value per month** on a risk-adjusted basis, nearly offsetting the contribution of the entire rest of the book. The central recommendation is to **remediate or contain the stressed segment**, which would nearly double risk-adjusted profitability, while **protecting and growing the transactor and premium-revolver segments** that drive efficient, low-risk return.

---

## The four segments

Behavioral K-means segmentation (on utilization, payment ratio, credit limit, delinquency, and spend) produced four economically distinct groups, validated by a wide spread in default rates (16%–60%):

**1. Stressed / At-Risk — 11% of accounts, 60% default rate.**
Chronic delinquency (4.6 months late on average), near-zero payment ratios. Generates NT$583K monthly revenue but **−NT$891K risk-adjusted** once expected losses are netted. This segment is the portfolio's primary source of credit loss.

**2. Healthy Revolvers — 40% of accounts, 19% default rate.**
The core interest-revenue engine. Moderate utilization, carries balances, pays them down. +NT$674K risk-adjusted — solid and stable.

**3. High-Utilization Revolvers — 11% of accounts, 17% default rate.**
High balances and limits, high spend, but low delinquency — affluent customers who use substantial credit and pay reliably. Highest revenue and the strongest risk-adjusted contribution (+NT$721K). A profitable, low-risk segment to protect.

**4. Transactors — 38% of accounts, 16% default rate.**
Pay in full, minimal carried balance, lowest expected loss in the book. Lower absolute revenue but the **most risk-efficient** segment — retains ~72% of revenue as risk-adjusted contribution versus ~26% for the revolver segments. Notably, this segment also has the **highest churn risk (20% dormant)**.

---

## Recommendations

**1. Contain the Stressed / At-Risk segment (highest priority).**
This segment is value-destroying after risk. Actions: freeze credit-line increases, deploy proactive hardship and restructuring outreach to reduce roll-to-default, and suppress all new-credit and balance-transfer marketing. Even a partial reduction in this segment's loss rate materially improves total portfolio profitability, given it nearly cancels the rest of the book's contribution.

**2. Protect and retain Transactors and High-Spend Premium accounts.**
These are the most risk-efficient relationships, yet transactors show the highest dormancy/churn risk. Actions: targeted retention and rewards programs to drive spend and prevent attrition. Losing a transactor means losing low-risk, fee-generating revenue — the cheapest profit in the portfolio.

**3. Grow the Healthy Revolver core selectively.**
Offer credit-line increases to the well-behaved subset (no recent delinquency, declining utilization) to expand interest revenue without materially raising risk.

**4. Address dormancy (9% of the portfolio).**
Run reactivation campaigns on dormant accounts where the expected reactivation value exceeds campaign cost; allow controlled attrition elsewhere.

---

## Model governance note

The default model was built for regulatory compliance: protected attributes (sex, marital status) were excluded from training, and predictions were tested for disparate impact using the four-fifths rule — the model passed with a 0.996 ratio (near-perfect parity by sex). An interpretable logistic regression is recommended for any customer-facing credit decision (explainability under ECOA), with the higher-performance gradient-boosting model reserved for portfolio monitoring. Recent and chronic delinquency were by far the most predictive features, consistent with established credit-scoring practice.

**Caveats.** Churn is proxied by account dormancy (no closure label was available); in production it should be validated against actual attrition outcomes. Revenue and LGD figures are illustrative assumptions. The dataset is a single 2005 Taiwan cross-section and would need recalibration to a current US portfolio.

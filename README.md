# Credit Risk Probability Model for Alternative Data

**10 Academy — KAIM 9 | Week 4 Challenge**
**Bati Bank × eCommerce Buy-Now-Pay-Later Credit Scoring**

---

## Project Overview

You are an Analytics Engineer at **Bati Bank**, tasked with building an end-to-end Credit Scoring Model for a buy-now-pay-later (BNPL) partnership with an eCommerce platform. The model transforms raw customer transaction data into a **risk probability score** that drives real-time loan approvals, credit limits, and loan terms.

The pipeline covers:
- Proxy target variable engineering (RFM-based K-Means clustering)
- Feature engineering with Weight of Evidence (WoE) and Information Value (IV)
- Model training, hyperparameter tuning, and MLflow experiment tracking
- Containerized REST API (FastAPI + Docker)
- CI/CD automation (GitHub Actions)

---

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml      # CI/CD pipeline
├── data/                          # add to .gitignore
│   ├── raw/                       # Raw data
│   └── processed/                 # Processed data for training
├── notebooks/
│   └── eda.ipynb                  # Exploratory analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Feature engineering
│   ├── train.py                   # Model training
│   ├── predict.py                 # Inference
│   └── api/
│       ├── main.py                # FastAPI application
│       └── pydantic_models.py     # Request/response schemas
├── tests/
│   └── test_data_processing.py    # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The **Basel II Capital Accord** (2004) requires financial institutions to hold regulatory capital proportional to their credit risk exposure. Under the **Advanced Internal Ratings-Based (A-IRB)** approach, banks may use their own internal models to estimate the three key risk parameters:

- **PD (Probability of Default):** The likelihood that a borrower will fail to repay.
- **LGD (Loss Given Default):** The estimated economic loss, as a percentage of exposure, if default occurs.
- **EAD (Exposure at Default):** The monetary exposure at the moment of default.

Because regulatory capital requirements are derived directly from these model outputs, Basel II imposes strict expectations around **model transparency, documentation, and auditability**. Specifically:

- **Interpretability is a regulatory requirement, not just a preference.** Regulators and internal risk committees must be able to understand *why* a model assigns a particular risk score to a borrower. A black-box ensemble that achieves high AUC but cannot explain individual predictions may be rejected outright during model validation — regardless of its predictive performance.

- **Documentation must justify every modeling choice.** Variable selection, treatment of missing data, the definition of "default," and the handling of class imbalance must all be recorded with business rationale. This protects the bank in supervisory reviews and stress-testing exercises.

- **Model risk management (MRM) requires ongoing monitoring.** Basel II mandates that models be periodically validated against new data and that performance degradation triggers recalibration. Interpretable models (e.g., Logistic Regression with WoE-encoded features) are far easier to monitor and recalibrate than complex models.

- **Scorecard explainability supports fair lending.** When a loan is declined, the institution may be legally required (depending on jurisdiction) to provide a reason. A scorecard built on WoE-encoded logistic regression produces a ranked list of contributing factors directly from model coefficients, satisfying this obligation naturally.

In summary, Basel II turns interpretability from a "nice-to-have" into a **compliance obligation**. For this project, all modeling choices — proxy variable design, feature encoding, model selection — are documented and justified with explicit reference to these regulatory constraints.

---

### 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

#### Why a Proxy Variable is Necessary

The raw Xente eCommerce dataset contains **no historical loan performance data** — there are no records of borrowers who received credit, made repayments, or defaulted. This is the fundamental challenge of building a credit model on **alternative data**: the behavioral signal exists, but the supervisory label (default / no-default) does not.

A **proxy variable** bridges this gap by using observable customer behavior as a *substitute signal* for creditworthiness. The rationale is grounded in established credit research: customers who are highly engaged (frequent, recent, high-value transactions) are empirically associated with lower default risk, while disengaged customers are associated with higher risk. We operationalize this through **RFM (Recency, Frequency, Monetary) analysis**:

- **Recency (R):** Days since the customer's most recent transaction. Low recency (recent activity) → lower risk.
- **Frequency (F):** Total number of transactions in the observation window. High frequency → lower risk.
- **Monetary (M):** Total transaction value. High monetary value → lower risk.

K-Means clustering on RFM features segments customers into behavioral groups, and the least-engaged cluster is labeled **`is_high_risk = 1`** — serving as our proxy for "likely to default."

#### Business Risks of Proxy-Based Prediction

Using a proxy introduces several risks that must be explicitly acknowledged:

| Risk | Description | Mitigation |
|---|---|---|
| **Label noise** | Disengaged customers are *not necessarily* defaulters. Some may be infrequent but reliable payers. | Document the assumption clearly; monitor false positive rates post-deployment. |
| **Proxy drift** | Customer behavior patterns change over time (e.g., seasonality, economic shocks). The proxy may stop correlating with true default. | Schedule periodic re-clustering; monitor cluster stability with each new data batch. |
| **Discrimination risk** | Behavioral proxies can inadvertently encode demographic or socioeconomic biases present in the transaction data. | Conduct fairness audits by protected attribute groups before deployment. |
| **Regulatory challenge** | Regulators may not accept a proxy-based model as equivalent to a model trained on verified default outcomes. | Position the model as a *risk prioritization tool*, not a final credit decision engine, until ground-truth labels become available. |
| **Self-fulfilling feedback** | If the model denies credit to flagged customers, those customers never get the chance to repay — making it impossible to validate whether the proxy was correct. | Implement a random holdout (e.g., 5%) of flagged customers who receive credit anyway, to generate ground-truth labels over time. |

**The proxy variable is a modeling assumption, not ground truth.** This project treats it as a pragmatic starting point, with the explicit intention of replacing it with verified default labels as Bati Bank accumulates loan performance data from the BNPL product.

---

### 3. What are the key trade-offs between a simple, interpretable model (e.g., Logistic Regression with WoE) and a high-performance model (e.g., Gradient Boosting) in a regulated financial context?

In most machine learning contexts, the dominant criterion for model selection is predictive performance (AUC, F1, etc.). In **regulated financial services**, the trade-off space is considerably richer:

| Dimension | Logistic Regression + WoE | Gradient Boosting (XGBoost / LightGBM) |
|---|---|---|
| **Predictive Performance** | Moderate — linear decision boundary may underfit complex patterns | High — captures non-linear interactions and feature dependencies |
| **Interpretability** | High — coefficients map directly to scorecard points; each feature's contribution is explicit | Low — individual predictions require post-hoc tools (SHAP, LIME); not natively auditable |
| **Regulatory Acceptance** | Strong — WoE scorecards are the industry standard accepted by regulators globally | Uncertain — regulators may require additional explainability documentation |
| **Adverse Action Notices** | Trivial — top contributing WoE bins are directly readable from the model | Requires SHAP values or surrogate models; adds implementation complexity |
| **Calibration** | Naturally well-calibrated probabilities | Tends to be poorly calibrated; requires Platt scaling or isotonic regression post-hoc |
| **Stability Under Distribution Shift** | More stable — fewer parameters, less prone to overfitting behavioral noise | More sensitive to data drift; requires closer monitoring and more frequent retraining |
| **Development & Validation Cost** | Low — fast to train, validate, and document | High — hyperparameter space is large; validation requires more resources |
| **Handling Missing Values** | Requires explicit imputation strategy | Native handling in tree-based implementations (e.g., LightGBM) |
| **Feature Engineering Dependency** | High — WoE encoding requires careful binning; garbage in, garbage out | Lower — trees can discover non-linear patterns without manual encoding |

#### Practical Recommendation for This Project

Given the **Basel II regulatory context** and the fact that this model will inform loan approvals at Bati Bank, the recommended approach is a **two-model strategy**:

1. **Primary model: Logistic Regression with WoE encoding** — deployed as the production scorecard. Interpretable, auditable, well-calibrated, and defensible to regulators. Forms the basis of the credit scorecard delivered to the loan origination team.

2. **Challenger model: Gradient Boosting (XGBoost or LightGBM)** — trained in parallel as a performance benchmark. If it significantly outperforms the logistic model, the performance gap informs whether more sophisticated feature engineering (rather than model complexity) could close the gap while preserving interpretability.

This approach satisfies both the **business need** (accurate risk scoring) and the **regulatory constraint** (interpretable, documented model), while leaving a clear upgrade path as the bank matures its model risk management function.

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/credit-risk-model.git
cd credit-risk-model

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the API

```bash
# Build and run with Docker Compose
docker-compose up --build

# The API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

## Running Tests

```bash
pytest tests/ -v
```

---

## Team

- Kerod
- Mahbubah
- Feven

## Key Dates

| Milestone | Date |
|---|---|
| Challenge Introduction | 28 May 2026 |
| Interim Submission (Tasks 1–2) | 31 May 2026, 8:00 PM UTC |
| Final Submission | 03 Jun 2026, 8:00 PM UTC |

---

## References

- [Basel II Capital Accord — FasterCapital](https://fastercapital.com/content/Basel-Accords--What-They-Are-and-How-They-Affect-Credit-Risk-Management.html)
- [Credit Scoring Approaches Guidelines — World Bank](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)
- [Credit Risk — Corporate Finance Institute](https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/)
- [Credit Scoring Through Predictive Modelling for Basel II — LinkedIn](https://www.linkedin.com/pulse/credit-scoring-through-predictive-modelling-basel-ii-using-odeneye/)
- [Weight of Evidence and Information Value — ListenData](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
- [Xente Challenge Dataset — Kaggle](https://www.kaggle.com/datasets/atwine/xente-challenge)

"""
src/data_processing.py
======================
Feature engineering pipeline for the Bati Bank Credit Risk Model.

Transforms raw Xente transaction data into a model-ready DataFrame via
a single, reproducible sklearn Pipeline.

Pipeline stages
---------------
1.  DropConstantColumns      – remove zero-variance columns (CountryCode, CurrencyCode)
2.  TemporalFeatureExtractor – hour, day, month, year from TransactionStartTime
3.  DebitFlagExtractor       – binary is_debit flag from sign of Amount
4.  CustomerAggregator       – per-customer RFM + statistical aggregates
5.  LogTransformer           – log1p on skewed numerical features
6.  CategoricalEncoder       – one-hot encoding of low-cardinality categoricals
7.  MissingValueImputer      – median imputation for numerics, 'Unknown' for objects
8.  FeatureScaler             – StandardScaler on all numerical columns
9.  WoEEncoder               – Weight of Evidence encoding (requires fitted target)

Usage
-----
    from src.data_processing import build_pipeline, run_pipeline

    # Full pipeline (fit + transform):
    X_ready, pipeline = run_pipeline("data/raw/data.csv", target_col="is_high_risk")

    # Or step by step:
    pipeline = build_pipeline(target_col="is_high_risk")
    X_ready  = pipeline.fit_transform(df, y=df["is_high_risk"])
"""

import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Drop constant / near-zero-variance columns
# ─────────────────────────────────────────────────────────────────────────────


class DropConstantColumns(BaseEstimator, TransformerMixin):
    """
    Remove columns whose unique-value count is 1 (constants) plus any
    extra columns explicitly named (e.g. ID cols, Amount after sign extraction).
    """

    DEFAULT_DROP = [
        # zero-variance confirmed in EDA
        "CountryCode",
        "CurrencyCode",
        # pure identifiers — no predictive signal
        "TransactionId",
        "BatchId",
        "AccountId",
        "SubscriptionId",
        # Amount replaced by Value + is_debit flag
        "Amount",
        # raw timestamp handled in TemporalFeatureExtractor
        "TransactionStartTime",
    ]

    def __init__(self, extra_drop=None):
        self.extra_drop = extra_drop or []
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        constant = [c for c in X.columns if X[c].nunique() <= 1]
        explicit = [c for c in self.DEFAULT_DROP + self.extra_drop if c in X.columns]
        self.cols_to_drop_ = list(set(constant + explicit))
        log.info(
            "DropConstantColumns: removing %d columns → %s",
            len(self.cols_to_drop_),
            self.cols_to_drop_,
        )
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.cols_to_drop_, errors="ignore").copy()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Extract temporal features from TransactionStartTime
# ─────────────────────────────────────────────────────────────────────────────


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Parse TransactionStartTime and derive:
        tx_hour, tx_day, tx_month, tx_year, tx_day_of_week
    Original timestamp column is dropped after extraction.
    """

    TIMESTAMP_COL = "TransactionStartTime"

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if self.TIMESTAMP_COL not in X.columns:
            log.warning(
                "TemporalFeatureExtractor: '%s' not found, skipping.",
                self.TIMESTAMP_COL,
            )
            return X

        ts = pd.to_datetime(X[self.TIMESTAMP_COL], utc=True, errors="coerce")
        X["tx_hour"] = ts.dt.hour
        X["tx_day"] = ts.dt.day
        X["tx_month"] = ts.dt.month
        X["tx_year"] = ts.dt.year
        X["tx_day_of_week"] = ts.dt.dayofweek  # 0=Monday … 6=Sunday
        X.drop(columns=[self.TIMESTAMP_COL], inplace=True)

        log.info("TemporalFeatureExtractor: extracted 5 temporal features.")
        return X


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Debit / Credit flag from Amount sign
# ─────────────────────────────────────────────────────────────────────────────


class DebitFlagExtractor(BaseEstimator, TransformerMixin):
    """
    EDA showed 40% of Amount values are negative (credits back to customer).
    Encode this as:
        is_debit = 1  →  Amount >= 0  (customer paying)
        is_debit = 0  →  Amount <  0  (credit / refund)
    Amount is expected to be dropped in DropConstantColumns; this step
    operates on Amount *before* it is dropped, so ordering matters —
    run this BEFORE DropConstantColumns, or keep Amount until this step.

    If Amount is already absent, the transformer is a no-op.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if "Amount" in X.columns:
            X["is_debit"] = (X["Amount"] >= 0).astype(int)
            log.info("DebitFlagExtractor: created 'is_debit' flag.")
        return X


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Customer-level aggregate features (RFM + statistical)
# ─────────────────────────────────────────────────────────────────────────────


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Compute per-customer aggregates and merge them back to transaction level.

    Features produced
    -----------------
    cust_total_value      : total transaction value (monetary M in RFM)
    cust_avg_value        : average transaction value
    cust_tx_count         : number of transactions  (frequency F in RFM)
    cust_std_value        : standard deviation of Value (spending variability)
    cust_max_value        : maximum single transaction value
    cust_min_value        : minimum single transaction value
    cust_recency_days     : days since last transaction from snapshot date (recency R)
    cust_unique_products  : number of distinct products purchased
    cust_unique_channels  : number of distinct channels used
    """

    SNAPSHOT_DATE = pd.Timestamp("2019-02-13", tz="UTC")  # dataset max date

    def __init__(
        self,
        customer_col="CustomerId",
        value_col="Value",
        ts_col="TransactionStartTime",
    ):
        self.customer_col = customer_col
        self.value_col = value_col
        self.ts_col = ts_col
        self.agg_df_ = None

    def fit(self, X, y=None):
        df = X.copy()
        ts = pd.to_datetime(df[self.ts_col], utc=True, errors="coerce")

        agg = (
            df.groupby(self.customer_col)
            .agg(
                cust_total_value=(self.value_col, "sum"),
                cust_avg_value=(self.value_col, "mean"),
                cust_tx_count=(self.value_col, "count"),
                cust_std_value=(self.value_col, "std"),
                cust_max_value=(self.value_col, "max"),
                cust_min_value=(self.value_col, "min"),
                cust_unique_products=("ProductId", "nunique"),
                cust_unique_channels=("ChannelId", "nunique"),
            )
            .reset_index()
        )

        # Recency: days since last transaction
        recency_df = (
            df.assign(_ts=ts).groupby(self.customer_col)["_ts"].max().reset_index()
        )
        recency_df["cust_recency_days"] = (
            self.SNAPSHOT_DATE - recency_df["_ts"]
        ).dt.days

        agg = agg.merge(
            recency_df[[self.customer_col, "cust_recency_days"]],
            on=self.customer_col,
            how="left",
        )
        agg["cust_std_value"] = agg["cust_std_value"].fillna(0)
        self.agg_df_ = agg
        log.info(
            "CustomerAggregator: computed 9 customer-level features "
            "for %d customers.",
            len(agg),
        )
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X = X.merge(self.agg_df_, on=self.customer_col, how="left")
        log.info("CustomerAggregator: merged aggregate features.")
        return X


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Log-transform skewed numerical features
# ─────────────────────────────────────────────────────────────────────────────


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Apply numpy.log1p to specified columns (handles zeros safely).
    Defaults to Value and all cust_* columns derived above.
    """

    DEFAULT_COLS = [
        "Value",
        "cust_total_value",
        "cust_avg_value",
        "cust_max_value",
        "cust_std_value",
    ]

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        cols = self.columns if self.columns else self.DEFAULT_COLS
        cols = [c for c in cols if c in X.columns]
        for c in cols:
            X[c] = np.log1p(X[c].clip(lower=0))
        log.info("LogTransformer: log1p applied to %d columns → %s", len(cols), cols)
        return X


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Categorical encoding  (One-Hot + Label)
# ─────────────────────────────────────────────────────────────────────────────


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    One-hot encode low-cardinality categoricals.
    High-cardinality categoricals (ProductId) get label encoding.
    Columns not found in X are silently skipped.
    """

    OHE_COLS = ["ProductCategory", "ChannelId", "ProviderId", "PricingStrategy"]
    LABEL_COLS = ["ProductId"]

    def __init__(self, ohe_cols=None, label_cols=None):
        self.ohe_cols = ohe_cols
        self.label_cols = label_cols
        self.ohe_maps_ = {}
        self.label_maps_ = {}

    def fit(self, X, y=None):
        ohe_cols = self.ohe_cols if self.ohe_cols else self.OHE_COLS
        label_cols = self.label_cols if self.label_cols else self.LABEL_COLS

        for col in ohe_cols:
            if col in X.columns:
                self.ohe_maps_[col] = sorted(X[col].astype(str).unique())

        for col in label_cols:
            if col in X.columns:
                cats = sorted(X[col].astype(str).unique())
                self.label_maps_[col] = {v: i for i, v in enumerate(cats)}

        log.info(
            "CategoricalEncoder: OHE cols=%s  Label cols=%s",
            list(self.ohe_maps_.keys()),
            list(self.label_maps_.keys()),
        )
        return self

    def transform(self, X, y=None):
        X = X.copy()

        # One-hot encode
        for col, categories in self.ohe_maps_.items():
            if col not in X.columns:
                continue
            for cat in categories:
                X[f"{col}_{cat}"] = (X[col].astype(str) == cat).astype(int)
            X.drop(columns=[col], inplace=True)

        # Label encode
        for col, mapping in self.label_maps_.items():
            if col not in X.columns:
                continue
            X[col] = X[col].astype(str).map(mapping).fillna(-1).astype(int)

        return X


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Handle missing values
# ─────────────────────────────────────────────────────────────────────────────


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Numeric columns  → median imputation
    Object columns   → constant 'Unknown'
    """

    def __init__(self):
        self.num_imputer_ = None
        self.num_cols_ = []
        self.obj_cols_ = []

    def fit(self, X, y=None):
        self.num_cols_ = X.select_dtypes(include="number").columns.tolist()
        self.obj_cols_ = X.select_dtypes(include="object").columns.tolist()

        if self.num_cols_:
            self.num_imputer_ = SimpleImputer(strategy="median")
            self.num_imputer_.fit(X[self.num_cols_])

        missing_num = X[self.num_cols_].isnull().sum().sum()
        missing_obj = X[self.obj_cols_].isnull().sum().sum() if self.obj_cols_ else 0
        log.info(
            "MissingValueImputer: %d numeric NaN | %d object NaN",
            missing_num,
            missing_obj,
        )
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if self.num_cols_ and self.num_imputer_:
            X[self.num_cols_] = self.num_imputer_.transform(X[self.num_cols_])
        for col in self.obj_cols_:
            if col in X.columns:
                X[col] = X[col].fillna("Unknown")
        return X


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Normalize / Standardize numerical features
# ─────────────────────────────────────────────────────────────────────────────


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    StandardScaler on all numeric columns except binary flags and the target.
    Binary columns (0/1 range) and integer-encoded categoricals are excluded.
    """

    EXCLUDE = {"is_debit", "FraudResult", "is_high_risk"}

    def __init__(self, exclude=None):
        self.exclude = exclude or set()
        self.scale_cols_ = []
        self.scaler_ = StandardScaler()

    def fit(self, X, y=None):
        skip = self.EXCLUDE | set(self.exclude)
        num = X.select_dtypes(include="number").columns.tolist()
        # exclude binary (only 0/1) and explicitly excluded
        self.scale_cols_ = [c for c in num if c not in skip and X[c].nunique() > 2]
        if self.scale_cols_:
            self.scaler_.fit(X[self.scale_cols_])
        log.info("FeatureScaler: scaling %d columns.", len(self.scale_cols_))
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if self.scale_cols_:
            X[self.scale_cols_] = self.scaler_.transform(X[self.scale_cols_])
        return X


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Weight of Evidence (WoE) encoding
# ─────────────────────────────────────────────────────────────────────────────


class WoEEncoder(BaseEstimator, TransformerMixin):
    """
    Apply Weight of Evidence encoding to high-signal categorical columns using
    the xverse library (preferred) with a fallback to manual WoE calculation.

    WoE(bin) = ln(Distribution of Events / Distribution of Non-Events)
    IV       = Σ (Events% - Non-Events%) × WoE(bin)

    Features with IV < 0.02 are considered unpredictive and excluded.

    Parameters
    ----------
    target_col : str
        Binary target column (1 = high risk / event, 0 = low risk).
    woe_cols   : list[str]
        Columns to WoE-encode.  Defaults to the three highest-signal
        categoricals identified in EDA.
    iv_threshold : float
        Drop encoded features whose IV falls below this threshold.
    """

    DEFAULT_WOE_COLS = ["ProductCategory", "ChannelId", "ProviderId"]

    def __init__(self, target_col="is_high_risk", woe_cols=None, iv_threshold=0.02):
        self.target_col = target_col
        self.woe_cols = woe_cols or self.DEFAULT_WOE_COLS
        self.iv_threshold = iv_threshold
        self.woe_maps_ = {}
        self.iv_values_ = {}
        self.kept_cols_ = []

    # ── manual WoE calculation (fallback) ───────────────────
    @staticmethod
    def _compute_woe_iv(series, target):
        df = pd.DataFrame({"x": series.astype(str), "y": target})
        total_events = max(df["y"].sum(), 1)
        total_nonevents = max((df["y"] == 0).sum(), 1)

        woe_map = {}
        iv = 0.0
        for cat, grp in df.groupby("x"):
            events = grp["y"].sum()
            nonevents = (grp["y"] == 0).sum()
            dist_e = max(events, 0.5) / total_events
            dist_ne = max(nonevents, 0.5) / total_nonevents
            woe_val = np.log(dist_e / dist_ne)
            iv += (dist_e - dist_ne) * woe_val
            woe_map[cat] = woe_val
        return woe_map, iv

    def fit(self, X, y=None):
        # Resolve target: prefer y argument, fall back to column in X
        if y is not None:
            target = pd.Series(y).reset_index(drop=True)
        elif self.target_col in X.columns:
            target = X[self.target_col].reset_index(drop=True)
        else:
            log.warning(
                "WoEEncoder: target '%s' not found — skipping WoE.", self.target_col
            )
            return self

        # Try xverse first
        try:
            from xverse.transformer import WOE

            woe_cols_present = [c for c in self.woe_cols if c in X.columns]
            if not woe_cols_present:
                return self
            clf = WOE()
            clf.fit(X[woe_cols_present], target)
            self._xverse_clf = clf
            self._use_xverse = True
            log.info("WoEEncoder: xverse fitted on %s", woe_cols_present)
            # Extract IV from xverse
            for col in woe_cols_present:
                try:
                    self.iv_values_[col] = float(
                        clf.woe_df[clf.woe_df["Variable_Name"] == col]["IV"].iloc[0]
                    )
                except Exception:
                    self.iv_values_[col] = 0.0
            self.kept_cols_ = [
                c
                for c in woe_cols_present
                if self.iv_values_.get(c, 0) >= self.iv_threshold
            ]
            log.info(
                "WoEEncoder: IV values %s | keeping %s",
                self.iv_values_,
                self.kept_cols_,
            )
        except Exception as e:
            log.warning("WoEEncoder: xverse failed (%s) — using manual WoE.", e)
            self._use_xverse = False
            for col in self.woe_cols:
                if col not in X.columns:
                    continue
                woe_map, iv = self._compute_woe_iv(
                    X[col].reset_index(drop=True), target
                )
                self.woe_maps_[col] = woe_map
                self.iv_values_[col] = iv
                if iv >= self.iv_threshold:
                    self.kept_cols_.append(col)
            log.info(
                "WoEEncoder (manual): IV=%s | keeping %s",
                self.iv_values_,
                self.kept_cols_,
            )
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if not self.kept_cols_:
            return X

        if getattr(self, "_use_xverse", False):
            try:
                woe_cols_present = [c for c in self.kept_cols_ if c in X.columns]
                transformed = self._xverse_clf.transform(X[woe_cols_present])
                for col in woe_cols_present:
                    woe_col = (
                        f"{col}_WoE" if f"{col}_WoE" in transformed.columns else col
                    )
                    X[f"{col}_woe"] = transformed[woe_col].values
                    X.drop(columns=[col], errors="ignore", inplace=True)
            except Exception as e:
                log.warning("WoEEncoder xverse transform failed (%s) — skipping.", e)
        else:
            for col, woe_map in self.woe_maps_.items():
                if col not in X.columns:
                    continue
                X[f"{col}_woe"] = X[col].astype(str).map(woe_map).fillna(0)
                X.drop(columns=[col], inplace=True)
        return X


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE BUILDER
# ─────────────────────────────────────────────────────────────────────────────


def build_pipeline(target_col: str = "is_high_risk") -> Pipeline:
    """
    Assemble and return the full feature engineering Pipeline.

    The pipeline is ordered so that each step receives the output
    of the previous one. Steps that need the raw columns (DebitFlagExtractor,
    TemporalFeatureExtractor) run before DropConstantColumns removes them.

    Parameters
    ----------
    target_col : str
        Name of the binary target column used by WoEEncoder.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline ready to call .fit_transform(df, y=df[target_col]).
    """
    steps = [
        # CustomerAggregator needs TransactionStartTime → must run first
        ("customer_agg", CustomerAggregator()),
        # DebitFlagExtractor needs Amount → must run before DropConstantColumns
        ("debit_flag", DebitFlagExtractor()),
        # TemporalFeatureExtractor parses & drops TransactionStartTime
        ("temporal", TemporalFeatureExtractor()),
        # Now safe to drop Amount, CountryCode, IDs, raw timestamp
        ("drop_const", DropConstantColumns()),
        ("log_transform", LogTransformer()),
        ("cat_encode", CategoricalEncoder()),
        ("imputer", MissingValueImputer()),
        ("scaler", FeatureScaler()),
        ("woe", WoEEncoder(target_col=target_col)),
    ]
    return Pipeline(steps=steps)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE RUNNER
# ─────────────────────────────────────────────────────────────────────────────


def run_pipeline(
    data_path: str,
    target_col: str = "is_high_risk",
    save_path: str = None,
) -> tuple:
    """
    Load raw data, attach a stub target if missing, fit and apply the pipeline.

    Parameters
    ----------
    data_path  : str   Path to the raw CSV file.
    target_col : str   Binary target column name.
    save_path  : str   Optional path to save the processed DataFrame as CSV.

    Returns
    -------
    (pd.DataFrame, Pipeline)  Processed feature matrix and the fitted pipeline.
    """
    log.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    log.info("Raw data shape: %s", df.shape)

    # ── Attach proxy target if not present ──────────────────
    # Real proxy target is built in Task 4 (RFM K-Means).
    # Here we use FraudResult as a stand-in so the pipeline can
    # be tested end-to-end before Task 4 produces is_high_risk.
    if target_col not in df.columns:
        log.warning(
            "'%s' column not found — using FraudResult as temporary proxy target.",
            target_col,
        )
        df[target_col] = df["FraudResult"]

    y = df[target_col].copy()

    # ── Drop target from feature matrix before fitting ──────
    X = df.drop(columns=[target_col], errors="ignore")

    pipeline = build_pipeline(target_col=target_col)
    X_processed = pipeline.fit_transform(X, y=y)

    # ── Re-attach target for downstream tasks ────────────────
    X_processed[target_col] = y.values

    log.info("Processed data shape: %s", X_processed.shape)
    log.info(
        "Final columns (%d): %s", len(X_processed.columns), X_processed.columns.tolist()
    )

    if save_path:
        X_processed.to_csv(save_path, index=False)
        log.info("Saved processed data to %s", save_path)

    return X_processed, pipeline


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — run directly to process data/raw/data.csv → data/processed/features.csv
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    RAW_PATH = os.path.join("data", "raw", "data.csv")
    SAVE_PATH = os.path.join("data", "processed", "features.csv")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    X_ready, fitted_pipeline = run_pipeline(
        data_path=RAW_PATH,
        target_col="is_high_risk",
        save_path=SAVE_PATH,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Output shape      : {X_ready.shape}")
    print(f"Output saved to   : {SAVE_PATH}")
    print("\nFirst 3 rows:")
    print(X_ready.head(3).to_string())

"""
src/proxy_target.py
===================
Proxy Target Variable Engineering for the Bati Bank Credit Risk Model.

Since the Xente dataset contains no verified loan default labels, this
module constructs a binary proxy target (is_high_risk) using RFM-based
K-Means customer segmentation.

Logic
-----
1. Compute per-customer RFM metrics from raw transaction history.
2. Scale RFM features with RobustScaler (handles extreme outliers).
3. Cluster customers into K=3 groups with K-Means (random_state=42).
4. Rank clusters by a composite engagement score; label the least
   engaged cluster as high-risk (is_high_risk = 1).
5. Merge is_high_risk back into the processed feature DataFrame.

Usage
-----
    # Standalone — produces data/processed/features_with_target.csv
    python src/proxy_target.py

    # Programmatic
    from src.proxy_target import build_target, attach_target

    rfm, labels = build_target("data/raw/data.csv")
    df_final    = attach_target("data/processed/features.csv", rfm)
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

# non-interactive backend for CI/CD
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
SNAPSHOT_DATE = pd.Timestamp("2019-02-13", tz="UTC")  # dataset max date
N_CLUSTERS = 3
RANDOM_STATE = 42
HIGH_RISK_LABEL = 1
LOW_RISK_LABEL = 0


# ─────────────────────────────────────────────────────────────
# STEP 1 — Calculate RFM Metrics
# ─────────────────────────────────────────────────────────────


def compute_rfm(raw_path: str) -> pd.DataFrame:
    """
    Load raw transaction data and compute per-customer RFM metrics.

    RFM definitions
    ---------------
    Recency   : Days between snapshot date and customer's last transaction.
                Lower is better (more recent = more engaged).
    Frequency : Total number of transactions in the observation window.
                Higher is better.
    Monetary  : Sum of all transaction Values (absolute amounts, UGX).
                Higher is better.

    Parameters
    ----------
    raw_path : str  Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame  One row per customer with columns:
                  [CustomerId, recency, frequency, monetary]
    """
    log.info("Computing RFM metrics from %s", raw_path)
    df = pd.read_csv(raw_path)
    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], utc=True, errors="coerce"
    )

    rfm = (
        df.groupby("CustomerId")
        .agg(
            recency=("TransactionStartTime", lambda x: (SNAPSHOT_DATE - x.max()).days),
            frequency=("TransactionId", "count"),
            monetary=("Value", "sum"),  # Value = |Amount|, always >= 0
        )
        .reset_index()
    )

    log.info(
        "RFM computed for %d customers | "
        "recency [%d–%d days] | frequency [%d–%d] | monetary [%.0f–%.0f UGX]",
        len(rfm),
        rfm["recency"].min(),
        rfm["recency"].max(),
        rfm["frequency"].min(),
        rfm["frequency"].max(),
        rfm["monetary"].min(),
        rfm["monetary"].max(),
    )
    return rfm


# ─────────────────────────────────────────────────────────────
# STEP 2 — Scale RFM Features
# ─────────────────────────────────────────────────────────────


def scale_rfm(rfm: pd.DataFrame) -> tuple:
    """
    Scale RFM features for K-Means clustering.

    Two-step approach to handle extreme skewness:
    1. Cap frequency and monetary at their 99th percentile.
       Rationale: EDA found frequency max = 4,091 vs median = 7,
       and monetary max = 104.9M UGX vs median = 32K. Without capping,
       K-Means collapses: the 7 extreme-outlier customers form their own
       micro-clusters and 99.8% of customers land in a single cluster,
       making segmentation meaningless for risk scoring.
    2. Apply RobustScaler on the capped values to further reduce the
       influence of remaining high values on cluster centroids.

    Parameters
    ----------
    rfm : pd.DataFrame  Output of compute_rfm().

    Returns
    -------
    (np.ndarray, RobustScaler)  Scaled feature matrix and fitted scaler.
    """
    features = rfm[["recency", "frequency", "monetary"]].copy()

    # Cap at 99th percentile to prevent outlier-driven cluster collapse
    for col in ["frequency", "monetary"]:
        cap = features[col].quantile(0.99)
        features[col] = features[col].clip(upper=cap)
        log.info("RFM scaling: capping %s at 99th pct = %.0f", col, cap)

    scaler = RobustScaler()
    scaled = scaler.fit_transform(features)
    log.info("RFM features scaled with RobustScaler (after 99th-pct capping).")
    return scaled, scaler


# ─────────────────────────────────────────────────────────────
# STEP 3 — Elbow Analysis (optional diagnostic)
# ─────────────────────────────────────────────────────────────


def plot_elbow(scaled: np.ndarray, max_k: int = 8, save_path: str = None) -> None:
    """
    Plot Within-Cluster Sum of Squares (WCSS) for k = 2..max_k.
    Saved to save_path if provided; otherwise displayed.
    """
    wcss = []
    ks = range(2, max_k + 1)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(scaled)
        wcss.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(ks), wcss, marker="o", color="steelblue", linewidth=2)
    ax.axvline(
        x=N_CLUSTERS,
        color="salmon",
        linestyle="--",
        linewidth=1.5,
        label=f"Chosen k={N_CLUSTERS}",
    )
    ax.set_title("Elbow Method — Optimal Number of Clusters", fontweight="bold")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("WCSS (Inertia)")
    ax.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
        log.info("Elbow plot saved to %s", save_path)
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────
# STEP 4 — Cluster Customers with K-Means
# ─────────────────────────────────────────────────────────────


def cluster_customers(rfm: pd.DataFrame, scaled: np.ndarray) -> tuple:
    """
    Fit K-Means (k=3) on scaled RFM and assign cluster labels.

    Parameters
    ----------
    rfm    : pd.DataFrame   Output of compute_rfm().
    scaled : np.ndarray     Scaled RFM matrix from scale_rfm().

    Returns
    -------
    (pd.DataFrame, KMeans)
        rfm DataFrame with added 'cluster' column, and the fitted model.
    """
    log.info("Fitting K-Means (k=%d, random_state=%d) ...", N_CLUSTERS, RANDOM_STATE)
    km = KMeans(
        n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10, max_iter=300
    )
    rfm = rfm.copy()
    rfm["cluster"] = km.fit_predict(scaled)

    # Log cluster sizes
    sizes = rfm["cluster"].value_counts().sort_index()
    log.info("Cluster sizes: %s", dict(sizes))
    return rfm, km


# ─────────────────────────────────────────────────────────────
# STEP 5 — Identify High-Risk Cluster & Assign Labels
# ─────────────────────────────────────────────────────────────


def assign_risk_labels(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Determine which cluster represents the least-engaged (highest-risk)
    customers and assign is_high_risk = 1 to that cluster.

    Ranking logic
    -------------
    A cluster is "more engaged" when it has:
      • Lower  recency   (more recent transactions)
      • Higher frequency (transacts more often)
      • Higher monetary  (spends more)

    We compute a composite engagement score per cluster:
        engagement = -recency_rank + frequency_rank + monetary_rank

    The cluster with the LOWEST engagement score is labelled high-risk.
    Rank ties are broken by frequency (most discriminative in EDA).

    Parameters
    ----------
    rfm : pd.DataFrame  Output of cluster_customers() with 'cluster' column.

    Returns
    -------
    pd.DataFrame  rfm with added 'is_high_risk' column (0 or 1).
    """
    # Cluster-level mean RFM
    cluster_profile = (
        rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean().round(2)
    )
    log.info("Cluster profiles (mean RFM):\n%s", cluster_profile.to_string())

    # Rank each dimension (1 = worst engagement)
    cluster_profile["recency_rank"] = cluster_profile["recency"].rank(
        ascending=False
    )  # higher recency  → worse
    cluster_profile["frequency_rank"] = cluster_profile["frequency"].rank(
        ascending=True
    )  # lower frequency → worse
    cluster_profile["monetary_rank"] = cluster_profile["monetary"].rank(
        ascending=True
    )  # lower monetary  → worse

    cluster_profile["engagement_score"] = (
        cluster_profile["recency_rank"]
        + cluster_profile["frequency_rank"]
        + cluster_profile["monetary_rank"]
    )

    high_risk_cluster = int(cluster_profile["engagement_score"].idxmin())
    log.info(
        "High-risk cluster identified: cluster %d " "(lowest engagement score = %.1f)",
        high_risk_cluster,
        cluster_profile.loc[high_risk_cluster, "engagement_score"],
    )
    log.info(
        "Cluster engagement scores:\n%s",
        cluster_profile[
            ["recency", "frequency", "monetary", "engagement_score"]
        ].to_string(),
    )

    rfm = rfm.copy()
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    n_high = rfm["is_high_risk"].sum()
    pct = n_high / len(rfm) * 100
    log.info(
        "High-risk customers: %d / %d  (%.1f%%)",
        n_high,
        len(rfm),
        pct,
    )
    return rfm, cluster_profile, high_risk_cluster


# ─────────────────────────────────────────────────────────────
# STEP 6 — Visualise RFM Clusters
# ─────────────────────────────────────────────────────────────


def plot_rfm_clusters(
    rfm: pd.DataFrame, high_risk_cluster: int, save_path: str = None
) -> None:
    """
    3-panel bar chart of mean Recency, Frequency, and Monetary
    value per cluster. High-risk cluster highlighted in salmon.
    """
    profile = (
        rfm.groupby("cluster")[["recency", "frequency", "monetary"]]
        .mean()
        .reset_index()
    )
    colors = [
        "salmon" if c == high_risk_cluster else "steelblue" for c in profile["cluster"]
    ]
    labels = [
        f"Cluster {c}\n(HIGH RISK)" if c == high_risk_cluster else f"Cluster {c}"
        for c in profile["cluster"]
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = [
        ("recency", "Recency (days)", "lower = more engaged"),
        ("frequency", "Frequency (count)", "higher = more engaged"),
        ("monetary", "Monetary (UGX)", "higher = more engaged"),
    ]

    for ax, (col, ylabel, note) in zip(axes, metrics):
        bars = ax.bar(labels, profile[col], color=colors, edgecolor="white", alpha=0.88)
        ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=9)
        ax.set_title(f"{col.capitalize()}\n({note})", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    plt.suptitle(
        f"K-Means RFM Cluster Profiles  |  "
        f"Cluster {high_risk_cluster} = HIGH RISK (salmon)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
        log.info("Cluster profile plot saved to %s", save_path)
    else:
        plt.show()
    plt.close()


def plot_risk_distribution(rfm: pd.DataFrame, save_path: str = None) -> None:
    """
    Pie chart showing high-risk vs low-risk customer split.
    """
    counts = rfm["is_high_risk"].value_counts().sort_index()
    labels = [f"Low Risk (0)\n{counts[0]:,}", f"High Risk (1)\n{counts[1]:,}"]
    colors = ["steelblue", "salmon"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        counts,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax.set_title("is_high_risk Distribution", fontweight="bold", fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
        log.info("Risk distribution plot saved to %s", save_path)
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────
# STEP 7 — Merge target back into processed feature dataset
# ─────────────────────────────────────────────────────────────


def attach_target(
    features_path: str, rfm: pd.DataFrame, save_path: str = None
) -> pd.DataFrame:
    """
    Merge is_high_risk from rfm into the processed feature DataFrame.

    The processed features CSV (output of data_processing.py) is
    transaction-level; we join on CustomerId so every transaction row
    inherits its customer's risk label.

    Parameters
    ----------
    features_path : str           Path to data/processed/features.csv
    rfm           : pd.DataFrame  Output of assign_risk_labels() with
                                  CustomerId and is_high_risk columns.
    save_path     : str           Optional output path.

    Returns
    -------
    pd.DataFrame  Transaction-level dataset with is_high_risk column.
    """
    log.info("Loading processed features from %s", features_path)
    features = pd.read_csv(features_path)

    # Drop the temporary FraudResult-based is_high_risk if present
    features.drop(columns=["is_high_risk"], errors="ignore", inplace=True)

    risk_map = rfm[["CustomerId", "is_high_risk"]].drop_duplicates()
    features = features.merge(risk_map, on="CustomerId", how="left")

    # Fill any unmatched rows as low risk (shouldn't happen with clean data)
    features["is_high_risk"] = features["is_high_risk"].fillna(0).astype(int)

    n_high = features["is_high_risk"].sum()
    pct = n_high / len(features) * 100
    log.info(
        "Target merged. Shape: %s | High-risk transactions: %d (%.1f%%)",
        features.shape,
        n_high,
        pct,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        features.to_csv(save_path, index=False)
        log.info("Final dataset saved to %s", save_path)

    return features


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────


def build_target(raw_path: str, plot_dir: str = None) -> tuple:
    """
    Full pipeline: raw CSV → RFM → K-Means → is_high_risk labels.

    Parameters
    ----------
    raw_path : str   Path to raw transaction CSV.
    plot_dir : str   Optional directory to save diagnostic plots.

    Returns
    -------
    (pd.DataFrame, dict)
        rfm DataFrame with [CustomerId, recency, frequency, monetary,
        cluster, is_high_risk] and a metadata dict.
    """
    # 1. Compute RFM
    rfm = compute_rfm(raw_path)

    # 2. Scale
    scaled, scaler = scale_rfm(rfm)

    # 3. Elbow diagnostic (optional)
    if plot_dir:
        plot_elbow(scaled, save_path=os.path.join(plot_dir, "elbow_curve.png"))

    # 4. Cluster
    rfm, km_model = cluster_customers(rfm, scaled)

    # 5. Assign labels
    rfm, cluster_profile, high_risk_cluster = assign_risk_labels(rfm)

    # 6. Plots
    if plot_dir:
        plot_rfm_clusters(
            rfm,
            high_risk_cluster,
            save_path=os.path.join(plot_dir, "rfm_cluster_profiles.png"),
        )
        plot_risk_distribution(
            rfm,
            save_path=os.path.join(plot_dir, "risk_distribution.png"),
        )

    meta = {
        "n_customers": len(rfm),
        "high_risk_cluster": high_risk_cluster,
        "n_high_risk": int(rfm["is_high_risk"].sum()),
        "pct_high_risk": round(rfm["is_high_risk"].mean() * 100, 2),
        "cluster_profile": cluster_profile.to_dict(),
        "km_model": km_model,
        "scaler": scaler,
    }
    return rfm, meta


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    RAW_PATH = os.path.join("data", "raw", "data.csv")
    FEATURES_PATH = os.path.join("data", "processed", "features.csv")
    FINAL_PATH = os.path.join("data", "processed", "features_with_target.csv")
    PLOT_DIR = os.path.join("data", "processed", "plots")

    # ── Step A: Build RFM + cluster labels ─────────────────
    rfm, meta = build_target(raw_path=RAW_PATH, plot_dir=PLOT_DIR)

    # ── Step B: Merge into processed feature dataset ───────
    df_final = attach_target(
        features_path=FEATURES_PATH,
        rfm=rfm,
        save_path=FINAL_PATH,
    )

    # ── Summary report ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("PROXY TARGET ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"Total customers     : {meta['n_customers']:,}")
    print(f"High-risk cluster   : Cluster {meta['high_risk_cluster']}")
    print(
        f"High-risk customers : {meta['n_high_risk']:,}  "
        f"({meta['pct_high_risk']:.1f}%)"
    )
    print(f"Final dataset shape : {df_final.shape}")
    print(f"Output saved to     : {FINAL_PATH}")
    print(f"Plots saved to      : {PLOT_DIR}/")

    print("\nCluster Profiles (mean RFM):")
    profile_display = (
        rfm.groupby(["cluster", "is_high_risk"])[["recency", "frequency", "monetary"]]
        .mean()
        .round(1)
    )
    print(profile_display.to_string())

    print("\nis_high_risk value counts (transaction level):")
    print(df_final["is_high_risk"].value_counts().to_string())

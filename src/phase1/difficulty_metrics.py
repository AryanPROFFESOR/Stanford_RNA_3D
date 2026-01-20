"""
Phase 1.4 – Structural Difficulty & Dataset Stratification
----------------------------------------------------------

Computes scientifically motivated difficulty metrics per target:

Inputs:
- outputs/phase1/sequence_statistics.csv

Outputs:
- outputs/phase1/difficulty_metrics.csv
- outputs/phase1/difficulty_summary.json

Metrics:
- length_log (log-scaled length)
- missing_fraction
- entropy_norm
- repeat_fraction
- max_homopolymer (scaled)
- difficulty_index (robust composite)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone


STEP = "PHASE1_04_DIFFICULTY_METRICS"

IN_PATH = Path("outputs/phase1/sequence_statistics.csv")
OUT_CSV = Path("outputs/phase1/difficulty_metrics.csv")
OUT_JSON = Path("outputs/phase1/difficulty_summary.json")


def log(msg):
    print(f"[{STEP}] {msg}")


def robust_zscore(x):
    """
    Robust normalization using median and IQR.
    Prevents long RNAs from dominating scale.
    """
    med = np.median(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    if iqr == 0:
        return np.zeros_like(x)
    return (x - med) / iqr


def main():
    start = datetime.now(timezone.utc)
    log(f"Started at {start.isoformat()}")

    if not IN_PATH.exists():
        raise RuntimeError(f"Missing input file: {IN_PATH}")

    df = pd.read_csv(IN_PATH)

    # ---- Feature transforms ----

    df["length_log"] = np.log10(df["length"])
    df["homopolymer_scaled"] = df["max_homopolymer"] / df["length"]

    # Core difficulty features
    features = {
        "length_log": df["length_log"].values,
        "missing_fraction": df["missing_fraction"].values,
        "entropy_norm": 1.0 - df["entropy_norm"].values,  # low entropy = easier
        "repeat_fraction": df["repeat_fraction"].values,
        "homopolymer_scaled": df["homopolymer_scaled"].values,
    }

    # Robust normalization
    for name, values in features.items():
        df[f"{name}_rz"] = robust_zscore(values)

    # Composite difficulty (equal-weighted, robust)
    rz_cols = [c for c in df.columns if c.endswith("_rz")]
    df["difficulty_index"] = df[rz_cols].mean(axis=1)

    # Normalize difficulty into 0–1 for interpretability
    dmin, dmax = df["difficulty_index"].min(), df["difficulty_index"].max()
    df["difficulty_norm"] = (df["difficulty_index"] - dmin) / (dmax - dmin)

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # Summary statistics
    summary = {
        "n_targets": int(len(df)),
        "difficulty_norm": {
            "mean": float(df["difficulty_norm"].mean()),
            "median": float(df["difficulty_norm"].median()),
            "min": float(df["difficulty_norm"].min()),
            "max": float(df["difficulty_norm"].max()),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features_used": list(features.keys()),
        "normalization": "robust z-score (median/IQR)",
        "composite_method": "equal-weighted mean of normalized components",
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    end = datetime.now(timezone.utc)
    log(f"Finished at {end.isoformat()}")
    log(f"Duration: {end - start}")
    log(f"Saved CSV → {OUT_CSV}")
    log(f"Saved JSON → {OUT_JSON}")


if __name__ == "__main__":
    main()

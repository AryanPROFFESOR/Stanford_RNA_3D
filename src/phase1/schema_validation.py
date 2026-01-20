"""
Phase 1 – Schema & Relational Validation (Final Robust Version)
---------------------------------------------------------------

Validates:
- Sequence integrity
- Label integrity
- Referential consistency
- Residue indexing continuity
- Coordinate completeness (biologically correct rules)
- Exports all residues missing coordinates

Outputs:
- outputs/phase1/schema_validation_report.json
- outputs/phase1/missing_coordinate_residues.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone


STEP = "PHASE1_02_SCHEMA_VALIDATION"
REPORT_PATH = Path("outputs/phase1/schema_validation_report.json")
MISSING_PATH = Path("outputs/phase1/missing_coordinate_residues.csv")


def log(msg):
    print(f"[{STEP}] {msg}")


def load_data():
    candidates = list(Path(".").rglob("train_sequences.csv"))
    if not candidates:
        raise RuntimeError("train_sequences.csv not found")

    root = candidates[0].parent
    seq = pd.read_csv(root / "train_sequences.csv", low_memory=False)
    lab = pd.read_csv(root / "train_labels.csv", low_memory=False)

    return seq, lab


def validate_sequences(seq_df, report):
    log("Validating sequence table...")
    issues = []

    if seq_df["target_id"].duplicated().any():
        issues.append("Duplicate target_id detected")

    valid = set("ACGU")
    bad = seq_df[
        ~seq_df["sequence"]
        .str.upper()
        .apply(lambda s: set(s).issubset(valid))
    ]

    if len(bad) > 0:
        issues.append(f"{len(bad)} sequences contain invalid characters")

    report["sequence_table"] = {
        "n_targets": len(seq_df),
        "issues": issues,
    }


def validate_labels(seq_df, lab_df, report):
    log("Validating label table...")

    issues = []
    warnings = []

    # Parse identifiers
    lab_df["target_id"] = lab_df["ID"].str.split("_").str[0]
    lab_df["resid"] = lab_df["ID"].str.split("_").str[1].astype(int)

    # Referential integrity
    missing = set(lab_df["target_id"]) - set(seq_df["target_id"])
    if missing:
        issues.append(f"{len(missing)} label targets missing from sequences")

    # Duplicate IDs
    dup = lab_df.duplicated(subset=["ID"])
    if dup.any():
        issues.append(f"{dup.sum()} duplicated residue IDs")

    # ---- Coordinate validation (robust vectorized) ----

    coord_cols = [c for c in lab_df.columns if c.startswith(("x_", "y_", "z_"))]

    # Force numeric
    lab_df[coord_cols] = lab_df[coord_cols].apply(pd.to_numeric, errors="coerce")

    coords = lab_df[coord_cols].to_numpy(dtype=float)

    n_structures = len(coord_cols) // 3

    # reshape: (rows, structures, xyz)
    coords = coords.reshape(coords.shape[0], n_structures, 3)

    # valid triplet = all finite across xyz
    valid_triplets = np.isfinite(coords).all(axis=2)

    # valid residue = any structure valid
    valid_rows = valid_triplets.any(axis=1)

    bad_mask = ~valid_rows
    bad_count = bad_mask.sum()

    if bad_count > 0:
        frac = bad_count / len(lab_df)
        warnings.append(
            f"{bad_count} residues lack coordinates ({frac:.2%} of dataset)"
        )

        bad_df = lab_df.loc[bad_mask, ["ID", "target_id", "resid"]]
        MISSING_PATH.parent.mkdir(parents=True, exist_ok=True)
        bad_df.to_csv(MISSING_PATH, index=False)

        log(f"Saved missing residue list to: {MISSING_PATH}")

    report["label_table"] = {
        "n_rows": len(lab_df),
        "issues": issues,
        "warnings": warnings,
    }


def validate_residue_continuity(lab_df, report):
    log("Validating residue indexing continuity...")

    issues = []

    lab_df["target_id"] = lab_df["ID"].str.split("_").str[0]
    lab_df["resid"] = lab_df["ID"].str.split("_").str[1].astype(int)

    for tid, group in lab_df.groupby("target_id"):
        expected = set(range(1, group["resid"].max() + 1))
        observed = set(group["resid"])
        if expected != observed:
            issues.append(f"{tid}: indexing inconsistency")

    report["residue_indexing"] = {
        "issues": issues
    }


def main():
    start = datetime.now(timezone.utc)
    log(f"Started at {start.isoformat()}")

    report = {
        "step": STEP,
        "generated_at": start.isoformat(),
        "status": "PASS"
    }

    seq_df, lab_df = load_data()

    validate_sequences(seq_df, report)
    validate_labels(seq_df, lab_df, report)
    validate_residue_continuity(lab_df, report)

    all_issues = (
        report["sequence_table"]["issues"]
        + report["label_table"]["issues"]
        + report["residue_indexing"]["issues"]
    )

    if all_issues:
        report["status"] = "FAIL"
        report["all_issues"] = all_issues
        log("Validation FAILED.")
    elif report["label_table"]["warnings"]:
        report["status"] = "PASS_WITH_WARNINGS"
        log("Validation PASSED with warnings.")
    else:
        log("Validation PASSED cleanly.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    end = datetime.now(timezone.utc)
    log(f"Finished at {end.isoformat()}")
    log(f"Duration: {end - start}")
    log(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()

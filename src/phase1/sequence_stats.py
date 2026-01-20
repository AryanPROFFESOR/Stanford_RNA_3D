"""
Phase 1.3 – Sequence Statistical & Biological Characterization (Research-Grade)
------------------------------------------------------------------------------

Computes robust, publication-quality sequence descriptors:
- Length
- GC content
- Normalized Shannon entropy (0–1)
- Dinucleotide frequencies (all 16, consistent schema)
- Max homopolymer length
- Repeat fraction (low complexity proxy)
- Missing coordinate fraction

Outputs:
- outputs/phase1/sequence_statistics.csv
- outputs/phase1/sequence_statistics_summary.json
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
from collections import Counter
from datetime import datetime, timezone
import math


STEP = "PHASE1_03_SEQUENCE_STATS"
OUT_CSV = Path("outputs/phase1/sequence_statistics.csv")
OUT_JSON = Path("outputs/phase1/sequence_statistics_summary.json")
MISSING_PATH = Path("outputs/phase1/missing_coordinate_residues.csv")

DINUCS = [a + b for a in "ACGU" for b in "ACGU"]


def log(msg):
    print(f"[{STEP}] {msg}")


def load_sequences():
    candidates = list(Path(".").rglob("train_sequences.csv"))
    if not candidates:
        raise RuntimeError("train_sequences.csv not found")
    root = candidates[0].parent
    return pd.read_csv(root / "train_sequences.csv")


def load_missing():
    if MISSING_PATH.exists():
        return pd.read_csv(MISSING_PATH)
    return pd.DataFrame(columns=["target_id", "resid"])


def normalized_entropy(seq):
    counts = Counter(seq)
    total = len(seq)
    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log2(p)
    return H / 2.0  # max entropy log2(4)=2


def longest_homopolymer(seq):
    max_run = current = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    return max_run


def repeat_fraction(seq):
    runs = 0
    i = 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        if j - i >= 3:
            runs += (j - i)
        i = j
    return runs / len(seq)


def dinucleotide_profile(seq):
    counts = Counter(seq[i:i+2] for i in range(len(seq)-1))
    total = sum(counts.values())
    return {f"di_{d}": counts.get(d, 0) / total for d in DINUCS}


def main():
    start = datetime.now(timezone.utc)
    log(f"Started at {start.isoformat()}")

    seq_df = load_sequences()
    missing_df = load_missing()

    missing_counts = (
        missing_df.groupby("target_id").size().to_dict()
        if not missing_df.empty else {}
    )

    rows = []

    for _, row in seq_df.iterrows():
        tid = row["target_id"]
        seq = row["sequence"]

        length = len(seq)
        gc = (seq.count("G") + seq.count("C")) / length
        ent = normalized_entropy(seq)
        homopoly = longest_homopolymer(seq)
        repeat_frac = repeat_fraction(seq)
        di = dinucleotide_profile(seq)

        rows.append({
            "target_id": tid,
            "length": length,
            "gc_content": gc,
            "entropy_norm": ent,
            "max_homopolymer": homopoly,
            "repeat_fraction": repeat_frac,
            "missing_residues": missing_counts.get(tid, 0),
            "missing_fraction": missing_counts.get(tid, 0) / length,
            **di
        })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    summary = {
        "n_targets": int(len(df)),
        "length": {
            "mean": float(df["length"].mean()),
            "median": float(df["length"].median()),
            "min": int(df["length"].min()),
            "max": int(df["length"].max())
        },
        "gc_content_mean": float(df["gc_content"].mean()),
        "entropy_mean": float(df["entropy_norm"].mean()),
        "repeat_fraction_mean": float(df["repeat_fraction"].mean()),
        "missing_fraction_mean": float(df["missing_fraction"].mean()),
        "generated_at": datetime.now(timezone.utc).isoformat()
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

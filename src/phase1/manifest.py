"""
Phase 1 – Data Manifest & Provenance Generator
---------------------------------------------

Research-grade dataset manifest generator for
Stanford RNA 3D Folding Challenge pipeline.

This script:
- Detects execution environment (local / Kaggle)
- Locates dataset root automatically
- Hashes all critical dataset files (SHA256)
- Captures file metadata (size, timestamps)
- Captures schema summaries for CSV files
- Produces reproducible audit artifacts

Outputs:
- outputs/phase1/data_manifest.json
- outputs/phase1/data_manifest.csv
"""

from pathlib import Path
import hashlib
import json
import pandas as pd
from datetime import datetime
import platform
import sys


# -------------------------------------------------
# Environment detection
# -------------------------------------------------

def detect_environment():
    if Path("/kaggle/input").exists():
        return "kaggle"
    return "local"


ENV = detect_environment()


# -------------------------------------------------
# Dataset root detection
# -------------------------------------------------

def locate_dataset_root():
    """
    Locate dataset directory robustly.
    Supports:
    - Kaggle competition mount
    - Local extracted folder
    - Local zip (warns user)
    """

    if ENV == "kaggle":
        return Path("/kaggle/input")

    # Local environment: search current project tree
    candidates = list(Path(".").rglob("train_sequences.csv"))
    if candidates:
        return candidates[0].parent

    # Zip present but not extracted
    zip_files = list(Path(".").glob("*.zip"))
    if zip_files:
        raise RuntimeError(
            f"Found dataset zip but not extracted: {zip_files[0]}\n"
            f"Please extract before running manifest.py"
        )

    raise RuntimeError(
        "Dataset not found. Ensure train_sequences.csv exists in project tree."
    )


DATA_ROOT = locate_dataset_root()
OUTPUT_ROOT = Path("outputs/phase1")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# Utility functions
# -------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_metadata(path: Path):
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def summarize_csv_schema(path: Path, max_rows: int = 1000):
    try:
        df = pd.read_csv(path, nrows=max_rows)
        return {
            "columns": list(df.columns),
            "n_columns": df.shape[1],
            "sample_rows_used": len(df),
            "null_counts": df.isnull().sum().to_dict(),
        }
    except Exception as e:
        return {"error": str(e)}


# -------------------------------------------------
# Main manifest generation
# -------------------------------------------------

def generate_manifest():
    manifest = {}

    manifest["generated_at"] = datetime.utcnow().isoformat() + "Z"
    manifest["environment"] = ENV
    manifest["platform"] = platform.platform()
    manifest["python_version"] = sys.version
    manifest["dataset_root"] = str(DATA_ROOT.resolve())

    files_of_interest = []

    # Core expected files
    expected = [
        "train_sequences.csv",
        "train_labels.csv",
        "validation_sequences.csv",
        "test_sequences.csv",
    ]

    for fname in expected:
        fpath = DATA_ROOT / fname
        if fpath.exists():
            files_of_interest.append(fpath)

    # Additional directories
    extra_dirs = ["MSA", "PDB_RNA", "extra"]
    for d in extra_dirs:
        dpath = DATA_ROOT / d
        if dpath.exists() and dpath.is_dir():
            for f in dpath.rglob("*"):
                if f.is_file():
                    files_of_interest.append(f)

    manifest["file_count"] = len(files_of_interest)
    manifest["files"] = []

    for f in sorted(files_of_interest):
        entry = {}
        entry.update(file_metadata(f))
        entry["sha256"] = sha256_file(f)

        if f.suffix.lower() == ".csv":
            entry["schema"] = summarize_csv_schema(f)

        manifest["files"].append(entry)

    return manifest


# -------------------------------------------------
# Save outputs
# -------------------------------------------------

def save_outputs(manifest: dict):
    json_path = OUTPUT_ROOT / "data_manifest.json"
    csv_path = OUTPUT_ROOT / "data_manifest.csv"

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save flattened CSV
    rows = []
    for f in manifest["files"]:
        flat = {
            "path": f["path"],
            "size_bytes": f["size_bytes"],
            "modified_time": f["modified_time"],
            "sha256": f["sha256"],
        }
        rows.append(flat)

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nManifest saved:")
    print(f"  JSON → {json_path}")
    print(f"  CSV  → {csv_path}")
    print(f"  Files indexed: {len(rows)}")


# -------------------------------------------------
# Entry point
# -------------------------------------------------

if __name__ == "__main__":
    print(f"Environment detected: {ENV}")
    print(f"Dataset root: {DATA_ROOT}")

    manifest = generate_manifest()
    save_outputs(manifest)

    print("\nManifest generation completed successfully.")

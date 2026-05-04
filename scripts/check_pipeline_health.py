from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def check_alignment():
    import pandas as pd
    print("Checking Pipeline Health & Data Alignment...")
    
    # 1. Check Chronos Predictions
    preds_path = ROOT / "results" / "lora" / "lora_predictions.csv" # The default output from train_lora.py
    if not preds_path.exists():
        # Check if v2 path exists
        preds_path = ROOT / "results" / "lora" / "lora_v2_predictions.csv"
        
    if not preds_path.exists():
        print("❌ FAILED: Chronos predictions not found.")
        return False
        
    df = pd.read_csv(preds_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"✅ Chronos Predictions found: {len(df)} rows.")
    print(f"   Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # 2. Check for Overlap with Backtest Dates (2024-2026)
    test_start = pd.Timestamp("2024-01-01")
    test_end = pd.Timestamp("2026-04-15")
    
    mask = (df['Date'] >= test_start) & (df['Date'] <= test_end)
    coverage = df[mask]
    
    if len(coverage) == 0:
        print(f"❌ FAILED: No predictions found in test window {test_start.date()} to {test_end.date()}.")
        return False
    
    print(f"✅ Data Coverage: Found {len(coverage)} rows in test window.")
    
    # 3. Check for duplicates
    dupes = df.duplicated(subset=['Date', 'Ticker']).sum()
    if dupes > 0:
        print(f"⚠️ WARNING: Found {dupes} duplicate Date/Ticker pairs in predictions.")
    else:
        print("✅ No duplicates found in prediction file.")

    print("\n🚀 Pipeline health is OK. Ready for Diagnostic Ablation.")
    return True

def verify_manifest(manifest_path: str = "data/manifest.csv",
                    processed_dir: str = "data/processed") -> bool:
    """Verify that every parquet in data/processed matches the md5 in data/manifest.csv.

    Stdlib-only so it runs even when the heavy ML deps are not yet installed.
    Returns True on success, raises AssertionError on hash mismatch, returns False
    if the manifest is missing.
    """
    import csv
    import hashlib
    import os
    if not os.path.exists(manifest_path):
        print(f"Manifest not found at {manifest_path}. Build with --build-manifest.")
        return False
    rows = list(csv.DictReader(open(manifest_path)))
    bad = []
    for row in rows:
        path = os.path.join(processed_dir, row["file"])
        if not os.path.exists(path):
            bad.append((row["file"], "missing"))
            continue
        with open(path, "rb") as fh:
            actual = hashlib.md5(fh.read()).hexdigest()
        if actual != row["md5"]:
            bad.append((row["file"], f"hash {actual} != expected {row['md5']}"))
    if bad:
        for f, reason in bad[:10]:
            print(f"BAD {f}: {reason}")
        raise AssertionError(f"Manifest verification failed for {len(bad)} of {len(rows)} files")
    print(f"Manifest verified: {len(rows)} files OK")
    return True


def build_manifest(manifest_path: str = "data/manifest.csv",
                   processed_dir: str = "data/processed") -> None:
    """Regenerate data/manifest.csv from the current state of data/processed."""
    import csv
    import hashlib
    import os
    rows = []
    for f in sorted(os.listdir(processed_dir)):
        if not f.endswith(".parquet"):
            continue
        p = os.path.join(processed_dir, f)
        with open(p, "rb") as fh:
            md5 = hashlib.md5(fh.read()).hexdigest()
        rows.append({
            "ticker": f.replace("_features.parquet", ""),
            "file": f,
            "bytes": os.path.getsize(p),
            "md5": md5,
        })
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=["ticker", "file", "bytes", "md5"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {manifest_path} with {len(rows)} entries")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-manifest", action="store_true")
    parser.add_argument("--build-manifest", action="store_true")
    args, _ = parser.parse_known_args()

    if args.build_manifest:
        build_manifest()
        sys.exit(0)
    if args.verify_manifest:
        sys.exit(0 if verify_manifest() else 1)
    if not check_alignment():
        sys.exit(1)

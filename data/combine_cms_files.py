"""
FirstWave — CMS File Combiner (Smart Version)
===============================================

Handles ALL download formats:

Option A — Pre-filtered files (recommended, smallest):
    part_d_rybelsus.csv              ← filtered to Rybelsus only
    part_d_prior_drugs.csv           ← filtered to Ozempic/Jardiance/Januvia etc.
    nppes_endocrinology.csv          ← one file per specialty
    nppes_internal_medicine.csv
    nppes_family_medicine.csv
    nppes_cardiology.csv
    nppes_geriatrics.csv
    nppes_nephrology.csv
    open_payments_novo_nordisk.csv   ← one file per manufacturer
    open_payments_eli_lilly.csv
    (any other open_payments_*.csv)

Option B — By-year files (medium):
    part_d_2020.csv ... part_d_2023.csv
    open_payments_2020.csv ... open_payments_2023.csv
    nppes.csv

Option C — Single combined files (already merged):
    part_d_combined.csv
    open_payments_combined.csv
    nppes.csv

Script auto-detects which option you have and handles it.
Mix and match is fine — e.g. Option A for Part D, Option B for Open Payments.

Run:
    python data/combine_cms_files.py

Output:
    data/raw/part_d_combined.csv
    data/raw/nppes_combined.csv
    data/raw/open_payments_combined.csv

Then:
    python data/load_real_data.py
    python ml/train.py
"""

import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)


# ── Column name normaliser ────────────────────────────────────────────────────
# CMS changes column names slightly across years — normalise them all
PART_D_COL_MAP = {
    "PRSCRBR_NPI":            "NPI",
    "Prscrbr_NPI":            "NPI",
    "BRND_NAME":              "Brnd_Name",
    "GEN_DRUG_NAME":          "Gnrc_Name",
    "GNRC_NAME":              "Gnrc_Name",
    "TOT_CLMS":               "Tot_Clms",
    "TOT_30DAY_FILLS":        "Tot_30day_Fills",
    "TOT_BENES":              "Tot_Benes",
    "TOT_DRUG_CST":           "Tot_Drug_Cst",
    "PRSCRBR_STATE_ABRVTN":   "Prscrbr_State_Abrvtn",
    "Prscrbr_State_Abrvtn":   "Prscrbr_State_Abrvtn",
}

OPEN_PAY_COL_MAP = {
    "covered_recipient_npi":                                           "Covered_Recipient_NPI",
    "total_amount_of_payment_usdollars":                               "Total_Amount_of_Payment_USDollars",
    "nature_of_payment_or_transfer_of_value":                         "Nature_of_Payment_or_Transfer_of_Value",
    "applicable_manufacturer_or_applicable_gpo_making_payment_name":  "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
}

NPPES_COL_MAP = {
    "NPI":                                                       "NPI",
    "Provider Business Practice Location Address State Name":    "state",
    "Provider Business Mailing Address State Name":              "state",
    "Healthcare Provider Taxonomy Code_1":                       "taxonomy",
    "Provider Gender Code":                                      "gender",
    "Provider Enumeration Date":                                 "enumeration_date",
}


def normalise_cols(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    df.columns = [col_map.get(c, c) for c in df.columns]
    return df


def read_csv_safe(path: Path, col_map: dict = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if col_map:
        df = normalise_cols(df, col_map)
    return df


# ── Part D ────────────────────────────────────────────────────────────────────
def combine_part_d() -> None:
    print("\n[1/3] Part D prescribing data")
    out = RAW / "part_d_combined.csv"

    # Already combined
    if out.exists():
        print(f"  ✓ part_d_combined.csv already exists — skipping")
        return

    frames = []

    # Option A — pre-filtered files
    pre_filtered = list(RAW.glob("part_d_rybelsus*.csv")) + \
                   list(RAW.glob("part_d_prior*.csv")) + \
                   list(RAW.glob("part_d_glp1*.csv"))
    if pre_filtered:
        print("  Detected: pre-filtered Part D files")
        for f in pre_filtered:
            df = read_csv_safe(f, PART_D_COL_MAP)
            frames.append(df)
            print(f"  ✓ {f.name} — {len(df):,} rows")

    # Option B — by-year files
    elif any((RAW / f"part_d_{y}.csv").exists() for y in [2020,2021,2022,2023]):
        print("  Detected: by-year Part D files")
        for year in [2020, 2021, 2022, 2023]:
            path = RAW / f"part_d_{year}.csv"
            if not path.exists():
                print(f"  ✗ {path.name} missing — skipping")
                continue
            df = read_csv_safe(path, PART_D_COL_MAP)
            df["year"] = year
            frames.append(df)
            print(f"  ✓ {path.name} — {len(df):,} rows")

    else:
        print("  ✗ No Part D files found in data/raw/")
        print("    Expected one of:")
        print("      part_d_rybelsus.csv   (pre-filtered)")
        print("      part_d_2020.csv       (by year)")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out, index=False)
    print(f"  ✓ Saved → part_d_combined.csv ({len(combined):,} rows)")


# ── NPPES ─────────────────────────────────────────────────────────────────────
def combine_nppes() -> None:
    print("\n[2/3] NPPES physician registry")
    out = RAW / "nppes_combined.csv"

    if out.exists():
        print(f"  ✓ nppes_combined.csv already exists — skipping")
        return

    frames = []

    # Option A — one file per specialty
    specialty_files = list(RAW.glob("nppes_*.csv"))
    if specialty_files:
        print("  Detected: per-specialty NPPES files")
        for f in specialty_files:
            df = read_csv_safe(f, NPPES_COL_MAP)
            frames.append(df)
            print(f"  ✓ {f.name} — {len(df):,} rows")

    # Option B — single full NPPES file
    elif (RAW / "nppes.csv").exists():
        print("  Detected: single NPPES file")
        df = read_csv_safe(RAW / "nppes.csv", NPPES_COL_MAP)
        frames.append(df)
        print(f"  ✓ nppes.csv — {len(df):,} rows")

    else:
        print("  ✗ No NPPES files found in data/raw/")
        print("    Expected one of:")
        print("      nppes_endocrinology.csv   (per specialty)")
        print("      nppes.csv                 (full file)")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["NPI"])
    combined.to_csv(out, index=False)
    print(f"  ✓ Saved → nppes_combined.csv ({len(combined):,} rows, deduplicated)")


# ── Open Payments ─────────────────────────────────────────────────────────────
def combine_open_payments() -> None:
    print("\n[3/3] Open Payments")
    out = RAW / "open_payments_combined.csv"

    if out.exists():
        print(f"  ✓ open_payments_combined.csv already exists — skipping")
        return

    frames = []

    # Option A — per-manufacturer files
    mfr_files = [f for f in RAW.glob("open_payments_*.csv")
                 if not f.name.startswith("open_payments_20")]
    if mfr_files:
        print("  Detected: per-manufacturer Open Payments files")
        for f in mfr_files:
            df = read_csv_safe(f, OPEN_PAY_COL_MAP)
            frames.append(df)
            print(f"  ✓ {f.name} — {len(df):,} rows")

    # Option B — by-year files
    elif any((RAW / f"open_payments_{y}.csv").exists() for y in [2020,2021,2022,2023]):
        print("  Detected: by-year Open Payments files")
        for year in [2020, 2021, 2022, 2023]:
            path = RAW / f"open_payments_{year}.csv"
            if not path.exists():
                print(f"  ✗ {path.name} missing — skipping")
                continue
            df = read_csv_safe(path, OPEN_PAY_COL_MAP)
            df["year"] = year
            frames.append(df)
            print(f"  ✓ {path.name} — {len(df):,} rows")

    else:
        print("  ✗ No Open Payments files found in data/raw/")
        print("    Expected one of:")
        print("      open_payments_novo_nordisk.csv   (per manufacturer)")
        print("      open_payments_2020.csv           (by year)")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out, index=False)
    print(f"  ✓ Saved → open_payments_combined.csv ({len(combined):,} rows)")


# ── Validation ────────────────────────────────────────────────────────────────
def validate() -> None:
    print("\n[Validation]")
    required = {
        "part_d_combined.csv":       "Part D prescribing",
        "nppes_combined.csv":        "NPPES physicians",
        "open_payments_combined.csv":"Open Payments",
    }
    all_ok = True
    for filename, label in required.items():
        path = RAW / filename
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)
            rows = sum(1 for _ in open(path)) - 1
            print(f"  ✓ {label}: {size:.1f}MB, {rows:,} rows")
        else:
            print(f"  ✗ MISSING: {label}")
            all_ok = False

    print()
    if all_ok:
        print("All files ready. Run:")
        print("  python data/load_real_data.py")
        print("  python ml/train.py")
    else:
        print("Fix missing files above. See comments in this script for expected filenames.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("FirstWave CMS File Combiner")
    print("=" * 40)
    print(f"Looking in: {RAW.resolve()}")

    combine_part_d()
    combine_nppes()
    combine_open_payments()
    validate()

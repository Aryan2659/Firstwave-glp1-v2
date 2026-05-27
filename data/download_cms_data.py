"""
FirstWave CMS Data Downloader
================================

Downloads exactly what FirstWave needs from CMS public APIs.
No manual downloads required.

What it pulls:
- Medicare Part D: Rybelsus, Ozempic, Trulicity, Jardiance, Januvia (2020-2023)
- NPPES: Endocrinology, Internal Medicine, Family Medicine, Cardiology, Geriatrics, Nephrology
- CMS Open Payments: GLP-1 related manufacturer payments (2020-2023)

Run:
    pip install requests pandas pyarrow tqdm
    python data/download_cms_data.py

Output:
    data/raw/part_d_rybelsus.csv
    data/raw/part_d_prior_drugs.csv
    data/raw/nppes_physicians.csv
    data/raw/open_payments.csv
"""

import time
from pathlib import Path

import pandas as pd
import requests

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

# ── CMS API config ────────────────────────────────────────────────────────────
CMS_BASE = "https://data.cms.gov/data-api/v1/dataset"
LIMIT    = 5000   # rows per page (CMS max)

# Dataset UUIDs — stable CMS identifiers
PART_D_DATASET_ID    = "e4e6-8hd3"  # Part D Prescribers by Provider and Drug
OPEN_PAY_DATASET_ID  = "muzy-jte4"  # Open Payments General

# Drugs to pull
RYBELSUS_FILTER    = "Rybelsus"
PRIOR_DRUGS        = ["Ozempic", "Trulicity", "Victoza", "Jardiance", "Farxiga", "Januvia", "Tradjenta"]

# NPPES taxonomy codes for target specialties
NPPES_TAXONOMIES = {
    "207RE0101X": "Endocrinology",
    "207RG0100X": "Endocrinology",
    "207Q00000X": "Family Medicine",
    "207R00000X": "Internal Medicine",
    "207RC0000X": "Cardiology",
    "208D00000X": "Geriatrics",
    "207RN0300X": "Nephrology",
}

GLP1_MANUFACTURERS = [
    "Novo Nordisk",
    "Eli Lilly",
    "AstraZeneca",
    "Boehringer Ingelheim",
    "Merck",
    "Pfizer",
]


# ── Generic paginated CMS API fetcher ─────────────────────────────────────────
def fetch_cms_api(
    dataset_id: str,
    filters: dict = None,
    columns: list = None,
    max_rows: int = 200_000,
    label: str = "",
) -> pd.DataFrame:
    """
    Pages through the CMS data API and returns a DataFrame.

    Args:
        dataset_id: CMS dataset UUID
        filters:    dict of {column: value} filters
        columns:    list of columns to return (None = all)
        max_rows:   safety cap
        label:      label for progress output
    """
    url = f"{CMS_BASE}/{dataset_id}/data"
    params = {"size": LIMIT, "offset": 0}

    if filters:
        for col, val in filters.items():
            params[f"filter[{col}]"] = val

    if columns:
        params["column"] = ",".join(columns)

    frames = []
    total_fetched = 0

    print(f"  Fetching {label}...")

    while True:
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Request failed: {e}")
            break

        data = r.json()

        if not data:
            break

        frames.append(pd.DataFrame(data))
        total_fetched += len(data)
        print(f"    {total_fetched:,} rows fetched...", end="\r")

        if len(data) < LIMIT or total_fetched >= max_rows:
            break

        params["offset"] += LIMIT
        time.sleep(0.3)   # be polite to CMS servers

    print(f"    ✓ {total_fetched:,} rows            ")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ── Step 1: Download Rybelsus Part D data ────────────────────────────────────
def download_rybelsus():
    print("\n[1/4] Medicare Part D — Rybelsus prescribing data")

    frames = []
    for year in [2020, 2021, 2022, 2023]:
        df = fetch_cms_api(
            dataset_id=PART_D_DATASET_ID,
            filters={"Brnd_Name": "Rybelsus", "year": str(year)},
            columns=[
                "Prscrbr_NPI", "Brnd_Name", "Gnrc_Name",
                "Tot_Clms", "Tot_30day_Fills", "Tot_Benes",
                "Tot_Drug_Cst", "Prscrbr_State_Abrvtn", "year",
            ],
            label=f"Rybelsus {year}",
        )
        if not df.empty:
            df["year"] = year
            frames.append(df)
        time.sleep(1)

    if frames:
        result = pd.concat(frames, ignore_index=True)
        out = RAW / "part_d_rybelsus.csv"
        result.to_csv(out, index=False)
        print(f"  ✓ Saved {len(result):,} rows → {out}")
    else:
        print("  ✗ No Rybelsus data retrieved — check dataset ID or try manual download")


# ── Step 2: Download prior drug history ──────────────────────────────────────
def download_prior_drugs():
    print("\n[2/4] Medicare Part D — Prior drug history (Ozempic, Jardiance, Januvia...)")

    frames = []
    for drug in PRIOR_DRUGS:
        df = fetch_cms_api(
            dataset_id=PART_D_DATASET_ID,
            filters={"Brnd_Name": drug},
            columns=[
                "Prscrbr_NPI", "Brnd_Name", "Tot_Clms",
                "Tot_Benes", "Prscrbr_State_Abrvtn",
            ],
            max_rows=100_000,
            label=drug,
        )
        if not df.empty:
            frames.append(df)
        time.sleep(1)

    if frames:
        result = pd.concat(frames, ignore_index=True)
        out = RAW / "part_d_prior_drugs.csv"
        result.to_csv(out, index=False)
        print(f"  ✓ Saved {len(result):,} rows → {out}")


# ── Step 3: Download NPPES physician data ────────────────────────────────────
def download_nppes():
    """
    NPPES full file is ~8GB — too large for API.
    We use the NPPES API to pull only target specialties.
    """
    print("\n[3/4] NPPES NPI Registry — target specialties")

    base_url = "https://npiregistry.cms.hhs.gov/api/"
    frames = []

    taxonomy_labels = {
        "207RE0101X": "Endocrinology",
        "207Q00000X": "Family Medicine",
        "207R00000X": "Internal Medicine",
        "207RC0000X": "Cardiology",
        "208D00000X": "Geriatrics",
        "207RN0300X": "Nephrology",
    }

    for taxonomy, specialty in taxonomy_labels.items():
        print(f"  Fetching {specialty}...")
        records = []
        skip = 0

        while True:
            try:
                r = requests.get(
                    base_url,
                    params={
                        "version": "2.1",
                        "taxonomy_description": taxonomy,
                        "limit": 200,
                        "skip": skip,
                        "enumeration_type": "NPI-1",
                    },
                    timeout=30,
                )
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                break

            results = data.get("results", [])
            if not results:
                break

            for res in results:
                basic = res.get("basic", {})
                addresses = res.get("addresses", [{}])
                practice_addr = next(
                    (a for a in addresses if a.get("address_purpose") == "LOCATION"),
                    addresses[0] if addresses else {},
                )
                taxonomies = res.get("taxonomies", [{}])
                primary_tax = next((t for t in taxonomies if t.get("primary")), taxonomies[0] if taxonomies else {})

                records.append({
                    "NPI": res.get("number"),
                    "specialty": specialty,
                    "state": practice_addr.get("state", ""),
                    "gender": basic.get("gender", "M"),
                    "enumeration_date": basic.get("enumeration_date", ""),
                })

            skip += len(results)
            total = data.get("result_count", 0)
            print(f"    {min(skip, total):,}/{total:,}", end="\r")
            time.sleep(0.5)

            if skip >= min(total, 50_000):
                break

        if records:
            frames.append(pd.DataFrame(records))
        print(f"    ✓ {len(records):,} {specialty} physicians")

    if frames:
        result = pd.concat(frames, ignore_index=True)
        result = result.drop_duplicates(subset=["NPI"])
        out = RAW / "nppes_physicians.csv"
        result.to_csv(out, index=False)
        print(f"  ✓ Saved {len(result):,} physicians → {out}")


# ── Step 4: Download Open Payments ───────────────────────────────────────────
def download_open_payments():
    print("\n[4/4] CMS Open Payments — GLP-1 manufacturer payments")

    frames = []
    for manufacturer in GLP1_MANUFACTURERS:
        df = fetch_cms_api(
            dataset_id=OPEN_PAY_DATASET_ID,
            filters={
                "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name": manufacturer
            },
            columns=[
                "Covered_Recipient_NPI",
                "Total_Amount_of_Payment_USDollars",
                "Nature_of_Payment_or_Transfer_of_Value",
                "Date_of_Payment",
                "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
            ],
            max_rows=50_000,
            label=manufacturer,
        )
        if not df.empty:
            frames.append(df)
        time.sleep(1)

    if frames:
        result = pd.concat(frames, ignore_index=True)
        out = RAW / "open_payments.csv"
        result.to_csv(out, index=False)
        print(f"  ✓ Saved {len(result):,} rows → {out}")


# ── Step 5: Validate downloads ───────────────────────────────────────────────
def validate():
    print("\n[Validation] Checking downloaded files...")
    files = {
        "part_d_rybelsus.csv": "Rybelsus Part D data",
        "part_d_prior_drugs.csv": "Prior drug history",
        "nppes_physicians.csv": "NPPES physician registry",
        "open_payments.csv": "Open Payments",
    }
    all_ok = True
    for filename, label in files.items():
        path = RAW / filename
        if path.exists():
            df = pd.read_csv(path, nrows=5)
            size = path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {label}: {size:.1f}MB, {df.shape[1]} columns")
        else:
            print(f"  ✗ MISSING: {label} ({filename})")
            all_ok = False

    if all_ok:
        print("\n✓ All files present. Run next:")
        print("  python data/load_real_data.py")
        print("  python ml/train.py")
    else:
        print("\n⚠ Some files missing — see notes above.")
        print("  If CMS API is down, try manual download at:")
        print("  https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("FirstWave CMS Data Downloader")
    print("=" * 40)
    print(f"Saving to: {RAW.resolve()}")

    download_rybelsus()
    download_prior_drugs()
    download_nppes()
    download_open_payments()
    validate()

    print("\nDone.")

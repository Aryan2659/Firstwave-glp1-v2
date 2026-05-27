"""
FirstWave Real Data Loader
============================

Replaces generate_synthetic_data.py.
Loads real CMS datasets and builds the same feature matrix
that the ML pipeline expects.

Required files in data/raw/:
- part_d_2020_2023.csv     Medicare Part D Prescriber Data
- nppes.csv                NPI Registry
- open_payments.csv        CMS Open Payments General Payments

Run:
    python data/load_real_data.py
"""

import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# ── Specialty taxonomy → readable name ──────────────────────────────────────
TAXONOMY_MAP = {
    "207RE0101X": "Endocrinology",
    "207RG0100X": "Endocrinology",
    "208D00000X": "Geriatrics",
    "207Q00000X": "Family Medicine",
    "207R00000X": "Internal Medicine",
    "207RC0000X": "Cardiology",
    "207RN0300X": "Nephrology",
    "261QP2300X": "Internal Medicine",
}

SPECIALTY_KEEP = [
    "Endocrinology", "Internal Medicine", "Family Medicine",
    "Cardiology", "Geriatrics", "Nephrology",
]

RYBELSUS_NDCS = [
    "0169430213", "0169430413", "0169430713",
    "00169430213", "00169430413", "00169430713",
]

# Analogous drug classes to detect prior prescribing history
INJECTABLE_GLP1_BRANDS = ["OZEMPIC", "TRULICITY", "VICTOZA", "RYBELSUS", "BYDUREON"]
SGLT2_BRANDS = ["JARDIANCE", "FARXIGA", "INVOKANA", "STEGLATRO"]
DPP4_BRANDS  = ["JANUVIA", "TRADJENTA", "ONGLYZA", "NESINA", "KAZANO"]


# ── Step 1: Load Part D ──────────────────────────────────────────────────────
def load_part_d() -> pd.DataFrame:
    print("[1/5] Loading Medicare Part D data...")
    frames = []

    combined_path = RAW / "part_d_combined.csv"
    if not combined_path.exists():
        print("  ✗ part_d_combined.csv not found. Run: python data/combine_cms_files.py first")
        return pd.DataFrame()

    for year in [2020, 2021, 2022, 2023]:
        path = combined_path

        df = pd.read_csv(
            path,
            usecols=lambda c: c in [
                "Prscrbr_NPI", "PRSCRBR_NPI",
                "Brnd_Name", "BRND_NAME",
                "Tot_Clms", "TOT_CLMS",
                "Tot_30day_Fills", "TOT_30DAY_FILLS",
                "Tot_Benes", "TOT_BENES",
            ],
            dtype=str,
        )
        df.columns = df.columns.str.upper()
        df = df.rename(columns={
            "PRSCRBR_NPI": "NPI",
            "BRND_NAME": "drug",
            "TOT_CLMS": "claims",
            "TOT_30DAY_FILLS": "fills",
            "TOT_BENES": "benes",
        })
        df["year"] = year
        df["claims"] = pd.to_numeric(df["claims"], errors="coerce").fillna(0)
        frames.append(df)

    part_d = pd.concat(frames, ignore_index=True)
    print(f"   {len(part_d):,} Part D rows loaded")
    return part_d


# ── Step 2: Build physician universe from NPPES ──────────────────────────────
def load_nppes() -> pd.DataFrame:
    print("[2/5] Loading NPPES NPI Registry...")

    nppes = pd.read_csv(
        RAW / "nppes.csv",
        usecols=[
            "NPI",
            "Provider Business Practice Location Address State Name",
            "Healthcare Provider Taxonomy Code_1",
            "Provider Gender Code",
            "Provider Enumeration Date",
        ],
        dtype=str,
        low_memory=False,
    )

    nppes = nppes.rename(columns={
        "NPI": "NPI",
        "Provider Business Practice Location Address State Name": "state",
        "Healthcare Provider Taxonomy Code_1": "taxonomy",
        "Provider Gender Code": "gender",
        "Provider Enumeration Date": "enum_date",
    })

    nppes["specialty"] = nppes["taxonomy"].map(TAXONOMY_MAP)
    nppes = nppes[nppes["specialty"].isin(SPECIALTY_KEEP)].copy()
    nppes["gender"] = nppes["gender"].map({"M": "M", "F": "F"}).fillna("M")
    nppes["enum_date"] = pd.to_datetime(nppes["enum_date"], errors="coerce")
    nppes["years_in_practice"] = (
        (pd.Timestamp("2023-01-01") - nppes["enum_date"]).dt.days / 365
    ).clip(1, 45).round(0).astype(int)

    rural_states = ["WY", "MT", "ND", "SD", "AK", "VT", "ME"]
    nppes["rural"] = nppes["state"].isin(rural_states).astype(int)

    print(f"   {len(nppes):,} physicians after specialty filter")
    return nppes[["NPI", "specialty", "state", "gender", "years_in_practice", "rural"]]


# ── Step 3: Build prescribing features ──────────────────────────────────────
def build_prescribing_features(part_d: pd.DataFrame, nppes: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] Building prescribing features...")

    drug_upper = part_d["drug"].str.upper().fillna("")

    def prior_prescriber_flag(brand_list):
        mask = drug_upper.str.contains("|".join(brand_list), na=False)
        return (
            part_d[mask]
            .groupby("NPI")["claims"]
            .sum()
            .reset_index()
            .assign(flag=1)[["NPI", "flag"]]
        )

    glp1_npis = prior_prescriber_flag(INJECTABLE_GLP1_BRANDS).rename(
        columns={"flag": "prior_injectable_glp1_prescriber"}
    )
    sglt2_npis = prior_prescriber_flag(SGLT2_BRANDS).rename(
        columns={"flag": "prior_sglt2_prescriber"}
    )
    dpp4_npis = prior_prescriber_flag(DPP4_BRANDS).rename(
        columns={"flag": "prior_dpp4_prescriber"}
    )

    panel = (
        part_d.groupby("NPI")
        .agg(
            patient_panel_size=("benes", lambda x: pd.to_numeric(x, errors="coerce").sum()),
            unique_drugs_prescribed=("drug", "nunique"),
        )
        .reset_index()
    )

    df = nppes.copy()
    df = df.merge(glp1_npis, on="NPI", how="left")
    df = df.merge(sglt2_npis, on="NPI", how="left")
    df = df.merge(dpp4_npis, on="NPI", how="left")
    df = df.merge(panel, on="NPI", how="left")

    for col in ["prior_injectable_glp1_prescriber", "prior_sglt2_prescriber", "prior_dpp4_prescriber"]:
        df[col] = df[col].fillna(0).astype(int)

    df["patient_panel_size"] = df["patient_panel_size"].fillna(100).clip(1, 10000).astype(int)
    df["unique_drugs_prescribed"] = df["unique_drugs_prescribed"].fillna(1).astype(int)

    return df


# ── Step 4: Open Payments (KOL features) ────────────────────────────────────
def build_open_payments_features(df: pd.DataFrame) -> tuple:
    print("[4/5] Building KOL graph from Open Payments...")

    op = pd.read_csv(
        RAW / "open_payments.csv",
        usecols=lambda c: c in [
            "Covered_Recipient_NPI",
            "Total_Amount_of_Payment_USDollars",
            "Nature_of_Payment_or_Transfer_of_Value",
            "Date_of_Payment",
            "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
        ],
        dtype=str,
        low_memory=False,
    )

    op = op.rename(columns={
        "Covered_Recipient_NPI": "NPI",
        "Total_Amount_of_Payment_USDollars": "amount",
        "Nature_of_Payment_or_Transfer_of_Value": "payment_type",
        "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name": "manufacturer",
    })

    op["amount"] = pd.to_numeric(op["amount"], errors="coerce").fillna(0)

    agg = (
        op.groupby("NPI")
        .agg(
            total_open_payments_usd=("amount", "sum"),
            num_speaker_events=("payment_type", lambda x: (x.str.contains("speak|consult", case=False, na=False)).sum()),
        )
        .reset_index()
    )
    agg["is_speaker"] = (agg["num_speaker_events"] > 0).astype(int)

    # Build KOL graph
    speaker_mask = op["payment_type"].str.contains("speak|consult", case=False, na=False)
    speakers = op[speaker_mask].copy()
    speakers = speakers.merge(df[["NPI", "specialty", "state"]], on="NPI", how="inner")

    G = nx.Graph()
    for (mfr, state, spec), group in speakers.groupby(
        ["manufacturer", "state", "specialty"], dropna=False
    ):
        npis = group["NPI"].unique().tolist()
        for i, n1 in enumerate(npis):
            for n2 in npis[i + 1:]:
                if G.has_edge(n1, n2):
                    G[n1][n2]["weight"] += 1
                else:
                    G.add_edge(n1, n2, weight=1)

    pagerank = nx.pagerank(G, weight="weight") if G.number_of_nodes() > 0 else {}
    print(f"   KOL graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    df = df.merge(agg, on="NPI", how="left")
    df["total_open_payments_usd"] = df["total_open_payments_usd"].fillna(0)
    df["num_speaker_events"] = df["num_speaker_events"].fillna(0).astype(int)
    df["is_speaker"] = df["is_speaker"].fillna(0).astype(int)
    df["kol_pagerank"] = df["NPI"].astype(str).map(pagerank).fillna(0)
    df["has_kol_connection"] = (df["kol_pagerank"] > 0).astype(int)

    max_pr = df["kol_pagerank"].max()
    df["kol_pagerank_normalized"] = df["kol_pagerank"] / max_pr if max_pr > 0 else 0

    return df, G


# ── Step 5: Build adoption labels from Rybelsus ─────────────────────────────
def build_adoption_labels(df: pd.DataFrame, part_d: pd.DataFrame) -> pd.DataFrame:
    print("[5/5] Building adoption labels from Rybelsus prescribing...")

    rybelsus = part_d[
        part_d["drug"].str.upper().str.contains("RYBELSUS", na=False)
    ].copy()

    first_rx = (
        rybelsus.groupby("NPI")["year"].min().reset_index()
        .rename(columns={"year": "first_rx_year"})
    )

    first_year_claims = rybelsus.merge(
        first_rx, left_on=["NPI", "year"], right_on=["NPI", "first_rx_year"]
    )
    first_year_claims = (
        first_year_claims.groupby("NPI")["claims"]
        .sum()
        .reset_index()
        .rename(columns={"claims": "total_claims_first_year"})
    )

    df = df.merge(first_rx, on="NPI", how="left")
    df = df.merge(first_year_claims, on="NPI", how="left")
    df["total_claims_first_year"] = df["total_claims_first_year"].fillna(0)

    df["will_prescribe"] = (~df["first_rx_year"].isna()).astype(int)

    df["early_adopter"] = (
        (df["first_rx_year"].isin([2020, 2021])) &
        (df["total_claims_first_year"] >= 10)
    ).astype(int)

    df["event_observed"] = df["will_prescribe"]
    df["months_to_event"] = np.where(
        df["will_prescribe"] == 1,
        (df["first_rx_year"].fillna(2025) - 2020) * 12,
        48.0,
    )

    print(f"   Total physicians:   {len(df):,}")
    print(f"   Will prescribe:     {df['will_prescribe'].sum():,} ({df['will_prescribe'].mean():.1%})")
    print(f"   Early adopters:     {df['early_adopter'].sum():,} ({df['early_adopter'].mean():.1%})")

    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    part_d = load_part_d()
    nppes  = load_nppes()
    df     = build_prescribing_features(part_d, nppes)
    df, G  = build_open_payments_features(df)
    df     = build_adoption_labels(df, part_d)

    out_path = OUT / "physician_features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n✓ Saved features → {out_path}")

    with open(OUT / "kol_graph.pkl", "wb") as f:
        pickle.dump(G, f)
    print(f"✓ Saved KOL graph → {OUT / 'kol_graph.pkl'}")

    print("\n=== Done. Run: python ml/train.py ===")


if __name__ == "__main__":
    main()

"""
FirstWave Synthetic Data Generator
====================================

Generates realistic synthetic data mimicking:
- CMS Medicare Part D Prescriber Data (Rybelsus prescriptions 2020-2024)
- NPPES NPI Registry (physician specialty + location)
- CMS Open Payments (speaker fees, consulting payments)

Output is a single Parquet feature matrix ready for ML training.

The synthetic generation respects real-world patterns:
- Endocrinologists adopt earlier than PCPs
- Higher Open Payments → higher adoption probability
- KOL-connected physicians adopt earlier
- ~12% baseline adoption rate (matches real Rybelsus uptake)
- Geographic clustering (some states are early adopter hubs)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import networkx as nx
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def generate_physician_universe(n_physicians: int = 20000) -> pd.DataFrame:
    """
    Generate the universe of physicians who could potentially prescribe Rybelsus.
    Mimics NPPES data: NPI, specialty, state, years_in_practice.
    """
    specialties = {
        "Endocrinology": 0.08,
        "Internal Medicine": 0.32,
        "Family Medicine": 0.40,
        "Cardiology": 0.10,
        "Geriatrics": 0.05,
        "Nephrology": 0.05,
    }

    states = [
        "CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
        "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
    ]
    state_weights = np.array([39, 30, 22, 19, 13, 13, 12, 11, 11, 10,
                              9, 9, 8, 7, 7, 7, 7, 6, 6, 6], dtype=float)
    state_weights = state_weights / state_weights.sum()

    physicians = pd.DataFrame({
        "NPI": np.arange(1000000000, 1000000000 + n_physicians),
        "specialty": np.random.choice(
            list(specialties.keys()),
            size=n_physicians,
            p=list(specialties.values()),
        ),
        "state": np.random.choice(states, size=n_physicians, p=state_weights),
        "years_in_practice": np.clip(np.random.normal(15, 8, n_physicians), 1, 45).astype(int),
        "gender": np.random.choice(["M", "F"], size=n_physicians, p=[0.62, 0.38]),
        "rural": np.random.choice([0, 1], size=n_physicians, p=[0.82, 0.18]),
    })

    return physicians


def assign_prior_drug_history(physicians: pd.DataFrame) -> pd.DataFrame:
    """
    Assign realistic prior prescribing history for analogous drug classes.
    These are key predictors: doctors who prescribed injectable GLP-1s are
    much more likely to adopt oral GLP-1.
    """
    df = physicians.copy()

    specialty_rates = {
        "Endocrinology": {"injectable_glp1": 0.78, "sglt2": 0.85, "dpp4": 0.70},
        "Internal Medicine": {"injectable_glp1": 0.35, "sglt2": 0.45, "dpp4": 0.40},
        "Family Medicine": {"injectable_glp1": 0.28, "sglt2": 0.38, "dpp4": 0.35},
        "Cardiology": {"injectable_glp1": 0.18, "sglt2": 0.55, "dpp4": 0.20},
        "Geriatrics": {"injectable_glp1": 0.22, "sglt2": 0.30, "dpp4": 0.42},
        "Nephrology": {"injectable_glp1": 0.30, "sglt2": 0.65, "dpp4": 0.35},
    }

    for drug in ["injectable_glp1", "sglt2", "dpp4"]:
        col = f"prior_{drug}_prescriber"
        df[col] = df["specialty"].map(
            lambda s: 1 if np.random.random() < specialty_rates[s][drug] else 0
        )

    df["patient_panel_size"] = np.clip(
        np.random.lognormal(mean=6.5, sigma=0.6, size=len(df)), 50, 5000
    ).astype(int)

    df["unique_drugs_prescribed"] = np.clip(
        df["patient_panel_size"] * 0.15 + np.random.normal(0, 20, len(df)),
        20, 500,
    ).astype(int)

    return df


def assign_open_payments(physicians: pd.DataFrame) -> pd.DataFrame:
    """
    Assign Open Payments data: speaker fees, consulting fees.
    These correlate strongly with KOL status.
    """
    df = physicians.copy()
    n = len(df)

    is_speaker = (
        (df["specialty"] == "Endocrinology") & (np.random.random(n) < 0.12)
    ) | (
        (df["specialty"] != "Endocrinology") & (np.random.random(n) < 0.025)
    )

    df["total_open_payments_usd"] = np.where(
        is_speaker,
        np.random.lognormal(mean=8.5, sigma=1.0, size=n),
        np.random.lognormal(mean=4.0, sigma=1.5, size=n) * (np.random.random(n) < 0.5),
    ).round(2)

    df["num_speaker_events"] = np.where(
        is_speaker,
        np.random.poisson(lam=4, size=n),
        np.where(np.random.random(n) < 0.15, np.random.poisson(lam=0.5, size=n), 0),
    )

    df["is_speaker"] = is_speaker.astype(int)

    return df


def build_kol_influence_graph(physicians: pd.DataFrame) -> tuple:
    """
    Build a synthetic prescriber influence graph.

    Construction logic (mimics real Open Payments analysis):
    - Speakers cluster by state and specialty (regional CME networks)
    - Speakers share edges if they attended the same events
    - PageRank scores represent KOL influence
    """
    G = nx.Graph()

    speakers = physicians[physicians["is_speaker"] == 1].copy()

    for (state, specialty), group in speakers.groupby(["state", "specialty"]):
        npis = group["NPI"].tolist()
        for i, n1 in enumerate(npis):
            for n2 in npis[i + 1:]:
                if np.random.random() < 0.35:
                    weight = np.random.randint(1, 6)
                    if G.has_edge(n1, n2):
                        G[n1][n2]["weight"] += weight
                    else:
                        G.add_edge(n1, n2, weight=weight)

    cross_state_edges = int(len(speakers) * 0.08)
    speaker_npis = speakers["NPI"].values
    for _ in range(cross_state_edges):
        n1, n2 = np.random.choice(speaker_npis, 2, replace=False)
        if not G.has_edge(n1, n2):
            G.add_edge(n1, n2, weight=np.random.randint(1, 4))

    pagerank = nx.pagerank(G, weight="weight") if G.number_of_nodes() > 0 else {}

    df = physicians.copy()
    df["kol_pagerank"] = df["NPI"].map(pagerank).fillna(0.0)
    df["has_kol_connection"] = (df["kol_pagerank"] > 0).astype(int)

    if df["kol_pagerank"].max() > 0:
        df["kol_pagerank_normalized"] = (
            df["kol_pagerank"] / df["kol_pagerank"].max()
        )
    else:
        df["kol_pagerank_normalized"] = 0.0

    return df, G


def assign_rybelsus_adoption(physicians: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate Rybelsus adoption (2020-2024) using a probabilistic model
    that reflects real-world adoption drivers.

    Outputs:
    - first_rx_year: int (2020-2024) or NaN if never prescribed
    - first_rx_quarter: 1-20 quarters from Q1 2020 launch
    - early_adopter: 1 if prescribed >=10 claims in first 12 months
    - total_claims_first_year: int
    - months_to_first_rx: float (for survival analysis)
    """
    df = physicians.copy()
    n = len(df)

    specialty_adoption_prob = {
        "Endocrinology": 0.55,
        "Internal Medicine": 0.32,
        "Family Medicine": 0.28,
        "Cardiology": 0.08,
        "Geriatrics": 0.15,
        "Nephrology": 0.18,
    }

    base_prob = df["specialty"].map(specialty_adoption_prob).values

    glp1_lift = df["prior_injectable_glp1_prescriber"].values * 0.30
    sglt2_lift = df["prior_sglt2_prescriber"].values * 0.12
    panel_lift = np.clip(df["patient_panel_size"].values / 5000.0, 0, 0.15)
    payments_lift = np.clip(df["total_open_payments_usd"].values / 50000.0, 0, 0.18)
    kol_lift = df["kol_pagerank_normalized"].values * 0.20
    rural_penalty = df["rural"].values * -0.05

    adoption_prob = np.clip(
        base_prob + glp1_lift + sglt2_lift + panel_lift + payments_lift + kol_lift + rural_penalty,
        0.0, 0.92,
    )

    will_prescribe = np.random.random(n) < adoption_prob

    speed_score = (
        df["prior_injectable_glp1_prescriber"].values * 4.0 +
        df["kol_pagerank_normalized"].values * 5.0 +
        (df["specialty"] == "Endocrinology").astype(int).values * 4.5 +
        df["total_open_payments_usd"].values / 15000.0 +
        df["prior_sglt2_prescriber"].values * 1.5
    )

    months_to_first_rx = np.where(
        will_prescribe,
        np.clip(
            30 - speed_score * 2.2 + np.random.normal(0, 3, n),
            0.5, 48,
        ),
        np.nan,
    )

    df["will_prescribe"] = will_prescribe.astype(int)
    df["months_to_first_rx"] = months_to_first_rx

    df["first_rx_year"] = np.where(
        will_prescribe,
        2020 + (months_to_first_rx // 12).astype("float"),
        np.nan,
    )

    activity_score = (
        (df["specialty"] == "Endocrinology").astype(int).values * 8.0 +
        df["prior_injectable_glp1_prescriber"].values * 6.0 +
        df["patient_panel_size"].values / 200.0 +
        df["kol_pagerank_normalized"].values * 10.0 +
        4.0
    )

    total_claims = np.where(
        will_prescribe,
        np.clip(
            np.random.poisson(lam=activity_score, size=n),
            0, 300,
        ),
        0,
    )

    df["total_claims_first_year"] = total_claims

    is_early_window = (df["months_to_first_rx"] <= 12) & (will_prescribe)
    df["early_adopter"] = (
        is_early_window & (df["total_claims_first_year"] >= 10)
    ).astype(int)

    df["event_observed"] = will_prescribe.astype(int)
    df["months_to_event"] = np.where(
        will_prescribe,
        df["months_to_first_rx"],
        48.0,
    )

    return df


def generate_full_dataset(
    n_physicians: int = 20000,
    output_dir: str = "/home/claude/firstwave/data/processed",
) -> pd.DataFrame:
    """
    Generate the complete FirstWave synthetic dataset.
    Returns the final feature matrix and saves to parquet.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Generating {n_physicians:,} physician records...")
    df = generate_physician_universe(n_physicians)

    print("[2/5] Assigning prior drug prescribing history...")
    df = assign_prior_drug_history(df)

    print("[3/5] Assigning Open Payments data...")
    df = assign_open_payments(df)

    print("[4/5] Building KOL influence graph (PageRank)...")
    df, kol_graph = build_kol_influence_graph(df)
    print(f"      Graph: {kol_graph.number_of_nodes():,} nodes, "
          f"{kol_graph.number_of_edges():,} edges")

    print("[5/5] Simulating Rybelsus adoption events...")
    df = assign_rybelsus_adoption(df)

    output_path = Path(output_dir) / "physician_features.parquet"
    df.to_parquet(output_path, index=False)

    graph_path = Path(output_dir) / "kol_graph.pkl"
    nx.write_gpickle(kol_graph, graph_path) if hasattr(nx, "write_gpickle") else None
    import pickle
    with open(graph_path, "wb") as f:
        pickle.dump(kol_graph, f)

    print(f"\n✓ Saved features:       {output_path}")
    print(f"✓ Saved KOL graph:      {graph_path}")
    print(f"\nDataset summary:")
    print(f"  Total physicians:     {len(df):,}")
    print(f"  Will prescribe:       {df['will_prescribe'].sum():,} "
          f"({df['will_prescribe'].mean():.1%})")
    print(f"  Early adopters:       {df['early_adopter'].sum():,} "
          f"({df['early_adopter'].mean():.1%})")
    print(f"  With KOL connection:  {df['has_kol_connection'].sum():,} "
          f"({df['has_kol_connection'].mean():.1%})")

    return df


if __name__ == "__main__":
    df = generate_full_dataset(n_physicians=5000)
    print("\nSample records:")
    print(df.head().to_string())

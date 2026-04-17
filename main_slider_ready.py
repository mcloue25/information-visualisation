import os
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


DATA_PATH = "data/csv/owid-energy-data.csv"
OUT_DIR = "data/results"
DATA_OUT_DIR = os.path.join(OUT_DIR, "data")

START_YEAR = 2000
MIN_POPULATION = 1_000_000

# Countries to keep highlighted in the scatter and line chart.
FOCUS_COUNTRIES = [
    "Ireland",
    "United Kingdom",
    "Germany",
    "Denmark",
    "China",
    "India",
    "United States",
    "Brazil",
    "Iceland",
    "Norway",
    "Sweden",
]

KEY_COLUMNS = [
    "country",
    "year",
    "iso_code",
    "population",
    "gdp",
    "primary_energy_consumption",
    "electricity_generation",
    "renewables_share_energy",
    "fossil_share_energy",
    "low_carbon_share_elec",
    "renewables_share_elec",
    "fossil_share_elec",
    "solar_share_elec",
    "wind_share_elec",
    "hydro_share_elec",
    "nuclear_share_elec",
    "coal_share_elec",
    "gas_share_elec",
    "oil_share_elec",
    "carbon_intensity_elec",
]


def ensure_dirs() -> None:
    os.makedirs(DATA_OUT_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[KEY_COLUMNS].copy()


def keep_countries(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["iso_code"].notna()].copy()
    df = df[df["iso_code"].str.len() == 3].copy()
    return df


def clean_data(df: pd.DataFrame, start_year: int = START_YEAR) -> pd.DataFrame:
    df = keep_countries(df)
    df = df[df["year"] >= start_year].copy()

    numeric_cols = [c for c in KEY_COLUMNS if c not in {"country", "iso_code"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["gdp_per_capita"] = df["gdp"] / df["population"]
    df["renewable_vs_fossil_gap_energy"] = (
        df["renewables_share_energy"] - df["fossil_share_energy"]
    )
    df["renewable_vs_fossil_gap_elec"] = (
        df["renewables_share_elec"] - df["fossil_share_elec"]
    )
    df["solar_wind_share_elec"] = (
        df["solar_share_elec"].fillna(0) + df["wind_share_elec"].fillna(0)
    )
    df["clean_share_elec"] = (
        df["renewables_share_elec"].fillna(0) + df["nuclear_share_elec"].fillna(0)
    )
    df["transition_gap"] = df["clean_share_elec"] - df["fossil_share_elec"]

    return df


def make_dashboard_yearly(
    df: pd.DataFrame,
    min_population: int = MIN_POPULATION,
) -> pd.DataFrame:
    cols = [
        "country",
        "year",
        "iso_code",
        "population",
        "renewables_share_energy",
        "fossil_share_energy",
        "clean_share_elec",
        "fossil_share_elec",
        "transition_gap",
        "carbon_intensity_elec",
    ]

    out = df[cols].copy()
    out = out.dropna(
        subset=[
            "country",
            "year",
            "population",
            "renewables_share_energy",
            "fossil_share_energy",
        ]
    )
    out = out[out["population"] >= min_population].copy()

    out["is_focus_country"] = out["country"].isin(FOCUS_COUNTRIES)
    out = out.sort_values(["year", "renewables_share_energy"], ascending=[True, False])

    return out


def make_story_countries_yearly(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "country",
        "year",
        "population",
        "renewables_share_energy",
        "fossil_share_energy",
        "clean_share_elec",
        "fossil_share_elec",
        "transition_gap",
    ]

    out = df[df["country"].isin(FOCUS_COUNTRIES)][cols].copy()
    out = out.dropna(subset=["year", "clean_share_elec", "fossil_share_elec"])
    out = out.sort_values(["country", "year"])

    return out


def make_latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    latest_year = int(df["year"].max())
    latest = df[df["year"] == latest_year].copy()
    latest = latest.dropna(
        subset=["renewables_share_energy", "fossil_share_energy", "population"]
    )
    latest = latest.sort_values("renewables_share_energy", ascending=False)
    return latest


def make_top_improvers_fixed(
    dashboard_yearly: pd.DataFrame,
    start_year: int,
    end_year: int,
    n: int = 15,
) -> pd.DataFrame:
    pair = build_pairwise_changes_for_years(
        dashboard_yearly, start_year=start_year, end_year=end_year
    )
    pair = pair.sort_values("renewables_share_energy_change", ascending=False).head(n)
    return pair


def build_pairwise_changes_for_years(
    dashboard_yearly: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    if start_year == end_year:
        raise ValueError("start_year and end_year must differ for a pairwise comparison")

    start_df = dashboard_yearly[dashboard_yearly["year"] == start_year][
        ["country", "iso_code", "renewables_share_energy", "population"]
    ].copy()
    start_df = start_df.rename(
        columns={
            "renewables_share_energy": "renewables_share_energy_start",
            "population": "population_start",
        }
    )

    end_df = dashboard_yearly[dashboard_yearly["year"] == end_year][
        ["country", "iso_code", "renewables_share_energy", "population"]
    ].copy()
    end_df = end_df.rename(
        columns={
            "renewables_share_energy": "renewables_share_energy_end",
            "population": "population_end",
        }
    )

    out = start_df.merge(end_df, on=["country", "iso_code"], how="inner")
    out["start_year"] = start_year
    out["end_year"] = end_year
    out["renewables_share_energy_change"] = (
        out["renewables_share_energy_end"] - out["renewables_share_energy_start"]
    )

    out = out.dropna(
        subset=[
            "renewables_share_energy_start",
            "renewables_share_energy_end",
            "renewables_share_energy_change",
        ]
    )

    if "population_end" in out.columns:
        out = out[out["population_end"] >= MIN_POPULATION].copy()

    cols = [
        "country",
        "iso_code",
        "start_year",
        "end_year",
        "renewables_share_energy_start",
        "renewables_share_energy_end",
        "renewables_share_energy_change",
        "population_start",
        "population_end",
    ]
    return out[cols].sort_values(
        ["start_year", "end_year", "renewables_share_energy_change"],
        ascending=[True, True, False],
    )


def make_all_year_pairs(dashboard_yearly: pd.DataFrame) -> pd.DataFrame:
    years = sorted(int(y) for y in dashboard_yearly["year"].dropna().unique())
    parts: List[pd.DataFrame] = []

    for start_year in years:
        for end_year in years:
            if end_year <= start_year:
                continue
            part = build_pairwise_changes_for_years(
                dashboard_yearly, start_year=start_year, end_year=end_year
            )
            parts.append(part)

    if not parts:
        raise ValueError("No year pairs could be generated.")

    out = pd.concat(parts, ignore_index=True)
    return out


def save_outputs(
    df: pd.DataFrame,
    dashboard_yearly: pd.DataFrame,
    story_yearly: pd.DataFrame,
    latest_snapshot: pd.DataFrame,
    top_improvers_fixed: pd.DataFrame,
    pairwise_changes: pd.DataFrame,
) -> None:
    df.to_csv(os.path.join(DATA_OUT_DIR, "energy_master.csv"), index=False)
    dashboard_yearly.to_csv(
        os.path.join(DATA_OUT_DIR, "energy_dashboard_yearly.csv"), index=False
    )
    story_yearly.to_csv(
        os.path.join(DATA_OUT_DIR, "energy_story_countries.csv"), index=False
    )
    latest_snapshot.to_csv(
        os.path.join(DATA_OUT_DIR, "energy_latest_snapshot.csv"), index=False
    )
    top_improvers_fixed.to_csv(
        os.path.join(DATA_OUT_DIR, "energy_top_improvers.csv"), index=False
    )
    pairwise_changes.to_csv(
        os.path.join(DATA_OUT_DIR, "energy_improver_pairs.csv"), index=False
    )


def main() -> None:
    ensure_dirs()

    raw = load_data(DATA_PATH)
    df = clean_data(raw, start_year=START_YEAR)

    latest_year = int(df["year"].max())
    dashboard_yearly = make_dashboard_yearly(df, min_population=MIN_POPULATION)
    story_yearly = make_story_countries_yearly(df)
    latest_snapshot = make_latest_snapshot(dashboard_yearly)
    top_improvers_fixed = make_top_improvers_fixed(
        dashboard_yearly,
        start_year=START_YEAR,
        end_year=latest_year,
        n=15,
    )
    pairwise_changes = make_all_year_pairs(dashboard_yearly)

    save_outputs(
        df=df,
        dashboard_yearly=dashboard_yearly,
        story_yearly=story_yearly,
        latest_snapshot=latest_snapshot,
        top_improvers_fixed=top_improvers_fixed,
        pairwise_changes=pairwise_changes,
    )

    print("Saved data to:", DATA_OUT_DIR)
    print("Latest year found:", latest_year)
    print("Dashboard yearly rows:", len(dashboard_yearly))
    print("Pairwise comparison rows:", len(pairwise_changes))
    print()
    print("Files created:")
    print(" - energy_master.csv")
    print(" - energy_dashboard_yearly.csv")
    print(" - energy_story_countries.csv")
    print(" - energy_latest_snapshot.csv")
    print(" - energy_top_improvers.csv")
    print(" - energy_improver_pairs.csv")


if __name__ == "__main__":
    main()

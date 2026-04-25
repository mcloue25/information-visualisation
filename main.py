import os
from typing import List, Optional

import pandas as pd

try:
    import pycountry
except ImportError:  # Map export still runs, but choropleth join IDs are omitted.
    pycountry = None


DATA_PATH = "data/csv/owid-energy-data.csv"
OUT_DIR = "data/results"
DATA_OUT_DIR = os.path.join(OUT_DIR, "data")

START_YEAR = 2000
MIN_POPULATION = 1_000_000

# Countries kept for detailed comparison panels.
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
    missing = [col for col in KEY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[KEY_COLUMNS].copy()


def keep_countries(df: pd.DataFrame) -> pd.DataFrame:
    """Keep normal country rows and remove aggregate regions."""
    out = df[df["iso_code"].notna()].copy()
    out = out[out["iso_code"].str.len() == 3].copy()
    return out


def iso3_to_numeric(iso3: str) -> Optional[int]:
    """Convert ISO-3 country codes to numeric IDs used by Vega world TopoJSON."""
    if pycountry is None or pd.isna(iso3):
        return None

    country = pycountry.countries.get(alpha_3=iso3)
    if country is None:
        return None

    return int(country.numeric)


def clean_data(df: pd.DataFrame, start_year: int = START_YEAR) -> pd.DataFrame:
    df = keep_countries(df)
    df = df[df["year"] >= start_year].copy()

    numeric_cols = [col for col in KEY_COLUMNS if col not in {"country", "iso_code"}]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

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
    """One row per country-year for interactive dashboard charts."""
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

    return out.sort_values(["year", "renewables_share_energy"], ascending=[True, False])


def make_story_countries_yearly(df: pd.DataFrame) -> pd.DataFrame:
    """Small time-series dataset for directly labelled line charts."""
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
    return out.sort_values(["country", "year"])


def make_latest_snapshot(dashboard_yearly: pd.DataFrame) -> pd.DataFrame:
    latest_year = int(dashboard_yearly["year"].max())
    latest = dashboard_yearly[dashboard_yearly["year"] == latest_year].copy()
    latest = latest.dropna(
        subset=["renewables_share_energy", "fossil_share_energy", "population"]
    )
    return latest.sort_values("renewables_share_energy", ascending=False)


def build_pairwise_changes_for_years(
    dashboard_yearly: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Compare renewable share between two selected years."""
    if start_year == end_year:
        raise ValueError("start_year and end_year must differ")

    start_df = dashboard_yearly[dashboard_yearly["year"] == start_year][
        ["country", "iso_code", "renewables_share_energy", "population"]
    ].rename(
        columns={
            "renewables_share_energy": "renewables_share_energy_start",
            "population": "population_start",
        }
    )

    end_df = dashboard_yearly[dashboard_yearly["year"] == end_year][
        ["country", "iso_code", "renewables_share_energy", "population"]
    ].rename(
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


def make_top_improvers_fixed(
    dashboard_yearly: pd.DataFrame,
    start_year: int,
    end_year: int,
    n: int = 15,
) -> pd.DataFrame:
    pairs = build_pairwise_changes_for_years(dashboard_yearly, start_year, end_year)
    return pairs.sort_values("renewables_share_energy_change", ascending=False).head(n)


def make_all_year_pairs(dashboard_yearly: pd.DataFrame) -> pd.DataFrame:
    years = sorted(int(year) for year in dashboard_yearly["year"].dropna().unique())
    parts: List[pd.DataFrame] = []

    for start_year in years:
        for end_year in years:
            if end_year > start_year:
                parts.append(build_pairwise_changes_for_years(dashboard_yearly, start_year, end_year))

    if not parts:
        raise ValueError("No year pairs could be generated")

    return pd.concat(parts, ignore_index=True)


def make_fossil_reduction_map(
    dashboard_yearly: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Country-level fossil reduction for the choropleth map."""
    start_df = dashboard_yearly[dashboard_yearly["year"] == start_year][
        ["country", "iso_code", "fossil_share_energy", "population"]
    ].rename(
        columns={
            "fossil_share_energy": "fossil_share_energy_start",
            "population": "population_start",
        }
    )

    end_df = dashboard_yearly[dashboard_yearly["year"] == end_year][
        [
            "country",
            "iso_code",
            "fossil_share_energy",
            "renewables_share_energy",
            "population",
        ]
    ].rename(
        columns={
            "fossil_share_energy": "fossil_share_energy_latest",
            "renewables_share_energy": "renewables_share_energy_latest",
            "population": "population_latest",
        }
    )

    out = start_df.merge(end_df, on=["country", "iso_code"], how="inner")
    out["start_year"] = start_year
    out["latest_year"] = end_year
    out["fossil_reduction"] = (
        out["fossil_share_energy_start"] - out["fossil_share_energy_latest"]
    )
    out["iso_numeric"] = out["iso_code"].apply(iso3_to_numeric)

    out = out.dropna(
        subset=[
            "iso_numeric",
            "fossil_share_energy_start",
            "fossil_share_energy_latest",
            "fossil_reduction",
        ]
    )
    out["iso_numeric"] = out["iso_numeric"].astype(int)

    cols = [
        "country",
        "iso_code",
        "iso_numeric",
        "start_year",
        "latest_year",
        "fossil_share_energy_start",
        "fossil_share_energy_latest",
        "fossil_reduction",
        "renewables_share_energy_latest",
        "population_start",
        "population_latest",
    ]
    return out[cols].sort_values("fossil_reduction", ascending=False)


def save_outputs(
    df: pd.DataFrame,
    dashboard_yearly: pd.DataFrame,
    story_yearly: pd.DataFrame,
    latest_snapshot: pd.DataFrame,
    top_improvers_fixed: pd.DataFrame,
    pairwise_changes: pd.DataFrame,
    fossil_reduction_map: pd.DataFrame,
) -> None:
    outputs = {
        "energy_master.csv": df,
        "energy_dashboard_yearly.csv": dashboard_yearly,
        "energy_story_countries.csv": story_yearly,
        "energy_latest_snapshot.csv": latest_snapshot,
        "energy_top_improvers.csv": top_improvers_fixed,
        "energy_improver_pairs.csv": pairwise_changes,
        "energy_fossil_reduction_map.csv": fossil_reduction_map,
    }

    for filename, frame in outputs.items():
        frame.to_csv(os.path.join(DATA_OUT_DIR, filename), index=False)


def main() -> None:
    ensure_dirs()

    raw = load_data(DATA_PATH)
    df = clean_data(raw, start_year=START_YEAR)

    dashboard_yearly = make_dashboard_yearly(df, min_population=MIN_POPULATION)
    latest_year = int(dashboard_yearly["year"].max())

    story_yearly = make_story_countries_yearly(df)
    latest_snapshot = make_latest_snapshot(dashboard_yearly)
    top_improvers_fixed = make_top_improvers_fixed(
        dashboard_yearly,
        start_year=START_YEAR,
        end_year=latest_year,
        n=15,
    )
    pairwise_changes = make_all_year_pairs(dashboard_yearly)
    fossil_reduction_map = make_fossil_reduction_map(
        dashboard_yearly,
        start_year=START_YEAR,
        end_year=latest_year,
    )

    save_outputs(
        df=df,
        dashboard_yearly=dashboard_yearly,
        story_yearly=story_yearly,
        latest_snapshot=latest_snapshot,
        top_improvers_fixed=top_improvers_fixed,
        pairwise_changes=pairwise_changes,
        fossil_reduction_map=fossil_reduction_map,
    )

    print("Saved data to:", DATA_OUT_DIR)
    print("Latest year found:", latest_year)
    print("Dashboard yearly rows:", len(dashboard_yearly))
    print("Pairwise comparison rows:", len(pairwise_changes))
    print("Fossil reduction map rows:", len(fossil_reduction_map))
    print("\nFiles created:")
    print(" - energy_master.csv")
    print(" - energy_dashboard_yearly.csv")
    print(" - energy_story_countries.csv")
    print(" - energy_latest_snapshot.csv")
    print(" - energy_top_improvers.csv")
    print(" - energy_improver_pairs.csv")
    print(" - energy_fossil_reduction_map.csv")

    if pycountry is None:
        print("\nWarning: install pycountry to generate iso_numeric values for the map.")


if __name__ == "__main__":
    main()

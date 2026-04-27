import os
from typing import Optional

import pandas as pd
import pycountry



DATA_PATH = "data/csv/owid-energy-data.csv"
DATA_OUT_DIR = "data/results/data"

START_YEAR = 2000
MIN_POPULATION = 1_000_000

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


def load_data(path: str):
    '''Load the OWID energy data and keep only cols used in dashboard
    '''
    df = pd.read_csv(path, low_memory=False)

    missing = [col for col in KEY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[KEY_COLUMNS].copy()


def keep_country_rows(df: pd.DataFrame):
    '''Remove regional aggregates and keep only country-level rows
    '''
    return df[df["iso_code"].notna() & (df["iso_code"].str.len() == 3)].copy()


def iso3_to_numeric(iso3: str):
    '''Convert ISO-3 codes to numeric country IDs for the Vega-Lite world map
    '''
    if pd.isna(iso3):
        return None

    country = pycountry.countries.get(alpha_3=iso3)
    return int(country.numeric) if country else None


def clean_data(df: pd.DataFrame):
    '''Clean the raw data and create derived transition metrics
    '''
    df = keep_country_rows(df)
    df = df[df["year"] >= START_YEAR].copy()

    numeric_cols = [col for col in KEY_COLUMNS if col not in {"country", "iso_code"}]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df["gdp_per_capita"] = df["gdp"] / df["population"]
    df["renewable_vs_fossil_gap_energy"] = (df["renewables_share_energy"] - df["fossil_share_energy"])
    df["renewable_vs_fossil_gap_elec"] = (df["renewables_share_elec"] - df["fossil_share_elec"])
    df["solar_wind_share_elec"] = (df["solar_share_elec"].fillna(0) + df["wind_share_elec"].fillna(0))
    df["clean_share_elec"] = (df["renewables_share_elec"].fillna(0) + df["nuclear_share_elec"].fillna(0))
    df["transition_gap"] = df["clean_share_elec"] - df["fossil_share_elec"]
    return df


def make_dashboard_yearly(df: pd.DataFrame):
    ''' Create one row per country-year for the dashboard charts
    '''
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

    out = df[cols].dropna(
        subset=[
            "country",
            "year",
            "population",
            "renewables_share_energy",
            "fossil_share_energy",
        ]
    ).copy()

    out = out[out["population"] >= MIN_POPULATION].copy()
    out["is_focus_country"] = out["country"].isin(FOCUS_COUNTRIES)

    return out.sort_values(
        ["year", "renewables_share_energy"],
        ascending=[True, False],
    )


def make_story_countries_yearly(df: pd.DataFrame):
    '''Create the smaller dataset used for line chart
    '''
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


def make_latest_snapshot(dashboard_yearly: pd.DataFrame):
    '''Create a latest-year snapshot for ranking countries
    '''
    latest_year = dashboard_yearly["year"].max()
    latest = dashboard_yearly[dashboard_yearly["year"] == latest_year].copy()
    latest = latest.dropna(subset=["renewables_share_energy", "fossil_share_energy", "population"])
    return latest.sort_values("renewables_share_energy", ascending=False)


def build_pairwise_changes(dashboard_yearly: pd.DataFrame, start_year: int, end_year: int):
    '''Compare renewable share between two selected years
    '''
    if start_year == end_year:
        raise ValueError("start_year and end_year must be different")

    start = dashboard_yearly[dashboard_yearly["year"] == start_year][
        ["country", "iso_code", "renewables_share_energy", "population"]
    ].rename(
        columns={
            "renewables_share_energy": "renewables_share_energy_start",
            "population": "population_start",
        }
    )

    end = dashboard_yearly[dashboard_yearly["year"] == end_year][
        ["country", "iso_code", "renewables_share_energy", "population"]
    ].rename(
        columns={
            "renewables_share_energy": "renewables_share_energy_end",
            "population": "population_end",
        }
    )

    out = start.merge(end, on=["country", "iso_code"], how="inner")

    out["start_year"] = start_year
    out["end_year"] = end_year
    out["renewables_share_energy_change"] = (out["renewables_share_energy_end"] - out["renewables_share_energy_start"])

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
    return out[cols].sort_values(["start_year", "end_year", "renewables_share_energy_change"], ascending=[True, True, False])


def make_all_year_pairs(dashboard_yearly: pd.DataFrame):
    '''Create pairwise renewable-change data for every valid year combination
    '''
    years = sorted(dashboard_yearly["year"].dropna().astype(int).unique())
    all_pairs = [build_pairwise_changes(dashboard_yearly, start_year, end_year)for start_year in years for end_year in years if end_year > start_year]
    if not all_pairs:
        raise ValueError("No valid year pairs could be generated")
    return pd.concat(all_pairs, ignore_index=True)



def make_top_improvers(dashboard_yearly: pd.DataFrame, start_year: int, end_year: int, n: int = 15):
    '''Create a fixed top-improvers dataset for the full period
    '''
    pairs = build_pairwise_changes(dashboard_yearly, start_year, end_year)
    return pairs.head(n)


def make_fossil_reduction_map(dashboard_yearly: pd.DataFrame, start_year: int, end_year: int):
    '''Create country-level fossil reduction data for the choropleth map
    '''
    start = dashboard_yearly[dashboard_yearly["year"] == start_year][
        ["country", "iso_code", "fossil_share_energy", "population"]
    ].rename(
        columns={
            "fossil_share_energy": "fossil_share_energy_start",
            "population": "population_start",
        }
    )

    end = dashboard_yearly[dashboard_yearly["year"] == end_year][
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

    out = start.merge(end, on=["country", "iso_code"], how="inner")
    out["start_year"] = start_year
    out["latest_year"] = end_year
    out["fossil_reduction"] = (out["fossil_share_energy_start"] - out["fossil_share_energy_latest"])
    out["iso_numeric"] = out["iso_code"].apply(iso3_to_numeric)

    out = out.dropna(
        subset=[
            "iso_numeric",
            "fossil_share_energy_start",
            "fossil_share_energy_latest",
            "fossil_reduction",
        ]
    ).copy()

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



def save_outputs(outputs: dict[str, pd.DataFrame]):
    ''' Save all processed datasets as CSV files
    '''
    os.makedirs(DATA_OUT_DIR, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(os.path.join(DATA_OUT_DIR, filename), index=False)


def main():
    ''' Main function for geneerating all feats that will be used in visualisation
    '''
    raw = load_data(DATA_PATH)
    cleaned = clean_data(raw)

    dashboard_yearly = make_dashboard_yearly(cleaned)
    latest_year = int(dashboard_yearly["year"].max())

    story_yearly = make_story_countries_yearly(cleaned)
    latest_snapshot = make_latest_snapshot(dashboard_yearly)
    top_improvers = make_top_improvers(dashboard_yearly, START_YEAR, latest_year)
    improver_pairs = make_all_year_pairs(dashboard_yearly)
    fossil_reduction_map = make_fossil_reduction_map(dashboard_yearly, START_YEAR, latest_year)

    outputs = {
        "energy_master.csv": cleaned,
        "energy_dashboard_yearly.csv": dashboard_yearly,
        "energy_story_countries.csv": story_yearly,
        "energy_latest_snapshot.csv": latest_snapshot,
        "energy_top_improvers.csv": top_improvers,
        "energy_improver_pairs.csv": improver_pairs,
        "energy_fossil_reduction_map.csv": fossil_reduction_map,
    }

    save_outputs(outputs)

    print(f"Saved processed data to: {DATA_OUT_DIR}")
    print(f"Latest year found: {latest_year}")
    print(f"Dashboard yearly rows: {len(dashboard_yearly)}")
    print(f"Pairwise comparison rows: {len(improver_pairs)}")
    print(f"Fossil reduction map rows: {len(fossil_reduction_map)}")


if __name__ == "__main__":
    main()
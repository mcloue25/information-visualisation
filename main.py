import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/csv/owid-energy-data.csv"
OUT_DIR = "data/results"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
DATA_OUT_DIR = os.path.join(OUT_DIR, "data")

START_YEAR = 2000
LATEST_YEAR = 2023

FOCUS_COUNTRIES = [
    "Ireland",
    "United Kingdom",
    "Germany",
    "Denmark",
    "China",
    "India",
    "United States",
    "Brazil"
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
    "carbon_intensity_elec"
]


def ensure_dirs():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DATA_OUT_DIR, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[KEY_COLUMNS].copy()


def keep_countries(df):
    df = df[df["iso_code"].notna()].copy()
    df = df[df["iso_code"].str.len() == 3].copy()
    return df


def clean_data(df):
    df = keep_countries(df)
    df = df[df["year"] >= START_YEAR].copy()

    for col in KEY_COLUMNS:
        if col not in {"country", "iso_code"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["gdp_per_capita"] = df["gdp"] / df["population"]
    df["renewable_vs_fossil_gap_energy"] = df["renewables_share_energy"] - df["fossil_share_energy"]
    df["renewable_vs_fossil_gap_elec"] = df["renewables_share_elec"] - df["fossil_share_elec"]
    df["solar_wind_share_elec"] = df["solar_share_elec"].fillna(0) + df["wind_share_elec"].fillna(0)
    df["clean_share_elec"] = (
        df["renewables_share_elec"].fillna(0) + df["nuclear_share_elec"].fillna(0)
    )

    return df


def add_change_features(df):
    base = df[["country", "year", "renewables_share_energy", "fossil_share_energy", "carbon_intensity_elec"]].copy()

    base_2000 = (
        base[base["year"] == START_YEAR]
        .rename(columns={
            "renewables_share_energy": "renewables_share_energy_2000",
            "fossil_share_energy": "fossil_share_energy_2000",
            "carbon_intensity_elec": "carbon_intensity_elec_2000"
        })
        .drop(columns=["year"])
    )

    base_latest = (
        base[base["year"] == LATEST_YEAR]
        .rename(columns={
            "renewables_share_energy": "renewables_share_energy_latest",
            "fossil_share_energy": "fossil_share_energy_latest",
            "carbon_intensity_elec": "carbon_intensity_elec_latest"
        })
        .drop(columns=["year"])
    )

    delta = base_latest.merge(base_2000, on="country", how="inner")
    delta["renewables_share_energy_change"] = (
        delta["renewables_share_energy_latest"] - delta["renewables_share_energy_2000"]
    )
    delta["fossil_share_energy_change"] = (
        delta["fossil_share_energy_latest"] - delta["fossil_share_energy_2000"]
    )
    delta["carbon_intensity_change"] = (
        delta["carbon_intensity_elec_latest"] - delta["carbon_intensity_elec_2000"]
    )

    return df.merge(delta, on="country", how="left")


def make_latest_snapshot(df):
    latest = df[df["year"] == LATEST_YEAR].copy()
    latest = latest.dropna(subset=["renewables_share_energy", "fossil_share_energy", "population"])
    latest = latest.sort_values("renewables_share_energy", ascending=False)
    return latest


def make_focus_countries(df):
    return df[df["country"].isin(FOCUS_COUNTRIES)].copy()


def make_energy_mix_long(df):
    mix_cols = [
        "coal_share_elec",
        "gas_share_elec",
        "oil_share_elec",
        "hydro_share_elec",
        "wind_share_elec",
        "solar_share_elec",
        "nuclear_share_elec"
    ]

    mix = df[["country", "year"] + mix_cols].copy()
    mix = mix.melt(
        id_vars=["country", "year"],
        value_vars=mix_cols,
        var_name="source",
        value_name="share"
    )
    mix["source"] = mix["source"].str.replace("_share_elec", "", regex=False)
    return mix


def make_top_improvers(df, n=15):
    latest = make_latest_snapshot(df)

    cols = [
        "country",
        "population",
        "renewables_share_energy_2000",
        "renewables_share_energy_latest",
        "renewables_share_energy_change"
    ]

    out = latest[cols].copy()

    out = out.dropna(subset=[
        "population",
        "renewables_share_energy_2000",
        "renewables_share_energy_latest",
        "renewables_share_energy_change"
    ])

    out = out[out["population"] >= 1_000_000]
    out = out.sort_values("renewables_share_energy_change", ascending=False).head(n)

    return out


def save_outputs(df, latest, focus, mix_long, improvers):
    df.to_csv(os.path.join(DATA_OUT_DIR, "energy_master.csv"), index=False)
    latest.to_csv(os.path.join(DATA_OUT_DIR, "energy_latest_snapshot.csv"), index=False)
    focus.to_csv(os.path.join(DATA_OUT_DIR, "energy_focus_countries.csv"), index=False)
    mix_long.to_csv(os.path.join(DATA_OUT_DIR, "energy_mix_long.csv"), index=False)
    improvers.to_csv(os.path.join(DATA_OUT_DIR, "energy_top_improvers.csv"), index=False)


def plot_scatter_latest(latest):
    plot_df = latest.dropna(subset=["renewables_share_energy", "fossil_share_energy", "population"]).copy()
    plot_df = plot_df[plot_df["population"] >= 1_000_000]

    sizes = np.sqrt(plot_df["population"]) / 150

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(
        plot_df["fossil_share_energy"],
        plot_df["renewables_share_energy"],
        s=sizes,
        alpha=0.7
    )

    for _, row in plot_df[plot_df["country"].isin(FOCUS_COUNTRIES)].iterrows():
        ax.annotate(
            row["country"],
            (row["fossil_share_energy"], row["renewables_share_energy"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points"
        )

    ax.set_title(f"Renewables vs Fossil Share of Primary Energy ({LATEST_YEAR})")
    ax.set_xlabel("Fossil share of primary energy (%)")
    ax.set_ylabel("Renewables share of primary energy (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scatter_latest.png"), dpi=300)
    plt.close()


def plot_focus_lines(focus):
    plot_df = focus.dropna(subset=["renewables_share_energy"]).copy()

    fig, ax = plt.subplots(figsize=(11, 7))
    for country in FOCUS_COUNTRIES:
        sub = plot_df[plot_df["country"] == country].sort_values("year")
        if not sub.empty:
            ax.plot(sub["year"], sub["renewables_share_energy"], label=country)

    ax.set_title("Renewables Share of Primary Energy Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Renewables share of primary energy (%)")
    ax.legend(ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "focus_lines_renewables.png"), dpi=300)
    plt.close()


def plot_focus_clean_vs_fossil(focus):
    plot_df = focus.dropna(subset=["clean_share_elec", "fossil_share_elec"]).copy()

    fig, ax = plt.subplots(figsize=(11, 7))
    for country in FOCUS_COUNTRIES:
        sub = plot_df[plot_df["country"] == country].sort_values("year")
        if not sub.empty:
            ax.plot(sub["year"], sub["clean_share_elec"] - sub["fossil_share_elec"], label=country)

    ax.axhline(0, linewidth=1)
    ax.set_title("Electricity Transition Gap: Clean Share Minus Fossil Share")
    ax.set_xlabel("Year")
    ax.set_ylabel("Gap in percentage points")
    ax.legend(ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "focus_lines_clean_vs_fossil.png"), dpi=300)
    plt.close()


def plot_heatmap_focus(mix_long):
    heat = (
        mix_long[mix_long["country"].isin(FOCUS_COUNTRIES)]
        .pivot_table(index="country", columns="year", values="share", aggfunc="sum")
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(heat.values, aspect="auto")

    ax.set_title("Electricity Mix Intensity Across Focus Countries")
    ax.set_xlabel("Year")
    ax.set_ylabel("Country")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=45)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)

    cbar = plt.colorbar(im)
    cbar.set_label("Share of electricity (%)")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "heatmap_focus.png"), dpi=300)
    plt.close()


def plot_top_improvers(improvers):

    print(improvers.shape)
    print(improvers.head())

    plot_df = improvers.sort_values(
        "renewables_share_energy_change",
        ascending=False
    ).reset_index(drop=True)

    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(11, 8))

    ax.hlines(
        y=y,
        xmin=plot_df["renewables_share_energy_2000"],
        xmax=plot_df["renewables_share_energy_latest"],
        linewidth=2
    )

    print(plot_df) 
    ax.scatter(plot_df["renewables_share_energy_2000"], y, s=40, label=str(START_YEAR))
    ax.scatter(plot_df["renewables_share_energy_latest"], y, s=40, label=str(LATEST_YEAR))

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["country"])

    ax.invert_yaxis()

    ax.set_xlabel("Renewables share of primary energy (%)")
    ax.set_title(f"Top {len(plot_df)} Improvers in Renewables Share")

    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "top_improvers.png"), dpi=300)
    plt.close()



def plot_latest_leaders(latest, n=15):

    plot_df = latest.sort_values(
        "renewables_share_energy",
        ascending=False
    ).head(n)

    fig, ax = plt.subplots(figsize=(11, 8))

    ax.barh(
        plot_df["country"],
        plot_df["renewables_share_energy"]
    )

    ax.invert_yaxis()

    ax.set_xlabel("Renewables share of primary energy (%)")
    ax.set_title(f"Top {n} Countries by Renewable Energy Share (Latest Year)")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "latest_leaders.png"), dpi=300)
    plt.close()



def plot_latest_lollipop(latest, n=15):

    # sort by latest renewable share
    plot_df = latest.sort_values(
        "renewables_share_energy",
        ascending=False
    ).head(n).reset_index(drop=True)

    # create y positions
    y = np.arange(len(plot_df))

    values = plot_df["renewables_share_energy"]

    fig, ax = plt.subplots(figsize=(11, 8))

    # draw lines (from 0 to value)
    ax.hlines(
        y=y,
        xmin=0,
        xmax=values,
        linewidth=2
    )

    # draw dots
    ax.scatter(values, y, s=60)

    # labels
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["country"])

    ax.invert_yaxis()

    ax.set_xlabel("Renewables share of primary energy (%)")
    ax.set_title(f"Top {n} Countries by Renewable Share (Latest Year)")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "latest_lollipop.png"), dpi=300)
    plt.close()


def plot_country_mix_area(mix_long, country="Ireland"):
    sub = mix_long[mix_long["country"] == country].copy()
    order = ["coal", "gas", "oil", "hydro", "wind", "solar", "nuclear"]
    sub["source"] = pd.Categorical(sub["source"], categories=order, ordered=True)
    wide = (
        sub.pivot(index="year", columns="source", values="share")
        .fillna(0)
        .reindex(columns=order)
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.stackplot(wide.index, wide.T, labels=wide.columns)

    ax.set_title(f"Electricity Mix Over Time: {country}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of electricity (%)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{country.lower().replace(' ', '_')}_mix_area.png"), dpi=300)
    plt.close()


def main():
    '''
    Links:
        data : https://github.com/owid/energy-data?tab=readme-ov-file
    '''
    ensure_dirs()

    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = add_change_features(df)

    latest = make_latest_snapshot(df)
    focus = make_focus_countries(df)
    mix_long = make_energy_mix_long(focus)
    improvers = make_top_improvers(df, n=15)
    
    save_outputs(df, latest, focus, mix_long, improvers)

    plot_scatter_latest(latest)
    plot_focus_lines(focus)
    plot_focus_clean_vs_fossil(focus)
    plot_heatmap_focus(mix_long)
    plot_top_improvers(improvers)
    plot_latest_leaders(latest, n=15)
    plot_latest_lollipop(latest, n=15)

    plot_country_mix_area(mix_long, country="Ireland")
    plot_country_mix_area(mix_long, country="Denmark")

    print("Saved data to:", DATA_OUT_DIR)
    print("Saved plots to:", PLOT_DIR)


if __name__ == "__main__":
    main()
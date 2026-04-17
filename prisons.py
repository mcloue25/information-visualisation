import re
import pdfplumber
import pandas as pd



import re
import pdfplumber
import pandas as pd


def extract_prison_data(pdf_path):
    """
    Extract prison nationality data from the PDF and save:
    - wide CSV: one row per year/nationality
    - long CSV: one row per year/nationality/gender

    Source:
    https://www.irishprisons.ie/wp-content/uploads/documents_pdf/PERSONS-COMMITTED-by-NATIONALITY-GROUP-Year-2007-to-Year-2024.pdf

    Args:
        pdf_path (str): Path to local PDF file
    """

    # Match year header like: "Nationality Group classified by gender in Year 2024"
    year_pattern = re.compile(r"Nationality Group classified by gender in Year (\d{4})")

    # Match table rows like: "Irish 717 4,702 5,419 75.4"
    row_pattern = re.compile(r"^(.*?)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d.]+)$")

    records = []
    current_year = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue

                # Update the current year when a new section starts
                year_match = year_pattern.search(line)
                if year_match:
                    current_year = int(year_match.group(1))
                    continue

                # Skip the table header row
                if line.startswith("Nationality Group Female Male Total %"):
                    continue

                # Try to parse a data row
                match = row_pattern.match(line)
                if not match or current_year is None:
                    continue

                nationality, female, male, total, percent = match.groups()

                # Skip the summary total row
                if nationality == "Total":
                    continue

                records.append({
                    "Year": current_year,
                    "Nationality": nationality,
                    "Female": int(female.replace(",", "")),
                    "Male": int(male.replace(",", "")),
                    "Total": int(total.replace(",", "")),
                    "Percent": float(percent)
                })

    # Wide format: one row per year + nationality
    df_wide = pd.DataFrame(records).sort_values(["Year", "Nationality"])
    df_wide.to_csv("data/results/prison_nationality_wide.csv", index=False)

    # Long format: one row per year + nationality + gender
    df_long = df_wide.melt(
        id_vars=["Year", "Nationality", "Total", "Percent"],
        value_vars=["Female", "Male"],
        var_name="Gender",
        value_name="Count"
    )
    df_long.to_csv("data/results/prison_nationality_long.csv", index=False)

    print("Wide data:")
    print(df_wide.head())
    print("\nLong data:")
    print(df_long.head())

    return df_wide, df_long


import os
import pandas as pd
import matplotlib.pyplot as plt


WIDE_CSV = "data/results/prison_nationality_wide.csv"
LONG_CSV = "data/results/prison_nationality_long.csv"
PLOTS_DIR = "plots"


def load_data():
    """Load cleaned prison datasets."""
    df_wide = pd.read_csv(WIDE_CSV)
    df_long = pd.read_csv(LONG_CSV)
    return df_wide, df_long


def setup_output_dir():
    """Create plots folder if it does not exist."""
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_stacked_area_share(df_wide):
    """
    Plot 100% stacked area chart:
    share of prison committals by nationality over time.
    """
    # Pivot to Year x Nationality
    pivot = df_wide.pivot(index="Year", columns="Nationality", values="Total").fillna(0)

    # Convert raw counts to shares
    share = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(share.index, share.T, labels=share.columns)

    ax.set_title("Share of Prison Committals by Nationality Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of Total Committals")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "stacked_area_share.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_grouped_bar_for_year(df_long, year=2024):
    """
    Plot grouped bar chart for one selected year:
    male vs female counts by nationality.
    """
    df_year = df_long[df_long["Year"] == year].copy()

    # Pivot to Nationality x Gender
    pivot = df_year.pivot(index="Nationality", columns="Gender", values="Count").fillna(0)

    ax = pivot.plot(kind="bar", figsize=(12, 6))

    ax.set_title(f"Prison Committals by Nationality and Gender ({year})")
    ax.set_xlabel("Nationality")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f"grouped_bar_{year}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_heatmap(df_wide):
    """
    Plot heatmap of total committals:
    nationality by year.
    """
    pivot = df_wide.pivot(index="Nationality", columns="Year", values="Total").fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_title("Heatmap of Prison Committals by Nationality and Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Nationality")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    cbar = plt.colorbar(im)
    cbar.set_label("Total Committals")

    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "heatmap_total.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()



def main():
    '''  Mian 
    '''
    pdf_path = "data/csv/PERSONS-COMMITTED-by-NATIONALITY-GROUP-Year-2007-to-Year-2024.pdf"
    # extract_prison_data(pdf_path)

    setup_output_dir()
    df_wide, df_long = load_data()

    print("Wide data preview:")
    print(df_wide.head(), "\n")

    print("Long data preview:")
    print(df_long.head(), "\n")

    plot_stacked_area_share(df_wide)
    plot_grouped_bar_for_year(df_long, year=2024)
    plot_heatmap(df_wide)


if __name__ == "__main__":
    main()

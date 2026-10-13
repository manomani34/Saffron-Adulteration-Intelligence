
# Import libraries for spectral data processing and plotting.
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Define the project root directory from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the input dataset and candidate-region report paths.
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "total-color-data.csv"
REGIONS_PATH = PROJECT_ROOT / "reports" / "spectral_region_lopo_details.csv"

# Define the output directory for the spectral inspection plots.
OUTPUT_DIR = PROJECT_ROOT / "reports" / "candidate_region_spectra"

# Define the dyes to inspect, in the order used by the previous analysis.
DYES = [
    "Orange 2",
    "Ponceau 4R",
    "Quinoline Yellow",
    "Tartrazine",
]

# Define the number of highest-ranked regions to inspect per dye.
TOP_REGIONS = 3


def load_analysis_data():
    # Read the original dataset and the previously selected spectral regions.
    data = pd.read_csv(DATA_PATH)
    regions = pd.read_csv(REGIONS_PATH)

    # Check that all required metadata columns exist.
    required_columns = {
        "Sample_ID",
        "Target_Label",
        "Adulterant_Type",
        "Dataset_Role",
    }
    missing_columns = required_columns.difference(data.columns)

    # Stop with a clear error if the dataset schema is unexpected.
    if missing_columns:
        raise ValueError(
            f"Missing required dataset columns: {sorted(missing_columns)}"
        )

    # Identify columns whose names represent numeric chemical shifts in ppm.
    ppm_columns = []
    for column in data.columns:
        # Skip metadata columns and keep numeric-looking spectral columns.
        try:
            ppm_value = float(column)
        except (TypeError, ValueError):
            continue

        # Store the original column name and its numeric ppm value.
        ppm_columns.append((column, ppm_value))

    # Sort the spectral columns by chemical shift.
    ppm_columns.sort(key=lambda item: item[1])

    # Reject the input if no spectral columns were detected.
    if not ppm_columns:
        raise ValueError("No numeric ppm spectral columns were found.")

    # Convert the spectral measurements to numeric values.
    spectral_columns = [item[0] for item in ppm_columns]
    data[spectral_columns] = data[spectral_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Keep only genuine and adulterated samples from Train and Test roles.
    role_mask = data["Dataset_Role"].astype(str).str.casefold().isin(
        ["train", "test"]
    )
    label_mask = data["Target_Label"].astype(str).str.casefold().isin(
        ["genuine", "adulterated"]
    )
    data = data.loc[role_mask & label_mask].copy()

    # Keep only spectral-region rankings calculated from the full training data.
    regions = regions.loc[regions["Analysis"].eq("FullData")].copy()

    # Convert the region boundaries and ranks into numeric values.
    for column in ["Region_Rank", "Start_PPM", "End_PPM"]:
        regions[column] = pd.to_numeric(regions[column], errors="coerce")

    # Return the filtered samples, candidate regions, and spectral axis.
    return data, regions, spectral_columns, ppm_columns


def plot_dye_regions(data, regions, spectral_columns, ppm_columns, dye):
    # Select the highest-ranked candidate regions for the current dye.
    dye_regions = (
        regions.loc[regions["Dye"].eq(dye)]
        .sort_values("Region_Rank")
        .head(TOP_REGIONS)
    )

    # Report a missing region definition instead of producing an empty plot.
    if dye_regions.empty:
        print(f"WARNING: No candidate regions found for {dye}.")
        return

    # Separate genuine samples from all adulterated samples.
    genuine_mask = data["Target_Label"].astype(str).str.casefold().eq(
        "genuine"
    )
    adulterated_mask = data["Target_Label"].astype(str).str.casefold().eq(
        "adulterated"
    )

    # Find adulterated samples whose recorded adulterant contains this dye name.
    dye_mask = data["Adulterant_Type"].astype(str).str.contains(
        re.escape(dye), case=False, regex=True, na=False
    )
    dye_positive_mask = adulterated_mask & dye_mask

    # Report the sample counts before plotting the spectral regions.
    genuine_count = int(genuine_mask.sum())
    adulterated_count = int(adulterated_mask.sum())
    dye_positive_count = int(dye_positive_mask.sum())

    print(
        f"{dye}: genuine={genuine_count}, "
        f"adulterated={adulterated_count}, "
        f"samples containing dye={dye_positive_count}"
    )

    # Skip the dye if no samples match its recorded label.
    if genuine_count == 0 or dye_positive_count == 0:
        print(f"WARNING: Insufficient matching groups for {dye}.")
        return

    # Create one subplot for each selected spectral region.
    figure, axes = plt.subplots(
        1,
        len(dye_regions),
        figsize=(16, 5),
        squeeze=False,
    )
    axes = axes.ravel()

    # Retrieve the complete spectral matrix for each comparison group.
    genuine_spectra = data.loc[genuine_mask, spectral_columns].to_numpy(
        dtype=float
    )
    adulterated_spectra = data.loc[
        adulterated_mask, spectral_columns
    ].to_numpy(dtype=float)
    dye_spectra = data.loc[
        dye_positive_mask, spectral_columns
    ].to_numpy(dtype=float)

    # Plot each selected region using the original spectral measurements.
    for axis, (_, region) in zip(axes, dye_regions.iterrows()):
        # Select ppm bins falling within the candidate region.
        start_ppm = float(region["Start_PPM"])
        end_ppm = float(region["End_PPM"])
        selected_bins = [
            index
            for index, (_, ppm) in enumerate(ppm_columns)
            if start_ppm <= ppm <= end_ppm
        ]

        # Warn if no spectral bins fall inside this region.
        if not selected_bins:
            axis.set_title("No spectral bins found")
            continue

        # Build the ppm axis and extract the corresponding spectra.
        ppm_axis = np.array(
            [ppm_columns[index][1] for index in selected_bins]
        )
        genuine_values = genuine_spectra[:, selected_bins]
        adulterated_values = adulterated_spectra[:, selected_bins]
        dye_values = dye_spectra[:, selected_bins]

        # Draw individual genuine spectra with transparency.
        for spectrum in genuine_values:
            axis.plot(
                ppm_axis,
                spectrum,
                color="steelblue",
                alpha=0.15,
                linewidth=0.8,
            )

        # Draw individual dye-containing spectra for direct inspection.
        for spectrum in dye_values:
            axis.plot(
                ppm_axis,
                spectrum,
                color="crimson",
                alpha=0.45,
                linewidth=1.0,
            )

        # Draw the genuine median spectrum as a robust group reference.
        axis.plot(
            ppm_axis,
            np.nanmedian(genuine_values, axis=0),
            color="navy",
            linewidth=2.2,
            label="Genuine median",
        )

        # Draw the median across all adulterated samples for context.
        axis.plot(
            ppm_axis,
            np.nanmedian(adulterated_values, axis=0),
            color="darkorange",
            linewidth=2.0,
            label="All adulterated median",
        )

        # Draw the median for samples recorded as containing this dye.
        axis.plot(
            ppm_axis,
            np.nanmedian(dye_values, axis=0),
            color="crimson",
            linewidth=2.2,
            linestyle="--",
            label=f"{dye} samples median",
        )

        # Add the region boundaries and labels to the subplot.
        axis.set_title(
            f"Rank {int(region['Region_Rank'])}\n"
            f"{start_ppm:.5f}–{end_ppm:.5f} ppm"
        )
        axis.set_xlabel("Chemical shift (ppm)")
        axis.set_ylabel("Raw spectral intensity")
        axis.invert_xaxis()
        axis.grid(alpha=0.25)

    # Add a shared legend and an explanatory figure title.
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3)
    figure.suptitle(
        f"{dye}: raw NMR spectra in candidate regions",
        y=1.05,
    )
    figure.tight_layout()

    # Save the figure under the project's reports directory.
    output_path = OUTPUT_DIR / f"{dye.lower().replace(' ', '_')}.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    # Print the saved plot location.
    print(f"Saved plot: {output_path}")


def main():
    # Create the output directory if it does not already exist.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the dataset and candidate spectral regions.
    data, regions, spectral_columns, ppm_columns = load_analysis_data()

    # Print the size of the dataset used for this visual inspection.
    print(f"Samples inspected: {len(data)}")
    print(f"Spectral bins: {len(spectral_columns)}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Generate a spectral comparison plot for each selected dye.
    for dye in DYES:
        plot_dye_regions(
            data,
            regions,
            spectral_columns,
            ppm_columns,
            dye,
        )


if __name__ == "__main__":
    # Start the spectral-region inspection workflow.
    main()

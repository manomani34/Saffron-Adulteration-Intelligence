
# Import libraries for numerical analysis and CSV reporting.
from pathlib import Path

import numpy as np
import pandas as pd


# Define the project root based on this script's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the source dataset and selected spectral-region report.
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "total-color-data.csv"
REGIONS_PATH = PROJECT_ROOT / "reports" / "spectral_region_lopo_details.csv"

# Define the directory where numerical results will be saved.
OUTPUT_DIR = PROJECT_ROOT / "reports"

# Define the dyes to evaluate.
DYES = [
    "Orange 2",
    "Ponceau 4R",
    "Quinoline Yellow",
    "Tartrazine",
]

# Define how many top-ranked regions to inspect for each dye.
TOP_REGIONS = 3


def load_training_data():
    # Load the dataset and the regions selected in the earlier analysis.
    data = pd.read_csv(DATA_PATH)
    regions = pd.read_csv(REGIONS_PATH)

    # Identify columns that represent chemical shifts in ppm.
    ppm_columns = []
    for column in data.columns:
        # Keep columns whose names can be interpreted as numeric ppm values.
        try:
            ppm_value = float(column)
        except (TypeError, ValueError):
            continue

        # Save the original column name and numeric chemical shift.
        ppm_columns.append((column, ppm_value))

    # Sort spectral columns in increasing ppm order.
    ppm_columns.sort(key=lambda item: item[1])

    # Stop if the dataset does not contain numeric spectral columns.
    if not ppm_columns:
        raise ValueError("No numeric ppm columns were found.")

    # Keep only training samples with genuine or adulterated labels.
    role_mask = data["Dataset_Role"].astype(str).str.casefold().eq("train")
    label_mask = data["Target_Label"].astype(str).str.casefold().isin(
        ["genuine", "adulterated"]
    )
    data = data.loc[role_mask & label_mask].copy()

    # Keep only candidate regions selected using the full training dataset.
    regions = regions.loc[regions["Analysis"].eq("FullData")].copy()

    # Convert the region rank and ppm boundaries to numeric values.
    for column in ["Region_Rank", "Start_PPM", "End_PPM"]:
        regions[column] = pd.to_numeric(regions[column], errors="coerce")

    # Return the training samples, regions, and ordered spectral axis.
    return data, regions, ppm_columns


def calculate_effect(genuine_values, positive_values):
    # Calculate medians and interquartile ranges for both groups.
    genuine_median = float(np.median(genuine_values))
    positive_median = float(np.median(positive_values))

    # Calculate the 25th and 75th percentiles for the genuine group.
    genuine_q25, genuine_q75 = np.percentile(genuine_values, [25, 75])

    # Calculate the 25th and 75th percentiles for dye-positive samples.
    positive_q25, positive_q75 = np.percentile(positive_values, [25, 75])

    # Calculate group standard deviations for a standardized difference.
    genuine_sd = float(np.std(genuine_values, ddof=1)) if len(genuine_values) > 1 else 0.0
    positive_sd = float(np.std(positive_values, ddof=1)) if len(positive_values) > 1 else 0.0

    # Calculate a pooled standard deviation when sample sizes allow it.
    total_degrees = len(genuine_values) + len(positive_values) - 2
    if total_degrees > 0:
        pooled_variance = (
            (len(genuine_values) - 1) * genuine_sd**2
            + (len(positive_values) - 1) * positive_sd**2
        ) / total_degrees
        pooled_sd = float(np.sqrt(max(pooled_variance, 0.0)))
    else:
        pooled_sd = 0.0

    # Calculate Cohen's d, leaving it undefined if both groups have no variation.
    cohen_d = (
        (positive_median - genuine_median) / pooled_sd
        if pooled_sd > 0
        else np.nan
    )

    # Determine whether the two groups' interquartile intervals overlap.
    iqr_overlap = max(
        0.0,
        min(genuine_q75, positive_q75) - max(genuine_q25, positive_q25),
    )

    # Calculate the width of the combined interquartile range.
    combined_iqr_width = max(genuine_q75, positive_q75) - min(
        genuine_q25, positive_q25
    )

    # Express IQR overlap as a proportion of the combined interval width.
    overlap_ratio = (
        iqr_overlap / combined_iqr_width
        if combined_iqr_width > 0
        else float(genuine_median == positive_median)
    )

    # Return the numerical comparison for this candidate region.
    return {
        "Genuine_Median": genuine_median,
        "Genuine_Q25": float(genuine_q25),
        "Genuine_Q75": float(genuine_q75),
        "Dye_Median": positive_median,
        "Dye_Q25": float(positive_q25),
        "Dye_Q75": float(positive_q75),
        "Median_Difference": positive_median - genuine_median,
        "Cohen_D_Approx": cohen_d,
        "IQR_Overlap_Ratio": float(overlap_ratio),
    }


def main():
    # Load training data and the previously selected spectral regions.
    data, regions, ppm_columns = load_training_data()

    # Extract the names and numeric values of all spectral columns.
    spectral_names = [item[0] for item in ppm_columns]
    spectral_axis = np.array([item[1] for item in ppm_columns])

    # Convert all training spectra to a numeric matrix.
    spectra = data[spectral_names].to_numpy(dtype=float)

    # Identify genuine samples for the negative comparison group.
    genuine_mask = data["Target_Label"].astype(str).str.casefold().eq(
        "genuine"
    )

    # Prepare lists to collect summary statistics and sample-level values.
    summary_rows = []
    detail_rows = []

    # Analyze the highest-ranked candidate regions for each dye.
    for dye in DYES:
        # Select the top-ranked regions for the current dye.
        dye_regions = (
            regions.loc[regions["Dye"].eq(dye)]
            .sort_values("Region_Rank")
            .head(TOP_REGIONS)
        )

        # Find adulterated samples whose recorded label contains this dye.
        adulterated_mask = data["Target_Label"].astype(str).str.casefold().eq(
            "adulterated"
        )
        dye_mask = data["Adulterant_Type"].astype(str).str.contains(
            dye, case=False, regex=False, na=False
        )
        positive_mask = adulterated_mask & dye_mask

        # Print the training group sizes for this dye.
        print(
            f"{dye}: genuine={int(genuine_mask.sum())}, "
            f"dye-positive={int(positive_mask.sum())}"
        )

        # Skip the dye if no positive training samples are found.
        if not positive_mask.any():
            print(f"WARNING: No training positives found for {dye}.")
            continue

        # Analyze each selected region independently.
        for _, region in dye_regions.iterrows():
            # Find the spectral bins inside the candidate ppm interval.
            start_ppm = float(region["Start_PPM"])
            end_ppm = float(region["End_PPM"])
            bin_mask = (spectral_axis >= start_ppm) & (
                spectral_axis <= end_ppm
            )

            # Skip a region if no spectral bins match its boundaries.
            if not bin_mask.any():
                continue

            # Reduce each sample's region to its mean spectral intensity.
            region_values = np.nanmean(spectra[:, bin_mask], axis=1)

            # Separate genuine and dye-positive sample values.
            genuine_values = region_values[genuine_mask.to_numpy()]
            positive_values = region_values[positive_mask.to_numpy()]

            # Calculate robust group statistics and effect-size estimates.
            metrics = calculate_effect(genuine_values, positive_values)

            # Store a summary row for this dye and region.
            summary_rows.append(
                {
                    "Dye": dye,
                    "Region_Rank": int(region["Region_Rank"]),
                    "Start_PPM": start_ppm,
                    "End_PPM": end_ppm,
                    "Spectral_Bins": int(bin_mask.sum()),
                    "Genuine_Count": len(genuine_values),
                    "Dye_Positive_Count": len(positive_values),
                    **metrics,
                }
            )

            # Save each training sample's value for later inspection.
            for index, sample_value in enumerate(region_values):
                # Identify the sample's actual label and dye-positive status.
                sample = data.iloc[index]
                detail_rows.append(
                    {
                        "Dye_Analyzed": dye,
                        "Region_Rank": int(region["Region_Rank"]),
                        "Start_PPM": start_ppm,
                        "End_PPM": end_ppm,
                        "Sample_ID": sample["Sample_ID"],
                        "Target_Label": sample["Target_Label"],
                        "Adulterant_Type": sample["Adulterant_Type"],
                        "Is_Genuine": bool(genuine_mask.iloc[index]),
                        "Is_Dye_Positive": bool(positive_mask.iloc[index]),
                        "Region_Mean_Intensity": float(sample_value),
                    }
                )

    # Define the output paths for the summary and sample-level reports.
    summary_path = OUTPUT_DIR / "candidate_region_numeric_summary.csv"
    details_path = OUTPUT_DIR / "candidate_region_numeric_details.csv"

    # Save both reports as UTF-8 CSV files.
    pd.DataFrame(summary_rows).to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(detail_rows).to_csv(
        details_path, index=False, encoding="utf-8-sig"
    )

    # Print the compact summary and output locations.
    print("\nNumerical summary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nSummary report: {summary_path}")
    print(f"Sample details: {details_path}")


if __name__ == "__main__":
    # Run the training-only numerical comparison.
    main()

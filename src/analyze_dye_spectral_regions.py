# Import paths for reading the dataset and writing analysis reports.
from pathlib import Path

# Import NumPy and Pandas for spectral calculations and report generation.
import numpy as np
import pandas as pd

# Reuse the existing training-data loader and dye definitions.
from cross_validate_multilabel_svm import DYES, load_training_data

# Define the dyes selected for this exploratory spectral comparison.
TARGET_DYES = [
    "Orange 2",
    "Ponceau 4R",
    "Quinoline Yellow",
    "Tartrazine",
]

# Define the number of neighboring spectral bins in each analysis window.
WINDOW_SIZE = 31

# Define the maximum number of non-overlapping regions to report per dye.
TOP_REGIONS = 10

# Define the output file for the ranked spectral regions.
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "dye_spectral_region_analysis.csv"
)


# Calculate Cohen's d while handling regions with zero pooled variance.
def calculate_cohens_d(positive_values, negative_values):
    # Calculate the number of samples in both label groups.
    positive_count = len(positive_values)
    negative_count = len(negative_values)

    # Calculate the mean signal intensity for each group.
    positive_mean = float(np.mean(positive_values))
    negative_mean = float(np.mean(negative_values))

    # Calculate the sample variance for each group.
    positive_variance = float(np.var(positive_values, ddof=1))
    negative_variance = float(np.var(negative_values, ddof=1))

    # Calculate the pooled standard deviation.
    pooled_variance = (
        ((positive_count - 1) * positive_variance)
        + ((negative_count - 1) * negative_variance)
    ) / (positive_count + negative_count - 2)

    # Return zero when both groups have no measurable variance.
    if pooled_variance <= 0:
        return 0.0

    # Return the standardized mean difference between the groups.
    return (positive_mean - negative_mean) / np.sqrt(pooled_variance)


# Rank spectral regions using local effect sizes and non-overlapping windows.
def analyze_dye(features, targets, sample_ids, dye):
    # Extract the presence or absence labels for the selected dye.
    labels = targets[dye].to_numpy(dtype=int)

    # Identify the positive and negative sample positions.
    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)

    # Report the sample counts before interpreting any spectral differences.
    positive_count = len(positive_indices)
    negative_count = len(negative_indices)

    # Require enough samples to calculate sample variances.
    if positive_count < 2 or negative_count < 2:
        print(f"Skipping {dye}: too few positive or negative samples.")
        return []

    # Convert spectral column names into their corresponding ppm positions.
    ppm_values = pd.to_numeric(
        pd.Index(features.columns).astype(str),
        errors="coerce",
    ).to_numpy(dtype=float)

    # Stop if the spectral axis cannot be recovered from the column names.
    if not np.isfinite(ppm_values).all():
        raise ValueError(
            "Some spectral column names are not numeric ppm values. "
            "Inspect the CSV header before continuing."
        )

    # Sort spectral bins by ppm so reported regions have consistent boundaries.
    sorted_indices = np.argsort(ppm_values)
    sorted_ppm = ppm_values[sorted_indices]

    # Convert the spectral matrix to a numeric array in ppm order.
    spectra = features.iloc[:, sorted_indices].to_numpy(dtype=float)

    # Select the spectra for samples containing the selected dye.
    positive_spectra = spectra[positive_indices]

    # Select the spectra for samples without the selected dye.
    negative_spectra = spectra[negative_indices]

    # Calculate the mean intensity at every spectral bin for both groups.
    positive_mean_spectrum = positive_spectra.mean(axis=0)
    negative_mean_spectrum = negative_spectra.mean(axis=0)

    # Calculate sample variances at every spectral bin for both groups.
    positive_variance_spectrum = positive_spectra.var(axis=0, ddof=1)
    negative_variance_spectrum = negative_spectra.var(axis=0, ddof=1)

    # Calculate pooled variance across the two label groups at each bin.
    pooled_variance_spectrum = (
        (positive_count - 1) * positive_variance_spectrum
        + (negative_count - 1) * negative_variance_spectrum
    ) / (positive_count + negative_count - 2)

    # Calculate the pooled standard deviation for each spectral bin.
    pooled_std_spectrum = np.sqrt(pooled_variance_spectrum)

    # Calculate the standardized difference between group means.
    effect_sizes = np.divide(
        positive_mean_spectrum - negative_mean_spectrum,
        pooled_std_spectrum,
        out=np.zeros_like(positive_mean_spectrum),
        where=pooled_std_spectrum > 0,
    )

    # Calculate a moving average of absolute effect sizes over local regions.
    local_effect_strength = (
        pd.Series(np.abs(effect_sizes))
        .rolling(WINDOW_SIZE, center=True, min_periods=WINDOW_SIZE)
        .mean()
        .to_numpy()
    )

    # Store candidate region records before selecting non-overlapping regions.
    candidate_rows = []

    # Visit every complete window along the sorted spectral axis.
    for start_index in range(len(sorted_ppm) - WINDOW_SIZE + 1):
        # Calculate the exclusive end index of the current spectral window.
        end_index = start_index + WINDOW_SIZE

        # Calculate the local mean absolute standardized difference.
        region_score = float(
            np.mean(np.abs(effect_sizes[start_index:end_index]))
        )

        # Calculate the mean spectrum intensity within this region per sample.
        positive_region_means = positive_spectra[
            :, start_index:end_index
        ].mean(axis=1)

        # Calculate the equivalent regional intensities for negative samples.
        negative_region_means = negative_spectra[
            :, start_index:end_index
        ].mean(axis=1)

        # Calculate a region-level standardized mean difference.
        region_cohens_d = calculate_cohens_d(
            positive_region_means,
            negative_region_means,
        )

        # Record the region boundaries and descriptive effect-size measures.
        candidate_rows.append(
            {
                "Dye": dye,
                "Positive_Samples": positive_count,
                "Negative_Samples": negative_count,
                "Start_PPM": float(sorted_ppm[start_index]),
                "End_PPM": float(sorted_ppm[end_index - 1]),
                "Center_PPM": float(
                    (sorted_ppm[start_index] + sorted_ppm[end_index - 1]) / 2
                ),
                "Mean_Absolute_Bin_Effect": region_score,
                "Region_Cohens_D": region_cohens_d,
                "Positive_Mean_Intensity": float(
                    positive_region_means.mean()
                ),
                "Negative_Mean_Intensity": float(
                    negative_region_means.mean()
                ),
                "Mean_Intensity_Difference": float(
                    positive_region_means.mean()
                    - negative_region_means.mean()
                ),
                "_start_index": start_index,
            }
        )

    # Sort candidate regions from strongest to weakest local spectral effect.
    candidate_rows.sort(
        key=lambda row: row["Mean_Absolute_Bin_Effect"],
        reverse=True,
    )

    # Select the strongest regions while preventing heavy overlap.
    selected_rows = []
    selected_starts = []

    # Examine candidate windows in descending order of local effect strength.
    for candidate in candidate_rows:
        # Get the starting bin for the candidate region.
        candidate_start = candidate["_start_index"]

        # Reject regions that overlap already selected windows.
        if any(
            abs(candidate_start - selected_start) < WINDOW_SIZE
            for selected_start in selected_starts
        ):
            continue

        # Keep the region and remember its starting position.
        selected_rows.append(candidate)
        selected_starts.append(candidate_start)

        # Stop after collecting the requested number of regions.
        if len(selected_rows) >= TOP_REGIONS:
            break

    # Remove internal ranking fields from the final report records.
    for row in selected_rows:
        # Remove the internal array index used only for overlap suppression.
        row.pop("_start_index", None)

    # Return the selected regions for the current dye.
    return selected_rows


# Run the exploratory analysis for all selected dyes and save the results.
def main():
    # Load the eligible training spectra and their multi-label targets.
    features, targets, sample_ids = load_training_data()

    # Create a list to collect spectral-region results across dyes.
    all_rows = []

    # Print the dataset dimensions for auditability.
    print(f"Training samples: {len(features)}")
    print(f"Spectral bins: {features.shape[1]}")
    print(f"Window size: {WINDOW_SIZE} bins")
    print(f"Selected dyes: {len(TARGET_DYES)}")

    # Analyze each selected dye independently.
    for dye in TARGET_DYES:
        # Rank the local spectral regions for this dye.
        dye_rows = analyze_dye(features, targets, sample_ids, dye)

        # Add the current dye's results to the combined report.
        all_rows.extend(dye_rows)

        # Print the number of positive samples to clarify evidence strength.
        positive_count = int(targets[dye].sum())
        print(
            f"{dye}: positives={positive_count}, "
            f"negatives={len(targets) - positive_count}, "
            f"regions={len(dye_rows)}"
        )

    # Convert all ranked regions into a tabular report.
    results = pd.DataFrame(all_rows)

    # Stop if no dye produced reportable regions.
    if results.empty:
        raise ValueError("No spectral regions were generated.")

    # Sort the report by dye and descending local spectral effect strength.
    results = results.sort_values(
        ["Dye", "Mean_Absolute_Bin_Effect"],
        ascending=[True, False],
    )

    # Create the output directory if necessary.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save the ranked spectral-region report.
    results.to_csv(OUTPUT_PATH, index=False)

    # Display a concise table of the strongest regions.
    display_columns = [
        "Dye",
        "Positive_Samples",
        "Start_PPM",
        "End_PPM",
        "Mean_Absolute_Bin_Effect",
        "Region_Cohens_D",
        "Mean_Intensity_Difference",
    ]

    # Print the first few rows for each selected dye.
    print("\nTop spectral regions:")
    for dye in TARGET_DYES:
        # Select the strongest regions for the current dye.
        dye_results = results[results["Dye"] == dye].head(5)

        # Display the regions with a consistent numeric format.
        print(f"\n{dye}")
        print(
            dye_results[display_columns]
            .round(5)
            .to_string(index=False)
        )

    # Print the report path and explain the exploratory limitation.
    print(f"\nReport: {OUTPUT_PATH}")
    print(
        "Note: These regions are exploratory associations, not proof of "
        "dye-specific chemical signals. Mixed samples, small class counts, "
        "and correlated neighboring bins can influence the ranking."
    )


# Execute the analysis only when this script is run directly.
if __name__ == "__main__":
    main()

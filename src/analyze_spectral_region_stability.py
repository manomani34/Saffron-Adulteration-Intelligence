from pathlib import Path

import numpy as np
import pandas as pd


# Define the project paths used by this analysis.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Point to the audited customer dataset.
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "total-color-data.csv"

# Define the folder and output files for the stability analysis.
REPORT_DIR = PROJECT_ROOT / "reports"
DETAILS_PATH = REPORT_DIR / "spectral_region_lopo_details.csv"
SUMMARY_PATH = REPORT_DIR / "spectral_region_lopo_summary.csv"

# Define the dyes whose spectral regions will be checked.
TARGET_DYES = [
    "Orange 2",
    "Ponceau 4R",
    "Quinoline Yellow",
    "Tartrazine",
]

# Configure the number of bins per window and the number of top windows.
WINDOW_BINS = 31
TOP_N_WINDOWS = 10

# Set the ppm tolerance for matching a region to its original location.
MATCH_TOLERANCE_PPM = 0.012


# Load training samples and identify spectral columns from numeric ppm names.
def load_training_data():
    # Read the source dataset without changing its original contents.
    dataset = pd.read_csv(DATA_PATH)

    # Normalize column names so metadata lookup is consistent.
    dataset.columns = [str(column).strip() for column in dataset.columns]

    # Define the metadata columns required by this script.
    required_columns = {
        "Sample_ID",
        "Target_Label",
        "Adulterant_Type",
        "Dataset_Role",
    }

    # Stop early if the dataset schema differs from the expected schema.
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(
            "Missing expected metadata columns: "
            + ", ".join(sorted(missing_columns))
        )

    # Keep only training rows with genuine or adulterated saffron labels.
    training_mask = (
        dataset["Dataset_Role"].astype(str).str.strip().str.casefold().eq("train")
        & dataset["Target_Label"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .isin({"genuine", "adulterated"})
    )
    training_data = dataset.loc[training_mask].copy()

    # Find columns whose names can be interpreted as numeric ppm values.
    spectral_columns = []
    for column in training_data.columns:
        try:
            float(column)
            spectral_columns.append(column)
        except (TypeError, ValueError):
            continue

    # Sort spectral columns by their numeric ppm values.
    spectral_columns = sorted(spectral_columns, key=float)

    # Require enough bins to calculate the configured spectral windows.
    if len(spectral_columns) < WINDOW_BINS:
        raise ValueError(
            f"Only {len(spectral_columns)} spectral bins were found; "
            f"at least {WINDOW_BINS} are required."
        )

    # Convert the spectral matrix to numeric values and reject invalid data.
    spectra = training_data[spectral_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if spectra.isna().any().any():
        raise ValueError(
            "Missing or non-numeric values were found in the training spectra."
        )

    # Store sample identifiers for traceability in the resulting reports.
    sample_ids = training_data["Sample_ID"].astype(str).to_numpy()

    # Normalize the target labels used to identify genuine samples.
    target_labels = (
        training_data["Target_Label"].astype(str).str.strip().str.casefold()
    )
    genuine_mask = target_labels.eq("genuine").to_numpy()

    # Return the spectral matrix, labels, sample IDs, and ppm axis.
    return (
        spectra.to_numpy(dtype=float),
        training_data["Adulterant_Type"].fillna("").astype(str).to_numpy(),
        sample_ids,
        genuine_mask,
        np.array([float(column) for column in spectral_columns]),
    )


# Determine whether an adulterant label contains a particular dye name.
def contains_dye(adulterant_label, dye_name):
    # Normalize spaces and case before matching the dye name.
    normalized_label = " ".join(str(adulterant_label).casefold().split())
    normalized_dye = " ".join(dye_name.casefold().split())

    # Match complete phrases to avoid confusing similarly named dyes.
    import re

    # Escape the dye name so punctuation is treated literally.
    dye_pattern = re.escape(normalized_dye)

    # Return true only when the complete dye phrase is present.
    return re.search(
        rf"(?<![a-z0-9]){dye_pattern}(?![a-z0-9])",
        normalized_label,
    ) is not None


# Calculate a per-bin standardized difference for positive versus negative samples.
def calculate_bin_scores(spectra, positive_mask):
    # Split the spectra into samples containing and not containing the dye.
    positive_spectra = spectra[positive_mask]
    negative_spectra = spectra[~positive_mask]

    # Count each group to document whether the comparison is well supported.
    positive_count = positive_spectra.shape[0]
    negative_count = negative_spectra.shape[0]

    # Refuse to calculate effects when either comparison group is empty.
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Both positive and negative samples are required.")

    # Calculate the average spectral intensity at each ppm bin.
    positive_mean = positive_spectra.mean(axis=0)
    negative_mean = negative_spectra.mean(axis=0)

    # Use the negative samples' standard deviation as a stable reference scale.
    negative_std = negative_spectra.std(axis=0, ddof=1)

    # Replace zero or invalid scales so constant bins cannot cause division errors.
    valid_scales = np.isfinite(negative_std) & (negative_std > 1e-12)
    safe_scales = np.where(valid_scales, negative_std, np.nan)

    # Calculate the absolute standardized mean difference at each spectral bin.
    bin_scores = np.abs(positive_mean - negative_mean) / safe_scales

    # Treat bins with no variation in the negative group as non-informative.
    bin_scores = np.nan_to_num(
        bin_scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Return the effect scores and group sizes for reporting.
    return bin_scores, positive_count, negative_count


# Select high-scoring spectral windows without overlapping previously selected windows.
def select_top_windows(bin_scores, ppm_axis):
    # Build candidate windows using the mean score across their bins.
    candidates = []
    for start_index in range(0, len(bin_scores) - WINDOW_BINS + 1):
        # Determine the last bin included in this candidate window.
        end_index = start_index + WINDOW_BINS - 1

        # Calculate the average score for the candidate window.
        window_score = float(np.mean(bin_scores[start_index : end_index + 1]))

        # Calculate the ppm boundaries and midpoint for this window.
        start_ppm = float(ppm_axis[start_index])
        end_ppm = float(ppm_axis[end_index])
        center_ppm = float((start_ppm + end_ppm) / 2.0)

        # Store the candidate's metrics and bin boundaries.
        candidates.append(
            {
                "start_index": start_index,
                "end_index": end_index,
                "start_ppm": start_ppm,
                "end_ppm": end_ppm,
                "center_ppm": center_ppm,
                "window_score": window_score,
            }
        )

    # Sort candidate windows from highest to lowest score.
    candidates.sort(key=lambda item: item["window_score"], reverse=True)

    # Keep only windows that do not overlap an already selected window.
    selected_windows = []
    for candidate in candidates:
        # Check whether the candidate overlaps any selected window.
        overlaps_existing = any(
            not (
                candidate["end_index"] < selected["start_index"]
                or candidate["start_index"] > selected["end_index"]
            )
            for selected in selected_windows
        )

        # Skip overlapping candidates to keep the final regions distinct.
        if overlaps_existing:
            continue

        # Add the candidate and stop after selecting the requested number.
        selected_windows.append(candidate)
        if len(selected_windows) >= TOP_N_WINDOWS:
            break

    # Return the selected windows in descending score order.
    return selected_windows


# Match a leave-one-out window to the nearest full-data reference window.
def find_reference_match(window, reference_windows):
    # Find the reference window with the closest ppm midpoint.
    nearest_window = min(
        reference_windows,
        key=lambda reference: abs(
            window["center_ppm"] - reference["center_ppm"]
        ),
    )

    # Calculate the absolute distance between the two window midpoints.
    distance_ppm = abs(window["center_ppm"] - nearest_window["center_ppm"])

    # Report a match only when the midpoint distance is within tolerance.
    if distance_ppm <= MATCH_TOLERANCE_PPM:
        return nearest_window["rank"], distance_ppm

    # Return empty match details when no reference region is sufficiently close.
    return None, distance_ppm


# Run the leave-one-positive-out spectral region stability analysis.
def main():
    # Create the report folder if it does not already exist.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the training spectra and associated labels.
    spectra, adulterant_labels, sample_ids, genuine_mask, ppm_axis = (
        load_training_data()
    )

    # Initialize collections for detailed region-level and summary results.
    detail_rows = []
    summary_rows = []

    # Analyze each requested dye independently.
    for dye_name in TARGET_DYES:
        # Identify training samples labelled as containing this dye.
        dye_positive_mask = np.array(
            [contains_dye(label, dye_name) for label in adulterant_labels],
            dtype=bool,
        )

        # Exclude genuine samples from the positive group even if metadata is noisy.
        dye_positive_mask = dye_positive_mask & ~genuine_mask

        # Count positives before performing any leave-one-out iteration.
        positive_indices = np.flatnonzero(dye_positive_mask)
        positive_count = len(positive_indices)

        # Stop for this dye if the training set has too few positive samples.
        if positive_count < 2:
            summary_rows.append(
                {
                    "Dye": dye_name,
                    "Positive_Samples": positive_count,
                    "Evaluated_Holdouts": 0,
                    "Mean_Region_Overlap": np.nan,
                    "Median_Region_Overlap": np.nan,
                    "Assessment": "Insufficient positive samples",
                }
            )
            continue

        # Calculate the reference ranking using all available training samples.
        full_scores, _, _ = calculate_bin_scores(spectra, dye_positive_mask)
        full_windows = select_top_windows(full_scores, ppm_axis)

        # Label the reference windows so they can be tracked after each omission.
        for reference_rank, reference_window in enumerate(full_windows, start=1):
            # Add the rank to the reference window metadata.
            reference_window["rank"] = reference_rank

            # Record the full-data regions as the baseline for comparisons.
            detail_rows.append(
                {
                    "Dye": dye_name,
                    "Analysis": "FullData",
                    "Omitted_Sample_ID": "",
                    "Positive_Samples_Remaining": positive_count,
                    "Negative_Samples": int((~dye_positive_mask).sum()),
                    "Region_Rank": reference_rank,
                    "Start_PPM": reference_window["start_ppm"],
                    "End_PPM": reference_window["end_ppm"],
                    "Center_PPM": reference_window["center_ppm"],
                    "Window_Score": reference_window["window_score"],
                    "Matched_FullData_Rank": reference_rank,
                    "Distance_To_Reference_PPM": 0.0,
                    "Region_Matched": True,
                }
            )

        # Initialize the per-dye overlap scores across held-out positive samples.
        overlap_fractions = []

        # Omit each positive sample once and recompute the region ranking.
        for omitted_index in positive_indices:
            # Keep every training sample except the current positive sample.
            retained_mask = np.ones(len(sample_ids), dtype=bool)
            retained_mask[omitted_index] = False

            # Apply the same omission to the dye's positive labels.
            retained_positive_mask = dye_positive_mask[retained_mask]

            # Skip a holdout if it leaves no positive samples for comparison.
            if retained_positive_mask.sum() == 0:
                continue

            # Recompute region scores using only the retained samples.
            heldout_spectra = spectra[retained_mask]
            heldout_ppm_axis = ppm_axis
            heldout_scores, remaining_positives, remaining_negatives = (
                calculate_bin_scores(
                    heldout_spectra,
                    retained_positive_mask,
                )
            )

            # Select the top non-overlapping windows for this held-out subset.
            heldout_windows = select_top_windows(
                heldout_scores,
                heldout_ppm_axis,
            )

            # Track how many held-out regions remain close to a full-data region.
            matched_reference_ranks = set()

            # Compare each held-out region with the full-data reference ranking.
            for heldout_rank, heldout_window in enumerate(
                heldout_windows,
                start=1,
            ):
                # Find the nearest reference region and its ppm distance.
                matched_rank, distance_ppm = find_reference_match(
                    heldout_window,
                    full_windows,
                )

                # Record successful matches for the overlap calculation.
                if matched_rank is not None:
                    matched_reference_ranks.add(matched_rank)

                # Record the held-out region and its match status.
                detail_rows.append(
                    {
                        "Dye": dye_name,
                        "Analysis": "LeaveOnePositiveOut",
                        "Omitted_Sample_ID": sample_ids[omitted_index],
                        "Positive_Samples_Remaining": remaining_positives,
                        "Negative_Samples": remaining_negatives,
                        "Region_Rank": heldout_rank,
                        "Start_PPM": heldout_window["start_ppm"],
                        "End_PPM": heldout_window["end_ppm"],
                        "Center_PPM": heldout_window["center_ppm"],
                        "Window_Score": heldout_window["window_score"],
                        "Matched_FullData_Rank": matched_rank,
                        "Distance_To_Reference_PPM": distance_ppm,
                        "Region_Matched": matched_rank is not None,
                    }
                )

            # Calculate the fraction of full-data regions recovered in this run.
            overlap_fraction = len(matched_reference_ranks) / max(
                len(full_windows),
                1,
            )
            overlap_fractions.append(overlap_fraction)

            # Store the stability result for this omitted sample.
            summary_rows.append(
                {
                    "Dye": dye_name,
                    "Positive_Samples": positive_count,
                    "Evaluated_Holdouts": 1,
                    "Omitted_Sample_ID": sample_ids[omitted_index],
                    "Matched_Reference_Regions": len(matched_reference_ranks),
                    "Reference_Regions": len(full_windows),
                    "Region_Overlap_Fraction": overlap_fraction,
                    "Assessment": "Exploratory; interpret with sample counts",
                }
            )

        # Summarize the region overlap across all holdouts for this dye.
        if overlap_fractions:
            # Calculate average and median overlap across the holdouts.
            mean_overlap = float(np.mean(overlap_fractions))
            median_overlap = float(np.median(overlap_fractions))
        else:
            # Leave the overlap undefined when no holdout could be evaluated.
            mean_overlap = np.nan
            median_overlap = np.nan

        # Add one overall stability summary row for this dye.
        summary_rows.append(
            {
                "Dye": dye_name,
                "Positive_Samples": positive_count,
                "Evaluated_Holdouts": len(overlap_fractions),
                "Omitted_Sample_ID": "ALL_HOLDOUTS",
                "Matched_Reference_Regions": np.nan,
                "Reference_Regions": len(full_windows),
                "Region_Overlap_Fraction": np.nan,
                "Mean_Region_Overlap": mean_overlap,
                "Median_Region_Overlap": median_overlap,
                "Assessment": (
                    "Higher overlap means more stable region ranking; "
                    "not proof of dye specificity"
                ),
            }
        )

        # Print a compact progress message for the current dye.
        print(
            f"{dye_name}: positives={positive_count}, "
            f"evaluated holdouts={len(overlap_fractions)}, "
            f"mean region overlap={mean_overlap:.3f}"
        )

    # Convert the collected details and summaries into tabular reports.
    details_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)

    # Save the detailed window rankings and the per-dye stability summary.
    details_df.to_csv(DETAILS_PATH, index=False, encoding="utf-8-sig")
    summary_df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    # Print the report paths so the generated files are easy to locate.
    print(f"Detailed results: {DETAILS_PATH}")
    print(f"Summary results:  {SUMMARY_PATH}")


# Execute the analysis only when this file is run as a script.
if __name__ == "__main__":
    main()

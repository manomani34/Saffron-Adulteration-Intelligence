# Imports tools for analyzing SVM decision scores.
from pathlib import Path

import numpy as np
import pandas as pd


# Defines the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Defines the existing test prediction report.
REPORT_PATH = PROJECT_ROOT / "reports" / "multilabel_svm_test_predictions.csv"

# Defines the output path for the score comparison report.
OUTPUT_PATH = PROJECT_ROOT / "reports" / "svm_score_analysis.csv"

# Defines the nine dye labels used by the multi-label classifier.
DYES = [
    "Amaranth",
    "Azorubine",
    "Erythrosine",
    "Orange 2",
    "Orange GGN",
    "Ponceau 4R",
    "Quinoline Yellow",
    "Sunset Yellow",
    "Tartrazine",
]


# Converts a sample's actual dye metadata into a set of dye names.
def parse_actual_dyes(value: object) -> set[str]:
    # Returns an empty set for genuine samples with no listed dyes.
    if pd.isna(value) or not str(value).strip():
        return set()

    # Splits the recorded dye names and removes surrounding whitespace.
    actual_dyes = {
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    }

    # Rejects labels that do not match the expected dye list.
    unexpected_dyes = actual_dyes - set(DYES)

    # Prevents incorrect comparisons if the report contains unexpected labels.
    if unexpected_dyes:
        raise ValueError(
            f"Unexpected dye labels in report: {sorted(unexpected_dyes)}"
        )

    # Returns the actual dyes recorded for this sample.
    return actual_dyes


# Compares SVM decision scores for samples with and without each dye.
def main() -> None:
    # Loads the previously generated test prediction report.
    report_df = pd.read_csv(REPORT_PATH)

    # Stores each test sample's actual dye set.
    actual_dye_sets = [
        parse_actual_dyes(value)
        for value in report_df["Actual_Dyes"]
    ]

    # Stores one summary row for each dye.
    summary_rows = []

    # Evaluates the score distribution for each dye independently.
    for dye in DYES:
        # Defines the score column corresponding to the current dye.
        score_column = f"Score_{dye}"

        # Stops if the expected score column is missing.
        if score_column not in report_df.columns:
            raise ValueError(f"Missing score column: {score_column}")

        # Extracts the decision scores for the current dye.
        scores = pd.to_numeric(
            report_df[score_column],
            errors="raise",
        )

        # Identifies test samples whose actual labels include this dye.
        positive_mask = np.array([
            dye in actual_dyes
            for actual_dyes in actual_dye_sets
        ])

        # Separates scores for samples that actually contain this dye.
        positive_scores = scores[positive_mask]

        # Separates scores for samples that do not contain this dye.
        negative_scores = scores[~positive_mask]

        # Counts actual positive test samples for this dye.
        positive_count = int(positive_mask.sum())

        # Counts actual negative test samples for this dye.
        negative_count = int((~positive_mask).sum())

        # Counts actual positive samples receiving a non-negative score.
        positive_detected = int((positive_scores >= 0).sum())

        # Counts actual negative samples receiving a non-negative score.
        false_positives = int((negative_scores >= 0).sum())

        # Calculates the median score among actual positive samples.
        positive_median = (
            float(positive_scores.median())
            if positive_count > 0
            else np.nan
        )

        # Calculates the median score among actual negative samples.
        negative_median = (
            float(negative_scores.median())
            if negative_count > 0
            else np.nan
        )

        # Measures the difference between the two median scores.
        median_difference = (
            positive_median - negative_median
            if positive_count > 0 and negative_count > 0
            else np.nan
        )

        # Records score distributions and current-threshold outcomes.
        summary_rows.append({
            "Dye": dye,
            "Actual_Positive": positive_count,
            "Actual_Negative": negative_count,
            "Positive_Score_Min": (
                float(positive_scores.min())
                if positive_count > 0 else np.nan
            ),
            "Positive_Score_Median": positive_median,
            "Positive_Score_Max": (
                float(positive_scores.max())
                if positive_count > 0 else np.nan
            ),
            "Negative_Score_Median": negative_median,
            "Median_Difference_Positive_Minus_Negative": median_difference,
            "Actual_Positive_Predicted_Positive": positive_detected,
            "Actual_Negative_Predicted_Positive": false_positives,
        })

    # Converts the score summaries into a report table.
    summary_df = pd.DataFrame(summary_rows)

    # Saves the score comparison for later inspection.
    summary_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Prints each test sample's actual dyes and all decision scores.
    print("\n--- Scores by test sample ---")
    score_columns = [f"Score_{dye}" for dye in DYES]
    sample_columns = ["Sample_ID", "Actual_Dyes"]

    # Displays scores rounded to four decimal places.
    print(
        report_df[sample_columns + score_columns]
        .round(4)
        .to_string(index=False)
    )

    # Prints the comparison of scores for actual positive and negative samples.
    print("\n--- Score comparison by dye ---")
    print(summary_df.round(4).to_string(index=False))

    # Prints the location of the saved analysis report.
    print(f"\nReport saved: {OUTPUT_PATH}")


# Runs the analysis when this file is executed directly.
if __name__ == "__main__":
    main()

# Import the libraries required for data processing and classification metrics.
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Define the project root directory based on this script's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the input file containing out-of-fold predictions.
INPUT_FILE = PROJECT_ROOT / "reports" / "multilabel_svm_cv_predictions.csv"

# Define the output file containing the best exploratory threshold per dye.
SUMMARY_FILE = PROJECT_ROOT / "reports" / "svm_threshold_summary.csv"

# Define the output file containing metrics for every tested threshold.
DETAIL_FILE = PROJECT_ROOT / "reports" / "svm_threshold_details.csv"

# Define the baseline threshold used by the current SVM model.
BASELINE_THRESHOLD = 0.0


# Find a column using a list of possible column names.
def find_column(dataframe, candidates):
    # Store the available column names in a case-insensitive lookup.
    column_lookup = {column.strip().lower(): column for column in dataframe.columns}

    # Return the first matching column name.
    for candidate in candidates:
        if candidate.lower() in column_lookup:
            return column_lookup[candidate.lower()]

    # Explain which candidate names were expected if no match exists.
    raise ValueError(
        f"Could not find any of these columns: {candidates}. "
        f"Available columns: {list(dataframe.columns)}"
    )


# Calculate classification metrics for one dye at a specific threshold.
def evaluate_threshold(actual, scores, threshold):
    # Convert continuous SVM scores into binary predictions.
    predicted = (scores >= threshold).astype(int)

    # Return the requested metrics for this threshold.
    return {
        "Threshold": float(threshold),
        "Precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "Recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "F1": f1_score(
            actual, predicted, zero_division=0
        ),
        "Balanced_Accuracy": balanced_accuracy_score(
            actual, predicted
        ),
        "Accuracy": accuracy_score(actual, predicted),
        "Predicted_Positive": int(predicted.sum()),
        "True_Positive": int(((actual == 1) & (predicted == 1)).sum()),
        "False_Positive": int(((actual == 0) & (predicted == 1)).sum()),
        "False_Negative": int(((actual == 1) & (predicted == 0)).sum()),
    }


# Load the cross-validation predictions and identify the required columns.
def main():
    # Read the out-of-fold prediction file.
    dataframe = pd.read_csv(INPUT_FILE)

    # Identify the dye, actual-label, and averaged-score columns.
    dye_column = find_column(
        dataframe, ["Dye", "Adulterant", "Adulterant_Type", "Label"]
    )
    actual_column = find_column(
        dataframe, ["Actual", "Actual_Label", "y_true", "True_Label"]
    )
    score_column = find_column(
        dataframe,
        ["Mean_OOF_Score", "OOF_Score", "Mean_Score", "Decision_Score", "Score"],
    )

    # Create lists to collect detailed metrics and summary rows.
    detail_rows = []
    summary_rows = []

    # Analyze each dye independently because this is a multi-label problem.
    for dye, dye_data in dataframe.groupby(dye_column):
        # Convert actual labels and SVM scores to numeric arrays.
        actual = pd.to_numeric(
            dye_data[actual_column], errors="raise"
        ).to_numpy(dtype=int)
        scores = pd.to_numeric(
            dye_data[score_column], errors="raise"
        ).to_numpy(dtype=float)

        # Reject malformed labels or scores before calculating metrics.
        if not np.isin(actual, [0, 1]).all():
            raise ValueError(f"Labels for {dye} must contain only 0 and 1.")
        if not np.isfinite(scores).all():
            raise ValueError(f"Scores for {dye} contain NaN or infinity.")
        if len(np.unique(actual)) != 2:
            print(f"Skipping {dye}: both positive and negative samples are required.")
            continue

        # Generate candidate thresholds from score boundaries and the baseline.
        unique_scores = np.unique(scores)
        midpoints = (
            (unique_scores[:-1] + unique_scores[1:]) / 2
            if len(unique_scores) > 1
            else np.array([], dtype=float)
        )
        thresholds = np.unique(
            np.concatenate(
                (
                    np.array([BASELINE_THRESHOLD]),
                    unique_scores,
                    midpoints,
                    np.array([unique_scores.min() - 1e-9]),
                    np.array([unique_scores.max() + 1e-9]),
                )
            )
        )

        # Calculate metrics for every candidate threshold.
        dye_detail_rows = []
        for threshold in thresholds:
            metrics = evaluate_threshold(actual, scores, threshold)
            metrics["Dye"] = dye
            dye_detail_rows.append(metrics)
            detail_rows.append(metrics)

        # Convert this dye's threshold results into a sortable table.
        dye_results = pd.DataFrame(dye_detail_rows)

        # Keep the exact baseline result for comparison.
        baseline_result = dye_results[
            np.isclose(dye_results["Threshold"], BASELINE_THRESHOLD)
        ].iloc[0]

        # Find the threshold with the highest F1 score.
        # Break ties by balanced accuracy, then by closeness to zero.
        best_f1_result = (
            dye_results.assign(
                _distance_from_zero=lambda table: table["Threshold"].abs()
            )
            .sort_values(
                ["F1", "Balanced_Accuracy", "_distance_from_zero"],
                ascending=[False, False, True],
            )
            .iloc[0]
        )

        # Record baseline and exploratory best-F1 metrics for this dye.
        summary_rows.append(
            {
                "Dye": dye,
                "Train_Positive": int(actual.sum()),
                "Train_Negative": int(len(actual) - actual.sum()),
                "Baseline_Threshold": BASELINE_THRESHOLD,
                "Baseline_Precision": baseline_result["Precision"],
                "Baseline_Recall": baseline_result["Recall"],
                "Baseline_F1": baseline_result["F1"],
                "Baseline_Balanced_Accuracy": baseline_result[
                    "Balanced_Accuracy"
                ],
                "Baseline_Predicted_Positive": baseline_result[
                    "Predicted_Positive"
                ],
                "Exploratory_Best_F1_Threshold": best_f1_result["Threshold"],
                "Exploratory_Best_Precision": best_f1_result["Precision"],
                "Exploratory_Best_Recall": best_f1_result["Recall"],
                "Exploratory_Best_F1": best_f1_result["F1"],
                "Exploratory_Best_Balanced_Accuracy": best_f1_result[
                    "Balanced_Accuracy"
                ],
                "Exploratory_Best_Predicted_Positive": best_f1_result[
                    "Predicted_Positive"
                ],
            }
        )

    # Convert the collected rows into output tables.
    summary_dataframe = pd.DataFrame(summary_rows)
    detail_dataframe = pd.DataFrame(detail_rows)

    # Create the reports directory if it does not already exist.
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Save the summary and detailed threshold reports.
    summary_dataframe.to_csv(SUMMARY_FILE, index=False)
    detail_dataframe.to_csv(DETAIL_FILE, index=False)

    # Print a concise comparison of baseline and exploratory best thresholds.
    print("\nThreshold analysis completed.")
    print(f"Summary report: {SUMMARY_FILE}")
    print(f"Detailed report: {DETAIL_FILE}")
    print("\nBaseline versus exploratory best F1:")
    print(
        summary_dataframe[
            [
                "Dye",
                "Baseline_F1",
                "Exploratory_Best_F1_Threshold",
                "Exploratory_Best_F1",
                "Exploratory_Best_Precision",
                "Exploratory_Best_Recall",
            ]
        ].to_string(index=False)
    )

    # Remind the user that selecting and scoring the best threshold on the same
    # out-of-fold results can overestimate future performance.
    print(
        "\nWARNING: Best thresholds are exploratory, not independent performance "
        "estimates. Do not treat these metrics as final test results."
    )


# Run the analysis only when this file is executed directly.
if __name__ == "__main__":
    main()

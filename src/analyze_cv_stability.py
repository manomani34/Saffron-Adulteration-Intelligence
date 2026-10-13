# Import paths for creating report files.
from pathlib import Path

# Import NumPy and Pandas for numerical calculations and report generation.
import numpy as np
import pandas as pd

# Import metrics for evaluating classification and ranking performance.
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Import stratified cross-validation to preserve each dye's class proportions.
from sklearn.model_selection import StratifiedKFold

# Import Pipeline to prevent preprocessing leakage between folds.
from sklearn.pipeline import Pipeline

# Import StandardScaler to standardize spectral intensities within each fold.
from sklearn.preprocessing import StandardScaler

# Import SVC as the binary classifier for each artificial dye.
from sklearn.svm import SVC

# Reuse the established dataset configuration and loader.
from cross_validate_multilabel_svm import (
    DYES,
    N_REPEATS,
    N_SPLITS,
    RANDOM_STATE,
    DECISION_THRESHOLD,
    load_training_data,
)

# Define the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the output file containing metrics for every repetition and dye.
DETAIL_PATH = PROJECT_ROOT / "reports" / "svm_cv_stability_details.csv"

# Define the output file summarizing mean and standard deviation per dye.
SUMMARY_PATH = PROJECT_ROOT / "reports" / "svm_cv_stability_summary.csv"


# Evaluate each dye across repeated stratified cross-validation runs.
def analyze_stability(features, targets):
    # Store one record for every dye and repetition.
    detail_rows = []

    # Process each dye as an independent binary classification problem.
    for dye in DYES:
        # Extract the actual presence or absence labels for this dye.
        actual = targets[dye].to_numpy(dtype=int)

        # Count positive and negative samples before splitting.
        positive_count = int(actual.sum())
        negative_count = int(len(actual) - positive_count)

        # Skip labels that cannot support the requested stratified folds.
        if min(positive_count, negative_count) < N_SPLITS:
            print(f"Skipping {dye}: insufficient samples in one class.")
            continue

        # Evaluate this dye independently in every repetition.
        for repeat_index in range(N_REPEATS):
            # Allocate one out-of-fold score for every sample in this repetition.
            oof_scores = np.full(len(actual), np.nan, dtype=float)

            # Create reproducible folds using the same seed convention as baseline.
            splitter = StratifiedKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=RANDOM_STATE + repeat_index,
            )

            # Train on each fold and score only its held-out samples.
            for train_indices, validation_indices in splitter.split(
                features, actual
            ):
                # Build a fresh pipeline so scaling is fitted only on training data.
                model = Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "svc",
                            SVC(
                                kernel="rbf",
                                C=1.0,
                                gamma="scale",
                                class_weight="balanced",
                            ),
                        ),
                    ]
                )

                # Fit the pipeline using only this fold's training samples.
                model.fit(
                    features.iloc[train_indices],
                    actual[train_indices],
                )

                # Generate continuous SVM scores for held-out samples.
                fold_scores = model.decision_function(
                    features.iloc[validation_indices]
                )

                # Save the scores in their original sample positions.
                oof_scores[validation_indices] = fold_scores

            # Ensure every sample received exactly one score in this repetition.
            if not np.isfinite(oof_scores).all():
                raise ValueError(
                    f"Missing or invalid OOF scores for {dye}, "
                    f"repetition {repeat_index + 1}."
                )

            # Apply the existing baseline threshold without tuning it.
            predicted = (
                oof_scores >= DECISION_THRESHOLD
            ).astype(int)

            # Record threshold-based and ranking metrics for this repetition.
            detail_rows.append(
                {
                    "Dye": dye,
                    "Repetition": repeat_index + 1,
                    "Positive_Samples": positive_count,
                    "Negative_Samples": negative_count,
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
                    "Average_Precision": average_precision_score(
                        actual, oof_scores
                    ),
                    "Predicted_Positive": int(predicted.sum()),
                }
            )

        # Print progress after all repetitions for this dye are completed.
        print(f"Completed {dye}: {N_REPEATS} repetitions")

    # Convert the repetition-level records into a DataFrame.
    detail_frame = pd.DataFrame(detail_rows)

    # Define the metrics that will be summarized across repetitions.
    metric_columns = [
        "Precision",
        "Recall",
        "F1",
        "Balanced_Accuracy",
        "Average_Precision",
    ]

    # Calculate mean, standard deviation, minimum, and maximum per dye.
    summary_frame = (
        detail_frame.groupby("Dye")[metric_columns]
        .agg(["mean", "std", "min", "max"])
    )

    # Flatten the multi-level column names into readable report headings.
    summary_frame.columns = [
        f"{metric}_{statistic.upper()}"
        for metric, statistic in summary_frame.columns
    ]

    # Include the number of repetitions used for each dye.
    summary_frame["Repetitions"] = (
        detail_frame.groupby("Dye")["Repetition"].nunique()
    )

    # Convert the dye index into an ordinary report column.
    summary_frame = summary_frame.reset_index()

    # Return the detailed and summary reports.
    return detail_frame, summary_frame


# Run the stability analysis and save both reports.
def main():
    # Load the same eligible training samples and targets as the baseline.
    features, targets, sample_ids = load_training_data()

    # Display the dataset dimensions before the repeated evaluation begins.
    print(f"Training samples: {len(features)}")
    print(f"Spectral features: {features.shape[1]}")
    print(f"Folds: {N_SPLITS}; repetitions: {N_REPEATS}")
    print(f"Fixed decision threshold: {DECISION_THRESHOLD}")

    # Run the per-repetition evaluation.
    detail_frame, summary_frame = analyze_stability(features, targets)

    # Ensure the reports directory exists before writing files.
    DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save all repetition-level metrics and the per-dye summary.
    detail_frame.to_csv(DETAIL_PATH, index=False)
    summary_frame.to_csv(SUMMARY_PATH, index=False)

    # Display the most useful mean and standard-deviation columns.
    display_columns = [
        "Dye",
        "F1_MEAN",
        "F1_STD",
        "Recall_MEAN",
        "Recall_STD",
        "Average_Precision_MEAN",
        "Average_Precision_STD",
    ]

    # Print the summary so results can be reviewed immediately.
    print("\nStability summary:")
    print(summary_frame[display_columns].round(3).to_string(index=False))

    # Print the locations of the generated reports.
    print(f"\nDetailed report: {DETAIL_PATH}")
    print(f"Summary report: {SUMMARY_PATH}")

    # Explain the limitations of repeated cross-validation on this small dataset.
    print(
        "\nNote: These are repeated cross-validation results, not an "
        "independent test. The standard deviation describes variation "
        "across repetitions and is not a confidence interval."
    )


# Execute the workflow only when this script is run directly.
if __name__ == "__main__":
    main()

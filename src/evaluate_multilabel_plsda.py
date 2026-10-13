# Import libraries for data processing, PLS regression, and evaluation.
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# Define the project root directory relative to this script.
BASE_DIR = Path(__file__).resolve().parent.parent

# Define the source dataset path.
DATA_PATH = BASE_DIR / "data" / "raw" / "total-color-data.csv"

# Define the report directory for PLS-DA-only outputs.
REPORTS_DIR = BASE_DIR / "reports"

# Define the output files without overwriting any SVM reports.
METRICS_PATH = REPORTS_DIR / "plsda_cv_metrics.csv"
PREDICTIONS_PATH = REPORTS_DIR / "plsda_cv_predictions.csv"
SUMMARY_PATH = REPORTS_DIR / "plsda_cv_summary.csv"

# Define the known artificial dyes used as multilabel targets.
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

# Define metadata columns that must never be used as spectral features.
METADATA_COLUMNS = [
    "Sample_ID",
    "Title",
    "Class",
    "Target_Label",
    "Adulterant_Type",
    "Dataset_Role",
]

# Keep the internal validation settings aligned with the existing SVM experiment.
N_SPLITS = 2
N_REPEATS = 20
RANDOM_STATE = 42

# Fix the number of PLS latent components for this first baseline experiment.
N_COMPONENTS = 2

# Use a fixed probability-like decision threshold for the PLS regression output.
DECISION_THRESHOLD = 0.5


# Parse a sample's adulterant description into a set of recognized dye labels.
def parse_dyes(value):
    # Return an empty set when the description is missing.
    if pd.isna(value):
        return set()

    # Convert the description to a string for safe processing.
    text = str(value).strip()

    # Return an empty set for blank or explicitly unknown labels.
    if not text or text.casefold() == "unknown":
        return set()

    # Match known dye names without depending on their order in the description.
    normalized_text = text.casefold()

    # Return only dyes that appear as recognized labels in the description.
    return {
        dye
        for dye in DYES
        if dye.casefold() in normalized_text
    }


# Load the customer dataset and validate the columns required by this experiment.
def load_training_data():
    # Read the raw CSV while supporting a possible UTF-8 BOM.
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    # Define the metadata columns required for sample selection and labeling.
    required_columns = {
        "Target_Label",
        "Adulterant_Type",
        "Dataset_Role",
    }

    # Identify missing required columns before attempting model training.
    missing_columns = required_columns - set(data.columns)

    # Stop early if the input schema does not match the expected dataset.
    if missing_columns:
        raise ValueError(
            "Missing required columns in the input dataset: "
            + ", ".join(sorted(missing_columns))
        )

    # Keep only training samples with recognized genuine/adulterated status.
    train_mask = (
        data["Dataset_Role"].astype(str).str.casefold().eq("train")
        & data["Target_Label"]
        .astype(str)
        .str.casefold()
        .isin(["genuine", "adulterated"])
        & data["Adulterant_Type"]
        .fillna("")
        .astype(str)
        .str.casefold()
        .ne("unknown")
    )

    # Create a separate copy so the source dataset remains unchanged.
    training_data = data.loc[train_mask].copy()

    # Identify spectral features by excluding metadata columns.
    feature_columns = [
        column
        for column in training_data.columns
        if column not in METADATA_COLUMNS
    ]

    # Convert all selected spectral columns to numeric values.
    features = training_data[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Stop if the selected spectral data contains missing or infinite values.
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError(
            "Spectral features contain missing or infinite values. "
            "Resolve these values before running PLS-DA."
        )

    # Build one binary target column for each artificial dye.
    targets = pd.DataFrame(
        {
            dye: training_data["Adulterant_Type"].apply(
                lambda value: int(dye in parse_dyes(value))
            )
            for dye in DYES
        },
        index=training_data.index,
    )

    # Return features, multilabel targets, and the original sample identifiers.
    return features, targets, training_data


# Evaluate one binary PLS-DA model using repeated stratified cross-validation.
def evaluate_one_dye(features, target, sample_ids, dye):
    # Count the positive and negative examples available for this dye.
    positive_count = int(target.sum())

    # Count the negative examples available for this dye.
    negative_count = int(len(target) - positive_count)

    # Initialize a list for collecting held-out scores from every repetition.
    score_runs = []

    # Initialize a list for recording individual out-of-fold predictions.
    prediction_records = []

    # Initialize a list for storing split-level diagnostic information.
    split_records = []

    # Skip the model when either class is too small for stratified validation.
    if min(positive_count, negative_count) < N_SPLITS:
        return None, None, None

    # Repeat stratified validation with different deterministic shuffle seeds.
    for repeat_index in range(N_REPEATS):
        # Create a stratified splitter for the current repetition.
        splitter = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + repeat_index,
        )

        # Store held-out predictions for every sample in this repetition.
        repeat_scores = np.full(len(target), np.nan, dtype=float)

        # Generate train/validation splits without using the external Test set.
        for fold_index, (train_indices, valid_indices) in enumerate(
            splitter.split(features, target)
        ):
            # Select the training-fold spectral data.
            x_train = features.iloc[train_indices].to_numpy(dtype=float)

            # Select the validation-fold spectral data.
            x_valid = features.iloc[valid_indices].to_numpy(dtype=float)

            # Select the training-fold binary labels.
            y_train = target.iloc[train_indices].to_numpy(dtype=float)

            # Select the validation-fold binary labels.
            y_valid = target.iloc[valid_indices].to_numpy(dtype=int)

            # Scale features using only the current training fold.
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_valid_scaled = scaler.transform(x_valid)

            # Limit component count to the dimensions supported by this fold.
            component_count = min(
                N_COMPONENTS,
                x_train_scaled.shape[0] - 1,
                x_train_scaled.shape[1],
            )

            # Fail clearly if the fold cannot support any latent component.
            if component_count < 1:
                raise ValueError(
                    f"Cannot fit PLS-DA for {dye}: "
                    "insufficient training-fold dimensions."
                )

            # Fit PLS regression to the binary target as the PLS-DA classifier.
            model = PLSRegression(
                n_components=component_count,
                scale=False,
                max_iter=500,
            )

            # Train the model only on the current training fold.
            model.fit(x_train_scaled, y_train)

            # Predict continuous scores for held-out validation samples.
            valid_scores = model.predict(x_valid_scaled).reshape(-1)

            # Save the held-out scores in their original sample positions.
            repeat_scores[valid_indices] = valid_scores

            # Record every held-out prediction for later error analysis.
            for local_index, sample_index in enumerate(valid_indices):
                # Store the original sample identifier for this prediction.
                sample_id = sample_ids.iloc[sample_index]

                # Store the actual binary label for this dye.
                actual_label = int(y_valid[local_index])

                # Store the continuous PLS regression score.
                predicted_score = float(valid_scores[local_index])

                # Convert the continuous score into a binary prediction.
                predicted_label = int(
                    predicted_score >= DECISION_THRESHOLD
                )

                # Append a detailed out-of-fold prediction record.
                prediction_records.append(
                    {
                        "Dye": dye,
                        "Sample_ID": sample_id,
                        "Repeat": repeat_index + 1,
                        "Fold": fold_index + 1,
                        "Actual": actual_label,
                        "PLSDA_Score": predicted_score,
                        "Predicted": predicted_label,
                    }
                )

            # Record the fold's sample counts for auditability.
            split_records.append(
                {
                    "Dye": dye,
                    "Repeat": repeat_index + 1,
                    "Fold": fold_index + 1,
                    "Train_Samples": len(train_indices),
                    "Validation_Samples": len(valid_indices),
                    "Train_Positives": int(y_train.sum()),
                    "Validation_Positives": int(y_valid.sum()),
                }
            )

        # Verify that all samples received an out-of-fold score.
        if np.isnan(repeat_scores).any():
            raise RuntimeError(
                f"Missing out-of-fold predictions for dye {dye}."
            )

        # Keep the completed repetition for the final per-sample average.
        score_runs.append(repeat_scores)

    # Average held-out scores across repetitions for each training sample.
    average_scores = np.mean(np.vstack(score_runs), axis=0)

    # Convert the averaged scores into binary predictions at the fixed threshold.
    average_predictions = (
        average_scores >= DECISION_THRESHOLD
    ).astype(int)

    # Calculate precision using the fixed threshold.
    precision = precision_score(
        target,
        average_predictions,
        zero_division=0,
    )

    # Calculate recall using the fixed threshold.
    recall = recall_score(
        target,
        average_predictions,
        zero_division=0,
    )

    # Calculate F1 using the fixed threshold.
    f1 = f1_score(
        target,
        average_predictions,
        zero_division=0,
    )

    # Calculate average precision from continuous scores rather than binary labels.
    average_precision = average_precision_score(
        target,
        average_scores,
    )

    # Summarize the internal validation results for this dye.
    metrics = {
        "Dye": dye,
        "Positive_Samples": positive_count,
        "Negative_Samples": negative_count,
        "N_Splits": N_SPLITS,
        "N_Repeats": N_REPEATS,
        "PLS_Components": N_COMPONENTS,
        "Decision_Threshold": DECISION_THRESHOLD,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Average_Precision": average_precision,
        "Predicted_Positive": int(average_predictions.sum()),
    }

    # Build a per-sample summary of the averaged out-of-fold scores.
    sample_predictions = pd.DataFrame(
        {
            "Dye": dye,
            "Sample_ID": sample_ids.to_numpy(),
            "Actual": target.to_numpy(dtype=int),
            "Mean_PLSDA_Score": average_scores,
            "Predicted": average_predictions,
        }
    )

    # Return the metrics, detailed fold predictions, and averaged predictions.
    return (
        metrics,
        pd.DataFrame(prediction_records),
        sample_predictions,
    )


# Run all nine binary PLS-DA experiments and save separate reports.
def main():
    # Create the report directory if it does not already exist.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load only the eligible training samples and their spectral features.
    features, targets, training_data = load_training_data()

    # Keep the original sample identifiers for traceable results.
    sample_ids = (
        training_data["Sample_ID"].reset_index(drop=True)
        if "Sample_ID" in training_data.columns
        else pd.Series(training_data.index.astype(str))
    )

    # Reset feature and target indices so positional CV splits remain aligned.
    features = features.reset_index(drop=True)
    targets = targets.reset_index(drop=True)

    # Initialize collections for per-dye metric reports.
    all_metrics = []

    # Initialize collections for detailed out-of-fold predictions.
    all_predictions = []

    # Initialize collections for per-sample averaged scores.
    all_sample_summaries = []

    # Evaluate each artificial dye independently using the same CV protocol.
    for dye in DYES:
        # Print progress so the user can monitor the current dye.
        print(f"Evaluating PLS-DA: {dye}")

        # Run the repeated cross-validation experiment for the current dye.
        metrics, predictions, sample_summary = evaluate_one_dye(
            features=features,
            target=targets[dye],
            sample_ids=sample_ids,
            dye=dye,
        )

        # Skip metric collection if there are insufficient class examples.
        if metrics is None:
            print(f"Skipped {dye}: insufficient positive/negative samples.")
            continue

        # Store the current dye's evaluation metrics.
        all_metrics.append(metrics)

        # Store the detailed held-out predictions.
        all_predictions.append(predictions)

        # Store the averaged per-sample prediction summary.
        all_sample_summaries.append(sample_summary)

    # Stop if none of the dye models could be evaluated.
    if not all_metrics:
        raise RuntimeError(
            "No PLS-DA results were produced. Check training labels and class counts."
        )

    # Combine metrics from all evaluated dyes into a single table.
    metrics_frame = pd.DataFrame(all_metrics)

    # Combine detailed out-of-fold predictions from all dyes.
    predictions_frame = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # Combine averaged per-sample predictions from all dyes.
    sample_summary_frame = pd.concat(
        all_sample_summaries,
        ignore_index=True,
    )

    # Save per-dye CV metrics to the PLS-DA-specific report.
    metrics_frame.to_csv(
        METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Save detailed predictions from every repetition and fold.
    predictions_frame.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Calculate descriptive means across the evaluated dye classes.
    summary_frame = pd.DataFrame(
        [
            {
                "Evaluated_Dyes": len(metrics_frame),
                "Mean_Precision": metrics_frame["Precision"].mean(),
                "Mean_Recall": metrics_frame["Recall"].mean(),
                "Mean_F1": metrics_frame["F1"].mean(),
                "Mean_Average_Precision": metrics_frame[
                    "Average_Precision"
                ].mean(),
                "Decision_Threshold": DECISION_THRESHOLD,
                "PLS_Components": N_COMPONENTS,
                "N_Splits": N_SPLITS,
                "N_Repeats": N_REPEATS,
                "Training_Samples": len(features),
                "Spectral_Features": features.shape[1],
            }
        ]
    )

    # Save the descriptive summary to its separate PLS-DA report.
    summary_frame.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Print the per-dye metrics table to the terminal.
    print("\nPLS-DA cross-validation metrics:")
    print(metrics_frame.to_string(index=False))

    # Print the paths of the generated PLS-DA reports.
    print("\nPLS-DA reports saved:")
    print(METRICS_PATH)
    print(PREDICTIONS_PATH)
    print(SUMMARY_PATH)

    # Clarify that this experiment does not evaluate the external Test set.
    print(
        "\nIMPORTANT: These are internal cross-validation results only. "
        "The independent Test set was not used."
    )


# Execute the experiment only when this script is run directly.
if __name__ == "__main__":
    main()
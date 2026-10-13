
# Import regular expressions to split dye names in multi-label values.
import re

# Import NumPy for numerical arrays and missing-value handling.
import numpy as np

# Import Pandas to read the dataset and write evaluation reports.
import pandas as pd

# Import Average Precision to evaluate ranking without fixing a threshold.
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

# Import StratifiedKFold to preserve positive and negative label proportions.
from sklearn.model_selection import StratifiedKFold

# Import Pipeline to prevent preprocessing leakage across folds.
from sklearn.pipeline import Pipeline

# Import StandardScaler to standardize spectral intensities.
from sklearn.preprocessing import StandardScaler

# Import SVC as the binary classifier for each dye.
from sklearn.svm import SVC


# Define the source dataset path.
DATA_PATH = "data/raw/total-color-data.csv"

# Define the directory where validation reports will be saved.
REPORT_DIR = "reports"

# Define the nine artificial dyes that the model must identify.
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

# Define the number of folds; two folds are necessary for very rare labels.
N_SPLITS = 2

# Define the number of repeated cross-validation rounds.
N_REPEATS = 20

# Define the random seed for reproducible fold assignments.
RANDOM_STATE = 42

# Define the decision-score threshold used by the existing baseline.
DECISION_THRESHOLD = 0.0

# Define the metadata columns that are not spectral features.
METADATA_COLUMNS = [
    "Sample_ID",
    "Target_Label",
    "Adulterant_Type",
    "Dataset_Role",
]


# Convert a multi-label text value into a set of dye names.
def parse_dyes(value):
    # Return an empty set for missing or explicitly unknown labels.
    if pd.isna(value) or str(value).strip().lower() in {"", "unknown", "nan"}:
        return set()

    # Split labels on plus signs, semicolons, or commas.
    parts = re.split(r"[+;,]", str(value))

    # Normalize whitespace and keep only recognized dye names.
    return {
        part.strip()
        for part in parts
        if part.strip() in DYES
    }


# Load the dataset and retain only eligible training samples.
def load_training_data():
    # Read the complete customer dataset.
    frame = pd.read_csv(DATA_PATH)

    # Keep training rows labeled as genuine or adulterated saffron.
    train_mask = (
        frame["Dataset_Role"].astype(str).str.strip().str.casefold().eq("train")
        & frame["Target_Label"].astype(str).str.strip().str.casefold().isin(
            {"genuine", "adulterated"}
        )
        & frame["Adulterant_Type"].astype(str).str.strip().str.casefold().ne(
            "unknown"
        )
    )

    # Select eligible training rows and create an independent copy.
    train_frame = frame.loc[train_mask].copy()

    # Identify the spectral columns by excluding metadata fields.
    feature_columns = [
        column
        for column in train_frame.columns
        if column not in METADATA_COLUMNS
    ]

    # Convert the spectral matrix to numeric values and reject invalid data.
    features = train_frame[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Stop early if the selected spectral matrix contains invalid values.
    if features.isna().any().any() or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Spectral features contain missing or non-finite values.")

    # Build a binary target column for each artificial dye.
    targets = pd.DataFrame(
        [
            [int(dye in parse_dyes(value)) for dye in DYES]
            for value in train_frame["Adulterant_Type"]
        ],
        columns=DYES,
        index=train_frame.index,
    )

    # Return the spectra, multi-label targets, and sample identifiers.
    return features, targets, train_frame["Sample_ID"].reset_index(drop=True)


# Evaluate each dye using repeated, stratified two-fold cross-validation.
def run_cross_validation(features, targets, sample_ids):
    # Create a list to collect one metrics record per dye.
    metrics_rows = []

    # Create a list to collect averaged out-of-fold scores per sample.
    prediction_rows = []

    # Evaluate each dye independently as a binary classification problem.
    for dye in DYES:
        # Extract the binary target vector for the current dye.
        y = targets[dye].to_numpy(dtype=int)

        # Count positive and negative examples for this dye.
        positive_count = int(y.sum())
        negative_count = int(len(y) - positive_count)

        # Require enough examples of both classes for stratified folds.
        if min(positive_count, negative_count) < N_SPLITS:
            print(f"Skipping {dye}: insufficient samples in one class.")
            continue

        # Store each sample's scores from repeated out-of-fold predictions.
        score_runs = np.full((N_REPEATS, len(y)), np.nan, dtype=float)

        # Repeat the stratified evaluation with different fold assignments.
        for repeat_index in range(N_REPEATS):
            # Create reproducible stratified folds for this repetition.
            splitter = StratifiedKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=RANDOM_STATE + repeat_index,
            )

            # Train and validate on each fold without mixing preprocessing data.
            for train_indices, validation_indices in splitter.split(features, y):
                # Create a fresh scaler and SVM for this fold.
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

                # Fit the model only on this fold's training samples.
                model.fit(features.iloc[train_indices], y[train_indices])

                # Calculate decision scores for held-out validation samples.
                fold_scores = model.decision_function(
                    features.iloc[validation_indices]
                )

                # Save scores only for samples not used to fit this fold.
                score_runs[repeat_index, validation_indices] = fold_scores

        # Average each sample's held-out scores across repeated evaluations.
        mean_scores = np.nanmean(score_runs, axis=0)

        # Convert averaged scores to binary predictions using the baseline threshold.
        predicted_labels = (mean_scores >= DECISION_THRESHOLD).astype(int)

        # Calculate threshold-based precision, recall, and F1.
        precision = precision_score(
            y,
            predicted_labels,
            zero_division=0,
        )
        recall = recall_score(
            y,
            predicted_labels,
            zero_division=0,
        )
        f1 = f1_score(
            y,
            predicted_labels,
            zero_division=0,
        )

        # Calculate threshold-independent ranking quality.
        average_precision = average_precision_score(y, mean_scores)

        # Record metrics and class counts for this dye.
        metrics_rows.append(
            {
                "Dye": dye,
                "Positive_Samples": positive_count,
                "Negative_Samples": negative_count,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Average_Precision": average_precision,
                "Predicted_Positive": int(predicted_labels.sum()),
            }
        )

        # Record the averaged out-of-fold scores for each sample.
        for sample_index, sample_id in enumerate(sample_ids):
            # Append the actual label and prediction for auditability.
            prediction_rows.append(
                {
                    "Sample_ID": sample_id,
                    "Dye": dye,
                    "Actual": int(y[sample_index]),
                    "Mean_OOF_Score": float(mean_scores[sample_index]),
                    "Predicted": int(predicted_labels[sample_index]),
                }
            )

        # Print a concise result for the current dye.
        print(
            f"{dye}: Precision={precision:.3f}, "
            f"Recall={recall:.3f}, F1={f1:.3f}, "
            f"AP={average_precision:.3f}"
        )

    # Convert collected results into report tables.
    metrics_frame = pd.DataFrame(metrics_rows)
    predictions_frame = pd.DataFrame(prediction_rows)

    # Save the per-dye metrics report.
    metrics_frame.to_csv(
        f"{REPORT_DIR}/multilabel_svm_cv_metrics.csv",
        index=False,
    )

    # Save the sample-level out-of-fold predictions.
    predictions_frame.to_csv(
        f"{REPORT_DIR}/multilabel_svm_cv_predictions.csv",
        index=False,
    )

    # Return both report tables for optional further analysis.
    return metrics_frame, predictions_frame


# Run the complete cross-validation workflow when this file is executed directly.
if __name__ == "__main__":
    # Load eligible training samples and their multi-label targets.
    X, Y, sample_ids = load_training_data()

    # Display the size of the dataset used for cross-validation.
    print(f"Training samples: {len(X)}")
    print(f"Spectral features: {X.shape[1]}")
    print(f"Dye labels: {len(DYES)}")
    print(f"Folds: {N_SPLITS}; repetitions: {N_REPEATS}")

    # Run cross-validation and save the evaluation reports.
    metrics, predictions = run_cross_validation(X, Y, sample_ids)

    # Confirm the report locations after successful completion.
    print("\nCross-validation completed.")
    print("Metrics: reports/multilabel_svm_cv_metrics.csv")
    print("Predictions: reports/multilabel_svm_cv_predictions.csv")

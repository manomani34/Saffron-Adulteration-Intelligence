# Imports tools for data processing, evaluation metrics, and SVM classification.
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Defines the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Defines the path to the source dataset.
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "total-color-data.csv"

# Defines the directory for generated reports.
REPORT_DIR = PROJECT_ROOT / "reports"

# Defines the fixed list of dye labels to predict.
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

# Defines the decision score threshold used to predict dye presence.
DECISION_THRESHOLD = 0.0

# Defines a reproducible random seed for the SVM classifier.
RANDOM_STATE = 42


# Extracts known dye names from a sample's adulterant metadata.
def parse_dyes(adulterant_type: object) -> set[str]:
    # Returns an empty set when the sample has no adulterant label.
    if pd.isna(adulterant_type):
        return set()

    # Splits the metadata into individual components.
    parts = {
        part.strip()
        for part in str(adulterant_type).split("+")
    }

    # Removes saffron and unknown labels, retaining dye names only.
    parts.discard("Saffron")
    parts.discard("Unknown")

    # Identifies unexpected labels to prevent silent target corruption.
    unexpected = parts - set(DYES)

    # Rejects unexpected labels instead of silently ignoring them.
    if unexpected:
        raise ValueError(
            f"Unexpected dye labels: {sorted(unexpected)}"
        )

    # Returns the set of dye labels present in the sample.
    return parts


# Builds a multi-label binary target matrix from sample metadata.
def build_target_matrix(frame: pd.DataFrame) -> np.ndarray:
    # Creates an initially empty target matrix for all samples and dyes.
    target_matrix = np.zeros(
        (len(frame), len(DYES)),
        dtype=int,
    )

    # Iterates over each sample and its metadata.
    for row_index, (_, row) in enumerate(frame.iterrows()):
        # Extracts the dyes recorded for the current sample.
        present_dyes = parse_dyes(row["Adulterant_Type"])

        # Marks each dye as present or absent in the current sample.
        for dye_index, dye in enumerate(DYES):
            target_matrix[row_index, dye_index] = int(
                dye in present_dyes
            )

    # Returns the binary target matrix.
    return target_matrix


# Loads the data, trains the baseline model, and evaluates known test samples.
def main() -> None:
    # Creates the report directory if it does not already exist.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Loads the dataset without modifying the original CSV.
    df = pd.read_csv(DATA_PATH)

    # Defines spectral feature columns after the four metadata columns.
    spectral_columns = df.columns[4:]

    # Converts all spectral features to numeric values.
    X_all = df[spectral_columns].apply(
        pd.to_numeric,
        errors="raise",
    ).to_numpy(dtype=float)

    # Stops execution if any spectral value is missing or non-finite.
    if not np.isfinite(X_all).all():
        raise ValueError(
            "Spectral data contains missing or non-finite values."
        )

    # Selects labeled saffron samples assigned to training.
    train_mask = (
        df["Dataset_Role"].eq("Train")
        & df["Target_Label"].isin(["Genuine", "Adulterated"])
        & ~df["Adulterant_Type"].eq("Unknown")
    )

    # Selects labeled saffron samples assigned to testing.
    test_mask = (
        df["Dataset_Role"].eq("Test")
        & df["Target_Label"].isin(["Genuine", "Adulterated"])
        & ~df["Adulterant_Type"].eq("Unknown")
    )

    # Extracts the training and test metadata independently.
    train_df = df.loc[train_mask].copy().reset_index(drop=True)
    test_df = df.loc[test_mask].copy().reset_index(drop=True)

    # Selects the spectral features corresponding to training samples.
    X_train = X_all[train_mask.to_numpy()]

    # Selects the spectral features corresponding to test samples.
    X_test = X_all[test_mask.to_numpy()]

    # Builds multi-label targets for the training samples.
    y_train = build_target_matrix(train_df)

    # Builds multi-label targets for the test samples.
    y_test = build_target_matrix(test_df)

    # Ensures that every dye has at least one positive training example.
    train_positive_counts = y_train.sum(axis=0)

    # Identifies dyes that cannot be learned without positive examples.
    missing_dyes = [
        DYES[index]
        for index, count in enumerate(train_positive_counts)
        if count == 0
    ]

    # Stops execution when any dye has no positive training examples.
    if missing_dyes:
        raise ValueError(
            "No positive training samples for these dyes: "
            f"{missing_dyes}"
        )

    # Ensures each dye classifier has both positive and negative examples.
    train_negative_counts = len(y_train) - train_positive_counts

    # Identifies dyes whose training labels contain only one class.
    single_class_dyes = [
        DYES[index]
        for index, count in enumerate(train_negative_counts)
        if count == 0
    ]

    # Stops execution if a binary classifier cannot be trained.
    if single_class_dyes:
        raise ValueError(
            "No negative training samples for these dyes: "
            f"{single_class_dyes}"
        )

    # Defines a standardized RBF-SVM classifier for each dye label.
    # Omits the deprecated probability parameter because decision scores are used.
    base_model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )

    # Creates one independent binary classifier for each dye.
    model = OneVsRestClassifier(base_model)

    # Fits all dye classifiers using training samples only.
    model.fit(X_train, y_train)

    # Calculates decision scores for every dye and test sample.
    decision_scores = model.decision_function(X_test)

    # Converts decision scores into binary dye predictions.
    y_pred = (
        decision_scores >= DECISION_THRESHOLD
    ).astype(int)

    # Stores the per-dye evaluation results.
    metric_rows = []

    # Evaluates each dye independently.
    for dye_index, dye in enumerate(DYES):
        # Extracts the actual labels for the current dye.
        y_true_dye = y_test[:, dye_index]

        # Extracts the predicted labels for the current dye.
        y_pred_dye = y_pred[:, dye_index]

        # Counts positive examples available in the test set.
        test_support = int(y_true_dye.sum())

        # Counts the number of negative examples in the test set.
        test_negative = int(len(y_true_dye) - test_support)

        # Counts how many test samples the model predicts contain this dye.
        predicted_positive = int(y_pred_dye.sum())

        # Checks whether the test set contains both positive and negative examples.
        evaluable = (
            test_support > 0
            and test_negative > 0
        )

        # Calculates precision when the test set contains both classes.
        precision = (
            precision_score(
                y_true_dye,
                y_pred_dye,
                zero_division=0,
            )
            if evaluable
            else np.nan
        )

        # Calculates recall when the test set contains both classes.
        recall = (
            recall_score(
                y_true_dye,
                y_pred_dye,
                zero_division=0,
            )
            if evaluable
            else np.nan
        )

        # Calculates F1 when the test set contains both classes.
        f1 = (
            f1_score(
                y_true_dye,
                y_pred_dye,
                zero_division=0,
            )
            if evaluable
            else np.nan
        )

        # Records the metrics and the test-set evaluation limitation.
        metric_rows.append({
            "Dye": dye,
            "Train_Positive": int(train_positive_counts[dye_index]),
            "Train_Negative": int(train_negative_counts[dye_index]),
            "Test_Positive": test_support,
            "Test_Negative": test_negative,
            "Predicted_Positive": predicted_positive,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Status": (
                "Evaluable with limited test data"
                if evaluable
                else (
                    "Not evaluable: test lacks positive or negative examples"
                )
            ),
        })

    # Converts the per-dye results into a report table.
    metrics_df = pd.DataFrame(metric_rows)

    # Defines the output path for the per-dye metrics report.
    metrics_path = REPORT_DIR / "multilabel_svm_metrics.csv"

    # Saves the metrics report as a UTF-8 CSV file.
    metrics_df.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Creates a prediction report with the test sample metadata.
    predictions_df = test_df[
        [
            "Sample_ID",
            "Target_Label",
            "Adulterant_Type",
            "Dataset_Role",
        ]
    ].copy()

    # Adds the actual known dye labels for each test sample.
    predictions_df["Actual_Dyes"] = [
        "; ".join(sorted(parse_dyes(value)))
        for value in test_df["Adulterant_Type"]
    ]

    # Adds the model's predicted dye labels for each test sample.
    predictions_df["Predicted_Dyes"] = [
        "; ".join(
            DYES[index]
            for index, value in enumerate(sample_predictions)
            if value == 1
        )
        for sample_predictions in y_pred
    ]

    # Adds the SVM decision score for every dye to the prediction report.
    for dye_index, dye in enumerate(DYES):
        predictions_df[f"Score_{dye}"] = (
            decision_scores[:, dye_index]
        )

    # Defines the output path for the sample-level prediction report.
    predictions_path = (
        REPORT_DIR / "multilabel_svm_test_predictions.csv"
    )

    # Saves the prediction report as a UTF-8 CSV file.
    predictions_df.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Prints the dataset sizes used in this baseline experiment.
    print("=" * 70)
    print("MULTI-LABEL SVM BASELINE - DECISION SCORES")
    print("=" * 70)
    print(f"Training samples: {len(train_df)}")
    print(f"Known test samples: {len(test_df)}")
    print(f"Spectral features: {len(spectral_columns)}")
    print(f"Dye labels: {len(DYES)}")
    print(f"Decision score threshold: {DECISION_THRESHOLD}")

    # Prints the positive and negative training counts for every dye.
    print("\n--- Training label counts ---")
    print(
        metrics_df[
            ["Dye", "Train_Positive", "Train_Negative"]
        ].to_string(index=False)
    )

    # Prints the per-dye evaluation results.
    print("\n--- Per-dye metrics ---")
    print(metrics_df.round(4).to_string(index=False))

    # Prints the actual and predicted dye labels for test samples.
    print("\n--- Test predictions ---")
    print(
        predictions_df[
            ["Sample_ID", "Actual_Dyes", "Predicted_Dyes"]
        ].to_string(index=False)
    )

    # Prints the paths of the generated reports.
    print("\n--- Reports saved ---")
    print(metrics_path)
    print(predictions_path)


# Runs the baseline experiment when this file is executed as a script.
if __name__ == "__main__":
    main()
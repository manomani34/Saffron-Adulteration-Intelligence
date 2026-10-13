
# Import libraries for training, evaluation, and report generation.
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Define input and output paths for the final held-out evaluation.
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "total-color-data.csv"
REPORT_DIR = ROOT / "reports"

# Define the nine dye labels evaluated by the multilabel models.
DYES = [
    "Amaranth", "Azorubine", "Erythrosine", "Orange 2",
    "Orange GGN", "Ponceau 4R", "Quinoline Yellow",
    "Sunset Yellow", "Tartrazine",
]

# Freeze model settings before evaluating the test set.
PLS_COMPONENTS = 4
SVM_THRESHOLD = 0.0
PLS_THRESHOLD = 0.5


# Check whether a dye is listed in a sample's adulterant description.
def contains_dye(description, dye):
    # Normalize text to make label matching case-insensitive.
    return dye.casefold() in str(description).casefold()


# Convert sample metadata into a binary target for a specific dye.
def build_dye_target(frame, dye):
    # Return one when the dye occurs in the adulterant label and zero otherwise.
    return frame["Adulterant_Type"].apply(
        lambda description: int(contains_dye(description, dye))
    ).to_numpy(dtype=int)


# Train a dye-specific model and return continuous test scores and predictions.
def fit_and_predict(X_train, y_train, X_test, model_name):
    # Fit spectral scaling on training data only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build the selected model with the previously evaluated settings.
    if model_name == "SVM":
        # Use the established RBF SVM configuration.
        model = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
        )
    else:
        # Use four-component PLS regression for binary PLS-DA.
        model = PLSRegression(
            n_components=PLS_COMPONENTS,
            scale=False,
        )

    # Train the model on the training split only.
    model.fit(X_train_scaled, y_train)

    # Generate continuous scores for the held-out test samples.
    if model_name == "SVM":
        # SVM decision scores use zero as the fixed classification threshold.
        scores = model.decision_function(X_test_scaled)
        predictions = (scores >= SVM_THRESHOLD).astype(int)
    else:
        # PLS predictions use the previously specified 0.5 threshold.
        scores = model.predict(X_test_scaled).ravel()
        predictions = (scores >= PLS_THRESHOLD).astype(int)

    # Return scores for ranking metrics and labels for classification metrics.
    return scores, predictions


# Evaluate one model on the known-label test samples for every dye.
def evaluate_model(X_train, train_data, X_test, test_data, model_name):
    # Store per-dye metrics and sample-level predictions.
    metric_rows = []
    prediction_rows = []

    # Evaluate each dye using a separate one-vs-rest classifier.
    for dye in DYES:
        # Build training and test targets from the recorded dye labels.
        y_train = build_dye_target(train_data, dye)
        y_test = build_dye_target(test_data, dye)

        # Skip a dye if its training labels contain only one class.
        if np.unique(y_train).size < 2:
            print(f"Skipping {model_name} / {dye}: training has one class.")
            continue

        # Fit the classifier and obtain held-out predictions.
        scores, predictions = fit_and_predict(
            X_train,
            y_train,
            X_test,
            model_name,
        )

        # Calculate AP only when the test set contains at least one positive.
        ap = (
            average_precision_score(y_test, scores)
            if y_test.sum() > 0
            else np.nan
        )

        # Record threshold-based metrics for this dye.
        metric_rows.append({
            "Model": model_name,
            "Dye": dye,
            "Test_Positive_Count": int(y_test.sum()),
            "Predicted_Positive_Count": int(predictions.sum()),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "Average_Precision": ap,
        })

        # Record each test sample's actual label, score, and prediction.
        for index, sample_id in enumerate(test_data["Sample_ID"]):
            # Save the dye-specific prediction in a long-format report.
            prediction_rows.append({
                "Model": model_name,
                "Sample_ID": sample_id,
                "Dye": dye,
                "Actual_Dye_Present": int(y_test[index]),
                "Score": float(scores[index]),
                "Predicted_Dye_Present": int(predictions[index]),
            })

    # Convert collected metrics and predictions into dataframes.
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


# Run the final evaluation without using test data for model selection.
def main():
    # Load the original customer dataset.
    data = pd.read_csv(DATA_PATH)

    # Normalize metadata fields for reliable split selection.
    role = data["Dataset_Role"].astype(str).str.strip().str.casefold()
    target = data["Target_Label"].astype(str).str.strip().str.casefold()
    adulterant = data["Adulterant_Type"].fillna("").astype(str)

    # Select known-label training samples and exclude unknown-label records.
    train_mask = (
        role.eq("train")
        & target.isin(["genuine", "adulterated"])
        & ~adulterant.str.strip().str.casefold().eq("unknown")
    )

    # Select only known-label test samples for the held-out evaluation.
    test_mask = (
        role.eq("test")
        & target.isin(["genuine", "adulterated"])
        & ~adulterant.str.strip().str.casefold().eq("unknown")
    )

    # Create independent training and test dataframes.
    train_data = data.loc[train_mask].reset_index(drop=True)
    test_data = data.loc[test_mask].reset_index(drop=True)

    # Exclude metadata columns and retain the spectral measurements.
    metadata_columns = data.columns[:4].tolist()
    spectral_columns = [
        column for column in data.columns
        if column not in metadata_columns
    ]

    # Extract spectral arrays without fitting preprocessing on test data.
    X_train = train_data[spectral_columns].to_numpy(dtype=float)
    X_test = test_data[spectral_columns].to_numpy(dtype=float)

    # Stop early if the expected known-label test set is not present.
    if len(test_data) != 9:
        raise ValueError(
            f"Expected 9 known-label test samples, found {len(test_data)}. "
            "Check Dataset_Role and label filters before proceeding."
        )

    # Display the split sizes before fitting either model.
    print(f"Training samples: {len(train_data)}")
    print(f"Known-label test samples: {len(test_data)}")
    print(f"Spectral features: {X_train.shape[1]}")

    # Store the reports produced by both model families.
    all_metrics = []
    all_predictions = []

    # Evaluate the frozen SVM and PLS-DA configurations.
    for model_name in ["SVM", "PLS-DA"]:
        # Fit on training data and evaluate on the held-out test set.
        metrics, predictions = evaluate_model(
            X_train,
            train_data,
            X_test,
            test_data,
            model_name,
        )

        # Collect each model's evaluation tables.
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    # Combine the two models' per-dye metrics and predictions.
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    # Create the report directory if it does not already exist.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the final test metrics separately from exploratory CV reports.
    metrics_df.to_csv(
        REPORT_DIR / "final_test_svm_vs_plsda_metrics.csv",
        index=False,
    )

    # Save per-sample, per-dye test predictions for detailed inspection.
    predictions_df.to_csv(
        REPORT_DIR / "final_test_svm_vs_plsda_predictions.csv",
        index=False,
    )

    # Print metrics with the number of actual positive test samples.
    print("\nFinal held-out test metrics:")
    print(metrics_df.to_string(index=False))

    # Show only AP values that can be evaluated on this test set.
    print("\nAverage Precision comparison:")
    print(
        metrics_df.pivot(
            index="Dye",
            columns="Model",
            values="Average_Precision",
        ).to_string()
    )

    # Confirm where the separate reports were saved.
    print("\nReports saved in:", REPORT_DIR)


# Execute the final evaluation when this script is run directly.
if __name__ == "__main__":
    main()

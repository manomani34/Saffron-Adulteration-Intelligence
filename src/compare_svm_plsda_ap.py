
# Import libraries for data processing, classification, and evaluation.
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Define project paths and shared validation settings.
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "total-color-data.csv"
REPORT_DIR = ROOT / "reports"

# Use the same dye labels and repeated splits for both models.
DYES = [
    "Amaranth", "Azorubine", "Erythrosine", "Orange 2",
    "Orange GGN", "Ponceau 4R", "Quinoline Yellow",
    "Sunset Yellow", "Tartrazine",
]
N_SPLITS = 2
N_REPEATS = 20
RANDOM_STATE = 42
PLS_COMPONENTS = 4


# Check whether a known dye name occurs in the sample's adulterant description.
def contains_dye(description, dye):
    # Normalize both values for case-insensitive matching.
    return dye.casefold() in str(description).casefold()


# Generate repeated out-of-fold scores for one model and one dye.
def evaluate_model(X, y, model_name, n_components=None):
    # Accumulate scores so each sample receives predictions from every repeat.
    score_sum = np.zeros(len(y), dtype=float)

    # Count the held-out predictions received by each sample.
    score_count = np.zeros(len(y), dtype=int)

    # Reuse identical random seeds across models for a fair comparison.
    for repeat in range(N_REPEATS):
        # Create stratified folds with matching settings for both algorithms.
        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + repeat,
        )

        # Fit a fresh model within each training fold.
        for train_idx, valid_idx in cv.split(X, y):
            # Scale spectral features using training-fold data only.
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_valid = scaler.transform(X[valid_idx])

            # Build the selected classification model.
            if model_name == "SVM":
                # Use the same RBF SVM configuration as the earlier experiment.
                model = SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                )
            else:
                # Use PLS regression as a binary PLS-DA model.
                model = PLSRegression(
                    n_components=n_components,
                    scale=False,
                )

            # Fit the model using only the training fold.
            model.fit(X_train, y[train_idx])

            # Obtain continuous scores without applying a classification threshold.
            if model_name == "SVM":
                # Decision scores rank samples by their distance from the SVM boundary.
                scores = model.decision_function(X_valid)
            else:
                # PLS predictions provide continuous scores for the positive class.
                scores = model.predict(X_valid).ravel()

            # Accumulate scores for the held-out samples.
            score_sum[valid_idx] += scores
            score_count[valid_idx] += 1

    # Average each sample's held-out scores over all repeats.
    averaged_scores = score_sum / np.maximum(score_count, 1)

    # Evaluate ranking quality independently of a fixed decision threshold.
    average_precision = average_precision_score(y, averaged_scores)

    # Return the sample counts and AP score for this experiment.
    return {
        "Model": (
            "PLS-DA-4" if model_name == "PLS-DA" else model_name
        ),
        "Positive_Samples": int(y.sum()),
        "Average_Precision": float(average_precision),
    }


# Load eligible training data and compare both models across all dyes.
def main():
    # Read the customer dataset.
    data = pd.read_csv(DATA_PATH)

    # Normalize metadata for consistent sample filtering.
    role = data["Dataset_Role"].astype(str).str.casefold()
    target = data["Target_Label"].astype(str).str.casefold()
    adulterant = data["Adulterant_Type"].fillna("").astype(str)

    # Exclude test, reference, and unknown-label samples from training evaluation.
    eligible = (
        role.eq("train")
        & target.isin(["genuine", "adulterated"])
        & ~adulterant.str.strip().str.casefold().eq("unknown")
    )
    train_data = data.loc[eligible].reset_index(drop=True)

    # Select spectral columns after excluding the first four metadata columns.
    metadata_columns = data.columns[:4].tolist()
    spectral_columns = [
        column for column in data.columns
        if column not in metadata_columns
    ]
    X = train_data[spectral_columns].to_numpy(dtype=float)

    # Store the per-dye comparison results.
    results = []

    # Compare both algorithms using the same binary target for each dye.
    for dye in DYES:
        # Build a binary target indicating whether this dye is present.
        y = train_data["Adulterant_Type"].apply(
            lambda description: int(contains_dye(description, dye))
        ).to_numpy()

        # Skip dyes with too few positive or negative examples for two-fold CV.
        if min(int(y.sum()), int(len(y) - y.sum())) < N_SPLITS:
            print(f"Skipping {dye}: insufficient class counts.")
            continue

        # Evaluate SVM and PLS-DA on the exact same fold definitions.
        for model_name, components in [
            ("SVM", None),
            ("PLS-DA", PLS_COMPONENTS),
        ]:
            # Save this model's out-of-fold average precision.
            result = evaluate_model(
                X,
                y,
                model_name,
                n_components=components,
            )

            # Attach the dye name to the result.
            result["Dye"] = dye
            results.append(result)

    # Convert all model results into a comparison table.
    details = pd.DataFrame(results)

    # Summarize mean AP and standard deviation across evaluated dyes.
    summary = (
        details.groupby("Model")["Average_Precision"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "Mean_Average_Precision",
            "std": "Std_Average_Precision",
        })
    )

    # Create the report folder if needed.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save separate files without overwriting previous experiments.
    details.to_csv(
        REPORT_DIR / "svm_vs_plsda_ap_details.csv",
        index=False,
    )
    summary.to_csv(
        REPORT_DIR / "svm_vs_plsda_ap_summary.csv",
        index=False,
    )

    # Display the overall and per-dye comparison.
    print("\nOverall comparison:")
    print(summary.to_string(index=False))
    print("\nPer-dye comparison:")
    print(
        details.pivot(
            index="Dye",
            columns="Model",
            values="Average_Precision",
        ).to_string()
    )
    print("\nReports saved in:", REPORT_DIR)


# Run the comparison when this file is executed directly.
if __name__ == "__main__":
    main()

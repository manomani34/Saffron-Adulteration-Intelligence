
# Import the libraries required for spectral classification and evaluation.
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Define project paths and reproducible experiment settings.
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "total-color-data.csv"
REPORT_DIR = ROOT / "reports"

# Define the dye labels that the experiment evaluates independently.
DYES = [
    "Amaranth", "Azorubine", "Erythrosine", "Orange 2",
    "Orange GGN", "Ponceau 4R", "Quinoline Yellow",
    "Sunset Yellow", "Tartrazine",
]

# Compare these candidate numbers of latent PLS components.
COMPONENT_COUNTS = [1, 2, 3, 4, 5]

# Keep the cross-validation settings consistent with the baseline experiment.
N_SPLITS = 2
N_REPEATS = 20
RANDOM_STATE = 42


# Convert an adulterant description into binary labels for the requested dye.
def contains_dye(description, dye):
    # Normalize text to make label matching case-insensitive.
    normalized_description = str(description).casefold()
    normalized_dye = dye.casefold()

    # Treat the dye as present when its name occurs in the description.
    return normalized_dye in normalized_description


# Evaluate one dye and one component count using repeated out-of-fold predictions.
def evaluate_components(X, y, dye, n_components):
    # Store each sample's predictions from its held-out folds.
    score_sum = np.zeros(len(y), dtype=float)

    # Count how many held-out predictions each sample receives.
    score_count = np.zeros(len(y), dtype=int)

    # Repeat stratified cross-validation with reproducible random seeds.
    for repeat in range(N_REPEATS):
        # Build folds while preserving the positive/negative proportions.
        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + repeat,
        )

        # Train and evaluate the model separately in each fold.
        for train_idx, valid_idx in cv.split(X, y):
            # Fit scaling only on the training fold to prevent preprocessing leakage.
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_valid = scaler.transform(X[valid_idx])

            # Create the PLS regression model used as a binary PLS-DA classifier.
            model = PLSRegression(n_components=n_components, scale=False)
            model.fit(X_train, y[train_idx])

            # Save continuous positive-class scores for held-out samples.
            fold_scores = model.predict(X_valid).ravel()
            score_sum[valid_idx] += fold_scores
            score_count[valid_idx] += 1

    # Average held-out scores across repeats for each sample.
    averaged_scores = score_sum / np.maximum(score_count, 1)

    # Calculate threshold-independent ranking performance.
    average_precision = average_precision_score(y, averaged_scores)

    # Return the experiment settings and evaluation result.
    return {
        "Dye": dye,
        "PLS_Components": n_components,
        "Positive_Samples": int(y.sum()),
        "Negative_Samples": int(len(y) - y.sum()),
        "Average_Precision": float(average_precision),
    }


# Load eligible training samples and run the component-count experiment.
def main():
    # Load the customer's spectral dataset.
    data = pd.read_csv(DATA_PATH)

    # Normalize metadata text for reliable filtering.
    role = data["Dataset_Role"].astype(str).str.casefold()
    target = data["Target_Label"].astype(str).str.casefold()
    adulterant = data["Adulterant_Type"].fillna("").astype(str)

    # Use training samples with known genuine/adulterated status only.
    eligible = (
        role.eq("train")
        & target.isin(["genuine", "adulterated"])
        & ~adulterant.str.casefold().str.strip().eq("unknown")
    )
    train_data = data.loc[eligible].reset_index(drop=True)

    # Exclude metadata and retain numeric spectral measurements.
    metadata_columns = data.columns[:4].tolist()
    spectral_columns = [
        column for column in data.columns
        if column not in metadata_columns
    ]
    X = train_data[spectral_columns].to_numpy(dtype=float)

    # Collect one-vs-rest results for every dye and component count.
    results = []

    # Evaluate only dyes with enough positive and negative samples for two folds.
    for dye in DYES:
        # Create the binary target for the current dye.
        y = train_data["Adulterant_Type"].apply(
            lambda description: int(contains_dye(description, dye))
        ).to_numpy()

        # Skip labels that cannot support stratified two-fold validation.
        if min(int(y.sum()), int(len(y) - y.sum())) < N_SPLITS:
            print(f"Skipping {dye}: insufficient positive or negative samples.")
            continue

        # Evaluate every candidate component count on identical split seeds.
        for component_count in COMPONENT_COUNTS:
            # Record performance for this dye and model complexity.
            results.append(
                evaluate_components(X, y, dye, component_count)
            )

    # Convert the results into a report table.
    details = pd.DataFrame(results)

    # Summarize mean and standard deviation across dyes for each component count.
    summary = (
        details.groupby("PLS_Components")["Average_Precision"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "Mean_Average_Precision",
            "std": "Std_Average_Precision",
        })
    )

    # Create the report directory if it does not exist.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save reports separately without overwriting previous SVM or PLS-DA results.
    details.to_csv(
        REPORT_DIR / "plsda_components_details.csv",
        index=False,
    )
    summary.to_csv(
        REPORT_DIR / "plsda_components_summary.csv",
        index=False,
    )

    # Display the component comparison for quick review.
    print("\nPLS-DA component comparison:")
    print(summary.to_string(index=False))
    print("\nPer-dye results:")
    print(details.to_string(index=False))
    print("\nReports saved in:", REPORT_DIR)


# Run the experiment when this script is executed directly.
if __name__ == "__main__":
    main()

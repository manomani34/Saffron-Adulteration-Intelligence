from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "total-color-data.csv"
OUT_DIR = ROOT / "reports" / "statistical_baseline"

# Number of equally sized spectral regions to summarize.
N_REGIONS = 20
TOP_FEATURES = 10


def load_data():
    """Load samples and identify numeric spectral columns."""
    df = pd.read_csv(DATA_PATH)

    metadata = {
        "Sample_ID",
        "Target_Label",
        "Adulterant_Type",
        "Dataset_Role",
    }

    spectral_cols = [
        c for c in df.columns
        if c not in metadata
        and pd.to_numeric(
            pd.Series([c]), errors="coerce"
        ).notna().all()
    ]

    if not spectral_cols:
        raise ValueError("No numeric spectral columns were found.")

    df[spectral_cols] = df[spectral_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    df = df.dropna(
        subset=["Sample_ID", "Target_Label", "Dataset_Role"]
    ).copy()

    return df, spectral_cols


def extract_features(df, spectral_cols):
    """Extract global and local summary statistics from each spectrum."""
    X = df[spectral_cols].to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, np.nan)

    features = pd.DataFrame(index=df.index)

    with np.errstate(all="ignore"):
        features["global_mean"] = np.nanmean(X, axis=1)
        features["global_std"] = np.nanstd(X, axis=1)
        features["global_min"] = np.nanmin(X, axis=1)
        features["global_max"] = np.nanmax(X, axis=1)
        features["global_range"] = (
            features["global_max"] - features["global_min"]
        )
        features["global_median"] = np.nanmedian(X, axis=1)
        features["global_q90"] = np.nanpercentile(X, 90, axis=1)
        features["global_q10"] = np.nanpercentile(X, 10, axis=1)

    # Split the spectrum into equal-index regions.
    edges = np.linspace(
        0, X.shape[1], N_REGIONS + 1, dtype=int
    )

    for i in range(N_REGIONS):
        start, end = edges[i], edges[i + 1]
        region = X[:, start:end]

        if region.shape[1] == 0:
            continue

        features[f"region_{i + 1:02d}_mean"] = np.nanmean(
            region, axis=1
        )
        features[f"region_{i + 1:02d}_max"] = np.nanmax(
            region, axis=1
        )
        features[f"region_{i + 1:02d}_std"] = np.nanstd(
            region, axis=1
        )
        features[f"region_{i + 1:02d}_area"] = np.trapezoid(
            np.nan_to_num(region, nan=0.0), axis=1
        )

    # Ratios can reveal relative spectral changes.
    eps = 1e-12

    for a, b in [(1, 5), (1, 10), (5, 15), (10, 20)]:
        ca = f"region_{a:02d}_mean"
        cb = f"region_{b:02d}_mean"

        if ca in features and cb in features:
            features[f"ratio_{a:02d}_{b:02d}"] = (
                features[ca] / (features[cb].abs() + eps)
            )

    return features.replace([np.inf, -np.inf], np.nan)


def choose_threshold(x, y):
    """Choose a feature threshold using training data only."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    valid = np.isfinite(x)
    x, y = x[valid], y[valid]

    unique = np.unique(x)

    if len(unique) < 2 or len(np.unique(y)) < 2:
        return np.nan, ">", np.nan

    candidates = (unique[:-1] + unique[1:]) / 2
    best = (-1.0, np.nan, ">")

    for threshold in candidates:
        for direction in (">", "<"):
            pred = (
                x > threshold
                if direction == ">"
                else x < threshold
            ).astype(int)

            score = balanced_accuracy_score(y, pred)

            if score > best[0]:
                best = (score, threshold, direction)

    return best[1], best[2], best[0]


def evaluate_feature(name, train_x, train_y, test_x, test_y):
    """Fit a single-feature threshold and evaluate held-out data."""
    threshold, direction, train_bacc = choose_threshold(
        train_x, train_y
    )

    if not np.isfinite(threshold):
        return None, None

    test_x = np.asarray(test_x, dtype=float)
    test_y = np.asarray(test_y, dtype=int)

    valid_test = np.isfinite(test_x)
    tx = test_x[valid_test]
    ty = test_y[valid_test]

    if len(ty) == 0 or len(np.unique(ty)) < 2:
        # AUC and balanced accuracy require both classes.
        # This also prevents misleading results if the test
        # selection is accidentally changed.
        return None, None

    pred = (
        tx > threshold
        if direction == ">"
        else tx < threshold
    ).astype(int)

    auc = roc_auc_score(ty, tx)

    row = {
        "Feature": name,
        "Threshold": threshold,
        "Direction": direction,
        "Train_Balanced_Accuracy": train_bacc,
        "Test_N": len(ty),
        "Test_Accuracy": accuracy_score(ty, pred),
        "Test_Balanced_Accuracy": balanced_accuracy_score(
            ty, pred
        ),
        "Test_Precision": precision_score(
            ty, pred, zero_division=0
        ),
        "Test_Recall": recall_score(
            ty, pred, zero_division=0
        ),
        "Test_F1": f1_score(
            ty, pred, zero_division=0
        ),
        "Test_Feature_AUC": auc,
    }

    predictions = pd.DataFrame({
        "Feature": name,
        "Actual": ty,
        "Predicted": pred,
        "Feature_Value": tx,
        "Threshold": threshold,
        "Direction": direction,
    })

    return row, predictions


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, spectral_cols = load_data()
    features = extract_features(df, spectral_cols)

    labels = (
        df["Target_Label"]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    roles = (
        df["Dataset_Role"]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    adulterant = (
        df["Adulterant_Type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    # Binary target: Genuine=0, Adulterated=1.
    known = labels.isin(["genuine", "adulterated"])
    y = labels.map({
        "genuine": 0,
        "adulterated": 1,
    })

    # Keep training selection consistent with the established
    # binary classification protocol.
    train_mask = (
        known
        & roles.eq("train")
        & ~adulterant.eq("unknown")
    )

    # Match the existing SVM/PLS-DA evaluation protocol:
    # exclude test samples whose adulterant identity is Unknown.
    # Genuine samples with missing Adulterant_Type remain included.
    test_mask = (
        known
        & roles.eq("test")
        & ~adulterant.eq("unknown")
    )

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print("Labels by role:")
        print(pd.crosstab(df["Dataset_Role"], df["Target_Label"]))

        raise ValueError(
            "Could not find labelled Train and Test samples. "
            "Review the label and role values."
        )

    X_train = features.loc[train_mask]
    y_train = y.loc[train_mask].astype(int)

    X_test = features.loc[test_mask]
    y_test = y.loc[test_mask].astype(int)

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Both Genuine and Adulterated labels must occur "
            "in the Train and Test groups."
        )

    # Rank features using training data only.
    ranking = []

    for col in features.columns:
        a = X_train[col].to_numpy(dtype=float)
        yy = y_train.to_numpy()
        valid = np.isfinite(a)

        if valid.sum() < 4 or len(np.unique(yy[valid])) < 2:
            continue

        try:
            auc = roc_auc_score(yy[valid], a)

            ranking.append({
                "Feature": col,
                "Train_AUC": auc,
                "Train_AUC_Symmetry": max(auc, 1 - auc),
                "Train_N": int(valid.sum()),
            })
        except ValueError:
            pass

    ranking_df = pd.DataFrame(ranking)

    if ranking_df.empty:
        raise ValueError(
            "No valid features could be ranked using training data."
        )

    ranking_df = ranking_df.sort_values(
        "Train_AUC_Symmetry",
        ascending=False,
    )

    ranking_df.to_csv(
        OUT_DIR / "feature_ranking.csv",
        index=False,
    )

    # Evaluate the top-ranked single features on held-out test data.
    rows = []
    prediction_frames = []

    top = ranking_df.head(TOP_FEATURES)["Feature"].tolist()

    for col in top:
        result, predictions = evaluate_feature(
            col,
            X_train[col].to_numpy(),
            y_train.to_numpy(),
            X_test[col].to_numpy(),
            y_test.to_numpy(),
        )

        if result is not None:
            rows.append(result)
            prediction_frames.append(predictions)

    metrics = pd.DataFrame(rows)

    if not metrics.empty:
        metrics = metrics.sort_values(
            "Test_Balanced_Accuracy",
            ascending=False,
        )

    metrics.to_csv(
        OUT_DIR / "metrics.csv",
        index=False,
    )

    if prediction_frames:
        pd.concat(
            prediction_frames,
            ignore_index=True,
        ).to_csv(
            OUT_DIR / "test_predictions.csv",
            index=False,
        )

    # Save derived features and sample metadata.
    feature_output = pd.concat(
        [
            df[[
                "Sample_ID",
                "Target_Label",
                "Adulterant_Type",
                "Dataset_Role",
            ]],
            features,
        ],
        axis=1,
    )

    feature_output.to_csv(
        OUT_DIR / "features.csv",
        index=False,
    )

    # Print the actual evaluation populations.
    print("\n=== Dataset split (all rows) ===")
    print(pd.crosstab(df["Dataset_Role"], df["Target_Label"]))

    print("\n=== Evaluation sample counts ===")
    print(f"Spectral points: {len(spectral_cols)}")
    print(f"Derived features: {features.shape[1]}")
    print(f"Training samples: {int(train_mask.sum())}")
    print(f"Test samples: {int(test_mask.sum())}")

    print("\nTest samples included:")
    print(
        df.loc[
            test_mask,
            [
                "Sample_ID",
                "Target_Label",
                "Adulterant_Type",
            ],
        ].to_string(index=False)
    )

    print("\nTest samples excluded because Adulterant_Type=Unknown:")
    print(
        df.loc[
            known & roles.eq("test") & adulterant.eq("unknown"),
            [
                "Sample_ID",
                "Target_Label",
                "Adulterant_Type",
            ],
        ].to_string(index=False)
    )

    print("\n=== Top features (training ranking) ===")
    print(ranking_df.head(TOP_FEATURES).to_string(index=False))

    print("\n=== Held-out test results ===")

    if metrics.empty:
        print("No valid feature results were produced.")
    else:
        print(metrics.to_string(index=False))

    print(f"\nReports saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
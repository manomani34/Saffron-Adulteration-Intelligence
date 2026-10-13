
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

from analyze_statistical_baseline import load_data, extract_features

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "statistical_baseline"
N_SPLITS = 2
N_REPEATS = 20


def best_threshold(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    valid = np.isfinite(x)
    x, y = x[valid], y[valid]

    unique = np.unique(x)
    if len(unique) < 2 or len(np.unique(y)) < 2:
        return None

    candidates = (unique[:-1] + unique[1:]) / 2
    best = None

    for threshold in candidates:
        for direction in (">", "<"):
            pred = (
                x > threshold if direction == ">"
                else x < threshold
            ).astype(int)

            score = balanced_accuracy_score(y, pred)
            if best is None or score > best["train_bacc"]:
                best = {
                    "threshold": threshold,
                    "direction": direction,
                    "train_bacc": score,
                }

    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    df, spectral_cols = load_data()
    features = extract_features(df, spectral_cols)

    labels = df["Target_Label"].astype(str).str.strip().str.lower()
    roles = df["Dataset_Role"].astype(str).str.strip().str.lower()

    mask = (
        roles.eq("train")
        & labels.isin(["genuine", "adulterated"])
    )

    X = features.loc[mask].reset_index(drop=True)
    y = labels.loc[mask].map({
        "genuine": 0,
        "adulterated": 1,
    }).to_numpy(dtype=int)

    if len(np.unique(y)) != 2:
        raise ValueError("Training data must contain both classes.")

    fold_rows = []
    prediction_rows = []

    for repeat in range(N_REPEATS):
        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=42 + repeat,
        )

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), 1):
            train_x = X.iloc[train_idx]
            valid_x = X.iloc[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            # Select the feature using only this fold's training data.
            ranking = []
            for col in X.columns:
                values = train_x[col].to_numpy(dtype=float)
                ok = np.isfinite(values)

                if ok.sum() < 4 or len(np.unique(y_train[ok])) < 2:
                    continue

                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_train[ok], values[ok])

                ranking.append((col, max(auc, 1 - auc)))

            if not ranking:
                continue

            ranking.sort(key=lambda item: item[1], reverse=True)
            feature = ranking[0][0]

            # Fit threshold on the training fold only.
            rule = best_threshold(
                train_x[feature].to_numpy(),
                y_train,
            )
            if rule is None:
                continue

            values = valid_x[feature].to_numpy(dtype=float)
            valid = np.isfinite(values)
            if not valid.any():
                continue

            actual = y_valid[valid]
            values = values[valid]

            pred = (
                values > rule["threshold"]
                if rule["direction"] == ">"
                else values < rule["threshold"]
            ).astype(int)

            fold_rows.append({
                "Repeat": repeat + 1,
                "Fold": fold,
                "Selected_Feature": feature,
                "Train_Balanced_Accuracy": rule["train_bacc"],
                "Validation_Accuracy": accuracy_score(actual, pred),
                "Validation_Balanced_Accuracy":
                    balanced_accuracy_score(actual, pred),
                "Validation_N": len(actual),
            })

            for a, p, v in zip(actual, pred, values):
                prediction_rows.append({
                    "Repeat": repeat + 1,
                    "Fold": fold,
                    "Feature": feature,
                    "Actual": int(a),
                    "Predicted": int(p),
                    "Value": float(v),
                    "Threshold": float(rule["threshold"]),
                    "Direction": rule["direction"],
                })

    results = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)

    results.to_csv(OUT / "statistical_cv_folds.csv", index=False)
    predictions.to_csv(
        OUT / "statistical_cv_predictions.csv", index=False
    )

    print("\n=== Repeated CV summary ===")
    print(f"Training samples: {len(y)}")
    print(f"Repeats: {N_REPEATS}; folds per repeat: {N_SPLITS}")

    if results.empty:
        print("No valid CV results.")
        return

    print("\nFold-level metrics:")
    print(
        results[
            ["Validation_Accuracy", "Validation_Balanced_Accuracy"]
        ].agg(["mean", "std"]).to_string()
    )

    print("\nFeature selection frequency:")
    print(
        results["Selected_Feature"]
        .value_counts()
        .head(15)
        .to_string()
    )

    print("\nResults saved to:", OUT)


if __name__ == "__main__":
    main()

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from analyze_statistical_baseline import (
    load_data,
    extract_features,
    choose_threshold,
)


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports" / "binary_model_comparison"

RANDOM_STATE = 42
N_SPLITS = 2
N_REPEATS = 20
PLS_COMPONENTS = 4


def auc_or_nan(y, scores):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, scores)


def calculate_metrics(y, pred, scores):
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "N": len(y),
        "Accuracy": accuracy_score(y, pred),
        "Balanced_Accuracy": balanced_accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": auc_or_nan(y, scores),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def rank_best_feature(X, y):
    """Select one feature using only the current training partition."""
    best_name = None
    best_auc = -np.inf

    for col in X.columns:
        values = X[col].to_numpy(dtype=float)
        valid = np.isfinite(values)

        if valid.sum() < 4 or len(np.unique(y[valid])) < 2:
            continue

        auc = roc_auc_score(y[valid], values[valid])
        symmetric_auc = max(auc, 1.0 - auc)

        if symmetric_auc > best_auc:
            best_auc = symmetric_auc
            best_name = col

    if best_name is None:
        raise ValueError("No valid statistical feature was found.")

    return best_name, best_auc


def fit_statistical_model(X_train, y_train, X_eval):
    """Select a feature and threshold using training data only."""
    feature, train_auc = rank_best_feature(X_train, y_train)

    train_x = X_train[feature].to_numpy(dtype=float)
    eval_x = X_eval[feature].to_numpy(dtype=float)

    threshold, direction, train_bacc = choose_threshold(
        train_x, y_train
    )

    if not np.isfinite(threshold):
        raise ValueError(
            f"Could not determine threshold for {feature}."
        )

    valid_eval = np.isfinite(eval_x)

    pred = np.full(len(eval_x), -1, dtype=int)
    pred[valid_eval] = (
        (eval_x[valid_eval] > threshold)
        if direction == ">"
        else (eval_x[valid_eval] < threshold)
    ).astype(int)

    # Higher scores must consistently mean more likely adulterated.
    scores = eval_x if direction == ">" else -eval_x

    return {
        "feature": feature,
        "train_auc": train_auc,
        "train_bacc": train_bacc,
        "threshold": threshold,
        "direction": direction,
        "pred": pred,
        "scores": scores,
    }


def make_svm():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
        ),
    )


def make_plsda():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        PLSRegression(
            n_components=PLS_COMPONENTS,
            scale=False,
            max_iter=1000,
        ),
    )


def evaluate_sklearn_model(model, X_train, y_train, X_eval):
    model.fit(X_train, y_train)

    if isinstance(model[-1], SVC):
        scores = model.decision_function(X_eval)
        pred = (scores > 0).astype(int)
    else:
        scores = np.asarray(model.predict(X_eval)).reshape(-1)
        pred = (scores >= 0.5).astype(int)

    return pred, scores


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, spectral_cols = load_data()

    labels = (
        df["Target_Label"].astype(str).str.strip().str.casefold()
    )
    roles = (
        df["Dataset_Role"].astype(str).str.strip().str.casefold()
    )
    adulterant = (
        df["Adulterant_Type"].fillna("").astype(str)
        .str.strip().str.casefold()
    )

    known = labels.isin(["genuine", "adulterated"])
    y_all = labels.map({
        "genuine": 0,
        "adulterated": 1,
    })

    # Keep the test protocol consistent with the existing evaluation:
    # remove samples with explicitly unknown adulterant identity.
    train_mask = (
        known & roles.eq("train") & ~adulterant.eq("unknown")
    )
    test_mask = (
        known & roles.eq("test") & ~adulterant.eq("unknown")
    )

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    y_train = y_all.loc[train_mask].to_numpy(dtype=int)
    y_test = y_all.loc[test_mask].to_numpy(dtype=int)

    X_train_raw = train_df[spectral_cols].to_numpy(dtype=float)
    X_test_raw = test_df[spectral_cols].to_numpy(dtype=float)

    X_train_features = extract_features(
        train_df, spectral_cols
    )
    X_test_features = extract_features(
        test_df, spectral_cols
    )

    if len(y_train) != 43 or len(y_test) != 9:
        raise ValueError(
            f"Expected 43 Train and 9 Test samples; got "
            f"{len(y_train)} and {len(y_test)}. "
            "Review the dataset split before continuing."
        )

    if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
        raise ValueError("Both classes must occur in Train and Test.")

    # ---------------------------------------------------------
    # 1. Final held-out Test evaluation.
    # ---------------------------------------------------------
    test_rows = []
    prediction_frames = []

    stat = fit_statistical_model(
        X_train_features, y_train, X_test_features
    )

    methods = {
        "Statistical_Threshold": (
            stat["pred"],
            stat["scores"],
        )
    }

    svm = make_svm()
    methods["SVM_RBF"] = evaluate_sklearn_model(
        svm, X_train_raw, y_train, X_test_raw
    )

    plsda = make_plsda()
    methods["PLS_DA"] = evaluate_sklearn_model(
        plsda, X_train_raw, y_train, X_test_raw
    )

    for method, (pred, scores) in methods.items():
        if np.any(pred < 0) or not np.isfinite(scores).all():
            raise ValueError(
                f"{method} produced invalid predictions or scores."
            )

        metrics = calculate_metrics(y_test, pred, scores)
        metrics["Method"] = method
        test_rows.append(metrics)

        prediction_frames.append(pd.DataFrame({
            "Method": method,
            "Sample_ID": test_df["Sample_ID"].to_numpy(),
            "Actual": y_test,
            "Predicted": pred,
            "Score": scores,
        }))

    test_results = pd.DataFrame(test_rows)
    test_results = test_results[
        ["Method"] + [
            c for c in test_results.columns if c != "Method"
        ]
    ]
    test_results.to_csv(
        OUT_DIR / "test_metrics.csv", index=False
    )

    pd.concat(
        prediction_frames, ignore_index=True
    ).to_csv(
        OUT_DIR / "test_predictions.csv", index=False
    )

    pd.DataFrame([{
        "Selected_Feature": stat["feature"],
        "Train_AUC_Symmetry": stat["train_auc"],
        "Train_Balanced_Accuracy": stat["train_bacc"],
        "Threshold": stat["threshold"],
        "Direction": stat["direction"],
    }]).to_csv(
        OUT_DIR / "selected_statistical_feature.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 2. Repeated CV on Train only.
    # Every fold independently selects the statistical feature
    # and learns its threshold from that fold's training samples.
    # ---------------------------------------------------------
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    cv_rows = []

    for fold_number, (fit_idx, val_idx) in enumerate(
        cv.split(X_train_raw, y_train), start=1
    ):
        y_fit = y_train[fit_idx]
        y_val = y_train[val_idx]

        raw_fit = X_train_raw[fit_idx]
        raw_val = X_train_raw[val_idx]

        feat_fit = X_train_features.iloc[fit_idx]
        feat_val = X_train_features.iloc[val_idx]

        fold_stat = fit_statistical_model(
            feat_fit, y_fit, feat_val
        )

        fold_methods = {
            "Statistical_Threshold": (
                fold_stat["pred"], fold_stat["scores"]
            )
        }

        fold_methods["SVM_RBF"] = evaluate_sklearn_model(
            make_svm(), raw_fit, y_fit, raw_val
        )

        fold_methods["PLS_DA"] = evaluate_sklearn_model(
            make_plsda(), raw_fit, y_fit, raw_val
        )

        for method, (pred, scores) in fold_methods.items():
            if np.any(pred < 0) or not np.isfinite(scores).all():
                raise ValueError(
                    f"Invalid CV output in fold {fold_number}: {method}"
                )

            row = calculate_metrics(y_val, pred, scores)
            row["Fold"] = fold_number
            row["Method"] = method
            row["Selected_Feature"] = (
                fold_stat["feature"]
                if method == "Statistical_Threshold"
                else ""
            )
            cv_rows.append(row)

    cv_results = pd.DataFrame(cv_rows)
    cv_results.to_csv(
        OUT_DIR / "cv_fold_metrics.csv", index=False
    )

    metric_columns = [
        "Accuracy",
        "Balanced_Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
    ]

    summary_rows = []

    for method, group in cv_results.groupby("Method"):
        row = {"Method": method}

        for metric in metric_columns:
            row[f"{metric}_Mean"] = group[metric].mean()
            row[f"{metric}_Std"] = group[metric].std(ddof=1)

        summary_rows.append(row)

    cv_summary = pd.DataFrame(summary_rows).sort_values(
        "Balanced_Accuracy_Mean", ascending=False
    )
    cv_summary.to_csv(
        OUT_DIR / "cv_summary.csv", index=False
    )

    feature_frequency = (
        cv_results.loc[
            cv_results["Method"].eq("Statistical_Threshold"),
            "Selected_Feature",
        ]
        .value_counts()
        .rename_axis("Feature")
        .reset_index(name="Selection_Count")
    )
    feature_frequency.to_csv(
        OUT_DIR / "cv_feature_selection_frequency.csv",
        index=False,
    )

    print("\n=== Dataset ===")
    print(f"Spectral points: {len(spectral_cols)}")
    print(f"Train samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print("Class encoding: Genuine=0, Adulterated=1")

    print("\n=== Held-out Test results ===")
    print(test_results.to_string(index=False))

    print("\n=== Repeated CV on Train: mean and standard deviation ===")
    print(cv_summary.to_string(index=False))

    print("\n=== Statistical feature selection frequency ===")
    print(feature_frequency.to_string(index=False))

    print(f"\nReports saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

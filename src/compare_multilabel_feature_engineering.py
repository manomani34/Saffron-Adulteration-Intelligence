from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from analyze_statistical_baseline import load_data, extract_features


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "multilabel_feature_engineering"

DYES = [
    "Amaranth", "Azorubine", "Erythrosine", "Orange 2",
    "Orange GGN", "Ponceau 4R", "Quinoline Yellow",
    "Sunset Yellow", "Tartrazine",
]

N_SPLITS = 2
N_REPEATS = 20
RANDOM_STATE = 42
PLS_COMPONENTS = 4
TOP_K = 10


def contains_dye(description, dye):
    return dye.casefold() in str(description).casefold()


def make_model(name):
    if name.startswith("SVM"):
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
        )

    return PLSRegression(
        n_components=PLS_COMPONENTS,
        scale=False,
    )


def get_scores(model, X, name):
    if name.startswith("SVM"):
        return model.decision_function(X)

    return model.predict(X).ravel()


def evaluate_cv(X, y, model_name, use_engineered):
    fold_rows = []

    for repeat in range(N_REPEATS):
        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + repeat,
        )

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), 1):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            if use_engineered:
                # Select features using this training fold only.
                k = min(TOP_K, X_train.shape[1])
                selector = SelectKBest(
                    score_func=lambda a, b: mutual_info_classif(
                        a, b, random_state=RANDOM_STATE
                    ),
                    k=k,
                )
                X_train = selector.fit_transform(X_train, y_train)
                X_valid = selector.transform(X_valid)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)

            model = make_model(model_name)

            # PLS needs enough training samples and selected features.
            if model_name == "PLS" and min(
                X_train.shape[0] - 1, X_train.shape[1]
            ) < PLS_COMPONENTS:
                continue

            model.fit(X_train, y_train)
            scores = get_scores(model, X_valid, model_name)

            fold_rows.append({
                "Repeat": repeat + 1,
                "Fold": fold,
                "AP": average_precision_score(y_valid, scores),
            })

    return fold_rows


def fit_test(X_train, y_train, X_test, model_name, use_engineered):
    if use_engineered:
        selector = SelectKBest(
            score_func=lambda a, b: mutual_info_classif(
                a, b, random_state=RANDOM_STATE
            ),
            k=min(TOP_K, X_train.shape[1]),
        )
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = make_model(model_name)

    if model_name == "PLS" and min(
        X_train.shape[0] - 1, X_train.shape[1]
    ) < PLS_COMPONENTS:
        raise ValueError("Not enough data for the configured PLS components.")

    model.fit(X_train, y_train)
    return get_scores(model, X_test, model_name)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df, spectral_cols = load_data()

    roles = df["Dataset_Role"].astype(str).str.strip().str.casefold()
    labels = df["Target_Label"].astype(str).str.strip().str.casefold()
    adulterant = (
        df["Adulterant_Type"].fillna("").astype(str).str.strip().str.casefold()
    )

    known = labels.isin(["genuine", "adulterated"])

    train_mask = (
        known & roles.eq("train") & ~adulterant.eq("unknown")
    )
    test_mask = (
        known & roles.eq("test") & ~adulterant.eq("unknown")
    )

    train_df = df.loc[train_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)

    if len(train_df) != 43 or len(test_df) != 9:
        raise ValueError(
            f"Expected 43 Train and 9 Test samples; got "
            f"{len(train_df)} and {len(test_df)}. Check dataset filters."
        )

    X_raw_train = train_df[spectral_cols].to_numpy(dtype=float)
    X_raw_test = test_df[spectral_cols].to_numpy(dtype=float)

    # Feature extraction uses each sample's spectrum only, not its label.
    X_eng_train = extract_features(train_df, spectral_cols).to_numpy(dtype=float)
    X_eng_test = extract_features(test_df, spectral_cols).to_numpy(dtype=float)

    if not np.isfinite(X_raw_train).all() or not np.isfinite(X_raw_test).all():
        raise ValueError("Raw spectra contain missing or infinite values.")
    if not np.isfinite(X_eng_train).all() or not np.isfinite(X_eng_test).all():
        raise ValueError("Engineered features contain missing or infinite values.")

    cv_results = []
    test_results = []
    test_predictions = []

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Raw spectral variables: {X_raw_train.shape[1]}")
    print(f"Engineered features: {X_eng_train.shape[1]}")

    for dye in DYES:
        y_train = train_df["Adulterant_Type"].apply(
            lambda value: int(contains_dye(value, dye))
        ).to_numpy(dtype=int)

        y_test = test_df["Adulterant_Type"].apply(
            lambda value: int(contains_dye(value, dye))
        ).to_numpy(dtype=int)

        if min(y_train.sum(), len(y_train) - y_train.sum()) < N_SPLITS:
            print(f"Skipping {dye}: insufficient training class counts.")
            continue

        if y_test.sum() == 0:
            print(f"Test AP unavailable for {dye}: no positive test samples.")

        for model_name in ["SVM", "PLS"]:
            for data_name, X_train, X_test, engineered in [
                ("Raw", X_raw_train, X_raw_test, False),
                ("Engineered_FS", X_eng_train, X_eng_test, True),
            ]:
                if model_name == "PLS" and engineered:
                    component_count = min(
                        PLS_COMPONENTS, TOP_K, len(y_train) - 1
                    )
                    if component_count < PLS_COMPONENTS:
                        continue

                folds = evaluate_cv(
                    X_train, y_train, model_name, engineered
                )

                for item in folds:
                    cv_results.append({
                        "Dye": dye,
                        "Model": model_name,
                        "Features": data_name,
                        **item,
                    })

                scores = fit_test(
                    X_train, y_train, X_test, model_name, engineered
                )

                ap_test = (
                    average_precision_score(y_test, scores)
                    if y_test.sum() > 0
                    else np.nan
                )

                # Preserve prior thresholds for reference only.
                threshold = 0.0 if model_name == "SVM" else 0.5
                predictions = (scores >= threshold).astype(int)

                test_results.append({
                    "Dye": dye,
                    "Model": model_name,
                    "Features": data_name,
                    "Train_Positive_Count": int(y_train.sum()),
                    "Test_Positive_Count": int(y_test.sum()),
                    "Test_AP": ap_test,
                    "Threshold": threshold,
                    "Test_Predicted_Positive_Count": int(predictions.sum()),
                })

                for i, sample_id in enumerate(test_df["Sample_ID"]):
                    test_predictions.append({
                        "Dye": dye,
                        "Model": model_name,
                        "Features": data_name,
                        "Sample_ID": sample_id,
                        "Actual": int(y_test[i]),
                        "Score": float(scores[i]),
                        "Predicted_at_old_threshold": int(predictions[i]),
                    })

    cv_df = pd.DataFrame(cv_results)
    test_df_out = pd.DataFrame(test_results)
    pred_df = pd.DataFrame(test_predictions)

    cv_df.to_csv(REPORT_DIR / "cv_fold_ap.csv", index=False)
    test_df_out.to_csv(REPORT_DIR / "test_ap.csv", index=False)
    pred_df.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    if not cv_df.empty:
        summary = (
            cv_df.groupby(["Model", "Features"])["AP"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={
                "mean": "CV_AP_Mean",
                "std": "CV_AP_Std",
                "count": "Fold_Count",
            })
        )
        summary.to_csv(REPORT_DIR / "cv_summary.csv", index=False)
        print("\n=== Cross-validation AP summary ===")
        print(summary.to_string(index=False))

    print("\n=== Held-out Test AP ===")
    print(test_df_out.to_string(index=False))
    print(f"\nReports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()

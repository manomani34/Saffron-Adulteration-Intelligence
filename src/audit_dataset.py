
# Imports tools for paths, JSON reports, and numerical analysis.
from pathlib import Path
import json

import numpy as np
import pandas as pd


# Defines the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Defines the source CSV file.
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "total-color-data.csv"

# Defines the directory for audit reports.
REPORT_DIR = PROJECT_ROOT / "reports"

# Defines the number of metadata columns at the beginning of the CSV.
METADATA_COLUMN_COUNT = 4

# Creates the report directory when necessary.
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# Loads the dataset using supported encodings.
def load_dataset(path: Path) -> pd.DataFrame:
    # Raises an error when the source file is missing.
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # Defines supported encodings for the input file.
    encodings = ("utf-8-sig", "utf-8", "cp1252", "latin1")

    # Attempts to read the CSV with each supported encoding.
    for encoding in encodings:
        try:
            # Returns the dataset when reading succeeds.
            return pd.read_csv(path, encoding=encoding)

        # Records common CSV reading failures and tries the next encoding.
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    # Raises a clear error if all reading attempts fail.
    raise ValueError("Could not read the dataset using supported encodings.")


# Audits metadata and spectral columns without modifying the source data.
def main() -> None:
    # Loads the dataset from the raw data directory.
    df = load_dataset(DATA_PATH)

    # Validates that the dataset contains four metadata columns.
    if len(df.columns) <= METADATA_COLUMN_COUNT:
        raise ValueError(
            "The dataset must contain four metadata columns "
            "followed by spectral columns."
        )

    # Selects the first four columns as sample metadata.
    metadata_columns = list(df.columns[:METADATA_COLUMN_COUNT])

    # Selects all remaining columns as spectral data.
    spectral_columns = list(df.columns[METADATA_COLUMN_COUNT:])

    # Separates metadata from spectral values for analysis.
    metadata = df.loc[:, metadata_columns].copy()

    # Converts spectral values to numeric values for quality checks.
    spectral = df.loc[:, spectral_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Identifies spectral cells that failed numeric conversion.
    non_numeric_counts = df.loc[:, spectral_columns].apply(
        lambda column: pd.to_numeric(column, errors="coerce").isna()
        & column.notna()
    ).sum()

    # Identifies metadata columns containing missing values.
    metadata_missing = metadata.isna().sum()

    # Identifies spectral columns containing missing values.
    spectral_missing = spectral.isna().sum()

    # Counts infinite values in each spectral column.
    spectral_infinite = pd.Series(
        np.isinf(spectral.to_numpy(dtype=float)).sum(axis=0),
        index=spectral.columns,
    )

    # Calculates descriptive statistics for numeric spectral columns.
    numeric_summary = spectral.describe().T

    # Adds missing-value counts to the numerical summary.
    numeric_summary["missing"] = spectral_missing

    # Adds infinite-value counts to the numerical summary.
    numeric_summary["infinite"] = spectral_infinite

    # Adds failed numeric conversion counts to the numerical summary.
    numeric_summary["non_numeric"] = non_numeric_counts

    # Builds the machine-readable audit report.
    report = {
        # Stores the source dataset path.
        "dataset": str(DATA_PATH),

        # Stores the number of sample rows.
        "rows": int(df.shape[0]),

        # Stores the total number of columns.
        "columns": int(df.shape[1]),

        # Stores the metadata column names.
        "metadata_columns": [str(column) for column in metadata_columns],

        # Stores the spectral column names.
        "spectral_columns": [str(column) for column in spectral_columns],

        # Stores the number of metadata columns.
        "metadata_column_count": int(len(metadata_columns)),

        # Stores the number of spectral columns.
        "spectral_column_count": int(len(spectral_columns)),

        # Stores the number of exact duplicate rows.
        "duplicate_rows": int(df.duplicated().sum()),

        # Stores the number of duplicate sample identifiers when available.
        "duplicate_sample_ids": (
            int(metadata.iloc[:, 0].duplicated().sum())
            if not metadata.empty
            else 0
        ),

        # Stores missing-value counts for metadata columns.
        "metadata_missing_values": {
            str(column): int(count)
            for column, count in metadata_missing.items()
        },

        # Stores spectral columns with missing values.
        "spectral_missing_values": {
            str(column): int(count)
            for column, count in spectral_missing.items()
            if count > 0
        },

        # Stores spectral columns with infinite values.
        "spectral_infinite_values": {
            str(column): int(count)
            for column, count in spectral_infinite.items()
            if count > 0
        },

        # Stores spectral columns with non-numeric values.
        "spectral_non_numeric_values": {
            str(column): int(count)
            for column, count in non_numeric_counts.items()
            if count > 0
        },

        # Stores the original dataset column names.
        "column_names": [str(column) for column in df.columns],

        # Stores the inferred data type of every original column.
        "dtypes": {
            str(column): str(dtype)
            for column, dtype in df.dtypes.items()
        },
    }

    # Defines the JSON audit report output path.
    report_path = REPORT_DIR / "dataset_audit.json"

    # Defines the numerical summary output path.
    summary_path = REPORT_DIR / "numeric_summary.csv"

    # Saves the audit report as readable UTF-8 JSON.
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Saves the numerical summary of the spectral columns.
    numeric_summary.to_csv(summary_path, encoding="utf-8-sig")

    # Prints the main dataset dimensions.
    print("=" * 70)
    print("SAFFRON DATASET AUDIT")
    print("=" * 70)
    print(f"Dataset: {DATA_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Metadata columns: {metadata_columns}")
    print(f"Spectral columns: {len(spectral_columns)}")
    print(f"Exact duplicate rows: {df.duplicated().sum()}")
    print(f"Duplicate sample IDs: {report['duplicate_sample_ids']}")

    # Displays missing metadata values.
    print("\n--- Missing metadata values ---")
    print(metadata_missing.to_string())

    # Displays the number of spectral cells that are missing.
    print("\n--- Spectral quality ---")
    print(f"Missing spectral cells: {int(spectral.isna().sum().sum())}")
    print(f"Infinite spectral cells: {int(np.isinf(spectral.to_numpy(dtype=float)).sum())}")
    print(f"Non-numeric spectral cells: {int(non_numeric_counts.sum())}")

    # Displays the first ten rows of metadata.
    print("\n--- First 10 sample records ---")
    print(metadata.head(10).to_string(index=False))

    # Displays the paths of generated reports.
    print("\n--- Reports saved ---")
    print(report_path)
    print(summary_path)


# Runs the audit when the module is executed directly.
if __name__ == "__main__":
    main()

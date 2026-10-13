
# Imports tools for paths and CSV data analysis.
from pathlib import Path

import pandas as pd


# Defines the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Defines the source dataset path.
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "total-color-data.csv"

# Defines the output directory for inspection reports.
REPORT_DIR = PROJECT_ROOT / "reports"

# Defines the expected number of metadata columns.
METADATA_COLUMN_COUNT = 4

# Creates the report directory when necessary.
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# Loads the CSV file while supporting common text encodings.
def load_dataset(path: Path) -> pd.DataFrame:
    # Raises an error when the dataset does not exist.
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # Lists the encodings to try when reading the CSV.
    encodings = ("utf-8-sig", "utf-8", "cp1252", "latin1")

    # Tries each supported encoding in order.
    for encoding in encodings:
        try:
            # Returns the loaded dataset without modifying its values.
            return pd.read_csv(path, encoding=encoding)

        # Continues when an encoding or CSV parsing error occurs.
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    # Raises an error when no supported encoding succeeds.
    raise ValueError("Unable to read the dataset with supported encodings.")


# Inspects metadata columns and generates sample reports.
def main() -> None:
    # Loads the source dataset.
    df = load_dataset(DATA_PATH)

    # Stops if the dataset has fewer than four columns.
    if len(df.columns) < METADATA_COLUMN_COUNT:
        raise ValueError(
            f"Expected at least {METADATA_COLUMN_COUNT} columns, "
            f"but found {len(df.columns)}."
        )

    # Selects the first four columns as the user-defined metadata.
    metadata_columns = list(df.columns[:METADATA_COLUMN_COUNT])

    # Copies metadata without changing the original dataset.
    metadata = df.loc[:, metadata_columns].copy()

    # Prints the dataset dimensions and metadata column names.
    print("\n--- Dataset overview ---")
    print(f"Rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Metadata columns: {metadata_columns}")
    print(f"Spectral columns: {len(df.columns) - METADATA_COLUMN_COUNT}")

    # Prints all sample metadata for manual verification.
    print("\n--- All sample metadata ---")
    print(metadata.to_string(index=False))

    # Creates a collection for metadata frequency records.
    label_counts = []

    # Counts values in each metadata column.
    for column in metadata_columns:
        # Includes missing values in the frequency counts.
        counts = metadata[column].value_counts(dropna=False)

        # Displays the distribution for the current column.
        print(f"\n--- Value counts: {column} ---")
        print(counts.to_string())

        # Converts each frequency into a report record.
        for value, count in counts.items():
            # Stores missing values explicitly instead of leaving them ambiguous.
            label_counts.append(
                {
                    "column": str(column),
                    "value": "<MISSING>" if pd.isna(value) else str(value),
                    "count": int(count),
                }
            )

    # Creates the metadata frequency report with stable column names.
    counts_df = pd.DataFrame(
        label_counts,
        columns=["column", "value", "count"],
    )

    # Defines the sample metadata report path.
    metadata_path = REPORT_DIR / "sample_metadata.csv"

    # Defines the metadata frequency report path.
    counts_path = REPORT_DIR / "label_counts.csv"

    # Saves the four metadata columns for inspection.
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    # Saves the frequency counts for all metadata columns.
    counts_df.to_csv(counts_path, index=False, encoding="utf-8-sig")

    # Reports the generated file paths.
    print("\n--- Reports saved ---")
    print(metadata_path)
    print(counts_path)


# Runs the inspection when this module is executed directly.
if __name__ == "__main__":
    main()

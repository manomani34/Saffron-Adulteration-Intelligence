from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Synthetic data quality check
# Non-destructive: never modifies input files or old reports
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "raw" / "SYNTHETIC_spectra.csv"
OUTPUT_DIR = ROOT / "reports" / "synthetic_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTPUT_DIR / "quality_summary.csv"
SAMPLE_CSV = OUTPUT_DIR / "quality_per_sample.csv"
AXIS_CSV = OUTPUT_DIR / "quality_axis_gaps.csv"
REPORT_MD = OUTPUT_DIR / "quality_report.md"
SUMMARY_JSON = OUTPUT_DIR / "quality_summary.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    print(f"Reading: {INPUT_FILE}")
    print("This may take a little time for a large CSV.")

    original_hash = sha256_file(INPUT_FILE)

    # Read the data without changing the source file.
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    if "SampleId" not in df.columns:
        raise ValueError(
            "Expected a 'SampleId' column. "
            "No analysis was performed because the file schema differs."
        )

    spectral_columns = [c for c in df.columns if c != "SampleId"]

    if not spectral_columns:
        raise ValueError("No spectral columns were found.")

    # Spectral column names should represent ppm values.
    ppm = pd.to_numeric(
        pd.Index(spectral_columns), errors="coerce"
    ).to_numpy(dtype=float)

    invalid_axis_columns = int((~np.isfinite(ppm)).sum())

    if invalid_axis_columns:
        raise ValueError(
            f"{invalid_axis_columns} spectral column names are not numeric ppm values."
        )

    if len(ppm) < 2:
        raise ValueError("At least two spectral columns are required.")

    print(f"Samples: {len(df)}")
    print(f"Spectral variables: {len(spectral_columns)}")

    # Convert spectral data to numeric values for QC only.
    X = df[spectral_columns].apply(pd.to_numeric, errors="coerce")
    values = X.to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(values)

    # A missing value and an infinite value are counted separately.
    missing_count = int(X.isna().sum().sum())
    nonfinite_count = int((~finite_mask).sum())

    axis_steps = np.diff(ppm)
    gap_indices = np.where(axis_steps > 0.0015)[0]

    axis_gaps = []
    for idx in gap_indices:
        axis_gaps.append({
            "left_ppm": float(ppm[idx]),
            "right_ppm": float(ppm[idx + 1]),
            "step_ppm": float(axis_steps[idx]),
            "approx_missing_bins_at_0.001_ppm":
                max(0, int(round(axis_steps[idx] / 0.001)) - 1),
        })

    gap_df = pd.DataFrame(
        axis_gaps,
        columns=[
            "left_ppm",
            "right_ppm",
            "step_ppm",
            "approx_missing_bins_at_0.001_ppm",
        ],
    )

    # Exact duplicate IDs and exact duplicate spectra are reported,
    # not removed.
    duplicate_id_rows = int(
        df["SampleId"].duplicated(keep=False).sum()
    )
    duplicate_spectrum_rows = int(
        X.duplicated(keep=False).sum()
    )

    constant_columns = int(
        (X.nunique(dropna=False) <= 1).sum()
    )

    row_missing = X.isna().sum(axis=1).to_numpy()
    row_nonfinite = (~finite_mask).sum(axis=1)

    # Per-sample summaries. They do not assign labels or delete samples.
    sample_min = np.full(len(df), np.nan)
    sample_max = np.full(len(df), np.nan)
    sample_mean = np.full(len(df), np.nan)
    sample_std = np.full(len(df), np.nan)
    negative_count = np.zeros(len(df), dtype=int)

    for i, row in enumerate(values):
        finite_row = row[np.isfinite(row)]
        if finite_row.size:
            sample_min[i] = float(finite_row.min())
            sample_max[i] = float(finite_row.max())
            sample_mean[i] = float(finite_row.mean())
            sample_std[i] = float(finite_row.std())
            negative_count[i] = int((finite_row < 0).sum())

    per_sample = pd.DataFrame({
        "SampleId": df["SampleId"].astype("string"),
        "missing_values": row_missing,
        "nonfinite_values": row_nonfinite,
        "negative_intensity_count": negative_count,
        "min_intensity": sample_min,
        "max_intensity": sample_max,
        "mean_intensity": sample_mean,
        "std_intensity": sample_std,
    })

    finite_values = values[np.isfinite(values)]

    if finite_values.size:
        intensity_stats = {
            "minimum": float(finite_values.min()),
            "p01": float(np.percentile(finite_values, 1)),
            "median": float(np.median(finite_values)),
            "p99": float(np.percentile(finite_values, 99)),
            "maximum": float(finite_values.max()),
            "negative_value_count": int((finite_values < 0).sum()),
            "zero_value_count": int((finite_values == 0).sum()),
        }
    else:
        intensity_stats = {
            "minimum": None,
            "p01": None,
            "median": None,
            "p99": None,
            "maximum": None,
            "negative_value_count": 0,
            "zero_value_count": 0,
        }

    summary = {
        "input_file": str(INPUT_FILE),
        "input_sha256_before": original_hash,
        "sample_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "spectral_variable_count": int(len(spectral_columns)),
        "unique_sample_id_count": int(df["SampleId"].nunique(dropna=False)),
        "missing_sample_id_count": int(df["SampleId"].isna().sum()),
        "duplicate_id_rows": duplicate_id_rows,
        "missing_spectral_value_count": missing_count,
        "nonfinite_spectral_value_count": nonfinite_count,
        "constant_spectral_variable_count": constant_columns,
        "exact_duplicate_spectrum_rows": duplicate_spectrum_rows,
        "ppm_min": float(ppm.min()),
        "ppm_max": float(ppm.max()),
        "ppm_median_step": float(np.median(axis_steps)),
        "ppm_min_step": float(axis_steps.min()),
        "ppm_max_step": float(axis_steps.max()),
        "ppm_strictly_increasing": bool(np.all(axis_steps > 0)),
        "ppm_gap_count_above_0_0015": int(len(gap_indices)),
        "intensity_statistics": intensity_stats,
        "interpretation_limit":
            "Unlabeled synthetic spectra: this QC does not assess dye-class accuracy.",
    }

    # Save only new output files in the dedicated folder.
    pd.DataFrame([{
        key: (json.dumps(value, ensure_ascii=False)
              if isinstance(value, (dict, list)) else value)
        for key, value in summary.items()
    }]).to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    per_sample.to_csv(SAMPLE_CSV, index=False, encoding="utf-8-sig")
    gap_df.to_csv(AXIS_CSV, index=False, encoding="utf-8-sig")

    SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    gap_lines = (
        "\n".join(
            f"- {g['left_ppm']:.6f} to {g['right_ppm']:.6f} ppm; "
            f"step = {g['step_ppm']:.6f} ppm"
            for g in axis_gaps
        )
        if axis_gaps else "- No gap above the configured threshold was found."
    )

    report = f"""# Synthetic spectra — quality control report

## Scope
This report describes structural and numerical checks only.
No sample was removed, no label was inferred, and no original file was changed.

## Dataset
- Samples: {len(df)}
- Spectral variables: {len(spectral_columns)}
- Ppm range: {ppm.min():.6f} to {ppm.max():.6f}
- Unique SampleId values: {df['SampleId'].nunique(dropna=False)}

## Data quality
- Rows with duplicate IDs: {duplicate_id_rows}
- Missing spectral values: {missing_count}
- Non-finite spectral values: {nonfinite_count}
- Constant spectral columns: {constant_columns}
- Rows in exact duplicate spectra: {duplicate_spectrum_rows}

## Ppm axis
- Median step: {np.median(axis_steps):.6f}
- Minimum step: {axis_steps.min():.6f}
- Maximum step: {axis_steps.max():.6f}
- Strictly increasing: {bool(np.all(axis_steps > 0))}
- Gaps above 0.0015 ppm: {len(axis_gaps)}

{gap_lines}

## Intensity statistics
```json
{json.dumps(intensity_stats, ensure_ascii=False, indent=2)}
```

## Interpretation limits
This file has no independent dye-type/concentration labels available to this script.
Therefore, this report does not calculate accuracy, F1, recall, or average precision.
Negative values are reported but are not automatically treated as errors.
Axis gaps are reported but are not interpolated or corrected.

## Input integrity
SHA-256 before analysis: `{original_hash}`
"""

    REPORT_MD.write_text(report, encoding="utf-8")

    final_hash = sha256_file(INPUT_FILE)
    if final_hash != original_hash:
        raise RuntimeError(
            "Input file hash changed during analysis; inspect the input file."
        )

    print("\nQC finished successfully.")
    print(f"Report folder: {OUTPUT_DIR}")
    print(f"- {SUMMARY_CSV.name}")
    print(f"- {SAMPLE_CSV.name}")
    print(f"- {AXIS_CSV.name}")
    print(f"- {SUMMARY_JSON.name}")
    print(f"- {REPORT_MD.name}")
    print("\nInput file integrity: unchanged")


if __name__ == "__main__":
    main()

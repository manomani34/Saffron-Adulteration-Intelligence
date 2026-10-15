from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

FILES = {
    "synthetic": ROOT / "data" / "raw" / "SYNTHETIC_spectra.csv",
    "existing": ROOT / "data" / "raw" / "total-color-data.csv",
}

METADATA = {
    "synthetic": {"SampleId"},
    "existing": {
        "Sample_ID",
        "Target_Label",
        "Adulterant_Type",
        "Dataset_Role",
    },
}


def inspect_file(name, path):
    if not path.exists():
        return {
            "file": str(path),
            "error": "File not found",
        }

    # Read headers first; intensity data is checked separately.
    headers = pd.read_csv(path, nrows=0).columns.tolist()
    spectral_columns = [
        c for c in headers if c not in METADATA[name]
    ]

    axis = pd.to_numeric(
        pd.Index(spectral_columns),
        errors="coerce",
    ).to_numpy(dtype=float)

    invalid_axis = int((~np.isfinite(axis)).sum())
    valid_axis = axis[np.isfinite(axis)]

    if len(valid_axis) == 0:
        return {
            "file": str(path),
            "total_columns": len(headers),
            "spectral_columns": len(spectral_columns),
            "invalid_axis_columns": invalid_axis,
            "error": "No valid numeric spectral axis",
        }

    differences = np.diff(valid_axis)

    ascending = bool(np.all(differences > 0))
    descending = bool(np.all(differences < 0))
    unique_count = int(np.unique(valid_axis).size)

    # Read data to inspect spectral intensities.
    frame = pd.read_csv(path, low_memory=False)
    intensity = frame[spectral_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    values = intensity.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]

    return {
        "file": str(path),
        "rows": int(len(frame)),
        "total_columns": int(len(headers)),
        "spectral_columns": int(len(spectral_columns)),
        "invalid_axis_columns": invalid_axis,
        "axis": {
            "minimum_ppm": float(valid_axis.min()),
            "maximum_ppm": float(valid_axis.max()),
            "strictly_ascending": ascending,
            "strictly_descending": descending,
            "unique_points": unique_count,
            "duplicate_points": int(len(valid_axis) - unique_count),
            "non_finite_steps": int((~np.isfinite(differences)).sum()),
            "minimum_step_ppm": (
                float(np.min(np.abs(differences)))
                if len(differences) else None
            ),
            "median_absolute_step_ppm": (
                float(np.median(np.abs(differences)))
                if len(differences) else None
            ),
            "maximum_step_ppm": (
                float(np.max(np.abs(differences)))
                if len(differences) else None
            ),
        },
        "intensity": {
            "numeric_finite_values": int(len(finite_values)),
            "non_finite_or_missing_values": int(
                values.size - len(finite_values)
            ),
            "negative_values": int(
                np.sum(finite_values < 0)
            ),
            "minimum": (
                float(np.min(finite_values))
                if len(finite_values) else None
            ),
            "maximum": (
                float(np.max(finite_values))
                if len(finite_values) else None
            ),
            "median": (
                float(np.median(finite_values))
                if len(finite_values) else None
            ),
        },
    }


report = {
    name: inspect_file(name, path)
    for name, path in FILES.items()
}

output = (
    ROOT / "reports" / "synthetic_evaluation"
    / "axis_intensity_diagnostics.json"
)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(report, ensure_ascii=False, indent=2))
print(f"\nSaved report: {output}")

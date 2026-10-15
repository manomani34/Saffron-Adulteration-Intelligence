from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[2]

files = {
    "synthetic": root / "data" / "raw" / "SYNTHETIC_spectra.csv",
    "existing": root / "data" / "raw" / "total-color-data.csv",
}

metadata_columns = {
    "synthetic": {"SampleId"},
    "existing": {
        "Sample_ID",
        "Target_Label",
        "Adulterant_Type",
        "Dataset_Role",
    },
}

results = {}

for name, path in files.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    # Read header only; do not load or modify spectral values.
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    meta = metadata_columns[name]
    spectral_columns = [c for c in columns if c not in meta]

    ppm = pd.to_numeric(
        pd.Index(spectral_columns), errors="coerce"
    ).to_numpy(dtype=float)

    invalid = ~np.isfinite(ppm)
    if invalid.any():
        raise ValueError(
            f"{name}: non-numeric spectral column names: "
            f"{np.asarray(spectral_columns)[invalid][:10].tolist()}"
        )

    steps = np.diff(ppm)
    gap_indices = np.where(steps > 0.0015)[0]

    gaps = [
        {
            "left_ppm": float(ppm[i]),
            "right_ppm": float(ppm[i + 1]),
            "step_ppm": float(steps[i]),
        }
        for i in gap_indices
    ]

    results[name] = {
        "file": path.name,
        "spectral_variables": len(ppm),
        "ppm_min": float(ppm.min()),
        "ppm_max": float(ppm.max()),
        "median_step": float(np.median(steps)),
        "minimum_step": float(steps.min()),
        "maximum_step": float(steps.max()),
        "strictly_increasing": bool(np.all(steps > 0)),
        "gaps_over_0_0015_ppm": gaps,
    }

syn = results["synthetic"]
real = results["existing"]

results["comparison"] = {
    "common_ppm_min": max(syn["ppm_min"], real["ppm_min"]),
    "common_ppm_max": min(syn["ppm_max"], real["ppm_max"]),
    "same_minimum_ppm": bool(
        np.isclose(syn["ppm_min"], real["ppm_min"], atol=1e-6)
    ),
    "same_maximum_ppm": bool(
        np.isclose(syn["ppm_max"], real["ppm_max"], atol=1e-6)
    ),
    "same_number_of_spectral_variables": bool(
        syn["spectral_variables"] == real["spectral_variables"]
    ),
}

output = root / "reports" / "synthetic_evaluation" / "axis_comparison.json"
output.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(results, ensure_ascii=False, indent=2))
print(f"\nSaved report: {output}")

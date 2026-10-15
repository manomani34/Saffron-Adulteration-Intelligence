from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[2]

files = {
    "synthetic": root / "data" / "raw" / "SYNTHETIC_spectra.csv",
    "existing": root / "data" / "raw" / "total-color-data.csv",
}

metadata = {
    "synthetic": {"SampleId"},
    "existing": {
        "Sample_ID",
        "Target_Label",
        "Adulterant_Type",
        "Dataset_Role",
    },
}

axes = {}

for name, path in files.items():
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    spectral_columns = [
        c for c in columns if c not in metadata[name]
    ]
    axes[name] = pd.to_numeric(
        pd.Index(spectral_columns), errors="raise"
    ).to_numpy(dtype=float)

syn = axes["synthetic"]
real = axes["existing"]

low = max(syn.min(), real.min())
high = min(syn.max(), real.max())

syn_common = syn[(syn >= low) & (syn <= high)]
real_common = real[(real >= low) & (real <= high)]

# For each point on the existing axis, find its nearest synthetic point.
nearest_indices = np.searchsorted(syn, real_common)
nearest_indices = np.clip(nearest_indices, 0, len(syn) - 1)

left_indices = np.maximum(nearest_indices - 1, 0)
use_left = (
    np.abs(syn[left_indices] - real_common)
    < np.abs(syn[nearest_indices] - real_common)
)
nearest_indices[use_left] = left_indices[use_left]

distances = np.abs(syn[nearest_indices] - real_common)

result = {
    "common_range_ppm": [float(low), float(high)],
    "synthetic_points_in_common_range": int(len(syn_common)),
    "existing_points_in_common_range": int(len(real_common)),
    "existing_points_with_synthetic_point_within_0_0001_ppm": int(
        np.sum(distances <= 0.0001)
    ),
    "existing_points_with_synthetic_point_within_0_0002_ppm": int(
        np.sum(distances <= 0.0002)
    ),
    "nearest_axis_distance_ppm": {
        "median": float(np.median(distances)),
        "p95": float(np.percentile(distances, 95)),
        "maximum": float(np.max(distances)),
    },
    "synthetic_gaps_in_common_range": [
        {
            "left_ppm": float(syn[i]),
            "right_ppm": float(syn[i + 1]),
            "step_ppm": float(syn[i + 1] - syn[i]),
        }
        for i in np.where(
            (np.diff(syn) > 0.0015)
            & (syn[:-1] <= high)
            & (syn[1:] >= low)
        )[0]
    ],
}

output = (
    root / "reports" / "synthetic_evaluation"
    / "common_axis_check.json"
)
output.write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(result, ensure_ascii=False, indent=2))
print(f"\nSaved report: {output}")

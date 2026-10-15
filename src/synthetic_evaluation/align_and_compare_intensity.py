from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[2]
raw = root / "data" / "raw"
out = root / "reports" / "synthetic_evaluation"
out.mkdir(parents=True, exist_ok=True)

synthetic_path = raw / "SYNTHETIC_spectra.csv"
real_path = raw / "total-color-data.csv"

# Read headers and identify spectral columns
syn_header = pd.read_csv(synthetic_path, nrows=0).columns.tolist()
real_header = pd.read_csv(real_path, nrows=0).columns.tolist()

syn_meta = ["SampleId"]
real_meta = [
    c for c in [
        "Sample_ID", "Target_Label",
        "Adulterant_Type", "Dataset_Role"
    ] if c in real_header
]

syn_cols = [c for c in syn_header if c not in syn_meta]
real_cols = [c for c in real_header if c not in real_meta]

syn_axis = np.asarray(syn_cols, dtype=float)
real_axis = np.asarray(real_cols, dtype=float)

if not np.all(np.diff(syn_axis) > 0):
    raise ValueError("Synthetic spectral axis is not strictly ascending.")

if not np.all(np.diff(real_axis) > 0):
    raise ValueError("Real spectral axis is not strictly ascending.")

if real_axis.min() < syn_axis.min() or real_axis.max() > syn_axis.max():
    raise ValueError("Real axis is not fully inside synthetic axis bounds.")

# Interpolate only onto the real dataset's axis.
# This creates a derived file; original data remain untouched.
syn_df = pd.read_csv(synthetic_path, low_memory=False)
real_df = pd.read_csv(real_path, low_memory=False)

syn_x = syn_df[syn_cols].to_numpy(dtype=np.float64)
real_x = real_df[real_cols].to_numpy(dtype=np.float64)

aligned = np.empty((len(syn_df), len(real_axis)), dtype=np.float64)
for i, row in enumerate(syn_x):
    aligned[i] = np.interp(real_axis, syn_axis, row)

aligned_cols = [f"{x:.5f}" for x in real_axis]
aligned_df = pd.DataFrame(aligned, columns=aligned_cols)
aligned_df.insert(0, "SampleId", syn_df["SampleId"].astype(str))

aligned_path = out / "synthetic_spectra_aligned_to_real_axis.csv"
aligned_df.to_csv(aligned_path, index=False, encoding="utf-8-sig")

def summarize(matrix):
    values = matrix.ravel()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"finite_values": 0}

    return {
        "spectra": int(matrix.shape[0]),
        "points_per_spectrum": int(matrix.shape[1]),
        "finite_values": int(len(values)),
        "missing_or_nonfinite": int(matrix.size - len(values)),
        "negative_fraction": float(np.mean(values < 0)),
        "minimum": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }

results = {
    "axis": {
        "points": int(len(real_axis)),
        "minimum_ppm": float(real_axis.min()),
        "maximum_ppm": float(real_axis.max()),
        "interpolation": "linear",
    },
    "synthetic_aligned": summarize(aligned),
    "real_by_role_and_label": {},
    "per_spectrum_median": {},
}

# Summarize real spectra separately by dataset role and target label.
if "Dataset_Role" in real_df.columns:
    for role, indices in real_df.groupby(
        "Dataset_Role", dropna=False
    ).groups.items():
        key = f"role::{role}"
        results["real_by_role_and_label"][key] = summarize(
            real_x[list(indices)]
        )

if "Target_Label" in real_df.columns:
    for label, indices in real_df.groupby(
        "Target_Label", dropna=False
    ).groups.items():
        key = f"label::{label}"
        results["real_by_role_and_label"][key] = summarize(
            real_x[list(indices)]
        )

# Per-spectrum summaries reveal whether a few spectra dominate intensity.
for name, matrix in [
    ("synthetic_aligned", aligned),
    ("real_all", real_x),
]:
    results["per_spectrum_median"][name] = {
        "median_of_spectrum_medians": float(np.median(np.median(matrix, axis=1))),
        "p05_of_spectrum_medians": float(np.quantile(np.median(matrix, axis=1), 0.05)),
        "p95_of_spectrum_medians": float(np.quantile(np.median(matrix, axis=1), 0.95)),
        "median_of_spectrum_maxima": float(np.median(np.max(matrix, axis=1))),
    }

report_path = out / "aligned_intensity_comparison.json"
report_path.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print(json.dumps(results, ensure_ascii=False, indent=2))
print(f"\nAligned spectra saved: {aligned_path}")
print(f"Comparison report saved: {report_path}")

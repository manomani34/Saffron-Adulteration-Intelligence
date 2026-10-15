from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[2]
syn_path = root / "data" / "raw" / "SYNTHETIC_spectra.csv"
real_path = root / "data" / "raw" / "total-color-data.csv"

# 1. Inspect the synthetic spectral axis
syn_header = pd.read_csv(syn_path, nrows=0).columns.tolist()
syn_spectral_cols = [c for c in syn_header if c != "SampleId"]
syn_axis = pd.to_numeric(
    pd.Index(syn_spectral_cols), errors="raise"
).to_numpy(dtype=float)

diff = np.diff(syn_axis)
gap_indices = np.where(diff > 0.0015)[0]

gaps = []
for i in gap_indices:
    gaps.append({
        "left_ppm": float(syn_axis[i]),
        "right_ppm": float(syn_axis[i + 1]),
        "gap_ppm": float(diff[i]),
        "left_column": str(syn_spectral_cols[i]),
        "right_column": str(syn_spectral_cols[i + 1]),
    })

# 2. Inspect synthetic sample IDs
syn_ids = pd.read_csv(syn_path, usecols=["SampleId"])["SampleId"]

# 3. Inspect metadata of existing samples
real_header = pd.read_csv(real_path, nrows=0).columns.tolist()
candidate_meta = [
    "Sample_ID", "Target_Label", "Adulterant_Type", "Dataset_Role"
]
real_meta = [c for c in candidate_meta if c in real_header]

real_df = pd.read_csv(
    real_path, usecols=real_meta, low_memory=False
)

metadata_summary = {}
for col in real_meta:
    values = real_df[col].value_counts(dropna=False)
    metadata_summary[col] = {
        str(key): int(value) for key, value in values.items()
    }

result = {
    "synthetic": {
        "rows": int(len(syn_ids)),
        "sample_id_unique": int(syn_ids.nunique(dropna=True)),
        "sample_id_missing": int(syn_ids.isna().sum()),
        "sample_id_examples": syn_ids.astype(str).head(10).tolist(),
        "axis_gap_count_over_0_0015_ppm": len(gaps),
        "axis_gaps": gaps,
    },
    "existing": {
        "rows": int(len(real_df)),
        "metadata_columns_found": real_meta,
        "metadata_unique_values": metadata_summary,
    },
}

print(json.dumps(result, ensure_ascii=False, indent=2))

out_dir = root / "reports" / "synthetic_evaluation"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "gap_and_labels_inspection.json"
out_path.write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"\nSaved report: {out_path}")

from pathlib import Path
import numpy as np
import pandas as pd

root = Path.cwd()
folder = root / "reports" / "synthetic_evaluation"
syn_file = folder / "synthetic_spectra_aligned_to_real_axis.csv"
real_file = root / "data" / "raw" / "total-color-data.csv"

def read_csv(path):
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")

syn = read_csv(syn_file)
real = read_csv(real_file)

meta_names = {
    "SampleId", "Sample_ID", "SampleID", "ID", "Sample",
    "Target_Label", "Adulterant_Type", "Dataset_Role"
}

def get_axis_columns(df):
    result = []
    for col in df.columns:
        if col in meta_names:
            continue
        try:
            ppm = float(str(col).strip())
            if np.isfinite(ppm):
                result.append((col, ppm))
        except (ValueError, TypeError):
            continue
    return result

syn_axis = get_axis_columns(syn)
real_axis = get_axis_columns(real)
real_cols = {round(ppm, 6): col for col, ppm in real_axis}
shared = [(col, ppm) for col, ppm in syn_axis
          if round(ppm, 6) in real_cols]

if not shared:
    raise RuntimeError(
        "No numeric ppm column names found in both datasets. "
        "Inspect the first row/column headers."
    )

records = []
for _, row in syn.iterrows():
    values = pd.to_numeric(
        row[[c for c, _ in shared]], errors="coerce"
    ).to_numpy(dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        continue
    positions = np.flatnonzero(valid)
    local_idx = positions[np.argmax(values[valid])]
    col, ppm = shared[local_idx]
    records.append({
        "Sample_ID": str(row.get("SampleId", row.get("Sample_ID", row.name))),
        "Peak_PPM": float(ppm),
        "Maximum_Intensity": float(values[local_idx]),
        "P95_Intensity": float(np.nanpercentile(values, 95)),
        "Median_Intensity": float(np.nanmedian(values)),
        "Negative_Fraction": float(np.mean(values[valid] < 0))
    })

peaks = pd.DataFrame(records).sort_values("Maximum_Intensity", ascending=False)
top_file = folder / "synthetic_peak_locations.csv"
peaks.to_csv(top_file, index=False, encoding="utf-8-sig")

# Group peak locations into 0.05 ppm bins to identify concentration regions.
peaks["PPM_Bin_Start"] = np.floor(peaks["Peak_PPM"] / 0.05) * 0.05
bins = (
    peaks.groupby("PPM_Bin_Start")
    .agg(
        Sample_Count=("Sample_ID", "count"),
        Median_Peak_Intensity=("Maximum_Intensity", "median"),
        Maximum_Peak_Intensity=("Maximum_Intensity", "max")
    )
    .reset_index()
    .sort_values(["Sample_Count", "Maximum_Peak_Intensity"], ascending=False)
)
bins_file = folder / "synthetic_peak_location_bins.csv"
bins.to_csv(bins_file, index=False, encoding="utf-8-sig")

print(f"Shared ppm points: {len(shared)}")
print(f"Synthetic spectra analyzed: {len(peaks)}")
print("\nTop 20 synthetic peak locations:")
print(peaks.head(20).to_string(index=False))
print("\nMost frequent 0.05 ppm bins for maximum peaks:")
print(bins.head(15).to_string(index=False))
print(f"\nSaved: {top_file}")
print(f"Saved: {bins_file}")

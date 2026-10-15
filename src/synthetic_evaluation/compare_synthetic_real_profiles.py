from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path.cwd()
out = root / "reports" / "synthetic_evaluation"
out.mkdir(parents=True, exist_ok=True)

syn_path = out / "synthetic_spectra_aligned_to_real_axis.csv"
real_path = root / "data" / "raw" / "total-color-data.csv"

syn = pd.read_csv(syn_path, low_memory=False, encoding="utf-8-sig")
real = pd.read_csv(real_path, low_memory=False, encoding="utf-8-sig")

metadata = {
    "SampleId", "Sample_ID", "SampleID", "ID", "Sample",
    "Target_Label", "Adulterant_Type", "Dataset_Role"
}

def numeric_axis(df):
    result = {}
    for col in df.columns:
        if col in metadata:
            continue
        try:
            ppm = float(str(col).strip())
            if np.isfinite(ppm):
                result[round(ppm, 6)] = col
        except (ValueError, TypeError):
            pass
    return result

syn_axis = numeric_axis(syn)
real_axis = numeric_axis(real)
shared_ppm = sorted(set(syn_axis) & set(real_axis))

if len(shared_ppm) < 100:
    raise RuntimeError(
        f"Only {len(shared_ppm)} shared numeric ppm columns found. "
        "Check the CSV headers before proceeding."
    )

ppm = np.array(shared_ppm, dtype=float)
syn_cols = [syn_axis[x] for x in shared_ppm]
real_cols = [real_axis[x] for x in shared_ppm]

S = syn[syn_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)

if "Dataset_Role" not in real.columns:
    raise RuntimeError("Dataset_Role column not found in real dataset.")

train_mask = real["Dataset_Role"].astype(str).str.strip().str.lower().eq("train")
R = real.loc[train_mask, real_cols].apply(
    pd.to_numeric, errors="coerce"
).to_numpy(float)

if len(S) == 0 or len(R) == 0:
    raise RuntimeError("No synthetic spectra or real Train spectra found.")

def profile(matrix, prefix):
    return {
        f"{prefix}_median": np.nanmedian(matrix, axis=0),
        f"{prefix}_p95": np.nanpercentile(matrix, 95, axis=0),
        f"{prefix}_p99": np.nanpercentile(matrix, 99, axis=0),
        f"{prefix}_mean": np.nanmean(matrix, axis=0),
    }

profiles = {}
profiles.update(profile(S, "synthetic"))
profiles.update(profile(R, "real_train"))

report = pd.DataFrame({"ppm": ppm, **profiles})
report["abs_median_difference"] = np.abs(
    report["synthetic_median"] - report["real_train_median"]
)
report["p95_ratio"] = (
    report["synthetic_p95"].abs()
    / report["real_train_p95"].abs().clip(lower=1e-12)
)
report.to_csv(
    out / "synthetic_vs_real_train_ppm_profile.csv",
    index=False, encoding="utf-8-sig"
)

# Plot median intensity across the spectrum
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(ppm, profiles["real_train_median"], label="Real Train median")
ax.plot(ppm, profiles["synthetic_median"], label="Synthetic median", alpha=0.85)
ax.set_xlabel("Chemical shift (ppm)")
ax.set_ylabel("Intensity")
ax.set_title("Median intensity profile: synthetic vs real Train")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(out / "median_profile_synthetic_vs_real.png", dpi=160)
plt.close(fig)

# Plot 95th percentile across the spectrum
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(ppm, profiles["real_train_p95"], label="Real Train P95")
ax.plot(ppm, profiles["synthetic_p95"], label="Synthetic P95", alpha=0.85)
ax.set_xlabel("Chemical shift (ppm)")
ax.set_ylabel("Intensity (95th percentile)")
ax.set_title("P95 intensity profile: synthetic vs real Train")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(out / "p95_profile_synthetic_vs_real.png", dpi=160)
plt.close(fig)

print(f"Shared ppm points: {len(ppm)}")
print(f"Synthetic spectra: {len(S)}")
print(f"Real Train spectra: {len(R)}")
print("\nLargest ppm-wise differences in median intensity:")
print(
    report.nlargest(15, "abs_median_difference")[
        ["ppm", "synthetic_median", "real_train_median",
         "abs_median_difference", "synthetic_p95", "real_train_p95"]
    ].to_string(index=False)
)
print("\nLargest synthetic/real P95 ratios:")
print(
    report.nlargest(15, "p95_ratio")[
        ["ppm", "synthetic_p95", "real_train_p95", "p95_ratio"]
    ].to_string(index=False)
)
print("\nSaved report and plots in:", out)

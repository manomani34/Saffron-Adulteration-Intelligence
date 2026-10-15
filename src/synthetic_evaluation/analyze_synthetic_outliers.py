from pathlib import Path
import pandas as pd
import numpy as np

root = Path.cwd()
folder = root / "reports" / "synthetic_evaluation"
source = folder / "per_spectrum_intensity_metrics.csv"

df = pd.read_csv(source)
synthetic = df[df["Group"] == "Synthetic"].copy()
train = df[df["Group"] == "Train"].copy()

metrics = ["median", "p95", "maximum", "negative_fraction"]
results = []

for metric in metrics:
    reference = pd.to_numeric(train[metric], errors="coerce").dropna()
    values = pd.to_numeric(synthetic[metric], errors="coerce")
    threshold = reference.quantile(0.95)
    valid = values.notna()

    results.append({
        "Metric": metric,
        "Train_P95_threshold": threshold,
        "Synthetic_samples": int(valid.sum()),
        "Synthetic_above_Train_P95": int((values[valid] > threshold).sum()),
        "Synthetic_above_Train_P95_percent":
            float((values[valid] > threshold).mean() * 100)
    })

result = pd.DataFrame(results)
out = folder / "synthetic_outlier_comparison.csv"
result.to_csv(out, index=False, encoding="utf-8-sig")

print("\nComparison against the 95th percentile of TRAIN samples:")
print(result.to_string(index=False))

print("\nTop 15 synthetic spectra by maximum intensity:")
cols = ["Sample_ID", "median", "p95", "p99", "maximum",
        "minimum", "negative_fraction"]
top = synthetic.sort_values("maximum", ascending=False)[cols].head(15)
print(top.to_string(index=False))

top.to_csv(folder / "synthetic_top_peak_samples.csv",
           index=False, encoding="utf-8-sig")

print(f"\nSaved: {out}")
print(f"Saved: {folder / 'synthetic_top_peak_samples.csv'}")

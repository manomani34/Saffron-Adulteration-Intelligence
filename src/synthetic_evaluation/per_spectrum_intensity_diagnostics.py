from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "synthetic_evaluation"

SYN_FILE = OUT / "synthetic_spectra_aligned_to_real_axis.csv"
REAL_FILE = RAW / "total-color-data.csv"
OUT.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, low_memory=False)

def detect_spectral_columns(df, metadata):
    cols = []
    for c in df.columns:
        if c in metadata:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
        else:
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().mean() >= 0.95:
                cols.append(c)
    return cols

def summarize_spectra(df, group, id_col, spectral_cols):
    rows = []
    for _, row in df.iterrows():
        x = pd.to_numeric(row[spectral_cols], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        item = {
            "Group": group,
            "Sample_ID": str(row.get(id_col, row.name)),
            "finite_points": int(x.size),
            "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)),
            "maximum": float(np.max(x)),
            "minimum": float(np.min(x)),
            "negative_fraction": float(np.mean(x < 0)),
        }
        for c in ("Target_Label", "Adulterant_Type", "Dataset_Role"):
            if c in df.columns:
                v = row[c]
                item[c] = None if pd.isna(v) else str(v)
        rows.append(item)
    return rows

def main():
    if not SYN_FILE.exists():
        raise FileNotFoundError(f"Aligned synthetic file not found: {SYN_FILE}")
    if not REAL_FILE.exists():
        raise FileNotFoundError(f"Real spectra file not found: {REAL_FILE}")

    syn = read_csv(SYN_FILE)
    real = read_csv(REAL_FILE)

    syn_id = next((c for c in ("SampleId", "Sample_ID", "SampleID", "ID")
                   if c in syn.columns), syn.columns[0])
    syn_meta = {"SampleId", "Sample_ID", "SampleID", "ID",
                "Target_Label", "Adulterant_Type", "Dataset_Role"}
    syn_cols = detect_spectral_columns(syn, syn_meta)

    real_id = next((c for c in ("Sample_ID", "SampleId", "SampleID", "ID", "Sample")
                    if c in real.columns), real.columns[0])
    real_meta = {"Sample_ID", "SampleId", "SampleID", "ID", "Sample",
                 "Target_Label", "Adulterant_Type", "Dataset_Role"}
    real_cols = detect_spectral_columns(real, real_meta)

    if not syn_cols or not real_cols:
        raise ValueError("Could not detect numeric spectral columns.")

    records = summarize_spectra(syn, "Synthetic", syn_id, syn_cols)

    if "Dataset_Role" in real.columns:
        roles = real["Dataset_Role"].astype(str).str.strip().str.casefold()
        for role in ("Train", "Test", "Reference"):
            part = real.loc[roles == role.casefold()]
            if not part.empty:
                records.extend(summarize_spectra(part, role, real_id, real_cols))
    else:
        records.extend(summarize_spectra(real, "Real", real_id, real_cols))

    metrics = pd.DataFrame(records)
    metrics_file = OUT / "per_spectrum_intensity_metrics.csv"
    metrics.to_csv(metrics_file, index=False, encoding="utf-8-sig")

    metric_names = [
        "median", "p95", "p99", "maximum", "minimum",
        "negative_fraction", "finite_points"
    ]
    summary_rows = []
    for group, part in metrics.groupby("Group", dropna=False):
        for metric in metric_names:
            v = pd.to_numeric(part[metric], errors="coerce").dropna()
            summary_rows.append({
                "Group": group,
                "Metric": metric,
                "Count": int(v.size),
                "Mean": float(v.mean()) if len(v) else None,
                "Median": float(v.median()) if len(v) else None,
                "P05": float(v.quantile(.05)) if len(v) else None,
                "P95": float(v.quantile(.95)) if len(v) else None,
                "Min": float(v.min()) if len(v) else None,
                "Max": float(v.max()) if len(v) else None,
            })

    summary = pd.DataFrame(summary_rows)
    summary_file = OUT / "per_spectrum_intensity_summary.csv"
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")

    groups = [g for g in ("Synthetic", "Train", "Test", "Reference", "Real")
              if g in set(metrics["Group"])]
    plot_specs = [
        ("median", "Per-spectrum median intensity", "distribution_median.png"),
        ("p95", "Per-spectrum 95th percentile intensity", "distribution_p95.png"),
        ("maximum", "Per-spectrum maximum intensity", "distribution_maximum.png"),
        ("negative_fraction", "Fraction of negative intensity points",
         "distribution_negativefraction.png"),
    ]
    plots = []
    for metric, title, filename in plot_specs:
        values, labels = [], []
        for group in groups:
            v = pd.to_numeric(
                metrics.loc[metrics["Group"] == group, metric], errors="coerce"
            ).dropna().to_numpy()
            if v.size:
                values.append(v)
                labels.append(f"{group}\n(n={len(v)})")
        if not values:
            continue
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.boxplot(values, tick_labels=labels, showfliers=True)
        ax.set_title(title)
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        dest = OUT / filename
        fig.savefig(dest, dpi=160)
        plt.close(fig)
        plots.append(str(dest))

    report = {
        "synthetic_file": str(SYN_FILE),
        "real_file": str(REAL_FILE),
        "synthetic_samples": int(len(syn)),
        "real_samples": int(len(real)),
        "synthetic_spectral_points": int(len(syn_cols)),
        "real_spectral_points": int(len(real_cols)),
        "samples_by_group": {
            str(k): int(v) for k, v in metrics["Group"].value_counts().items()
        },
        "metrics_csv": str(metrics_file),
        "summary_csv": str(summary_file),
        "plots": plots,
        "notes": [
            "Descriptive intensity comparison only; not model performance.",
            "No normalization, baseline correction, clipping, or other transformation applied.",
            "Negative intensity values retained."
        ]
    }
    report_file = OUT / "per_spectrum_intensity_diagnostics.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    print("\nAnalysis completed.")
    print(f"Metrics: {metrics_file}")
    print(f"Summary: {summary_file}")
    print(f"Report:  {report_file}")
    for p in plots:
        print(f"Plot:    {p}")

    print("\nKey per-spectrum summary:")
    print(summary[summary["Metric"].isin(
        ["median", "p95", "maximum", "negative_fraction"]
    )].to_string(index=False))

if __name__ == "__main__":
    main()


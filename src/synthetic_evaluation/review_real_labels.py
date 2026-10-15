from pathlib import Path
import json
import pandas as pd

root = Path(__file__).resolve().parents[2]
path = root / "data" / "raw" / "total-color-data.csv"
out_dir = root / "reports" / "synthetic_evaluation"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(path, low_memory=False)

required = ["Sample_ID", "Target_Label", "Adulterant_Type", "Dataset_Role"]
available = [c for c in required if c in df.columns]

for col in available:
    df[col] = df[col].fillna("<MISSING>").astype(str).str.strip()
    df.loc[df[col] == "", col] = "<EMPTY>"

print("\n=== TARGET LABEL x DATASET ROLE ===")
if {"Target_Label", "Dataset_Role"}.issubset(df.columns):
    print(pd.crosstab(
        df["Target_Label"],
        df["Dataset_Role"],
        margins=True,
        dropna=False
    ).to_string())

print("\n=== TARGET LABEL x ADULTERANT TYPE ===")
if {"Target_Label", "Adulterant_Type"}.issubset(df.columns):
    print(pd.crosstab(
        df["Target_Label"],
        df["Adulterant_Type"],
        dropna=False
    ).to_string())

print("\n=== ALL SAMPLE METADATA ===")
print(df[available].sort_values(
    by=[c for c in ["Dataset_Role", "Target_Label", "Sample_ID"] if c in available]
).to_string(index=False))

# Save metadata only; do not duplicate or modify raw spectra.
metadata_path = out_dir / "real_sample_metadata_review.csv"
df[available].to_csv(metadata_path, index=False, encoding="utf-8-sig")

print(f"\nSaved metadata review: {metadata_path}")

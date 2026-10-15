# Synthetic spectra — quality control report

## Scope
This report describes structural and numerical checks only.
No sample was removed, no label was inferred, and no original file was changed.

## Dataset
- Samples: 513
- Spectral variables: 9617
- Ppm range: 0.500998 to 10.499900
- Unique SampleId values: 513

## Data quality
- Rows with duplicate IDs: 0
- Missing spectral values: 0
- Non-finite spectral values: 0
- Constant spectral columns: 0
- Rows in exact duplicate spectra: 0

## Ppm axis
- Median step: 0.001000
- Minimum step: 0.000970
- Maximum step: 0.383990
- Strictly increasing: True
- Gaps above 0.0015 ppm: 1

- 4.666970 to 5.050960 ppm; step = 0.383990 ppm

## Intensity statistics
```json
{
  "minimum": -28.778328,
  "p01": -0.93647224,
  "median": 0.70477799,
  "p99": 90.70623559999973,
  "maximum": 1937.201,
  "negative_value_count": 468647,
  "zero_value_count": 0
}
```

## Interpretation limits
This file has no independent dye-type/concentration labels available to this script.
Therefore, this report does not calculate accuracy, F1, recall, or average precision.
Negative values are reported but are not automatically treated as errors.
Axis gaps are reported but are not interpolated or corrected.

## Input integrity
SHA-256 before analysis: `3be9ec8270d5b037d6b8835abc7be61929ce679609b31050b20f944497ce1d68`

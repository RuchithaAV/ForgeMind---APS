# Data

This folder is intentionally empty in version control (see `.gitignore`).

## Source

APS Failure at Scania Trucks — UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks

## Files expected here (not committed)

- `aps_failure_training_set.csv` — 60,000 rows, ~59,000 negative / ~1,000 positive
- `aps_failure_test_set.csv` — 16,000 rows (official test set — held out until Phase 21)

## Known parsing gotchas (handled explicitly in Phase 2, not silently)

1. Both raw CSV files begin with a ~21-line metadata/comment header before
   the actual column row — a naive `pd.read_csv()` will misparse this.
2. Missing values are encoded as the literal string `"na"`, not an empty
   cell — must be passed to `na_values=["na"]` at load time or they will
   be read as a string category instead of NaN.

## Target variable

`class`: `pos` (APS-related failure) / `neg` (failure unrelated to APS
or no failure). This is the only column with a known, documented meaning —
all 170 predictors (`aa_000`, `ab_000`, ...) are anonymized and their
physical meaning is not disclosed by UCI. We will not speculate about it.

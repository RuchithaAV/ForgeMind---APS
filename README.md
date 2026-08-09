# ForgeMind APS

**Cost-Sensitive AI Decision Intelligence for Heavy-Vehicle Maintenance**

A portfolio machine learning project investigating how multiple ML
approaches can be benchmarked and combined with cost-sensitive
decision-making to identify rare APS (Air Pressure System) failures in
Scania heavy trucks, using the [APS Failure at Scania Trucks
dataset](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks)
(UCI Machine Learning Repository).

> **Status:** Phase 0–1 complete, dataset verified. Not a production
> system — a research/portfolio project on a public benchmark dataset.

---

## The problem

Given that a Scania truck required maintenance, was the root cause its
**APS (Air Pressure System)**, or something unrelated?

- `class = pos` → failure was APS-related
- `class = neg` → failure was unrelated to APS (or no failure)

This is a binary classification problem under **extreme class imbalance**
and an **explicit, asymmetric business cost**:

| Error type | Cost | Meaning |
|---|---|---|
| False Positive | 10 | Healthy truck flagged — unnecessary inspection |
| False Negative | 500 | Real APS failure missed — the expensive mistake |

A missed failure is **50x** more costly than a false alarm. This
asymmetry is treated as central to model selection, not an afterthought
— **accuracy is never used as the primary metric** in this project.

```
Expected Cost = 10 × FP + 500 × FN
```

## Dataset (verified against the actual files, not assumed)

| Property | Value |
|---|---|
| Training rows | 60,000 (59,000 `neg` / 1,000 `pos`) |
| Test rows | 16,000 (15,625 `neg` / 375 `pos`) |
| Total columns | 171 (`class` + 170 anonymized predictors) |
| Histogram-style feature groups | 7 groups, 10 bins each |
| Missing value encoding | literal string `"na"` |
| Missingness | 169/170 features have missing values; 8 columns >50% missing |

**The 170 predictor features are anonymized.** This project does not
invent physical meanings for them (e.g. claiming a column represents
brake pressure or temperature) — only documented facts from the official
UCI description are treated as ground truth about feature semantics.

Raw data is not committed to this repository — see
[`data/README.md`](data/README.md) for how to obtain it and two parsing
gotchas to handle at load time.

## Project philosophy

This is **not** a "train Random Forest → report accuracy → wrap in
Streamlit" project. It's structured as a rigorous, sequential
experimentation project covering:

- **Model benchmarking** across families — linear models, KNN/Naive
  Bayes, SVM (linear + RBF), tree ensembles, gradient boosting (incl.
  XGBoost/LightGBM where computationally justified), MLP
- **Class imbalance handling** — class weights, resampling, balanced
  ensembles, compared experimentally rather than assumed
- **Dimensionality reduction** (PCA) and **unsupervised learning**
  (K-Means, DBSCAN) as complementary analysis layers, not primary
  predictors
- **Anomaly detection** (Isolation Forest) as a supplementary signal
- **Cost-sensitive threshold optimization** against the expected-cost
  function above, rather than a default 0.5 cutoff
- **Probability calibration** (Platt scaling / isotonic regression)
- **Explainability** via SHAP, using exact anonymized feature names —
  never invented semantics
- **Maintenance prioritization** under limited inspection capacity
  (e.g. "only 100 trucks can be inspected — which ones?")
- **A what-if decision simulator** for exploring cost/threshold/capacity
  tradeoffs
- **A Streamlit dashboard** tying the full pipeline into a
  decision-support tool

## Repository structure

```
ForgeMind-APS/
├── README.md            # this file
├── LICENSE               # MIT
├── requirements.txt      # core deps only — heavier packages (xgboost,
│                          # lightgbm, imbalanced-learn, shap, streamlit)
│                          # are added only when a phase justifies them
├── .gitignore
├── data/
│   └── README.md          # dataset provenance + parsing notes
├── notebooks/              # created phase-by-phase as the project progresses
├── src/                     # reusable pipeline code
├── models/                  # trained model artifacts (gitignored)
├── reports/
│   ├── figures/
│   └── results/
└── app/                      # Streamlit decision-support dashboard
```

## Setup

```bash
git clone https://github.com/<your-username>/ForgeMind-APS.git
cd ForgeMind-APS
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then follow [`data/README.md`](data/README.md) to obtain the dataset and
place it in `data/`.

## Roadmap

- [x] Phase 0 — Project setup and environment
- [x] Phase 1 — Industrial problem and dataset understanding
- [ ] Phase 2 — Data loading and data-quality analysis
- [ ] Phase 3 — Preprocessing pipeline
- [ ] Phase 4 — Baseline models
- [ ] Phase 5 — SVM experiments
- [ ] Phase 6 — Tree models
- [ ] Phase 7 — Boosting models
- [ ] Phase 8 — Neural network
- [ ] Phase 9 — PCA experiments
- [ ] Phase 10 — Unsupervised learning
- [ ] Phase 11 — Anomaly detection
- [ ] Phase 12 — Cost-sensitive learning
- [ ] Phase 13 — Threshold optimization
- [ ] Phase 14 — Model calibration
- [ ] Phase 15 — Explainability
- [ ] Phase 16 — Final evaluation
- [ ] Maintenance prioritization system
- [ ] What-if decision simulator
- [ ] Streamlit application

## What this project does not claim

- Not a Scania production deployment
- No invented meanings for anonymized features
- No fabricated results or business impact
- Not presented as a novel predictive-maintenance algorithm — the
  dataset is an established benchmark; the contribution here is the
  integrated cost-sensitive decision-support system built on top of it

## License

MIT — see [`LICENSE`](LICENSE).

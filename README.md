# ForgeMind APS

**Cost-Sensitive AI Decision Intelligence for Heavy-Vehicle Maintenance**

A portfolio machine learning project investigating how multiple ML approaches can be benchmarked and combined with cost-sensitive decision-making to identify rare APS (Air Pressure System) failures in Scania heavy trucks, using the [APS Failure at Scania Trucks dataset](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks) (UCI Machine Learning Repository).

> **Status:** Phase 0–4 complete (Data Quality, Leak-Free Preprocessing Pipeline, Baseline Models & Cost Optimization Benchmarked). Not a production system — a research/portfolio project on a public benchmark dataset.

---

## The Problem

Given that a Scania truck required maintenance, was the root cause its **APS (Air Pressure System)**, or something unrelated?

- `class = pos` → failure was APS-related
- `class = neg` → failure was unrelated to APS (or no failure)

This is a binary classification problem under **extreme class imbalance** (59:1 ratio) and an **explicit, asymmetric business cost matrix**:

| Error Type | Cost | Meaning |
|---|---|---|
| **False Positive (FP)** | **$10** | Healthy truck flagged — unnecessary inspection |
| **False Negative (FN)** | **$500** | Real APS failure missed — breakdown / expensive damage |

A missed failure is **50x** more costly than a false alarm. This asymmetry is central to model selection and decision thresholding — **accuracy is never used as the primary metric** in this project.

$$\text{Expected Cost} = 10 \times \text{FP} + 500 \times \text{FN}$$

---

## Dataset Properties (Verified Against Source)

| Property | Value |
|---|---|
| **Training Set** | 60,000 rows (59,000 `neg` / 1,000 `pos`) |
| **Test Set** | 16,000 rows (15,625 `neg` / 375 `pos`) — *held out until Phase 16* |
| **Total Columns** | 171 (`class` target + 170 anonymized predictors) |
| **Histogram Feature Groups** | 7 groups, 10 bins each |
| **Missing Values** | Encoded as `"na"`; 169/170 features have missing values (8 columns >50% missing) |
| **Constant Feature** | `cd_000` is completely invariant (0.0 across all rows) |
| **Zero-IQR Features** | 46 features have an IQR of 0 with extreme positive tails up to $6.35 \times 10^9$ |

> **Note on Feature Names:** The 170 predictor features are anonymized. This project strictly relies on documented facts from the UCI specification and does not invent speculative physical meanings for anonymized sensor channels.

---

## Phase 4 Baseline & Cost Optimization Benchmark

All models are evaluated on the 60,000-row training set using **5-fold Stratified Cross-Validation**:

| Model / Strategy | Threshold ($t$) | ROC-AUC | PR-AUC | FP | FN | Total Cost | Cost / Truck |
|---|---|---|---|---|---|---|---|
| **All-Positive Heuristic** | 0.50 | 0.5000 | 0.0167 | 59,000 | 0 | **$590,000** | $9.83 |
| **All-Negative Heuristic** | 0.50 | 0.5000 | 0.0167 | 0 | 1,000 | **$500,000** | $8.33 |
| **Logistic Regression (Default, Log OFF)** | 0.50 | 0.5833 | 0.2751 | 13,776 | 536 | **$405,760** | $6.76 |
| **Logistic Regression (Tuned, Log OFF)** | 0.518 | 0.5833 | 0.2751 | 4,809 | 583 | **$339,590** | $5.66 |
| **Balanced LogReg + Signed-Log (Dups dropped)** | 0.468 | 0.9837 | 0.7798 | 2,109 | 58 | **$50,090** | $0.83 |
| **Unweighted LogReg + Signed-Log (OOF-tuned)** | 0.024 | **0.9853** | **0.8000** | 1,968 | 54 | **$46,680** | **$0.78** |

### Key Diagnostic Takeaways:
1. **Signed-Log Transformation ($\text{sign}(x) \cdot \log(1 + |x|)$):** Resolves solver non-convergence caused by zero-IQR heavy-tailed features, boosting ROC-AUC from **0.5833 $\to$ 0.9853** and cutting cost from **$339k $\to$ $46k**.
2. **Duplicate Indicator Pruning:** `DuplicateDropper` removes 112 identical missingness masks, reducing the feature space from 337 to 225 without sacrificing predictive ranking.
3. **Class Weighting vs. Threshold Tuning:** Class weighting shifts the unweighted decision boundary from $t \approx 0.024$ towards $t \approx 0.468$, demonstrating that both yield equivalent ranking power once optimization converges.

---

## Repository Structure

```
ForgeMind-APS/
├── README.md                          # Project overview, benchmark results, roadmap
├── LICENSE                            # MIT License
├── requirements.txt                   # Core project dependencies
├── .gitignore
├── dataset/                           # Raw CSV dataset files (or obtain via data/README.md)
├── data/
│   └── README.md                      # Dataset provenance and parsing gotchas
├── src/                               # Modular, leak-free pipeline codebase
│   ├── __init__.py
│   ├── data_loader.py                 # load_aps_data() with header skipping & NA parsing
│   ├── preprocessing.py              # ColumnDropper, DuplicateDropper, signed_log1p, build_linear_preprocessor, make_cv_splitter
│   └── evaluation.py                 # calculate_aps_cost, calculate_classification_metrics, generate_oof_predictions, best_cost_threshold, evaluate_at_threshold
├── notebooks/                         # Sequential phase research notebooks
│   ├── 02_data_quality_report.ipynb   # Phase 2: Missingness, distributions & invariant column analysis
│   ├── 03_preprocessing_pipeline.ipynb# Phase 3: Preprocessing pipeline, scaling & duplicate indicator tests
│   └── 04_baseline_models.ipynb       # Phase 4: Baseline benchmarks, scaling diagnostics & threshold tuning
├── reports/
│   ├── figures/                       # Generated diagnostic and ROC/PR plots
│   └── results/                       # CSV export artifacts for benchmark metrics
│       ├── phase4_baseline_results.csv
│       ├── phase4_logistic_fold_results.csv
│       ├── phase4_preprocessing_comparison.csv
│       ├── phase4_threshold_results.csv
│       └── phase4_variant_results.csv
├── models/                            # Trained model artifacts (gitignored)
└── app/                               # Future Streamlit decision-support dashboard
```

---

## Setup & Execution

### 1. Environment Installation

```bash
git clone https://github.com/RuchithaAV/ForgeMind---APS.git ForgeMind-APS
cd ForgeMind-APS
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset Preparation

Follow [`data/README.md`](data/README.md) to place `aps_failure_training_set.csv` and `aps_failure_test_set.csv` into the `dataset/` directory.

### 3. Running Notebooks

Launch Jupyter to run through the verified pipeline phases:

```bash
jupyter lab
```

---

## Project Roadmap

- [x] **Phase 0** — Project setup and environment configuration
- [x] **Phase 1** — Industrial problem and cost matrix definition ($10 FP / $500 FN)
- [x] **Phase 2** — Robust data loading and data-quality inspection
- [x] **Phase 3** — Leak-free preprocessing pipeline & signed-log heavy-tail compaction
- [x] **Phase 4** — Baseline models, scaling diagnostics & out-of-fold cost threshold tuning
- [ ] **Phase 5** — Linear & Non-linear SVM experiments
- [ ] **Phase 6** — Decision Tree and Random Forest ensembles
- [ ] **Phase 7** — Gradient Boosting (XGBoost / LightGBM / CatBoost)
- [ ] **Phase 8** — Multilayer Perceptron (MLP) Neural Network
- [ ] **Phase 9** — Principal Component Analysis (PCA) & dimensionality reduction
- [ ] **Phase 10** — Unsupervised clustering & latent pattern discovery
- [ ] **Phase 11** — Anomaly detection (Isolation Forest / Local Outlier Factor)
- [ ] **Phase 12** — Cost-sensitive learning & class-weight optimization
- [ ] **Phase 13** — Global and fold-wise decision threshold optimization
- [ ] **Phase 14** — Probability calibration (Platt Scaling & Isotonic Regression)
- [ ] **Phase 15** — Model explainability via SHAP (using exact anonymized features)
- [ ] **Phase 16** — Final evaluation on held-out test set
- [ ] **Phase 17** — Maintenance prioritization system under capacity constraints
- [ ] **Phase 18** — What-if industrial decision simulator
- [ ] **Phase 19** — Interactive Streamlit decision-support dashboard

---

## What This Project Does Not Claim

- Not a Scania production deployment
- No invented meanings for anonymized sensor channels
- No fabricated business metrics or synthetic shortcuts
- Not presented as a novel predictive-maintenance algorithm — the dataset is an established public benchmark; the contribution is an integrated, leak-free, cost-sensitive decision-intelligence system.

---

## License

MIT — see [`LICENSE`](LICENSE).

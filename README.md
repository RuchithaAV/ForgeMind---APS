# ForgeMind APS

**Cost-Sensitive AI Decision Intelligence for Heavy-Vehicle Maintenance**

A portfolio machine learning project investigating how multiple ML approaches can be benchmarked and combined with cost-sensitive decision-making to identify rare APS (Air Pressure System) failures in Scania heavy trucks, using the [APS Failure at Scania Trucks dataset](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks) (UCI Machine Learning Repository).

> **Status:** Phase 0–5 complete (Data Quality, Leak-Free Preprocessing Pipeline, Baseline Models, Linear & Kernel SVM Benchmarks & Cost Optimization). Not a production system — a research/portfolio project on a public benchmark dataset.

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

## Current Model Benchmark & Cost Leaderboard

All models are evaluated on the 60,000-row training set using **5-fold Stratified Cross-Validation** with leak-free preprocessing (signed-log compaction + median imputation + standard scaling + duplicate indicator pruning). Out-of-fold decision boundaries are evaluated using **fold-wise honest cost** (threshold chosen on outer folds, scored on the holdout fold):

| Model Family | Configuration / Variant | ROC-AUC | PR-AUC | Honest FP | Honest FN | Total Honest Cost | Cost / Truck | Cost Saved vs. Baseline |
|---|---|---|---|---|---|---|---|---|
| **Heuristic** | All-Positive Strategy | 0.5000 | 0.0167 | 59,000 | 0 | **$590,000** | $9.83 | -18.0% |
| **Heuristic** | Do Nothing (All-Negative) | 0.5000 | 0.0167 | 0 | 1,000 | **$500,000** | $8.33 | 0.0% (Ref) |
| **Logistic Regression** | Default Solver (Log OFF, $t=0.50$) | 0.5833 | 0.2751 | 13,776 | 536 | **$405,760** | $6.76 | +18.8% |
| **Logistic Regression** | Tuned Threshold (Log OFF) | 0.5833 | 0.2751 | 4,809 | 583 | **$339,590** | $5.66 | +32.1% |
| **Logistic Regression** | Balanced + Signed-Log | 0.9837 | 0.7798 | 2,143 | 61 | **$51,930** | $0.87 | +89.6% |
| **Logistic Regression** | Unweighted + Signed-Log | 0.9853 | 0.8000 | 1,975 | 58 | **$48,750** | $0.81 | +90.2% |
| **Kernel SVM** | RBF Nystroem ($\gamma=0.03/D, m=1000, C=10$) | 0.9846 | 0.8061 | 2,005 | 55 | **$47,550** | $0.79 | +90.5% |
| **Kernel SVM** | Linear + RBF Nystroem ($\gamma=0.03/D, C=10$) | 0.9848 | 0.8343 | 1,744 | 60 | **$47,440** | $0.79 | +90.5% |
| **Linear SVM (Best)** | **LinearSVC ($C=1.0$, unweighted, signed-log)** | **0.9855** | **0.8371** | **2,110** | **48** | **$45,100** | **$0.75** | **+91.0%** |

---

## Key Experimental Findings (Phases 2 – 5)

1. **Signed-Log Transformation ($\text{sign}(x) \cdot \log(1 + |x|)$):**
   - Resolves solver non-convergence caused by 46 zero-IQR heavy-tailed features.
   - Boosts baseline ROC-AUC from **0.5833 $\to$ 0.9855** and slashes total maintenance cost from **$339,590 $\to$ $45,100** (an **86.7% reduction** in cost over raw scaling).
2. **Duplicate Indicator Pruning:**
   - `DuplicateDropper` eliminates 112 redundant missingness masks, condensing the feature space from 337 to 225 predictors with zero degradation in ranking quality or stability.
3. **Linear Margin Maximization Outperforms Probabilistic Logistic Regression:**
   - The unweighted `LinearSVC` ($C=1.0$) establishes a new cost minimum of **$45,100** ($0.75 / truck), outperforming unweighted Logistic Regression ($48,750) by **7.5%** and balanced Logistic Regression ($51,930) by **13.1%**.
   - Linear SVM won in **5 of 5 folds** against balanced Logistic Regression and **4 of 5 folds** against unweighted Logistic Regression.
4. **Kernel Approximations & Boundary Geometry:**
   - Non-linear kernel approximations using Nystroem RBF features (300 to 1,000 components) yielded **$47,550**, while concatenated Linear + RBF embeddings achieved **$47,440**.
   - Because pure Linear SVM beats non-linear kernel expansions with lower computational overhead (57s vs 388s) and higher PR-AUC (0.8371 vs 0.8061), it demonstrates that after signed-log compaction, the APS classification boundary in 225-dimensional space is predominantly linearly separable.
5. **Exact OOF Threshold Sweep:**
   - Upgraded `best_cost_threshold` from a grid-based search to an exact $O(N \log N)$ cumulative sum sweep over unbounded decision scores, ensuring precise cost-optimal cutoff without threshold tie splitting.

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
│   ├── 04_baseline_models.ipynb       # Phase 4: Baseline benchmarks, scaling diagnostics & threshold tuning
│   └── 05_svm_experiments.ipynb       # Phase 5: LinearSVC, RBF Nystroem kernel approximation & fold-wise cost evaluation
├── reports/
│   ├── figures/                       # Generated diagnostic and ROC/PR plots
│   └── results/                       # CSV export artifacts for benchmark metrics
│       ├── phase4_baseline_results.csv
│       ├── phase4_logistic_fold_results.csv
│       ├── phase4_preprocessing_comparison.csv
│       ├── phase4_threshold_results.csv
│       ├── phase4_variant_results.csv
│       ├── phase5_fold_costs.csv
│       ├── phase5_linear_plus_rbf_results.csv
│       ├── phase5_linear_svm_results.csv
│       ├── phase5_rbf_nystroem_results.csv
│       └── phase5_summary.csv
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
- [x] **Phase 5** — Linear & Non-linear SVM experiments (LinearSVC, Nystroem RBF kernel approximation & fold-wise honest cost evaluation)
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

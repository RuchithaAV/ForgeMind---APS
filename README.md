# ForgeMind---APS

ForgeMind APS is a portfolio machine learning project investigating how multiple ML approaches can be benchmarked and combined with cost-sensitive decision-making to identify rare APS (Air Pressure System) failures in Scania heavy trucks.

Built on the APS Failure at Scania Trucks dataset (UCI Machine Learning Repository), the project is centered on extreme class imbalance (~1.7% positive class) and explicit business cost asymmetry:

- False Negative (missed failure): **50x** higher cost
- False Positive (unnecessary inspection): baseline unit cost

Because of this, accuracy is not treated as the primary metric; model selection is driven by expected business cost.

## What this project covers

- Systematic benchmarking across model families (linear models, SVM, tree ensembles, gradient boosting, MLP)
- Extreme class imbalance handling via class weighting, resampling, and balanced ensembles (compared experimentally)
- Dimensionality reduction (PCA) and unsupervised learning (K-Means, DBSCAN) as complementary analysis layers
- Anomaly detection (Isolation Forest) alongside supervised predictions
- Cost-sensitive threshold optimization against a defined expected-cost function
- Probability calibration (Platt scaling / isotonic regression)
- Explainability via SHAP using only documented feature semantics (the 170 predictors are anonymized, with no invented physical meaning)
- Maintenance prioritization under limited inspection capacity
- An interactive what-if decision simulator
- A Streamlit dashboard integrating the full pipeline

## What this project is not

- A Scania production deployment
- A claim of a novel predictive-maintenance algorithm
- A Kaggle-style accuracy leaderboard

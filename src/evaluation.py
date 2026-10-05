import time
import warnings


import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_aps_cost(
    y_true,
    y_pred,
    false_positive_cost=10,
    false_negative_cost=500,
):
    """
    Calculate confusion-matrix values and total APS maintenance cost.
    Cost = 10 * FP + 500 * FN
    """
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    total_cost = (
        false_positive_cost * fp
        + false_negative_cost * fn
    )

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "aps_cost": int(total_cost),
    }


def calculate_classification_metrics(
    model_name,
    y_true,
    y_pred,
    y_probability,
    threshold=0.50,
):
    """
    Calculate classification metrics and APS cost.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_prob_arr = np.asarray(y_probability)

    cost_result = calculate_aps_cost(
        y_true_arr,
        y_pred_arr,
    )

    metrics = {
        "model": model_name,
        "threshold": threshold,
        "precision": float(
            precision_score(
                y_true_arr,
                y_pred_arr,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true_arr,
                y_pred_arr,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true_arr,
                y_pred_arr,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true_arr,
                y_prob_arr,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true_arr,
                y_prob_arr,
            )
        ),
        **cost_result,
    }

    metrics["cost_per_truck"] = (
        metrics["aps_cost"] / len(y_true_arr)
    )

    return metrics


def get_scores(model, X):
    """
    Return positive-class probabilities if model supports predict_proba,
    otherwise decision function scores.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if hasattr(proba, "ndim") and proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return np.asarray(proba).ravel()
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def generate_oof_predictions(
    model,
    X,
    y,
    cv,
    threshold=0.50,
):
    """
    Generate out-of-fold probabilities and predictions.
    """
    oof_probabilities = np.zeros(
        len(y),
        dtype=float,
    )

    oof_predictions = np.zeros(
        len(y),
        dtype=int,
    )

    fold_results = []

    for fold, (train_index, valid_index) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        print(f"Training fold {fold}...")

        X_fold_train = X.iloc[train_index] if hasattr(X, "iloc") else X[train_index]
        X_fold_valid = X.iloc[valid_index] if hasattr(X, "iloc") else X[valid_index]

        y_fold_train = y.iloc[train_index] if hasattr(y, "iloc") else y[train_index]
        y_fold_valid = y.iloc[valid_index] if hasattr(y, "iloc") else y[valid_index]

        fold_model = clone(model)

        start_time = time.perf_counter()

        fold_model.fit(
            X_fold_train,
            y_fold_train,
        )

        training_seconds = (
            time.perf_counter()
            - start_time
        )

        fold_probabilities = get_scores(
            fold_model,
            X_fold_valid,
        )

        fold_predictions = (
            fold_probabilities >= threshold
        ).astype(int)

        oof_probabilities[valid_index] = fold_probabilities
        oof_predictions[valid_index] = fold_predictions

        y_fold_valid_arr = np.asarray(y_fold_valid)
        fold_metrics = calculate_classification_metrics(
            model_name=f"Fold {fold}",
            y_true=y_fold_valid_arr,
            y_pred=fold_predictions,
            y_probability=fold_probabilities,
            threshold=threshold,
        )

        fold_metrics.update({
            "fold": fold,
            "validation_rows": len(valid_index),
            "positive_rate": float(np.mean(y_fold_valid_arr)),
            "training_seconds": training_seconds,
        })

        fold_results.append(fold_metrics)

        print(
            f"Fold {fold} complete — "
            f"cost={fold_metrics['aps_cost']}, "
            f"time={training_seconds:.2f}s"
        )

    fold_results = pd.DataFrame(fold_results)

    return (
        oof_predictions,
        oof_probabilities,
        fold_results,
    )


def evaluate_at_threshold(
    model_name,
    y_true,
    y_probability,
    threshold=0.50,
):
    """
    Evaluate binary classification metrics and APS cost at a specific probability threshold.
    """
    y_pred = (np.asarray(y_probability) >= threshold).astype(int)
    return calculate_classification_metrics(
        model_name=model_name,
        y_true=y_true,
        y_pred=y_pred,
        y_probability=y_probability,
        threshold=threshold,
    )


def best_cost_threshold(y_true, y_probability, thresholds=None):
    """
    Cost-minimising threshold (predict positive if score >= threshold).

    Default: exact sweep over every distinct score (works for probabilities
    AND unbounded scores such as SVM decision values), including predicting all negatives.
    If `thresholds` is given, evaluates on the provided grid.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    scores = np.asarray(y_probability, dtype=float)
    total_pos = int(np.sum(y_true_arr))

    if thresholds is None:
        order = np.argsort(-scores, kind="stable")
        s_sorted = scores[order]
        tp = np.cumsum(y_true_arr[order])
        fp = np.arange(1, len(scores) + 1) - tp
        fn = total_pos - tp
        cost = 10 * fp + 500 * fn

        # Only cut where the score changes, so ties are never split
        valid_cut = np.r_[s_sorted[1:] != s_sorted[:-1], True]
        cost = np.where(valid_cut, cost, np.inf)

        cost_all_negative = 500 * total_pos
        min_idx = int(np.argmin(cost))
        if cost_all_negative < cost[min_idx]:
            best_thresh = float(s_sorted[0] + 1.0)
        else:
            best_thresh = float(s_sorted[min_idx])
    else:
        costs = [
            calculate_aps_cost(y_true_arr, (scores >= t).astype(int))["aps_cost"]
            for t in thresholds
        ]
        best_thresh = float(thresholds[int(np.argmin(costs))])

    best_metrics = calculate_aps_cost(y_true_arr, (scores >= best_thresh).astype(int))
    return {
        **best_metrics,
        "threshold": best_thresh,
        "cost_per_truck": best_metrics["aps_cost"] / len(y_true_arr),
    }


def oof_scores(model, X, y, cv):
    """
    Out-of-fold decision scores: each row is scored by a model that never saw it.
    """
    scores = np.zeros(len(y))
    n_warn = 0
    t0 = time.time()
    for tr, va in cv.split(X, y):
        m = clone(model)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            X_tr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
            y_tr = y.iloc[tr] if hasattr(y, "iloc") else y[tr]
            X_va = X.iloc[va] if hasattr(X, "iloc") else X[va]
            m.fit(X_tr, y_tr)
        n_warn += any(issubclass(x.category, ConvergenceWarning) for x in w)
        scores[va] = get_scores(m, X_va)
    return scores, n_warn, time.time() - t0


def honest_cost(y, scores, cv):
    """
    Per fold: threshold chosen on the other folds' scores, scored on this fold.
    Returns (total_cost, FP, FN).
    """
    y_arr = np.asarray(y)
    scores_arr = np.asarray(scores)
    total, FP, FN = 0, 0, 0
    for tr, va in cv.split(np.zeros(len(y_arr)), y_arr):
        t = best_cost_threshold(y_arr[tr], scores_arr[tr])["threshold"]
        pred = scores_arr[va] >= t
        fp = int((pred & (y_arr[va] == 0)).sum())
        fn = int((~pred & (y_arr[va] == 1)).sum())
        FP += fp
        FN += fn
        total += 10 * fp + 500 * fn
    return total, FP, FN


def robust_cv_predictions(model, X, y, cv, warm_start=False, max_iter=None):
    """
    Generate out-of-fold scores and honest CV evaluation metrics.
    """
    n = len(y)
    scores = np.zeros(n)
    t0 = time.time()

    for tr, va in cv.split(X, y):
        m = clone(model)
        if max_iter is not None:
            if hasattr(m, "max_iter"):
                m.set_params(max_iter=max_iter)
            elif hasattr(m, "steps"):
                last_step_name = m.steps[-1][0]
                last_estimator = m.steps[-1][1]
                if hasattr(last_estimator, "max_iter"):
                    m.set_params(**{f"{last_step_name}__max_iter": max_iter})

        X_tr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
        y_tr = y.iloc[tr] if hasattr(y, "iloc") else y[tr]
        X_va = X.iloc[va] if hasattr(X, "iloc") else X[va]
        m.fit(X_tr, y_tr)
        scores[va] = get_scores(m, X_va)

    y_arr = np.asarray(y)
    cost, FP, FN = honest_cost(y_arr, scores, cv)

    best_thresh_info = best_cost_threshold(y_arr, scores)
    best_thresh = best_thresh_info["threshold"]

    total_pos = int((y_arr == 1).sum())
    total_neg = int((y_arr == 0).sum())

    tp = total_pos - FN
    tn = total_neg - FP

    precision = float(tp / max(1, tp + FP))
    recall = float(tp / max(1, total_pos))
    f1 = float((2 * precision * recall) / max(1e-9, precision + recall))

    metrics = {
        "threshold": best_thresh,
        "aps_cost": int(cost),
        "tn": int(tn),
        "fp": int(FP),
        "fn": int(FN),
        "tp": int(tp),
        "cost_per_truck": cost / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": float(roc_auc_score(y_arr, scores)),
        "pr_auc": float(average_precision_score(y_arr, scores)),
    }

    return scores, metrics, time.time() - t0

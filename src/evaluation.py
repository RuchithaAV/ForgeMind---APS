import time

import numpy as np
import pandas as pd

from sklearn.base import clone

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def calculate_aps_cost(
    y_true,
    y_pred,
    false_positive_cost=10,
    false_negative_cost=500
):
    """
    Calculate confusion-matrix values and
    total APS maintenance cost.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
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
        "aps_cost": int(total_cost)
    }


def calculate_classification_metrics(
    model_name,
    y_true,
    y_pred,
    y_probability,
    threshold=0.50
):
    """
    Calculate classification metrics and APS cost.
    """

    cost_result = calculate_aps_cost(
        y_true,
        y_pred
    )

    metrics = {
        "model": model_name,
        "threshold": threshold,

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_true,
            y_probability
        ),

        "pr_auc": average_precision_score(
            y_true,
            y_probability
        ),

        **cost_result
    }

    metrics["cost_per_truck"] = (
        metrics["aps_cost"] / len(y_true)
    )

    return metrics


def generate_oof_predictions(
    model,
    X,
    y,
    cv,
    threshold=0.50
):
    """
    Generate out-of-fold probabilities and predictions.
    """

    oof_probabilities = np.zeros(
        len(y),
        dtype=float
    )

    oof_predictions = np.zeros(
        len(y),
        dtype=int
    )

    fold_results = []

    for fold, (train_index, valid_index) in enumerate(
        cv.split(X, y),
        start=1
    ):
        print(f"Training fold {fold}...")

        X_fold_train = X.iloc[train_index]
        X_fold_valid = X.iloc[valid_index]

        y_fold_train = y.iloc[train_index]
        y_fold_valid = y.iloc[valid_index]

        fold_model = clone(model)

        start_time = time.perf_counter()

        fold_model.fit(
            X_fold_train,
            y_fold_train
        )

        training_seconds = (
            time.perf_counter()
            - start_time
        )

        fold_probabilities = (
            fold_model.predict_proba(
                X_fold_valid
            )[:, 1]
        )

        fold_predictions = (
            fold_probabilities >= threshold
        ).astype(int)

        oof_probabilities[valid_index] = (
            fold_probabilities
        )

        oof_predictions[valid_index] = (
            fold_predictions
        )

        fold_metrics = (
            calculate_classification_metrics(
                model_name=f"Fold {fold}",
                y_true=y_fold_valid,
                y_pred=fold_predictions,
                y_probability=fold_probabilities,
                threshold=threshold
            )
        )

        fold_metrics.update({
            "fold": fold,
            "validation_rows": len(valid_index),
            "positive_rate":
                y_fold_valid.mean(),
            "training_seconds":
                training_seconds
        })

        fold_results.append(
            fold_metrics
        )

        print(
            f"Fold {fold} complete — "
            f"cost={fold_metrics['aps_cost']}, "
            f"time={training_seconds:.2f}s"
        )

    fold_results = pd.DataFrame(
        fold_results
    )

    return (
        oof_predictions,
        oof_probabilities,
        fold_results
    )
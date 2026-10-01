import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    A simple transformer that removes selected columns.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(
            columns=self.columns,
            errors="ignore"
        )


class DuplicateDropper(BaseEstimator, TransformerMixin):
    """
    Transformer that removes duplicate columns across feature/indicator matrices.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.keep_indices_ = np.where(~df.T.duplicated().values)[0]
        return self

    def transform(self, X):
        if self.keep_indices_ is None:
            raise RuntimeError("DuplicateDropper must be fitted before transform.")
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.keep_indices_]
        return X[:, self.keep_indices_]


def signed_log1p(X):
    """
    Signed logarithm transformation: sign(x) * log1p(|x|).
    Compacts heavy-tailed distributions without losing negative signs or zero baseline.
    """
    return np.sign(X) * np.log1p(np.abs(X))


def build_linear_preprocessor(use_log=True, drop_duplicates=True):
    """
    Build preprocessing pipeline for scale-sensitive models.

    Steps:
    1. Drop constant column ('cd_000')
    2. Median imputation + missing value indicators
    3. (Optional) Duplicate feature/indicator removal
    4. (Optional) Signed-log transformation to compact heavy tails
    5. Robust scaling (median / IQR)

    Suitable for:
    - Logistic Regression
    - SVM (Phase 5)
    - KNN
    - MLP neural network
    """
    steps = [
        (
            "drop_constant",
            ColumnDropper(columns=["cd_000"])
        ),
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True
            )
        ),
    ]

    if drop_duplicates:
        steps.append(("drop_duplicates", DuplicateDropper()))

    if use_log:
        steps.append((
            "signed_log",
            FunctionTransformer(signed_log1p, validate=False)
        ))

    steps.append((
        "scaler",
        RobustScaler()
    ))

    return Pipeline(steps)


def make_cv_splitter():
    """
    Create reproducible stratified 5-fold cross validation splitter.
    """
    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
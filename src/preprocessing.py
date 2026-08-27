from sklearn.preprocessing import RobustScaler
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer


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
            errors = "ignore"
        )

def build_linear_preprocessor():
    """
    Build preprocessing for scale-sensitive models.

    Suitable for:
    - Logistic Regression
    - SVM
    - KNN
    - MLP neural network
    """
    preprocessor = Pipeline([
        (
            "drop_constant",
            ColumnDropper(columns=["cd_000"])
        ),
        (
            "scaler",
            RobustScaler()
        ),
        (
            "imputer", 
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True
            )
        )
    ])
      
    return preprocessor

def make_cv_splitter():
    """
    Create reproductible stratified five fold cross validation
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    return cv
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin

class APSFeaturePreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_missing_indicators:bool = True):
        self.add_missing_indicators = add_missing_indicators
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
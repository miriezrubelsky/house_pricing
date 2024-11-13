import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
from pathlib import Path
from collections import Counter
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

class MeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        
    
    def fit(self, X, y=None):
        self.mean_dict = {}
        for col in self.variables:
            if X[col].dtype in [np.float64, np.int64]:  # Check if the column is numeric
                self.mean_dict[col] = X[col].mean()
        return self

    
    
    def transform(self, X):
      X = X.copy()  # Avoid modifying the original DataFrame
      for col in self.variables:
        if col in X.columns:
            if X[col].dtype in [np.float64, np.int64]:  # For numeric columns
                mean_value = self.mean_dict.get(col, 0)
                X[col] = X[col].fillna(mean_value)

      return X


class MedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        
    
    def fit(self, X, y=None):
        self.median_dict = {}
        for col in self.variables:
            if X[col].dtype in [np.float64, np.int64]:  # Check if the column is numeric
                self.median_dict[col] = X[col].median()
        return self

    
    
    def transform(self, X):
      X = X.copy()  # Avoid modifying the original DataFrame
      for col in self.variables:
         if col in X.columns:
            if X[col].dtype in [np.float64, np.int64]:  # For numeric columns
                median_value = self.median_dict.get(col, 0)
                X[col] = X[col].fillna(median_value)

      return X

    
    



class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables_to_drop = variables_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()  # Make a copy to avoid modifying the original data
        X = X.drop(columns=self.variables_to_drop, errors='ignore')  # Drop columns safely
        return X



    
class NoneImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()  # Make a copy to avoid modifying the original data
        for col in self.variables:
            if col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = X[col].fillna("None")
                elif pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(0)
                 
        return X

    
      

class MostCommonImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables  

    def fit(self, X, y=None):
        
        self.most_common_values_ = {}  
        for col in self.variables:
            self.most_common_values_[col] = self._most_common_term(X[col].dropna())
        
        return self

    def transform(self, X):
        X = X.copy()  # Avoid modifying the original DataFrame
        for col in self.variables:
            most_common_value = self.most_common_values_.get(col)
            X[col].fillna(most_common_value, inplace=True)
        
        return X
    def _most_common_term(self, lst):
    # Use Counter to count occurrences and return the most common element
        return Counter(lst).most_common(1)[0][0]
    

class LogTransforms(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
                
        X = X.copy()  # Ensure we don't modify the original data
        for col in self.variables:
            X[col] = np.log(X[col].astype(float))  # Apply log transformation

        return X


    
class CategoricalToNumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        self.category_map_ = {}
        for col in self.variables:
            feature_set = set(X[col])
            self.category_map_[col] = {cat: idx for idx, cat in enumerate(feature_set)}
        return self
    
    def transform(self, X):
        
        X_transformed = X.copy()
        for col in self.variables:
            print(self.category_map_[col])
            X_transformed[col] = X_transformed[col].map(self.category_map_[col])
            print(X_transformed[col])
        return X_transformed

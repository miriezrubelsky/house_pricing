import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np



import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        self.mean_dict = {}

        # Check if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            # Loop over the columns and calculate mean only for numeric columns
            for col in self.variables:
                if X[col].dtype in [np.float64, np.int64]:  # Check if the column is numeric
                    self.mean_dict[col] = X[col].mean()
        elif isinstance(X, np.ndarray):
            # Handle numpy arrays: We assume that variables are indexed correctly
            for i, col in enumerate(self.variables):
                # We need to check the dtype of the array column
                if np.issubdtype(X[:, i].dtype, np.number):  # Check if column is numeric
                    self.mean_dict[i] = np.nanmean(X[:, i])  # Compute mean ignoring NaN values
        else:
            raise TypeError("Input data should be a Pandas DataFrame or NumPy array")
        
        return self
    
    def transform(self, X):
        X = X.copy()  # Make a copy to avoid modifying the original data

        # Check if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            for col in self.variables:
                if col in X.columns and col in self.mean_dict:
                    # Replace NaN values with the mean only for numeric columns
                    X[col].fillna(self.mean_dict[col], inplace=True)
        elif isinstance(X, np.ndarray):
            for i, col in enumerate(self.variables):
                if i in self.mean_dict:
                    # Replace NaN values in the column with the computed mean
                    col_mean = self.mean_dict[i]
                    X[:, i] = np.where(np.isnan(X[:, i]), col_mean, X[:, i])
        else:
            raise TypeError("Input data should be a Pandas DataFrame or NumPy array")
        
        return X

    
    def transform(self, X):
        X = X.copy()  # Make a copy to avoid modifying the original data

        # Check if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            for col in self.variables:
                if col in X.columns:
                    # Replace NaN values in the specified columns with the computed mean
                    X[col].fillna(self.mean_dict[col], inplace=True)
        elif isinstance(X, np.ndarray):
            for i, col in enumerate(self.variables):
                # For NumPy arrays, use column indices to replace NaN values
                col_mean = self.mean_dict.get(i)
                if col_mean is not None:
                    # Replace NaNs in the NumPy array column with the mean
                    X[:, i] = np.where(np.isnan(X[:, i]), col_mean, X[:, i])
        else:
            raise TypeError("Input data should be a Pandas DataFrame or NumPy array")
        
        return X

    



class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables_to_drop = variables_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Make a copy to avoid modifying the original data
        X = X.copy()

        # Check if X is a Pandas DataFrame
        if isinstance(X, pd.DataFrame):
            # If X is a DataFrame, use drop method
            X = X.drop(columns=self.variables_to_drop, errors='ignore')  # `errors='ignore'` handles non-existing columns safely
        elif isinstance(X, np.ndarray):
            # If X is a NumPy array, we don't have column names, so we drop by index
            # Assuming self.variables_to_drop is a list of column names (we need to convert them to indices if necessary)
            if isinstance(self.variables_to_drop, list):
                # If the list contains column names and X is a DataFrame before it becomes an ndarray,
                # we need to match column names to indices (in case of NumPy ndarray).
                # Note: You need to know the column names if you want to drop by name, otherwise use indices directly.
                pass  # The column names to index mapping should be handled earlier if using ndarray.
            else:
                # If `variables_to_drop` contains column indices
                X = np.delete(X, self.variables_to_drop, axis=1)
        else:
            raise TypeError("Input should be either a Pandas DataFrame or a NumPy array")
        
        return X


    


class NoneImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        # No fitting needed, just return the transformer
        return self
    
    def transform(self, X):
        # Make a copy of X to avoid modifying the original data
        X = X.copy()

        # If X is a pandas DataFrame, we can use columns
        if isinstance(X, pd.DataFrame):
            for col in self.variables:
                if col in X.columns:
                    # Print the column name and its dtype to debug
                    print(f"Processing column: {col}, dtype: {X[col].dtype}")

                    # Check if dtype is 'object' (categorical)
                    if X[col].dtype == 'object':
                        X[col].fillna("None", inplace=True)
                    elif pd.api.types.is_numeric_dtype(X[col]):
                        X[col].fillna("None", inplace=True)  # If numeric, fill with "None"
                else:
                    print(f"Column '{col}' not found in the DataFrame")
        else:
            # If X is a numpy array, assume the column order corresponds to variables
            for i, col in enumerate(self.variables):
                try:
                    # Check if the column is numeric or categorical
                    if np.issubdtype(X[:, i].dtype, np.number):  # Check if numeric
                        X[:, i] = np.where(np.isnan(X[:, i]), "None", X[:, i])  # Replace NaNs with "None"
                    else:
                        # For non-numeric columns, assume it's categorical
                        X[:, i] = np.where(X[:, i] == None, "None", X[:, i])  # Handle None values for categorical columns
                except IndexError:
                    print(f"Column '{col}' does not exist in the array.")
                    raise ValueError(f"Column '{col}' not found in the input array.")
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
        return max(set(lst), key=lst.count)



import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransforms(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()  # Ensure we don't modify the original data
        
        # Check if X is a pandas DataFrame or numpy ndarray
        if isinstance(X, pd.DataFrame):
            # If it's a DataFrame, we can use column names
            for col in self.variables:
                # Convert to float before applying np.log
                X[col] = np.log(X[col].astype(float))
        elif isinstance(X, np.ndarray):
            # If it's a NumPy array, we'll need to use integer indexing for columns
            for col in self.variables:
                # Find the column index by name (assuming variables list has column names)
                col_index = self.variables.index(col)
                # Convert to float before applying np.log
                X[:, col_index] = np.log(X[:, col_index].astype(float))
        else:
            raise TypeError("Input data should be a Pandas DataFrame or NumPy ndarray")
        
        return X


    



class CategoricalToNumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        self.category_map_ = {}
        # If X is a DataFrame, we use columns directly
        if isinstance(X, pd.DataFrame):
            for col in self.variables:
                feature_set = set(X[col])
                self.category_map_[col] = {cat: idx for idx, cat in enumerate(feature_set)}
        else:
            # Handle NumPy array (assuming column names are provided externally or inferred)
            # We assume that variables are passed by column index when working with NumPy arrays
            for idx in self.variables:
                feature_set = set(X[:, idx])  # Using the column index
                self.category_map_[idx] = {cat: idx for idx, cat in enumerate(feature_set)}

        return self
    
    def transform(self, X):
        # Check if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            for col in self.variables:
                X_transformed[col] = X_transformed[col].map(self.category_map_[col])
        elif isinstance(X, np.ndarray):
            # If X is a NumPy array, we convert it to a DataFrame temporarily for processing
            X_transformed = pd.DataFrame(X)
            for idx in self.variables:
                X_transformed[idx] = X_transformed[idx].map(self.category_map_[idx])
            X_transformed = X_transformed.values  # Convert back to a NumPy array if needed
        else:
            raise ValueError("Input must be a pandas DataFrame or NumPy array.")

        return X_transformed


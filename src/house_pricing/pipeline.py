from sklearn.pipeline import Pipeline
from pathlib import Path
import os
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from house_pricing.config import config
import house_pricing.processing.preprocessing as pp 



classification_pipeline = Pipeline(
    [
        ('NullReplaceProcessing',pp.NoneImputer(variables=config.NULL_HAS_MEANING)),
        ('MedianImputation', pp.MedianImputer(variables=config.NUM_FEATURES)),
        ('MeanImputation', pp.MeanImputer(variables=config.NUMERIC_COLUMNS)),
        ('MostCommonImputation', pp.MostCommonImputer(variables=config.CATEGORICAL_COLUMNS)),
      #  ('LogTransform',pp.LogTransforms(variables=config.LOG_FEATURES)),
        
  
        ('CategoricalToNumericTransform',pp.CategoricalToNumericTransformer(variables=config.CATEGORICAL_COLUMNS)),
        ('DropFeatures', pp.DropColumns(variables_to_drop=config.COLUMNS_TO_DROP))
       # ('LinearRegression', GridSearchCV(
       #     LinearRegression(),
       #     param_grid={"fit_intercept": [True, False], "copy_X": [True, False]},
       #     scoring="r2",
      #      verbose=1
       # ))
        
    ]
)
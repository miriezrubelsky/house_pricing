from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

import sys
from pathlib import Path
import os

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))


from house_pricing.config import config
import house_pricing.pipeline as pipe 
from house_pricing.processing.data_handling import load_dataset,save_pipeline
from house_pricing.processing.preprocessing import MeanImputer
def compute_correlations(X, target_column="SalePrice", top_n=5):
    correlations = X.corr()
    correlations = correlations[target_column].sort_values(ascending=False)
    top_features = correlations.index[1:top_n+1]  # Skip the target column itself
    return top_features


def perform_training():
     train_data = load_dataset(config.TRAIN_FILE)
     train_data = train_data.head(5)
     file_path = "transformed_data.xlsx"

# Save the DataFrame to an Excel file
     train_data.to_excel(file_path, index=False)
     columns_to_drop = ["Id","SalePrice"]
    
    # Drop only existing columns
     columns_to_drop = [col for col in columns_to_drop if col in train_data.columns]
    
    # Drop columns safely
     train_data_for_columns = train_data.drop(columns=columns_to_drop)
     config.NUMERIC_COLUMNS = get_numerical_columns(train_data_for_columns)
     config.CATEGORICAL_COLUMNS =get_categorical_columns(train_data_for_columns)
     
     pipe.classification_pipeline.set_params(NullReplaceProcessing__variables=config.NULL_HAS_MEANING)
     pipe.classification_pipeline.set_params(MedianImputation__variables=config.NUM_FEATURES)
     pipe.classification_pipeline.set_params(MeanImputation__variables=config.NUMERIC_COLUMNS)
     pipe.classification_pipeline.set_params(MostCommonImputation__variables=config.CATEGORICAL_COLUMNS)
     pipe.classification_pipeline.set_params(CategoricalToNumericTransform__variables=config.CATEGORICAL_COLUMNS)
     pipe.classification_pipeline.set_params(DropFeatures__variables_to_drop=config.COLUMNS_TO_DROP)
       
     X_train = train_data
     if "SalePrice" in X_train.columns:
        X_train["TransformedPrice"] = np.log(X_train["SalePrice"].astype(float)) 
     y_train = train_data["TransformedPrice"]   
     X_train = X_train.drop(["Id","TransformedPrice","SalePrice"], axis=1)
     pipe.classification_pipeline.fit(X_train,y_train)
     save_pipeline(pipe.classification_pipeline)
     
     #X_transformed = pipe.classification_pipeline.named_steps['NullReplaceProcessing'].transform(X_train) 
     #X_transformed = pipe.classification_pipeline.named_steps['MedianImputation'].transform(X_transformed) 
     #X_transformed = pipe.classification_pipeline.named_steps['MeanImputation'].transform(X_transformed) 
     #X_transformed = pipe.classification_pipeline.named_steps['MostCommonImputation'].transform(X_transformed) 
    # X_transformed = pipe.classification_pipeline.named_steps['LogTransform'].transform(X_transformed) 
     #X_transformed = pipe.classification_pipeline.named_steps['CategoricalToNumericTransform'].transform(X_transformed) 

     #X_transformed = pipe.classification_pipeline.named_steps['DropFeatures'].transform(X_transformed)

     #file_path = "transformed_data1.xlsx"
    # X_transformed_df = pd.DataFrame(X_transformed, columns=train_data.drop(["Id", "SalePrice"], axis=1).columns)
     #X_transformed.to_excel(file_path, index=False)
     




def get_numerical_columns(df):
    # Get the data types of each column
    types = df.dtypes
    
    # Filter for columns that are either 'int64' or 'float64'
    num_columns = types[(types == 'int64') | (types == 'float64')]
    
    # Return the index (which are the column names) as a list
    return num_columns.index.tolist()



def get_categorical_columns(df):
    # Get the data types of each column
    types = df.dtypes
    
    # Filter for columns that are of type 'object' (typically categorical data in pandas)
    cat_columns = types[types == 'object']
    
    # Return the index (column names) as a list
    return cat_columns.index.tolist()



if __name__=='__main__':
    perform_training()
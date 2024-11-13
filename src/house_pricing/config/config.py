
import pathlib
import os
import house_pricing


PACKAGE_ROOT = pathlib.Path(house_pricing.__file__).resolve().parent



NULL_HAS_MEANING = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature','MasVnrType']
COLUMNS_TO_DROP = ['LotFrontage']
NUM_FEATURES = ['GarageYrBlt', 'MasVnrArea']

NUMERIC_COLUMNS = []
CATEGORICAL_COLUMNS = []

LOG_FEATURES = ['SalePrice'] # taking log of numerical columns


CAT_FEATURES_MOST_COMMON = ["Electrical", "Exterior1st", "Exterior2nd", "Functional", "KitchenQual", "MSZoning", "SaleType", "Utilities", "MasVnrType"]

LOGGING_DIR = os.path.join(PACKAGE_ROOT, 'logs')  # Set logs directory at the root level
LOGGING_FILENAME = os.path.join(LOGGING_DIR, 'pipeline_logs.log')
LOGGING_FILEMODE = 'w'  # 'w' for overwrite, 'a' for append
LOGGING_LEVEL = 'DEBUG'  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


DATAPATH = os.path.join(PACKAGE_ROOT,"data")

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'


MODEL_NAME = 'classification.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_model')

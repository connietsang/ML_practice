import sys
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

training_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'
test_file_path = '../input/house-prices-advanced-regression-techniques/test.csv'

test_X_full = pd.read_csv(test_file_path, index_col='Id')

training_data_full = pd.read_csv(training_file_path, index_col='Id')
# remove rows with missing SalePrice
training_data_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = training_data_full.SalePrice

# drop target from training data
training_data_full.drop(['SalePrice'], axis=1, inplace=True)
test_data = pd.read_csv(test_file_path)

# training features (columns used for predictions) and X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd',
            'Neighborhood', 'Condition1', 'Condition2', 'HouseStyle', 'OverallQual', 'OverallCond',
            'YearRemodAdd', 'ExterQual', 'ExterCond', 'BsmtCond', 'GarageArea', 'GarageQual', 'YrSold']

training_data = training_data_full[features]

X_train_full, X_valid_full, y_train, y_valid = train_test_split(training_data, y,
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)

categorical_cols = [cname for cname in training_data.columns if
                    training_data[cname].nunique() < 10 and
                    training_data[cname].dtype == "object"]
numerical_cols = [cname for cname in training_data.columns if
                  training_data[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = test_X_full[my_cols].copy()

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = XGBRegressor(n_estimators=400, learning_rate=0.05, random_state=1)

# Bundle preprocessing and modeling code in a pipeline
pl = Pipeline(steps=[('preprocessor', preprocessor),
                     ('model', model)
                     ])

pl.fit(X_train, y_train)
predictions = pl.predict(X_valid)
score = mean_absolute_error(y_valid, predictions)


def best_learning_rate():
    min = float('inf')
    index = 1
    for i in range(1, 10):
        test_model = XGBRegressor(
            n_estimators=400, learning_rate=0.01*i, random_state=1)
#         test_model = RandomForestRegressor(n_estimators=i*50,random_state=1)
        test_pl = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', test_model)
                                  ])
        test_pl.fit(X_train, y_train)
        preds = test_pl.predict(X_valid)
        if (mean_absolute_error(y_valid, preds) < min):
            min = mean_absolute_error(y_valid, preds)
            index = i
    return index, min

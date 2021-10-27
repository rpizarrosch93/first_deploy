from posixpath import split
import pandas as pd
import numpy as np
# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import MinMaxScaler

# to save the trained scaler class
import joblib

import scipy.stats as stats

import os
os.getcwd()


# Load Dataset
# config.yml global variable path to data

data = pd.read_csv("~/projects/learning/deployment/repo/data/train.csv")

# config.yml
test_size_param = 0.1

# for reproducibility
random_state_param = 0

# target variable
target_variable = 'SalePrice'

# drop Id Column
data = data.drop(['Id'], axis=1)


def split_data_train_test(data, test_size_param, random_state_param):
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop([target_variable], axis=1),  # predictive variables
        data[target_variable],  # target
        test_size=test_size_param,  # portion of dataset to allocate to test set
        random_state=random_state_param,  # we are setting the seed here
    )

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = split_data_train_test(
    data, test_size_param, random_state_param)

# Missing Values imputation 

# Categorical Data with nas

categorical_vars_with_na_missing = ['Alley', 'FireplaceQu',
                          'PoolQC', 'Fence', 'MiscFeature']

categorical_vars_with_na_frequent = ['MasVnrType',
                                'BsmtQual',
                                'BsmtCond',
                                'BsmtExposure',
                                'BsmtFinType1',
                                'BsmtFinType2',#
                                'Electrical',
                                'GarageType',
                                'GarageFinish',
                                'GarageQual',
                                'GarageCond']


# Numerical Data with nas

numerical_vars_with_na = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']

# Binary missing indicator

# New Features

year_data_vars = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']

# Create diff between YrSold and each of year_data_vars

# Drop YrSold

# Normalize non standard distribuited variables using log

not_standard_distribuited_vars = ["LotFrontage", "1stFlrSF", "GrLivArea"]

# We will apply the Yeo-Johnson transformation to LotArea.

#the yeo-johnson transformation learns the best exponent to transform the variable
# it needs to learn it from the train set: 
X_train['LotArea'], param = stats.yeojohnson(X_train['LotArea'])

# and then apply the transformation to the test set with the same
# parameter: see who this time we pass param as argument to the 
# yeo-johnson
X_test['LotArea'] = stats.yeojohnson(X_test['LotArea'], lmbda=param)

# Ordered categorical values needs to be mapped

# re-map strings to numbers, which determine quality

qual_mappings = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, 'Missing': 0, 'NA': 0}

qual_vars = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
             'HeatingQC', 'KitchenQual', 'FireplaceQu',
             'GarageQual', 'GarageCond',
            ]

for var in qual_vars:
    X_train[var] = X_train[var].map(qual_mappings)
    X_test[var] = X_test[var].map(qual_mappings)


exposure_mappings = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}

var = 'BsmtExposure'

X_train[var] = X_train[var].map(exposure_mappings)
X_test[var] = X_test[var].map(exposure_mappings)


finish_mappings = {'Missing': 0, 'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}

finish_vars = ['BsmtFinType1', 'BsmtFinType2']

for var in finish_vars:
    X_train[var] = X_train[var].map(finish_mappings)
    X_test[var] = X_test[var].map(finish_mappings)

garage_mappings = {'Missing': 0, 'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}

var = 'GarageFinish'

X_train[var] = X_train[var].map(garage_mappings)
X_test[var] = X_test[var].map(garage_mappings)

fence_mappings = {'Missing': 0, 'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}

var = 'Fence'

X_train[var] = X_train[var].map(fence_mappings)
X_test[var] = X_test[var].map(fence_mappings)

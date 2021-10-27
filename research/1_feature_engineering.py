from posixpath import split
import pandas as pd
import numpy as np
# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import MinMaxScaler

# to save the trained scaler class
import joblib

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
data = data.drop(['Id'],axis = 1)


def split_data_train_test(data,test_size_param,random_state_param):
    X_train, X_test, y_train, y_test = train_test_split(
    data.drop([ target_variable], axis=1), # predictive variables
    data[target_variable], # target
    test_size=test_size_param, # portion of dataset to allocate to test set
    random_state=random_state_param, # we are setting the seed here
    )

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data_train_test(data,test_size_param,random_state_param)


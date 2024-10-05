# data_preprocessing.py

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from scipy import stats

def load_data(file_path):
    return pd.read_csv(file_path)

def handle_missing_values(df):
    # Separating numerical and categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # Imputing numerical columns with mean
    num_imputer = SimpleImputer(strategy='mean')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Imputing categorical columns with most frequent value
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df

def remove_outliers(df, z_thresh=3):
    # Removing outliers based on Z-score
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    filtered_entries = (z_scores < z_thresh).all(axis=1)
    df = df[filtered_entries]
    return df

def encode_categorical_features(df):
    df = pd.get_dummies(df, drop_first=True)
    return df

def create_new_features(df):
    # Creating new features
    df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']
    return df

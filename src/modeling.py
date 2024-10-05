# modeling.py

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_linear_regression(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def train_ridge_regression(X_train, y_train):
    ridge = Ridge()
    param_grid = {'alpha': [0.1, 1.0, 10.0]}
    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5,
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_lasso_regression(X_train, y_train):
    lasso = Lasso()
    param_grid = {'alpha': [0.001, 0.01, 0.1]}
    grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=5,
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_random_forest(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5,
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_rf, best_params

def train_xgboost(X_train, y_train):
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0]
    }
    grid_search = GridSearchCV(estimator=xg_reg, param_grid=param_grid, cv=5,
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_xg = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_xg, best_params

def evaluate_model(model, X_test, y_test, return_predictions=False):
    y_pred = model.predict(X_test)
    # Transforming back to original scale
    y_pred = np.expm1(y_pred)
    y_test_exp = np.expm1(y_test)
    mse = mean_squared_error(y_test_exp, y_pred)
    rmse = mse ** 0.5
    if return_predictions:
        return rmse, y_pred
    else:
        return rmse

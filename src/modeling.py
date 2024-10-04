import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_linear_regression(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def train_random_forest(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train):
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    xg_reg.fit(X_train, y_train)
    return xg_reg

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

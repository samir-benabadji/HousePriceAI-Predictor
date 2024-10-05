# main.py

import os
import sys
import numpy as np
import pandas as pd

# Ensuring that the src directory is in the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import (
    load_data, handle_missing_values, encode_categorical_features,
    remove_outliers, create_new_features
)
from feature_engineering import feature_selection
from modeling import (
    split_data, train_linear_regression, train_ridge_regression,
    train_lasso_regression, train_random_forest, train_xgboost, evaluate_model
)
from utils import (
    plot_correlation_matrix, plot_feature_importance, plot_actual_vs_predicted,
    plot_residuals, plot_saleprice_distribution
)

def main():
    # Setting up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_path = os.path.join(project_dir, 'data', 'raw', 'house_prices.csv')
    processed_data_path = os.path.join(project_dir, 'data', 'processed', 'processed_data.csv')
    plots_dir = os.path.join(project_dir, 'plots')

    # Ensuring directories exist
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Loading Data
    df = load_data(data_path)
    print("Data loaded successfully.")

    # Visualize the distribution of SalePrice
    plot_saleprice_distribution(df, plots_dir)

    # Data Preprocessing
    df_clean = handle_missing_values(df)
    print("Missing values handled.")

    # Remove outliers
    df_clean = remove_outliers(df_clean)
    print(f"Outliers removed. Data shape: {df_clean.shape}")

    # Create new features
    df_clean = create_new_features(df_clean)
    print("New features created.")

    # Encoding categorical features
    df_encoded = encode_categorical_features(df_clean)
    print("Categorical features encoded.")

    # Saving processed data
    df_encoded.to_csv(processed_data_path, index=False)
    print("Processed data saved.")

    # Feature Engineering
    df_selected = feature_selection(df_encoded)
    print("Feature selection completed.")

    # Visualize the correlation matrix
    plot_correlation_matrix(df_selected, plots_dir)

    # Applying log transformation to 'SalePrice'
    df_selected['SalePrice'] = np.log1p(df_selected['SalePrice'])

    # Modeling
    # Splitting features and target
    X = df_selected.drop('SalePrice', axis=1)
    y = df_selected['SalePrice']

    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Data split into training and testing sets.")

    # Training models
    print("Training Linear Regression model...")
    lr_model = train_linear_regression(X_train, y_train)

    print("Training Ridge Regression model with hyperparameter tuning...")
    ridge_model = train_ridge_regression(X_train, y_train)

    print("Training Lasso Regression model with hyperparameter tuning...")
    lasso_model = train_lasso_regression(X_train, y_train)

    print("Tuning Random Forest model...")
    rf_model, rf_best_params = train_random_forest(X_train, y_train)
    print(f"Best Random Forest parameters: {rf_best_params}")

    print("Tuning XGBoost model...")
    xg_model, xg_best_params = train_xgboost(X_train, y_train)
    print(f"Best XGBoost parameters: {xg_best_params}")

    # Evaluating models
    print("\nModel Performance:")

    # Linear Regression
    lr_rmse, lr_y_pred = evaluate_model(lr_model, X_test, y_test, return_predictions=True)
    print(f'Linear Regression RMSE: {lr_rmse:.2f}')

    # Ridge Regression
    ridge_rmse, ridge_y_pred = evaluate_model(ridge_model, X_test, y_test, return_predictions=True)
    print(f'Ridge Regression RMSE: {ridge_rmse:.2f}')

    # Lasso Regression
    lasso_rmse, lasso_y_pred = evaluate_model(lasso_model, X_test, y_test, return_predictions=True)
    print(f'Lasso Regression RMSE: {lasso_rmse:.2f}')

    # Random Forest
    rf_rmse, rf_y_pred = evaluate_model(rf_model, X_test, y_test, return_predictions=True)
    print(f'Random Forest RMSE: {rf_rmse:.2f}')

    # XGBoost
    xg_rmse, xg_y_pred = evaluate_model(xg_model, X_test, y_test, return_predictions=True)
    print(f'XGBoost RMSE: {xg_rmse:.2f}')

    # Visualizations (Optional)
    # You can choose which models to visualize
    plot_feature_importance(rf_model, X_train, 'Random Forest', plots_dir)
    plot_feature_importance(xg_model, X_train, 'XGBoost', plots_dir)
    plot_actual_vs_predicted(y_test, rf_y_pred, 'Random Forest', plots_dir)
    plot_residuals(y_test, rf_y_pred, 'Random Forest', plots_dir)

    print("\nAll evaluations and visualizations are complete.")

if __name__ == '__main__':
    main()

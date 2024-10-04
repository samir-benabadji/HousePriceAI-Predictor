import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_matrix(df, plots_dir):
    plt.figure(figsize=(12,10))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'))
    plt.close()

def plot_feature_importance(model, X_train, model_name, plots_dir):
    feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    feat_importances.nlargest(20).plot(kind='barh', figsize=(10,8))
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Top 20 Feature Importances - {model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'feature_importance_{model_name}.png'))
    plt.close()

def plot_actual_vs_predicted(y_test, y_pred, model_name, plots_dir):
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(f'Actual vs Predicted Prices - {model_name}')
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'actual_vs_predicted_{model_name}.png'))
    plt.close()

def plot_residuals(y_test, y_pred, model_name, plots_dir):
    residuals = y_test - y_pred
    plt.figure(figsize=(8,6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals')
    plt.title(f'Residuals Plot - {model_name}')
    plt.axhline(0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'residuals_{model_name}.png'))
    plt.close()

def plot_saleprice_distribution(df, plots_dir):
    plt.figure(figsize=(8,6))
    sns.histplot(df['SalePrice'], kde=True)
    plt.title('Distribution of SalePrice')
    plt.xlabel('SalePrice')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'saleprice_distribution.png'))
    plt.close()

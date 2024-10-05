# feature_engineering.py

def feature_selection(df, threshold=0.05):
    # Removing features with low correlation with target
    corr = df.corr()
    target_corr = corr['SalePrice'].abs()
    relevant_features = target_corr[target_corr > threshold].index
    return df[relevant_features]

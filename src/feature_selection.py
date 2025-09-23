from catboost import CatBoostRegressor
import numpy as np

from config import Config


def top_k_feature_selection(X, y, k=Config.top_features):
    """
    Select top k features based on feature importance from a CatBoostRegressor.
    Args:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Target variable array.
        k (int): Number of top features to select.
    Returns:
        list: List of top k feature names. 
    """
    model = CatBoostRegressor(**Config.cat_params)
    model.fit(X, y)

    feature_importances = model.get_feature_importance(prettified=True)
    top_features = feature_importances.head(k)['Feature Id'].tolist()

    return top_features

def drop_highly_correlated_features(X, threshold=0.9):
    """Drop features that are highly correlated above the given threshold.
    Args:
        X (pd.DataFrame): Feature matrix.
        threshold (float): Correlation threshold to drop features.
    Returns:
        list: List of features after dropping highly correlated ones.
    """
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    print(f'Dropping {len(to_drop)} highly correlated features: {to_drop}')
    reduced_features = [feature for feature in X.columns.tolist() if feature not in to_drop]
    return reduced_features

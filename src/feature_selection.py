from catboost import CatBoostRegressor
import numpy as np

from config import Config


def top_k_feature_selection(X, y, k=Config.top_features, method='catboost'):
    """
    Select top k features based on feature importance from a CatBoostRegressor.
    Args:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Target variable array.
        k (int): Number of top features to select.
    Returns:
        list: List of top k feature names. 
    """
    print(f'Selecting top {k} features using {method} method')

    if method == 'catboost':
        model = CatBoostRegressor(**Config.cat_params)
        model.fit(X, y)

        feature_importances = model.get_feature_importance(prettified=True)
        top_features = feature_importances.head(k)['Feature Id'].tolist()

        return top_features

    elif method == 'anova':
        from sklearn.feature_selection import SelectKBest, f_classif
        # Select top 20 features using ANOVA F-test (works for classification)
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, y)

        # Get the selected feature names
        selected_features = X.columns[selector.get_support()]
        top_features = selected_features.tolist()

        return top_features

    elif method == 'lasso':
        from sklearn.linear_model import LassoCV
        lasso = LassoCV(cv=5).fit(X, y)
        selected_features = X.columns[lasso.coef_ != 0]
        top_features = selected_features.tolist()
        return top_features

    elif method == 'permutation':
        from sklearn.inspection import permutation_importance

        model = CatBoostRegressor(**Config.cat_params)
        model.fit(X, y, verbose=0)

        perm_importance = permutation_importance(
            model, X, y, n_repeats=10, random_state=42, n_jobs=-1
        )

        # Pair each feature with its importance
        feature_scores = list(zip(X.columns, perm_importance.importances_mean))

        # Sort by importance (descending)
        feature_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top-k features
        top_features = [f for f, score in feature_scores[:k]]
        return top_features

    else:
        raise ValueError(f"Invalid selection method: {method}")

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

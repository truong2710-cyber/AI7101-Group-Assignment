import numpy as np

from feature_selection import drop_highly_correlated_features, top_k_feature_selection


class EnsembleModel:
    def __init__(self, top_features=30, corr_threshold=0.9, clip_threshold=0.99, feature_selection_method='top_k'):
        """
        Args:
        top_features: Number of top features to select based on importance.
        corr_threshold: Threshold to drop highly correlated features.
        clip_threshold: Quantile threshold to clip target variable.
        """
        self.top_features = top_features
        self.corr_threshold = corr_threshold
        self.clip_threshold = clip_threshold
        self.models = None
        self.reduced_features = None
        self.best_params = None
        self.feature_selection_method = feature_selection_method

    def _clip_target(self, y):
        """Clip target values at the specified quantile threshold.
        Args:
            y (np.ndarray): Target variable array.
        Returns:
            np.ndarray: Clipped target variable array.
        """
        clip_val = np.quantile(y, self.clip_threshold)
        return np.where(y >= clip_val, clip_val, y)

    def _feature_selection(self, X, y):
        """Select top features and drop highly correlated ones.
        Args:
            X (pd.DataFrame): Feature matrix.
            y (np.ndarray): Target variable array.
        """
        top_feats = top_k_feature_selection(X, y, k=self.top_features, method=self.feature_selection_method)
        self.reduced_features = drop_highly_correlated_features(X[top_feats], threshold=self.corr_threshold)

    def _fit_models(self, X, y, models):
        """Fit all models and return the fitted models.
        Args:
            X (pd.DataFrame): Feature matrix.
            y (np.ndarray): Target variable array.
            models (dict): Dictionary of model instances.
        Returns:
            dict: Dictionary of fitted models.
        """
        for name, model in models.items():
            model.fit(X[self.reduced_features], y)
        return models

    def cross_val_fit(self, X, y, folds, models, X_test=None):
        """Run CV to evaluate hyperparameters; returns OOF, test predictions, mean RMSE.
        Args:
            X (pd.DataFrame): Feature matrix.
            y (np.ndarray): Target variable array.
            folds (np.ndarray): Array indicating fold assignments.
            models (dict): Dictionary of model instances.
            X_test (pd.DataFrame, optional): Test feature matrix. Defaults to None.
        Returns:
            oof_preds (np.ndarray): Out-of-fold predictions.
            final_test_preds (np.ndarray or None): Test predictions if X_test is provided, else None.
            mean_rmse (float): Mean RMSE across folds.
        """
        oof_preds = np.zeros(len(y))
        fold_test_preds = [] if X_test is not None else None
        fold_rmse_list = []

        unique_folds = np.unique(folds)
        self._feature_selection(X, y)

        for fold in unique_folds:
            train_idx = folds != fold
            val_idx = folds == fold
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            y_train_clipped = self._clip_target(y_train)
            
            val_preds = np.zeros((len(X_val), len(models)))
            test_preds = np.zeros((len(X_test), len(models))) if X_test is not None else None

            # Train each model
            models_fold = self._fit_models(X_train, y_train_clipped, models)
            for i, (name, model) in enumerate(models_fold.items()):
                val_preds[:, i] = model.predict(X_val[self.reduced_features])
                if X_test is not None:
                    test_preds[:, i] = model.predict(X_test[self.reduced_features])

            # Ensemble
            oof_preds[val_idx] = val_preds.mean(axis=1)
            if X_test is not None:
                fold_test_preds.append(test_preds.mean(axis=1))

            # Fold RMSE
            fold_rmse = np.sqrt(np.mean((y_val - oof_preds[val_idx])**2))
            fold_rmse_list.append(fold_rmse)

        mean_rmse = np.mean(fold_rmse_list)
        final_test_preds = np.mean(fold_test_preds, axis=0) if X_test is not None else None
        return oof_preds, final_test_preds, mean_rmse

    def fit_final(self, X, y, models):
        """Train final ensemble on full training data after hyperparameter selection.
        Args:
            X (pd.DataFrame): Feature matrix.
            y (np.ndarray): Target variable array.
            models (dict): Dictionary of model instances.
        """
        y_clipped = self._clip_target(y)
        self._feature_selection(X, y_clipped)
        self.models = self._fit_models(X, y_clipped, models)

    def predict(self, X):
        """Predict using the ensemble (mean of all models).
        Args:
            X (pd.DataFrame): Feature matrix.
        Returns:
            np.ndarray: Ensemble predictions.
        """
        test_preds = np.zeros((len(X), len(self.models)))
        for i, (name, model) in enumerate(self.models.items()):
            test_preds[:, i] = model.predict(X[self.reduced_features])
        return np.mean(test_preds, axis=1)

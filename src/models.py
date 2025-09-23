import numpy as np

from feature_selection import drop_highly_correlated_features, top_k_feature_selection


class EnsembleModel:
    def __init__(self, top_features=30, corr_threshold=0.9, clip_threshold=0.99):
        self.top_features = top_features
        self.corr_threshold = corr_threshold
        self.clip_threshold = clip_threshold
        self.models = None
        self.reduced_features = None
        self.best_params = None

    def _clip_target(self, y):
        clip_val = np.quantile(y, self.clip_threshold)
        return np.where(y >= clip_val, clip_val, y)

    def _feature_selection(self, X, y):
        top_feats = top_k_feature_selection(X, y, k=self.top_features)
        self.reduced_features = drop_highly_correlated_features(X[top_feats], threshold=self.corr_threshold)

    def _fit_models(self, X, y, models):
        for name, model in models.items():
            model.fit(X[self.reduced_features], y)
        return models

    def cross_val_fit(self, X, y, folds, models, X_test=None):
        """Run CV to evaluate hyperparameters; returns OOF, test predictions, mean RMSE."""
        oof_preds = np.zeros(len(y))
        fold_test_preds = [] if X_test is not None else None
        fold_rmse_list = []

        unique_folds = np.unique(folds)
        for fold in unique_folds:
            train_idx = folds != fold
            val_idx = folds == fold
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            y_train_clipped = self._clip_target(y_train)
            self._feature_selection(X_train, y_train_clipped)

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
        """Train final ensemble on full training data after hyperparameter selection."""
        y_clipped = self._clip_target(y)
        self._feature_selection(X, y_clipped)
        self.models = self._fit_models(X, y_clipped, models)

    def predict(self, X):
        """Predict using the ensemble (mean of all models)."""
        test_preds = np.zeros((len(X), len(self.models)))
        for i, (name, model) in enumerate(self.models.items()):
            test_preds[:, i] = model.predict(X[self.reduced_features])
        return np.mean(test_preds, axis=1)

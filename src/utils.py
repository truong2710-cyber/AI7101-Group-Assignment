import json
import os
import joblib
import numpy as np

from config import Config


def prepare_submission(test, best_test_pred, save_path='output/submission.csv'):
    """Prepare submission file.
    Args:
        test (pd.DataFrame): Test dataset.
        best_test_pred (np.ndarray): Predictions for the test dataset.
        save_path (str): Path to save the submission CSV file.
    """
    test[Config.target_col] = best_test_pred
    submission = test[[Config.id_col, Config.target_col]]
    submission.to_csv(save_path, index=False)
    submission.head()

def load_and_predict(test, rmse=27.84, model_names=None):
    """
    Load best models, make predictions on test set, and ensemble them.

    Args:
        test (pd.DataFrame): Test dataset.
        rmse (float): RMSE value to locate model folder.
        model_names (list): List of model names to load. Defaults to ['cat', 'lgb', 'xgb', 'lasso', 'svr'].

    Returns:
        np.ndarray: Ensemble predictions for the test set.
    """
    if model_names is None:
        model_names = ['cat', 'lgb', 'xgb', 'lasso', 'svr']

    # Load models
    folder_name = f"./weights/{rmse:.2f}"
    models = {name: joblib.load(f"{folder_name}/best_model_{name}.pkl") for name in model_names}

    # Load reduced features
    with open(f"{folder_name}/reduced_features.json", "r") as f:
        features = json.load(f)

    # Prepare test predictions
    test_preds = np.zeros((len(test), len(models)))

    # Predict
    for i, (name, model) in enumerate(models.items()):
        test_preds[:, i] = model.predict(test[features])

    # Ensemble with equal weights
    final_test_pred = np.mean(test_preds, axis=1)

    return final_test_pred

# Save models and reduced features
def save_models_and_features(models, reduced_features, rmse, save_dir='weights'):
    """Save models and reduced features to disk.
    Args:
        models (dict): Dictionary of trained models.
        reduced_features (list): List of selected features.
        rmse (float): RMSE value to name the folder.
        save_dir (str): Directory to save the models and features.
    """
    folder_name = os.path.join(save_dir, f"{rmse:.2f}")
    os.makedirs(folder_name, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, f"{folder_name}/best_model_{name}.pkl")
    with open(f"{folder_name}/reduced_features.json", "w") as f:
        json.dump(reduced_features, f)

# Apply best hyperparameters to Config
def apply_best_params(best_params):
    """Update Config with best hyperparameters.
    Args:
        best_params (dict): Dictionary of best hyperparameters.
    """
    Config.cat_params['learning_rate'] = best_params['cat_lr']
    Config.cat_params['depth'] = best_params['cat_depth']
    Config.lgb_params['learning_rate'] = best_params['lgb_lr']
    Config.lgb_params['max_depth'] = best_params['lgb_depth']
    Config.xgb_params['learning_rate'] = best_params['xgb_lr']   
    Config.xgb_params['max_depth'] = best_params['xgb_depth']
    Config.lasso_params['alpha'] = best_params['lasso_alpha']
    Config.svr_params['C'] = best_params['svr_C']
    Config.svr_params['epsilon'] = best_params['svr_eps']

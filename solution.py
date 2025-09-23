
import pandas as pd

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR

import optuna

import warnings

import os
import sys
sys.path.append(os.path.abspath("src"))

from config import Config
from data import create_folds, load_data, remove_high_missing_cols
from feature_engineering import feature_engineering
from model_selection import objective
from models import EnsembleModel
from utils import apply_best_params, load_and_predict, prepare_submission, save_models_and_features
warnings.filterwarnings("ignore")

pd.options.display.max_rows = 500
pd.options.display.max_rows = 500

def main():
    # Load data
    train, test = load_data('Train.csv', 'Test.csv')

    # Remove columns with too many missing values
    train, test = remove_high_missing_cols(train, test, threshold=Config.missing_threshold)

    # Create GroupKFold
    train = create_folds(train)

    # Feature engineering
    train, test, features = feature_engineering(train, test)

    ensemble = EnsembleModel(top_features=Config.top_features,
                            corr_threshold=Config.corr_threshold,
                            clip_threshold=Config.clip_threshold)

    # Run Optuna to find best hyperparameters
    study = optuna.create_study(direction="minimize")
    study.set_user_attr("ensemble", ensemble)
    study.set_user_attr("X", train[features])
    study.set_user_attr("y", train[Config.target_col].values)
    study.set_user_attr("folds", train['folds'].values)
    study.optimize(objective, n_trials=100)

    # Apply best hyperparameters
    apply_best_params(study.best_params)

    # Define final models using best hyperparameters
    best_models = {
        "cat": CatBoostRegressor(**Config.cat_params),
        "lgb": LGBMRegressor(**Config.lgb_params),
        "xgb": XGBRegressor(**Config.xgb_params),
        "lasso": Lasso(**Config.lasso_params),
        "svr": SVR(**Config.svr_params)
    }

    # Fit final ensemble and predict
    ensemble.fit_final(train[features], train[Config.target_col].values, best_models)
    final_test_preds = ensemble.predict(test[features])

    # Save everything
    save_models_and_features(best_models, ensemble.reduced_features, study.best_value)
    prepare_submission(test, final_test_preds)

    # Load best models and save submission
    # final_test_pred = load_and_predict(test, rmse=27.84)
    # prepare_submission(test, final_test_pred)

if __name__ == "__main__":
    main()



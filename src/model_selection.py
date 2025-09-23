import os
import json
import joblib
import optuna
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR

from config import Config

best_rmse = float("inf")  # track best score

# Function to create model dict given a trial
def create_model_dict(trial):
    return {
        "cat": CatBoostRegressor(
            iterations=1000,
            learning_rate=trial.suggest_float('cat_lr', 0.01, 0.1, log=True),
            depth=trial.suggest_int('cat_depth', 4, 10),
            random_seed=Config.random_state,
            verbose=0,
            early_stopping_rounds=250
        ),
        "lgb": LGBMRegressor(
            n_estimators=100,
            learning_rate=trial.suggest_float('lgb_lr', 0.01, 0.1, log=True),
            max_depth=trial.suggest_int('lgb_depth', 3, 12),
            random_state=Config.random_state,
            verbosity=-1
        ),
        "xgb": XGBRegressor(
            n_estimators=100,
            learning_rate=trial.suggest_float('xgb_lr', 0.01, 0.1, log=True),
            max_depth=trial.suggest_int('xgb_depth', 3, 12),
            random_state=Config.random_state,
            objective='reg:squarederror'
        ),
        "lasso": Lasso(alpha=trial.suggest_float('lasso_alpha', 1e-4, 1.0, log=True),
                       random_state=Config.random_state),
        "svr": SVR(
            C=trial.suggest_float('svr_C', 0.1, 10.0, log=True),
            epsilon=trial.suggest_float('svr_eps', 0.01, 1.0, log=True)
        )
    }

# Optuna objective function
def objective(trial):
    ensemble = trial.study.user_attrs["ensemble"]
    X = trial.study.user_attrs["X"]
    y = trial.study.user_attrs["y"]
    folds = trial.study.user_attrs["folds"]

    models = create_model_dict(trial)
    _, _, mean_rmse = ensemble.cross_val_fit(
        X,
        y,
        folds,
        models=models
    )

    return mean_rmse





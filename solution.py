import os
import sys
import warnings

import pandas as pd
import optuna

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath("src"))

# Import project-specific modules
from config import Config
from data import create_folds, load_data, remove_high_missing_cols
from feature_engineering import feature_engineering
from model_selection import objective
from models import EnsembleModel
from utils import apply_best_params, load_and_predict, prepare_submission, save_models_and_features
import argparse
# Suppress warnings
warnings.filterwarnings("ignore")

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='default')
    parser.add_argument("--missing_threshold", type=float, default=None)
    parser.add_argument("--top_features", type=int, default=None)
    parser.add_argument("--corr_threshold", type=float, default=None)
    parser.add_argument("--clip_threshold", type=float, default=None)
    parser.add_argument("--models", type=str, default='cat,lgb,xgb,lasso,svr')
    parser.add_argument("--feature_selection_method", type=str, default='catboost')
    parser.add_argument("--drop_location", default=False, action='store_true')
    parser.add_argument("--augment_date", default=False, action='store_true')
    parser.add_argument("--use_unify", default=False, action='store_true')
    parser.add_argument("--use_cloud_diff", default=False, action='store_true')
    parser.add_argument("--scale_target", default=False, action='store_true')
    parser.add_argument("--clip_target", default=False, action='store_true')
    return parser.parse_args()

def main(args):
    # Load data
    train, test = load_data('dataset/Train.csv', 'dataset/Test.csv')

    # Remove columns with too many missing values
    missing_threshold = args.missing_threshold if args.missing_threshold else Config.missing_threshold
    train, test = remove_high_missing_cols(train, test, threshold=missing_threshold)

    # Create GroupKFold
    train = create_folds(train)

    # Feature engineering
    train, test, features = feature_engineering(train, test, drop_location=args.drop_location, augment_date=args.augment_date, \
                                            use_unify=args.use_unify, use_cloud_diff=args.use_cloud_diff)

    ensemble = EnsembleModel(top_features=args.top_features,
                            corr_threshold=args.corr_threshold,
                            clip_threshold=args.clip_threshold,
                            feature_selection_method=args.feature_selection_method,
                            scale_target=args.scale_target,
                            clip_target=args.clip_target)
    ensemble._feature_selection(train[features], train[Config.target_col].values)

    # Run Optuna to find best hyperparameters
    study = optuna.create_study(direction="minimize")
    study.set_user_attr("ensemble", ensemble)
    study.set_user_attr("X", train[features])
    study.set_user_attr("y", train[Config.target_col].values)
    study.set_user_attr("folds", train['folds'].values)
    study.set_user_attr("models", args.models)
    study.optimize(objective, n_trials=5)

    # Apply best hyperparameters
    apply_best_params(study.best_params, args.models)

    # Define final models using best hyperparameters
    best_models = {}
    if "cat" in args.models:
        best_models["cat"] = CatBoostRegressor(**Config.cat_params)
    if "lgb" in args.models:
        best_models["lgb"] = LGBMRegressor(**Config.lgb_params)
    if "xgb" in args.models:
        best_models["xgb"] = XGBRegressor(**Config.xgb_params)
    if "lasso" in args.models:
        best_models["lasso"] = Lasso(**Config.lasso_params)
    if "svr" in args.models:
        best_models["svr"] = SVR(**Config.svr_params)

    # Fit final ensemble and predict
    ensemble.fit_final(train[features], train[Config.target_col].values, best_models)
    final_test_preds = ensemble.predict(test[features])

    # Save everything
    save_models_and_features(best_models, ensemble.reduced_features, study.best_value, save_dir=f'weights/{args.exp_name}')
    prepare_submission(test, final_test_preds, save_path=f'output/submission_{study.best_value:.2f}_{args.exp_name}.csv')


if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)
    os.makedirs('weights', exist_ok=True)
    args = arg_parser()
    main(args)



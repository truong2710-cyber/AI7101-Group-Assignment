import os

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR

import matplotlib.pyplot as plt
import seaborn as sns

import optuna

import joblib

import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_rows = 500
pd.options.display.max_rows = 500


class Config:
    target_col = 'pm2_5'
    n_splits = 4
    random_state = 42
    id_col = 'id'
    missing_threshold = 0.7
    top_features = 70
    clip_threshold = 0.97

    # Default hyperparameters
    cat_params = {
        'iterations': 1000,
        'learning_rate': 0.03,
        'depth': 6,
        'eval_metric': 'RMSE',
        'random_seed': random_state,
        'early_stopping_rounds': 250,
        'verbose': 100
    }
    lgb_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': -1,
        'random_state': random_state,
        'verbosity': -1
    }
    xgb_params = {
        'n_estimators': 100,
        'learning_rate': 0.3,
        'max_depth': 6,
        'random_state': random_state,
        'objective': 'reg:squarederror'
    }
    lasso_params = {'alpha': 0.001, 'random_state': random_state}
    svr_params = {'C': 1.0, 'epsilon': 0.1}

# Create GroupKFold
def create_folds(data):
    data['folds'] = -1
    gkf = GroupKFold(n_splits=Config.n_splits)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X=data, groups=data['city']), start=1):
        data.loc[val_idx, 'folds'] = fold
    return data

# Feature engineering
def feature_engineering(train, test):
    le = LabelEncoder()
    data = pd.concat([train, test])
    data['location'] = data['site_latitude'].astype('str') + '_' + data['site_longitude'].astype('str')
    data = data.sort_values(by = ['city','location', 'date', 'hour'])
    categorical_cols = data.select_dtypes(include='object').columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['date', 'id', 'city', 'country']]
    print(f'Categorical columns: {categorical_cols}')

    # Date features
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = data['date'].dt.month
    data['week'] = data['date'].dt.isocalendar().week
    data['day'] = data['date'].dt.day
    data['dayofweek'] = data['date'].dt.dayofweek
    data['is_weekend'] = data['dayofweek'].isin([5,6]).astype(int)

    numerical_cols = data.select_dtypes(exclude='object').columns.tolist()
    numerical_cols.remove(Config.target_col)
    numerical_cols.remove('folds')
    numerical_cols.remove('hour')
    numerical_cols.remove('site_latitude')
    numerical_cols.remove('site_longitude') 
    print(f'Numerical columns: {numerical_cols}')

    # Fill in missing values by forward and backward fill within each city and location
    nan_cols = [col for col in numerical_cols if data[col].isnull().sum() > 0 and col not in [Config.target_col, "folds"]]
    for col in nan_cols:
        for col in nan_cols:
            data[col] = (
                data.groupby(["city", "location"])[col]
                    .transform(lambda x: x.ffill().bfill())
                    .fillna(data[col].median())  # global fallback
                )

    # Encode categorical features
    for col in categorical_cols + ['date']:
        data[col] = le.fit_transform(data[col])

    # Split back into train and test
    train  = data[data['id'].isin(train['id'].unique())]
    test = data[data['id'].isin(test['id'].unique())]

    features = [col for col in data.columns if col not in 
                [Config.target_col, Config.id_col, 'folds', 'country', 'city', 'site_id', 'site_latitude', 'site_longitude']]
  
    return train, test, features

def drop_highly_correlated_features(data, features, threshold=0.9):
    corr_matrix = data[features].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    print(f'Dropping {len(to_drop)} highly correlated features: {to_drop}')
    reduced_features = [feature for feature in features if feature not in to_drop]
    return reduced_features

def prepare_submission(test, best_test_pred):
    test['pm2_5'] = best_test_pred
    submission = test[[Config.id_col, 'pm2_5']]
    submission.to_csv('submission.csv', index=False)
    submission.head()

def load_and_predict(test, reduced_features, rmse=27.84, model_names=None):
    """
    Load best models, make predictions on test set, and ensemble them.

    Args:
        test (pd.DataFrame): Test dataset.
        reduced_features (list): List of features to use for prediction.
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

    # Prepare test predictions
    test_preds = np.zeros((len(test), len(models)))

    # Predict
    for i, (name, model) in enumerate(models.items()):
        test_preds[:, i] = model.predict(test[reduced_features])

    # Ensemble with equal weights
    final_test_pred = np.mean(test_preds, axis=1)

    return final_test_pred

def cross_val_predict(train, test, reduced_features, Config, model_class, model_params=None):
    """
    Perform cross-validated training using a given model and return OOF and test predictions.

    Args:
        train (pd.DataFrame): Training dataset containing 'folds' column and target column.
        test (pd.DataFrame): Test dataset.
        reduced_features (list): List of features to use for training.
        Config: Configuration object with attributes:
            - n_splits (int)
            - target_col (str)
            - clip_threshold (float)
        model_class: Model class to instantiate (e.g., CatBoostRegressor, LGBMRegressor).
        model_params (dict): Parameters to pass to the model_class.

    Returns:
        oof_predictions (np.ndarray): Out-of-fold predictions for training set.
        test_predictions (np.ndarray): Averaged predictions for test set.
        mean_rmse (float): Mean RMSE across all folds.
        fold_rmse_list (list): RMSE per fold.
    """
    if model_params is None:
        model_params = {}

    oof_predictions = np.zeros(len(train))
    test_predictions = np.zeros(len(test))
    fold_rmse_list = []

    for fold in range(1, Config.n_splits + 1):
        print(f'Training fold {fold}...')
        
        train_set = train[train['folds'] != fold].copy()
        val_set = train[train['folds'] == fold].copy()

        # Clip target in training set to remove outliers
        clip_value = train_set[Config.target_col].quantile(Config.clip_threshold)
        train_set[Config.target_col] = np.where(
            train_set[Config.target_col] >= clip_value, 
            clip_value, 
            train_set[Config.target_col]
        )

        # Instantiate and train the model
        model = model_class(**model_params)
        model.fit(
            train_set[reduced_features], 
            train_set[Config.target_col], 
            eval_set=(val_set[reduced_features], val_set[Config.target_col]) if hasattr(model, 'eval_set') else None,
            verbose=False if hasattr(model, 'verbose') else None
        )

        # Predict
        oof_predictions[val_set.index] = model.predict(val_set[reduced_features])
        test_predictions += model.predict(test[reduced_features]) / Config.n_splits

        # Compute fold RMSE
        fold_rmse = root_mean_squared_error(val_set[Config.target_col], oof_predictions[val_set.index])
        print(f'Fold {fold} RMSE: {fold_rmse}')
        fold_rmse_list.append(fold_rmse)

        print('-' * 112)

    mean_rmse = np.mean(fold_rmse_list)
    print(f'Mean RMSE across all folds: {mean_rmse}')

    return test_predictions, mean_rmse

# Load data
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

# Remove columns with too many missing values
train = train.loc[:, train.isnull().mean() < Config.missing_threshold]
test = test.loc[:, test.isnull().mean() < Config.missing_threshold]  

train = create_folds(train)

train, test, features = feature_engineering(train, test)

# Feature selection
# Embedded method

# Initialize CatBoost Regressor
model = CatBoostRegressor(**Config.cat_params)

train_set  = train[train['folds'].isin([1.0, 3.0, 4.0])]
val_set = train[train['folds'].isin([2.0])]

# Train the model on the training data
model.fit(train_set[features], train_set[Config.target_col], eval_set=(val_set[features], val_set[Config.target_col]), verbose=100,  early_stopping_rounds=250)

# Get feature importance
feature_importances = model.get_feature_importance(prettified=True)

# Display the top features
print(feature_importances)

# Select top K features based on feature importance
top_features = feature_importances.head(Config.top_features)['Feature Id'].tolist()
print(f'Top {Config.top_features} features: {top_features}')

# Drop highly correlated features
reduced_features = drop_highly_correlated_features(train, top_features, threshold=0.9)

print(f'Reduced features: {reduced_features}')

# Cross Validation with GroupKFold
# Perform cross-validation with GroupKFold
test_preds, mean_rmse = cross_val_predict(
    train, 
    test, 
    reduced_features, 
    Config, 
    model_class=CatBoostRegressor, 
    model_params=Config.cat_params
)

print(f'Mean RMSE across all folds: {mean_rmse}')


# Model ensemble
# Global variable to store best test predictions
best_test_pred = None
best_rmse = float("inf")  # track best score
best_models = None

def objective(trial):
    global best_test_pred, best_rmse, best_models

    # Suggest hyperparameters for each model
    cat_params = {
        'iterations': 1000,
        'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('cat_depth', 4, 10),
        'early_stopping_rounds': 250,
        'random_seed': Config.random_state,
        'verbose': 0
    }
    lgb_params = {
        'n_estimators': 100,
        'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('lgb_max_depth', 3, 12),
        'random_state': Config.random_state,
        'verbosity': -1
    }
    xgb_params = {
        'n_estimators': 100,
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
        'random_state': Config.random_state,
        'objective': 'reg:squarederror'
    }
    lasso_params = {'alpha': trial.suggest_float('lasso_alpha', 1e-4, 1.0, log=True), 'random_state': Config.random_state}
    svr_params = {
        'C': trial.suggest_float('svr_C', 0.1, 10.0, log=True),
        'epsilon': trial.suggest_float('svr_epsilon', 0.01, 1.0, log=True)
    }

    # Cross-validation
    oof_predictions = np.zeros(len(train))
    fold_rmse_list = []
    fold_test_preds = []

    for fold in range(1, Config.n_splits + 1):
        train_set = train[train['folds'] != fold].copy()
        val_set = train[train['folds'] == fold].copy()

        # Clip target
        clip_val = train_set[Config.target_col].quantile(Config.clip_threshold)
        train_set[Config.target_col] = np.where(
            train_set[Config.target_col] >= clip_val, clip_val, train_set[Config.target_col]
        )

        # Define models 
        models = {
            "cat": CatBoostRegressor(**cat_params),
            "lgb": LGBMRegressor(**lgb_params),
            "xgb": XGBRegressor(**xgb_params),
            "lasso": Lasso(**lasso_params),
            "svr": SVR(**svr_params)
        }

        val_preds = np.zeros((len(val_set), len(models)))
        test_preds = np.zeros((len(test), len(models)))

        # Train each model and predict
        for i, (name, model) in enumerate(models.items()):
            model.fit(train_set[reduced_features], train_set[Config.target_col])
            val_preds[:, i] = model.predict(val_set[reduced_features])
            test_preds[:, i] = model.predict(test[reduced_features])

        # Ensemble with equal weights
        oof_predictions[val_set.index] = np.mean(val_preds, axis=1)
        fold_test_preds.append(np.mean(test_preds, axis=1))

        # Fold RMSE
        fold_rmse = root_mean_squared_error(val_set[Config.target_col], oof_predictions[val_set.index])
        fold_rmse_list.append(fold_rmse)
    
    mean_rmse = np.mean(fold_rmse_list)
    mean_test_pred = np.mean(fold_test_preds, axis=0)

    # Save best test prediction
    if mean_rmse < best_rmse:
        best_rmse = mean_rmse
        best_test_pred = mean_test_pred
        best_models = {name: model for name, model in models.items()}

    return mean_rmse

# Run Optuna study to optimize hyperparameters
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)  # adjust n_trials

print("Best hyperparameters:", study.best_params)
print("Best RMSE:", study.best_value)

# Save best models
folder_name = f"./weights/{best_rmse:.2f}"  # keep 4 decimal places
os.makedirs(folder_name, exist_ok=True)

for name, model in best_models.items():
    joblib.dump(model, f"{folder_name}/best_model_{name}.pkl")

# Prepare test submission
prepare_submission(test, best_test_pred)

# Load best models and save submission
# final_test_pred = load_and_predict(test, reduced_features, rmse=27.84)
# prepare_submission(test, final_test_pred)



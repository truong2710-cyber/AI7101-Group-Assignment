class Config:
    """Configuration parameters for the project."""
    target_col = 'pm2_5'
    n_splits = 4
    random_state = 42
    id_col = 'id'
    missing_threshold = 0.7
    top_features = 70
    clip_threshold = 0.97
    corr_threshold = 0.9

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

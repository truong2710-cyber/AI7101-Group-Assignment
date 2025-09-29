# feature engineering
python solution.py --exp_name no_location --missing_threshold 0.7 --top_features 70 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost --not_use_location
                    
python solution.py --exp_name no_date --missing_threshold 0.7 --top_features 70 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost --not_use_date


# feature selection methods
python solution.py --exp_name catboost --missing_threshold 0.7 --top_features 70 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost

python solution.py --exp_name anova --missing_threshold 0.7 --top_features 70 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method anova

python solution.py --exp_name lasso --missing_threshold 0.7 --top_features 70 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method lasso

python solution.py --exp_name permutation --missing_threshold 0.7 --top_features 70 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method permutation



# hyperparameter tuning
python solution.py --exp_name missing_threshold_1.0 --missing_threshold 1.0 --top_features 70 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost


python solution.py --exp_name top_features_40 --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost


python solution.py --exp_name corr_threshold_0.6 --missing_threshold 0.7 --top_features 40 --corr_threshold 0.6 \
                    --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost




# only one model
python solution.py --exp_name cat_only --missing_threshold 0.7 --top_features 70 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models cat --feature_selection_method catboost

python solution.py --exp_name lgb_only --missing_threshold 0.7 --top_features 70 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models lgb --feature_selection_method catboost

python solution.py --exp_name xgb_only --missing_threshold 0.7 --top_features 70 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models xgb --feature_selection_method catboost

python solution.py --exp_name lasso_only --missing_threshold 0.7 --top_features 70 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models lasso --feature_selection_method catboost

python solution.py --exp_name svr_only --missing_threshold 0.7 --top_features 70 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models svr --feature_selection_method catboost




# ensemble models
python solution.py --exp_name cat_lgb_only --missing_threshold 0.7 --top_features 70 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models cat,lgb --feature_selection_method catboost

python solution.py --exp_name cat_lgb_xgb_only --missing_threshold 0.7 --top_features 70 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models cat,lgb,xgb --feature_selection_method catboost

python solution.py --exp_name cat_lgb_xgb_lasso_only --missing_threshold 0.7 --top_features 70 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models cat,lgb,xgb,lasso --feature_selection_method catboost

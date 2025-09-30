python solution.py --exp_name no_clip_target_no_scale_target --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost --not_use_location

python solution.py --exp_name clip_target --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost --not_use_location --clip_target

python solution.py --exp_name scale_target --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost --not_use_location --scale_target

python solution.py --exp_name clip_target_scale_target --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
                    --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost --not_use_location --clip_target --scale_target

# # feature engineering
# python solution.py --exp_name no_location --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost --not_use_location
                    
# python solution.py --exp_name no_date --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost --not_use_date

# python solution.py --exp_name no_unify --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost --not_use_unify

# python solution.py --exp_name no_cloud_diff --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost --not_use_cloud_diff


# # feature selection methods
# python solution.py --exp_name catboost --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost

# python solution.py --exp_name anova --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method anova

# python solution.py --exp_name lasso --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method lasso

# python solution.py --exp_name permutation --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method permutation



# # hyperparameter tuning
# python solution.py --exp_name missing_threshold_1.0 --missing_threshold 1.0 --top_features 40 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost


# python solution.py --exp_name top_features_70 --missing_threshold 0.7 --top_features 70 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost


# python solution.py --exp_name top_features_20 --missing_threshold 0.7 --top_features 20 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost


# python solution.py --exp_name corr_threshold_0.6 --missing_threshold 0.7 --top_features 40 --corr_threshold 0.6 \
#                     --clip_threshold 0.97 --models cat,lgb,xgb,lasso,svr --feature_selection_method catboost




# # only one model
# python solution.py --exp_name cat_only --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models cat --feature_selection_method catboost

# python solution.py --exp_name lgb_only --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models lgb --feature_selection_method catboost

# python solution.py --exp_name xgb_only --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models xgb --feature_selection_method catboost

# python solution.py --exp_name lasso_only --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models lasso --feature_selection_method catboost

# python solution.py --exp_name svr_only --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models svr --feature_selection_method catboost




# # ensemble models
# python solution.py --exp_name cat_lgb_only --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models cat,lgb --feature_selection_method catboost

# python solution.py --exp_name cat_lgb_xgb_only --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models cat,lgb,xgb --feature_selection_method catboost

# python solution.py --exp_name cat_lgb_xgb_lasso_only --missing_threshold 0.7 --top_features 40 --corr_threshold 0.9 \
#                     --clip_threshold 0.97 --models cat,lgb,xgb,lasso --feature_selection_method catboost

python solution.py --exp_name baseline --missing_threshold 1.0 --top_features 1000 --corr_threshold 1.0 \
                    --clip_threshold 1.0 --models svr --feature_selection_method all

# feature selection methods
python solution.py --exp_name feat_select_catboost --missing_threshold 1.0 --top_features 40 --corr_threshold 1.0 \
                    --clip_threshold 1.0 --models svr --feature_selection_method catboost \

python solution.py --exp_name feat_select_anova --missing_threshold 1.0 --top_features 40 --corr_threshold 1.0 \
                    --clip_threshold 1.0 --models svr --feature_selection_method anova \

python solution.py --exp_name feat_select_lasso --missing_threshold 1.0 --top_features 40 --corr_threshold 1.0 \
                    --clip_threshold 1.0 --models svr --feature_selection_method lasso \

python solution.py --exp_name feat_select_permutation --missing_threshold 1.0 --top_features 40 --corr_threshold 1.0 \
                    --clip_threshold 1.0 --models svr --feature_selection_method permutation \


# feature engineering
python solution.py --exp_name feat_eng_drop_location --missing_threshold 1.0 --top_features 1000 --corr_threshold 1.0 \
                    --clip_threshold 1.0 --models svr --feature_selection_method all \
                    --drop_location

python solution.py --exp_name feat_eng_augmented_date --missing_threshold 1.0 --top_features 1000 --corr_threshold 1.0 \
                    --clip_threshold 1.0 --models svr --feature_selection_method all \
                    --augment_date

python solution.py --exp_name feat_eng_unify --missing_threshold 1.0 --top_features 1000 --corr_threshold 1.0 \
                    --clip_threshold 1.0 --models svr --feature_selection_method all \
                    --use_unify

python solution.py --exp_name feat_eng_cloud_diff --missing_threshold 1.0 --top_features 1000 --corr_threshold 1.0 \
                    --clip_threshold 1.0 --models svr --feature_selection_method all \
                    --use_cloud_diff



# ensemble models
python solution.py --exp_name model_svr_cat --missing_threshold 1.0 --top_features 1000 --corr_threshold 1.0 \
                    --clip_threshold 1.0 --models svr,cat --feature_selection_method all


python solution.py --exp_name model_svr_cat_lgb --missing_threshold 1.0 --top_features 1000 --corr_threshold 1.0 \
                    --clip_threshold 1.0 --models svr,cat,lgb --feature_selection_method top_k
                    
python solution.py --exp_name model_svr_cat_lgb_xgb --missing_threshold 1.0 --top_features 1000 --corr_threshold 1.0 \
                    --clip_threshold 1.0 --models svr,cat,lgb,xgb --feature_selection_method top_k
                    
python solution.py --exp_name model_svr_cat_lgb_xgb_lasso --missing_threshold 1.0 --top_features 1000 --corr_threshold 1.0 \
                    --clip_threshold 1.0 --models svr,cat,lgb,xgb,lasso --feature_selection_method top_k

# Hyperparameter tuning
python solution.py --exp_name tunning_top_features_40 --missing_threshold 1.0 --top_features 40 --corr_threshold 1.0 \
                    --clip_threshold 1.0 --models svr --feature_selection_method permutation \

python solution.py --exp_name tunning_top_features_20 --missing_threshold 1.0 --top_features 20 --corr_threshold 1.0 \
                    --clip_threshold 1.0 --models svr --feature_selection_method permutation \

python solution.py --exp_name tunning_missing_threshold_07 --missing_threshold 0.7 --top_features 1000 --corr_threshold 1.0 \
                    --clip_threshold 1.0 --models svr --feature_selection_method all     

python solution.py --exp_name tunning_corr_threshold_09 --missing_threshold 1.0 --top_features 1000 --corr_threshold 0.9 \
                    --clip_threshold 1.0 --models svr --feature_selection_method all

python solution.py --exp_name tunning_corr_threshold_07 --missing_threshold 1.0 --top_features 1000 --corr_threshold 0.7 \
                    --clip_threshold 1.0 --models svr --feature_selection_method all


# target preprocessing
python solution.py --exp_name clip_target --missing_threshold 1.0 --top_features 1000 --corr_threshold 1.0 \
                    --clip_threshold 0.97 --models svr --feature_selection_method all --clip_target

python solution.py --exp_name scale_target --missing_threshold 1.0 --top_features 1000 --corr_threshold 1.0 \
                    --clip_threshold 1.0 --models svr --feature_selection_method all --scale_target

python solution.py --exp_name clip_target_scale_target --missing_threshold 1.0 --top_features 1000 --corr_threshold 1.0 \
                    --clip_threshold 0.97 --models svr --feature_selection_method all --clip_target --scale_target


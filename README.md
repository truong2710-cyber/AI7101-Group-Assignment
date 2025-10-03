# African Air Quality Prediction
Final project - AI7101: Machine Learning with Python @ MBZUAI, Fall 2025

[[Slides](assets/AI7101%20Project.pdf) | [Report](assets/AI7101%20Report.pdf)]

Competition on [Zindi](https://zindi.africa/competitions/airqo-african-air-quality-prediction-challenge)

## Installing dependencies
`pip install -r requirements.txt`

## File structure
`dataset`: contains the train and test sets in 2 .csv files.

`eda`: contains a notebook for EDA, running will create a new folder named `imgs` to store the result figures.

`src`: main structure of the project.

`output`: stores .csv files per run for submission.

`solution.py`: main file for execution of the whole ML pipeline, including preparing a file for submission.

## Script for training, validation, and producing test file
```bash
python solution.py \
  --exp_name <your_exp_name> \
  --missing_threshold <missing_threshold> \
  --top_features <top_features> \
  --corr_threshold <corr_threshold> \
  --clip_threshold <clip_threshold> \
  --models <models> \
  --feature_selection_method <feature_selection_method> \
  --drop_location <True/False> \
  --augment_date <True/False> \
  --use_unify <True/False> \
  --use_cloud_diff <True/False> \
  --scale_target <True/False> \
  --clip_target <True/False>
```

Example
```bash
python solution.py \
  --exp_name baseline_pm25 \
  --missing_threshold 1.0 \
  --top_features 40 \
  --corr_threshold 0.9 \
  --clip_threshold 0.97 \
  --models cat,lgb,xgb,lasso,svr \
  --feature_selection_method catboost \
  --drop_location False \
  --augment_date True \
  --use_unify False \
  --use_cloud_diff False \
  --scale_target False \
  --clip_target True
```
Explanation of variables:

`--exp_name`: name of your experiment to be saved

`--missing_threshold`: threshold to eliminate null columns

`--top_features`: number of features kept after feature selection

`--corr_threshold`: correlation threshold to eliminate features

`--clip_threshold`: target variable quantile to be clipped, used with `--clip_target`

`--models`: list of models to be ensembled, a comma seperated list such as 'cat,lgb,xgb,lasso,svr'

`--feature_selection_method`: one of 'catboost','anova','lasso','permutation','all'

`--drop_location`, `--augment_date`, `--use_unify`, `--use_cloud_diff`: feature engineering methods

`--scale_target`: whether to log-scale the target variable

`--clip_target`: whether to clip the target variable by log scale

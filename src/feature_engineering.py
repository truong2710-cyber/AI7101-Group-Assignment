import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from collections import Counter
from config import Config

def unify_with_majority(df, new_col='sensor_zenith_angle', decimals=3):
    """Perform filling null and feature unification by majority voting on rounded values.
    Args:
        df (pd.DataFrame): Input dataset.
        new_col (str): The name of the new unified column. Also the source columns should contain this as a substring.
        decimals (int): Number of decimal places to round for majority voting.
    Returns:
        train (pd.DataFrame): Transformed training dataset.
        test (pd.DataFrame): Transformed test dataset.
        features (list): List of features used."""
    src_cols = [c for c in df.columns if new_col in c and c != new_col]
    if not src_cols:
        return df

    # Force numeric for row_majority function
    src_raw = df[src_cols].apply(pd.to_numeric, errors='coerce')
    src_rnd = src_raw.round(decimals)

    def row_majority(i):
        rnd = src_rnd.iloc[i].dropna()
        raw = src_raw.iloc[i].dropna()
        if rnd.empty:
            return np.nan

        cnt = Counter(rnd.values.tolist())
        maxc = max(cnt.values())
        # Take majority values
        tops = sorted([v for v, k in cnt.items() if k == maxc])
        chosen_r = tops[0]

        # Median of original values corresponding to the chosen rounded value
        mask = (src_rnd.iloc[i] == chosen_r) & src_raw.iloc[i].notna()
        cand = src_raw.iloc[i][mask].values
        if cand.size:
            return float(np.median(cand))
        # Fallback: first value
        return float(raw.iloc[0]) if not raw.empty else np.nan

    df[new_col] = [row_majority(i) for i in range(len(df))]

    # Drop source columns
    df = df.drop(columns=src_cols)
    return df

def feature_engineering(train, test, drop_location=False, augment_date=False, use_unify=False, use_cloud_diff=False):
    """Perform feature engineering on train and test datasets.
    Args:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Test dataset.
    Returns:
        train (pd.DataFrame): Transformed training dataset.
        test (pd.DataFrame): Transformed test dataset.
        features (list): List of features used."""
    
    le = LabelEncoder()
    data = pd.concat([train, test])

    if drop_location:
        data = data.sort_values(by = ['city', 'date', 'hour'])
    else:
        data['location'] = data['site_latitude'].astype('str') + '_' + data['site_longitude'].astype('str')
        data = data.sort_values(by = ['city','location', 'date', 'hour'])

    categorical_cols = data.select_dtypes(include='object').columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['date', 'id', 'city', 'country']]
    print(f'Categorical columns: {categorical_cols}')

    # Date features
    data['date'] = pd.to_datetime(data['date'])
    if augment_date:
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
    
    # Unify features with majority voting
    if use_unify:
        unify_cols = ['sensor_zenith_angle', 'sensor_azimuth_angle', 'solar_zenith_angle', 'solar_azimuth_angle','altitude']
        for col in unify_cols:
            data = unify_with_majority(data, new_col=col)

        print(f'Numerical columns before unify: {len(numerical_cols)}')
        numerical_cols = [col for col in numerical_cols if col in data.columns]
        print(f'Numerical columns after unify: {len(numerical_cols)}')

    # Fill in missing values by forward and backward fill within each city and location
    nan_cols = [col for col in numerical_cols if data[col].isnull().sum() > 0 and col not in [Config.target_col, "folds"]]
    for col in nan_cols:
        if drop_location:
            data[col] = (
                data.groupby(["city"])[col]
                    .transform(lambda x: x.ffill().bfill())
                    .fillna(data[col].median())  # global fallback
                )
        else:
            data[col] = (
                data.groupby(["city", "location"])[col]
                        .transform(lambda x: x.ffill().bfill())
                        .fillna(data[col].median())  # global fallback
                    )

    # Special case: Cloud has base vs top pressure/height, which are highly correlated.
    # Create a new feature to capture the difference.
    if use_cloud_diff:
        data['cloud_cloud_diff_pressure'] = data['cloud_cloud_top_pressure'] - data['cloud_cloud_base_pressure']
        data['cloud_cloud_diff_height'] = data['cloud_cloud_top_height'] - data['cloud_cloud_base_height']
        
    # Encode categorical features
    for col in categorical_cols + ['date']:
        data[col] = le.fit_transform(data[col])

    # Split back into train and test
    train  = data[data['id'].isin(train['id'].unique())]
    test = data[data['id'].isin(test['id'].unique())]

    features = [col for col in data.columns if col not in 
                [Config.target_col, Config.id_col, 'folds', 'country', 'city', 'site_id', 'site_latitude', 'site_longitude']]

    return train, test, features

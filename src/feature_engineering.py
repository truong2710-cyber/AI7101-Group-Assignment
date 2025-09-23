# Feature engineering
import pandas as pd
from sklearn.calibration import LabelEncoder

from config import Config


def feature_engineering(train, test):
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

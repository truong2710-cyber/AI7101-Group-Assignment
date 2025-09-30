import pandas as pd
from sklearn.model_selection import GroupKFold

from config import Config


def load_data(train_path='Train.csv', test_path='Test.csv'):
    """Load training and test data from CSV files.
    Args:
        train_path (str): Path to the training data CSV file.
        test_path (str): Path to the test data CSV file.
    Returns:
        train (pd.DataFrame): Loaded training dataset.
        test (pd.DataFrame): Loaded test dataset.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def create_folds(data):
    """Create GroupKFold splits based on 'city'.
    Args:
        data (pd.DataFrame): Dataset with a 'city' column for grouping.
    Returns:
        data (pd.DataFrame): Dataset with an additional 'folds' column indicating fold assignment
    """
    data['folds'] = -1
    gkf = GroupKFold(n_splits=Config.n_splits)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X=data, groups=data['city']), start=1):
        data.loc[val_idx, 'folds'] = fold
    return data

def remove_high_missing_cols(train, test, threshold=Config.missing_threshold):
    """Remove columns with missing values above the specified threshold.
    Args:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Test dataset.
        threshold (float): Proportion threshold to drop columns.
    Returns:
        train (pd.DataFrame): Training dataset with high-missing columns removed.
        test (pd.DataFrame): Test dataset with high-missing columns removed.
    """
    before_removing_columns = train.columns
    null_percentages = train.isnull().mean()
    train = train.loc[:, null_percentages < threshold]
    after_removing_columns = train.columns
    removed_columns = set(before_removing_columns) - set(after_removing_columns)
    if removed_columns:
        removed_null_percentages = null_percentages[list(removed_columns)]
        print(f"Null percentage of removed columns:")
        print(removed_null_percentages)
        print(f"Number of removed columns: {len(removed_columns)}")
        print(f"Columns removed: {list(removed_columns)}")

    if removed_columns:
        # Get columns to keep in test (all columns except the ones removed from train)
        test_columns_to_keep = [col for col in test.columns if col not in removed_columns]
        test = test[test_columns_to_keep]
        print(f"Test columns after removing same columns as train: {len(test.columns)}")
    else:
        print("No columns removed from test (no columns were removed from train)")
        
    return train, test
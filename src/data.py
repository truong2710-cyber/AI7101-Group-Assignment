import pandas as pd
from sklearn.model_selection import GroupKFold

from config import Config


def load_data(train_path='Train.csv', test_path='Test.csv'):
    """Load training and test data from CSV files."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

# Create GroupKFold
def create_folds(data):
    data['folds'] = -1
    gkf = GroupKFold(n_splits=Config.n_splits)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X=data, groups=data['city']), start=1):
        data.loc[val_idx, 'folds'] = fold
    return data

def remove_high_missing_cols(train, test, threshold=Config.missing_threshold):
    """Remove columns with missing values above the specified threshold."""
    train = train.loc[:, train.isnull().mean() < threshold]
    test = test.loc[:, test.isnull().mean() < threshold]  
    return train, test
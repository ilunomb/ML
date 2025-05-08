import numpy as np
from models.constants import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED

def train_val_test_split(X, y, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, shuffle=True, seed=RANDOM_SEED):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios deben sumar 1"

    np.random.seed(seed)
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]

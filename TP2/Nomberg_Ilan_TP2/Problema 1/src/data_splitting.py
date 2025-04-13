import numpy as np
import pandas as pd
from utils import calculate_stats
from preprocessing import handle_missing_values, one_hot_encoding, normalize_df
from models import LogisticRegression
from metrics import f1_score
from tqdm import tqdm

RANDOM_STATE = 42
TARGET = 'Diagnosis'

TRUE_INTERVALS = {
    'CellSize': (60, 80),
    'CellShape': (0.4, 0.58),
    'NucleusDensity': (1.2, 1.6),
    'ChromatinTexture': (20, 25),
    'CytoplasmSize': (20, 34.4),
    'CellAdhesion': (0.4, 0.6),
    'MitosisRate': (2.4, 4),
    'NuclearMembrane': (2, 2.8),
    'GrowthFactor': (52, 68),
    'OxygenSaturation': (78, 84),
    'Vascularization': (4, 6),
    'InflammationMarkers': (30, 50)
}

def train_validate_split(df, validation_size=0.2, random_state=42):
    """
    Splits the given DataFrame into training and validation sets based on the specified ratio.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    validation_size (float): The ratio of the training set.

    Returns:
    pd.DataFrame: The training set.
    pd.DataFrame: The validation set.
    """
    validate = df.sample(frac=validation_size, random_state=random_state)
    train = df.drop(validate.index)
    return train, validate

def cross_validate(df, k=10, random_state=42, lambdas=np.logspace(0, 4, 100)):
    """
    Implements K-fold Cross-Validation following the algorithm in the image.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing both training and validation data.
    k (int): The number of folds.
    lambdas (np.ndarray): The array of lambda values to test.
    metric (Metrics): The metric to use for evaluation.
    L2 (bool): Whether to use L2 regularization.
    training_method (str): The training method ('pinv' or 'gradient_descent').
    
    Returns:
    float: The optimal lambda value.
    list: The list of scores for each lambda value.
    """

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    folds = np.array_split(df, k)
    
    best_lambda = None
    best_score = float('-inf')
    
    for lambda_ in tqdm(lambdas, desc="Cross-Validation Progress"):
        preds = []
        true_vals = []
        
        for i in range(k):
            df_ho = folds[i]
            df_tr = pd.concat([folds[j] for j in range(k) if j != i])

            df_ho_copy = df_ho.copy()
            df_tr_copy = df_tr.copy()

            train_stats = calculate_stats(df_tr_copy)

            df_tr_copy_filled = handle_missing_values(df_tr_copy, true_intervals=TRUE_INTERVALS)
            df_tr_copy_filled_OHE = one_hot_encoding(df_tr_copy_filled)
            df_tr_copy_filled_OHE_normalized, normalized_train_stats = normalize_df(df_tr_copy_filled_OHE, train=True)

            df_ho_copy_filled = handle_missing_values(df_ho_copy, train=False, stats=train_stats, true_intervals=TRUE_INTERVALS, reference_df=df_tr_copy_filled)
            df_ho_copy_filled_OHE = one_hot_encoding(df_ho_copy_filled)
            df_ho_copy_filled_OHE_normalized, _ = normalize_df(df_ho_copy_filled_OHE, train=False, stats=normalized_train_stats)

            X_train = df_tr_copy_filled_OHE_normalized.drop(columns=[TARGET])
            y_train = df_tr_copy_filled_OHE_normalized[TARGET]

            X_val = df_ho_copy_filled_OHE_normalized.drop(columns=[TARGET])
            y_val = df_ho_copy_filled_OHE_normalized[TARGET]

            model = LogisticRegression(l2=lambda_)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)

            preds.append(y_pred)
            true_vals.append(y_val.values)

        preds = np.concatenate(preds)
        true_vals = np.concatenate(true_vals)
        
        loss = f1_score(true_vals, preds)
        
        if loss > best_score:
            best_lambda = lambda_
            best_score = loss
    
    return best_lambda, best_score

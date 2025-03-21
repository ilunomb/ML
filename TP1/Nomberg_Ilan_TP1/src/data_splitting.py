import numpy as np
import pandas as pd
from models import LinearRegression
from metrics import MSE


def train_validate_split(df, validation_size, random_state=42):
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

def k_fold_split(df_train, df_validate, k, random_state=42):
    """
    Splits the given DataFrame into k folds for cross-validation.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    k (int): The number of folds.

    Returns:
    List[pd.DataFrame]: A list of k DataFrames, each containing a different fold.
    """
    df_train = df_train.sample(frac=1, random_state=random_state)  # Shuffle the data
    df_validate = df_validate.sample(frac=1, random_state=random_state)  # Shuffle the data
    return np.array_split(df_train, k), np.array_split(df_validate, k)

def cross_validate(df_train, df_validate, k=10, random_sate=42, lambdas=np.logspace(-2, 2, 100), metric=MSE(), L2=True, training_method='pinv'):
    """
    Performs k-fold cross-validation on the given DataFrame using the specified number of folds.

    Parameters:
    df_train (pd.DataFrame): The training DataFrame.
    df_validate (pd.DataFrame): The validation DataFrame.
    k (int): The number of folds.
    lambdas (np.ndarray): The array of lambda values to test.
    metric (Metrics): The metric to use for evaluation.

    Returns:
    float: The optimal lambda value.
    """
    # Split the data into k folds
    train_folds, validate_folds = k_fold_split(df_train, df_validate, k, random_state=random_sate)
    best_lambda = None
    best_score = float('inf')
    global_scores = []
    for lambda_ in lambdas:
        scores = []
        for i in range(k):
            # Train the model
            if L2:
                model = LinearRegression(train_folds[i].drop(columns='price'), train_folds[i]['price'], L2=lambda_)
                if training_method == 'pinv':
                    model.pinv_fit()
                else:
                    model.gradient_descent_fit()
            elif training_method == 'gradient_descent':
                model = LinearRegression(train_folds[i].drop(columns='price'), train_folds[i]['price'], L1=lambda_)
                model.gradient_descent_fit()
            else:
                print("Can't use L1 regularization with pinv method")
            # Calculate the loss
            loss = model.loss(validate_folds[i].drop(columns='price'), validate_folds[i]['price'], metric)
            scores.append(loss)
        # Calculate the mean loss
        mean_score = np.mean(scores)
        global_scores.append(mean_score)
        # Update the best lambda if this one is better
        if mean_score < best_score:
            best_score = mean_score
            best_lambda = lambda_
    return best_lambda, global_scores
    
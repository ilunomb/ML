import numpy as np
import pandas as pd

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

# def cross_validate(df, k=10, random_state=42, lambdas=np.logspace(-2, 2, 100), metric=MSE(), L2=True, training_method='pinv'):
#     """
#     Implements K-fold Cross-Validation following the algorithm in the image.
    
#     Parameters:
#     df (pd.DataFrame): The input DataFrame containing both training and validation data.
#     k (int): The number of folds.
#     lambdas (np.ndarray): The array of lambda values to test.
#     metric (Metrics): The metric to use for evaluation.
#     L2 (bool): Whether to use L2 regularization.
#     training_method (str): The training method ('pinv' or 'gradient_descent').
    
#     Returns:
#     float: The optimal lambda value.
#     list: The list of scores for each lambda value.
#     """

#     df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
#     folds = np.array_split(df, k)
    
#     best_lambda = None
#     best_score = float('inf')
#     global_scores = []
    
#     for lambda_ in lambdas:
#         scores = []
        
#         for i in range(k):
#             df_ho = folds[i]
#             df_tr = pd.concat([folds[j] for j in range(k) if j != i])

#             df_ho_copy = df_ho.copy()
#             df_tr_copy = df_tr.copy()

#             handle_missing_values(df_tr_copy, "age")
#             handle_missing_values(df_tr_copy, "rooms")

#             df_tr_copy_encoded = one_hot_encoding(df_tr_copy, "area_units")

#             convert_sqft_to_m2(df_tr_copy_encoded, "area_units_sqft", "area")
            
#             df_tr_copy_encoded = feature_engineer(df_tr_copy_encoded)

#             df_tr_copy_normalized, stats_dict_train = normalize_df(df_tr_copy_encoded, train=True)

#             handle_missing_values(df_ho_copy, "age", train=False, stats=stats_dict_train)
#             handle_missing_values(df_ho_copy, "rooms", train=False, stats=stats_dict_train)

#             df_ho_copy_encoded = one_hot_encoding(df_ho_copy, "area_units")

#             convert_sqft_to_m2(df_ho_copy_encoded, "area_units_sqft", "area")
            
#             df_ho_copy_encoded = feature_engineer(df_ho_copy_encoded)

#             df_ho_copy_normalized, _ = normalize_df(df_ho_copy_encoded, train=False, stats=stats_dict_train)

#             if L2:
#                 model = LinearRegression(df_tr_copy_normalized.drop(columns='price'), df_tr_copy_normalized['price'], L2=lambda_)
#                 if training_method == 'pinv':
#                     model.pinv_fit()
#                 else:
#                     model.gradient_descent_fit()
#             elif training_method == 'gradient_descent':
#                 model = LinearRegression(df_tr_copy_normalized.drop(columns='price'), df_tr_copy_normalized['price'], L1=lambda_)
#                 model.gradient_descent_fit()
#             else:
#                 raise ValueError("Can't use L1 regularization with pinv method")
            
#             loss = model.loss(df_ho_copy_normalized.drop(columns='price'), df_ho_copy_normalized['price'], metric)
#             scores.append(loss)

#         mean_score = np.mean(scores)
#         global_scores.append(mean_score)
        
#         if mean_score < best_score:
#             best_score = mean_score
#             best_lambda = lambda_
    
#     return best_lambda, global_scores

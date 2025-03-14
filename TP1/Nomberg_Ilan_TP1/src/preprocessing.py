import pandas as pd

def one_hot_encoding(df, column):
    """
    Performs one-hot encoding on the specified column of the given DataFrame and converts boolean values to 0/1.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column to encode.

    Returns:
    pd.DataFrame: The DataFrame with the encoded column.
    """
    df_encoded = pd.get_dummies(df, columns=[column])
    # Convert boolean columns to 0/1
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)
    return df_encoded

def handle_missing_values(df, column):
    mean = df[column].mean()
    df[column] = df[column].fillna(mean)  # Evita el problema del chained assignment


def normalize_df(df, train = True, stats = {}):
    """
    Normalizes the numeric columns of the given DataFrame by subtracting the mean and dividing by the standard deviation.
    Also saves the mean and standard deviation of each column in a dictionary.
    Skips columns that are categorical (contain only 0s and 1s).

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with normalized numeric columns.
    dict: A dictionary with the mean and standard deviation of each numeric column.
    """
    numeric_df = df.select_dtypes(include=['number'])
    for column in numeric_df.columns:
        unique_values = numeric_df[column].unique()
        if set(unique_values).issubset({0, 1}):
            continue  # Skip normalization for categorical columns
        if train:
            mean = numeric_df[column].mean()
            std = numeric_df[column].std()
            stats[column] = {'mean': mean, 'std': std}
            df[column] = (numeric_df[column] - mean) / std
        else:
            df[column] = (numeric_df[column] - stats[column]['mean']) / stats[column]['std']
    return df, stats

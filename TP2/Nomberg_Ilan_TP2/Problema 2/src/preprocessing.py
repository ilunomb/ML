import pandas as pd
import numpy as np
from tqdm import tqdm

TARGET = 'Diagnosis'

def one_hot_encoding_util(df, column, drop_first = True):

    if len(df[column].unique()) == 1:
        df_encoded = pd.get_dummies(df, columns=[column])
    else:
        df_encoded = pd.get_dummies(df, columns=[column], drop_first=drop_first)
    return df_encoded


def one_hot_encoding(df):
    new_df = df.copy()
    for col in new_df.columns:
        ## check if the column is categorical
        if new_df[col].dtype == 'object' or new_df[col].dtype == 'bool':
            new_df = one_hot_encoding_util(new_df, col)

    for col in new_df.columns:
        if new_df[col].dtype == 'bool':
            new_df[col] = new_df[col].astype(int)

    return new_df
            
def handle_missing_values(df, train=True, stats={}, true_intervals={}, reference_df=None, k=5):
    new_df = replace_invalid_values_with_nan(df, true_intervals)
    df_to_fill = new_df.copy()

    for col in df_to_fill.columns:
        if df_to_fill[col].isnull().sum() > 0:
            if train:
                if df_to_fill[col].dtype == 'object' or df_to_fill[col].dtype == 'bool':
                    mode = df_to_fill[col].mode()[0]
                    df_to_fill[col] = df_to_fill[col].fillna(mode)
                else:
                    mean = df_to_fill[col].mean()
                    df_to_fill[col] = df_to_fill[col].fillna(mean)
            else:
                if df_to_fill[col].dtype == 'object' or df_to_fill[col].dtype == 'bool':
                    mode = stats[col]['mode']
                    df_to_fill[col] = df_to_fill[col].fillna(mode)
                else:
                    df_to_fill[col] = df_to_fill[col].fillna(stats[col]['mean'])

    final_df = fill_with_knn(
        df_to_fill,
        k=5,
        train=train,
        stats=stats,
        original_df=new_df,
        reference_df=reference_df if not train else None  # usar solo si no estamos en training
    )

    return final_df


def fill_with_knn(df, k=5, train=True, stats={}, original_df=None, reference_df=None):
    result_df = df.copy()
    if original_df is None:
        original_df = df
    if train:
        reference_df = df  # cuando estamos entrenando, usamos el mismo DF como referencia

    # Columnas categóricas y numéricas
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns

    # Normalizar referencia
    df_normalized = df.copy()
    ref_normalized = reference_df.copy()
    for col in numerical_cols:
        if col == TARGET or set(df[col].unique()).issubset({0, 1}):
            continue
        if train:
            mean = df[col].mean()
            std = df[col].std()
            stats[col] = {'mean': mean, 'std': std}
        else:
            mean = stats[col]['mean']
            std = stats[col]['std']
        df_normalized[col] = (df[col] - mean) / std
        ref_normalized[col] = (reference_df[col] - mean) / std

    # Convertir a NumPy
    ref_data = ref_normalized[numerical_cols].values
    target_data = df_normalized[numerical_cols].values

    # Imputar por fila
    for idx in tqdm(original_df.index[original_df.isnull().any(axis=1)]):
        row = target_data[df.index.get_loc(idx)]
        valid_mask = ~np.isnan(row)

        if not np.any(valid_mask):
            continue

        diff = ref_data[:, valid_mask] - row[valid_mask]
        distances = np.sqrt(np.sum(diff ** 2, axis=1))

        neighbor_pos = np.argpartition(distances, k)[:k]
        neighbor_idx = reference_df.index[neighbor_pos]

        nan_columns = original_df.columns[original_df.loc[idx].isnull()]
        for col in nan_columns:
            neighbor_values = reference_df.loc[neighbor_idx, col].dropna()
            if len(neighbor_values) > 0:
                if col in numerical_cols:
                    result_df.loc[idx, col] = neighbor_values.mean()
                else:
                    result_df.loc[idx, col] = neighbor_values.mode()[0]

    return result_df


def replace_invalid_values_with_nan(df, true_intervals):
    """
    Replaces values outside specified intervals with NaN, preserving row structure.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    true_intervals (dict): A dictionary where keys are column names and values are tuples (min, max).

    Returns:
    pd.DataFrame: The DataFrame with out-of-bound values replaced with NaN.
    """
    new_df = df.copy()
    for column, (min_val, max_val) in true_intervals.items():
        mask = (new_df[column] < min_val) | (new_df[column] > max_val)
        new_df.loc[mask, column] = np.nan
    
    return new_df

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
    normalize_df = df.copy()
    for column in numeric_df.columns:
        unique_values = numeric_df[column].unique()
        if set(unique_values).issubset({0, 1}) or column == TARGET:
            continue
        if train:
            mean = numeric_df[column].mean()
            std = numeric_df[column].std()
            stats[column] = {'mean': mean, 'std': std}
            normalize_df[column] = (numeric_df[column] - mean) / std
        else:
            normalize_df[column] = (numeric_df[column] - stats[column]['mean']) / stats[column]['std']
    return normalize_df, stats

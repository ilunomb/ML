import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

def convert_sqft_to_m2(df, typeColumn, valueColumn):
    for index, row in df.iterrows():
        if row[typeColumn] == True:
            df.at[index, valueColumn] = df.at[index, valueColumn] * 0.092903

def plot_pairplot(df):
    """
    Generates a pair plot for the given DataFrame to visualize relationships
    between numerical columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    None
    """
    # Ensure the dataframe contains numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.shape[1] < 2:
        print("Not enough numerical columns to create a pair plot.")
        return

    sns.pairplot(numeric_df, diag_kind='kde')  # Use kde for diagonal plots
    plt.show()

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
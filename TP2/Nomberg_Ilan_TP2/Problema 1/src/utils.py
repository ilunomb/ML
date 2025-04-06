import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

TARGET = 'Diagnosis'

def plot_pairplot(df):
    """
    Generates a pair plot for the given DataFrame to visualize relationships
    between numerical columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    None
    """

    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.shape[1] < 2:
        print("Not enough numerical columns to create a pair plot.")
        return

    sns.pairplot(numeric_df, diag_kind='kde', hue=TARGET)
    plt.show()

def calculate_stats(df):
    stats = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            stats[column] = {
                'mode': df[column].mode()[0]
            }
        else:
            stats[column] = {
                'mean': df[column].mean(),
                'std': df[column].std()
            }
    return stats
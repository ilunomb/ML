import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

TARGET = 'Diagnosis'

def plot_pairplot(df, title="Pairplot of Features"):
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
    
    # Your existing pairplot
    pairplot = sns.pairplot(numeric_df, diag_kind='kde', hue=TARGET)

    # Set a better-looking title using the fig object
    plt.suptitle(title, fontsize=50, weight='bold')

    # Tighter layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Avoid overlap with title

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

def correlation_with_target(df, target_column, plot=True, top_n=None, cmap="twilight"):
    # Filtrar columnas numéricas
    numeric_df = df.select_dtypes(include=['float64', 'int64']).copy()
    
    # Asegurar que el target esté en el DataFrame
    if target_column not in numeric_df.columns:
        numeric_df[target_column] = df[target_column]
    
    # Calcular correlaciones
    correlations = numeric_df.corr()[target_column].drop(target_column)

    # Ordenar por valor absoluto (más fuertes primero)
    sorted_corr = correlations.sort_values(key=abs, ascending=True)

    # Mostrar gráfico si se pide
    if plot:
        data_to_plot = sorted_corr if top_n is None else sorted_corr.head(top_n)
        plt.figure(figsize=(6, max(1.5, 0.4 * len(data_to_plot))))
        sns.heatmap(data_to_plot.to_frame().T, annot=True, cmap=cmap, center=0,
                    cbar_kws={'label': 'Correlation'}, fmt=".2f")
        plt.title(f"Correlation of Features with Target: {target_column}")
        plt.yticks([])
        plt.tight_layout()
        plt.show()

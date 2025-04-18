import numpy as np
import pandas as pd
from itertools import product
from metrics import f1_score
from models import RandomForest
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def compute_mutual_information(x, y):
    joint = pd.crosstab(x, y, normalize=True)
    px = joint.sum(axis=1).values
    py = joint.sum(axis=0).values

    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            pxy = joint.iat[i, j]
            if pxy > 0:
                mi += pxy * np.log(pxy / (px[i] * py[j]))
    return mi

def feature_relevance_manual_mi(df, target_column, top_n=None, plot=True, cmap="twilight"):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    relevance_scores = {}
    for col in X.columns:
        col_data = X[col]

        # Convert numeric columns to discrete bins (optional, helps for MI)
        if np.issubdtype(col_data.dtype, np.number):
            col_data = pd.qcut(col_data, q=10, duplicates='drop')

        relevance_scores[col] = compute_mutual_information(col_data, y)

    relevance_series = pd.Series(relevance_scores).sort_values(ascending=False)

    if plot:
        data_to_plot = relevance_series if top_n is None else relevance_series.head(top_n)
        plt.figure(figsize=(6, max(1.5, 0.4 * len(data_to_plot))))
        sns.heatmap(data_to_plot.to_frame().T, annot=True, cmap=cmap,
                    cbar_kws={'label': 'Mutual Information'})
        plt.title(f"Mutual Information with Target: {target_column}")
        plt.yticks([])
        plt.tight_layout()
        plt.show()

    return relevance_series


def grid_search_random_forest(X_train, y_train, X_val, y_val, param_grid, metric=f1_score):
    """
    Performs grid search over a parameter grid using a pre-split validation set.

    Parameters:
    - X_train, y_train: Training data.
    - X_val, y_val: Validation data.
    - param_grid: Dict with hyperparameters, e.g. {'n_trees': [5, 10], 'max_depth': [None, 3], ...}
    - metric: Scoring function (e.g. accuracy_score).
    - verbose: Whether to print progress.

    Returns:
    - best_params: Dict with best hyperparameters.
    - best_score: Best metric value achieved.
    - best_model: Trained RandomForest model with best parameters.
    """
    keys, values = zip(*param_grid.items())
    combinations = list(product(*values))

    best_score = -float("inf")
    best_params = None
    best_model = None

    for combo in tqdm(combinations, desc="Grid Search"):
        params = dict(zip(keys, combo))

        model = RandomForest(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = metric(y_val, preds)

        if score > best_score:
            best_score = score
            best_params = params
            best_model = model

    return best_params, best_score, best_model

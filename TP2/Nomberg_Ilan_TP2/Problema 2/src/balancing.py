import pandas as pd
import numpy as np

TARGET = 'Diagnosis'

def undersampling(df, random_state=42):
    """
    Undersampling to balance classes in a DataFrame.

    Returns:
    X, y: features and labels balanced
    """
    class_counts = df[TARGET].value_counts()
    minority_count = class_counts.min()

    samples = []
    for cls in class_counts.index:
        sampled_df = df[df[TARGET] == cls].sample(n=minority_count, random_state=random_state)
        samples.append(sampled_df)

    balanced_df = pd.concat(samples, ignore_index=True)
    X = balanced_df.drop(columns=[TARGET])
    y = balanced_df[TARGET]

    return X.reset_index(drop=True), y.reset_index(drop=True)


def oversampling(df, random_state=42):
    """
    Oversampling to balance classes in a DataFrame.

    Returns:
    X, y: features and labels balanced
    """
    class_counts = df[TARGET].value_counts()
    majority_count = class_counts.max()

    samples = []
    for cls in class_counts.index:
        sampled_df = df[df[TARGET] == cls].sample(n=majority_count, replace=True, random_state=random_state)
        samples.append(sampled_df)

    balanced_df = pd.concat(samples, ignore_index=True)
    X = balanced_df.drop(columns=[TARGET])
    y = balanced_df[TARGET]

    return X.reset_index(drop=True), y.reset_index(drop=True)


def smote(df, k=5, random_state=42):
    """
    Manual SMOTE implementation (only for binary classification).
    """
    np.random.seed(random_state)
    new_df = df.copy()

    # Separate classes
    minority_class = new_df[TARGET].value_counts().idxmin()
    df_minority = new_df[new_df[TARGET] == minority_class]
    df_majority = new_df[new_df[TARGET] != minority_class]

    X_minority = df_minority.drop(columns=TARGET).to_numpy()
    n_minority = len(X_minority)
    n_majority = len(df_majority)

    n_synthetic = n_majority - n_minority
    if n_synthetic <= 0:
        X = new_df.drop(columns=[TARGET])
        y = new_df[TARGET]
        return X.reset_index(drop=True), y.reset_index(drop=True)

    synthetic_samples = []

    for _ in range(n_synthetic):
        idx = np.random.randint(0, n_minority)
        x_i = X_minority[idx]

        distances = np.linalg.norm(X_minority - x_i, axis=1)
        neighbor_indices = np.argsort(distances)[1:k+1]

        neighbor_idx = np.random.choice(neighbor_indices)
        x_neighbor = X_minority[neighbor_idx]

        gap = np.random.rand()
        x_synthetic = x_i + gap * (x_neighbor - x_i)

        synthetic_samples.append(x_synthetic)

    df_synthetic = pd.DataFrame(synthetic_samples, columns=df_minority.columns.drop(TARGET))
    df_synthetic[TARGET] = minority_class

    df_final = pd.concat([new_df, df_synthetic], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=random_state).reset_index(drop=True)

    X = df_final.drop(columns=[TARGET])
    y = df_final[TARGET]
    return X, y

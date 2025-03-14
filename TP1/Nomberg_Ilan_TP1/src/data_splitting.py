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
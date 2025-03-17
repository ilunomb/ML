import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# Determine city for each row
def get_city(lat, lon):
    if lat > 0:  # Northern Hemisphere -> New York
        return "New York"
    else:  # Southern Hemisphere -> Buenos Aires
        return "Buenos Aires"
    
# Function to calculate Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

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

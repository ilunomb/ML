import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

city_centers = {
    "New York": (40.7128, -74.0060),
    "Buenos Aires": (-34.6037, -58.3816)
}

def get_city(lat, lon):
    if lat > 0:
        return "New York"
    else:
        return "Buenos Aires"
    
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
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

    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.shape[1] < 2:
        print("Not enough numerical columns to create a pair plot.")
        return

    sns.pairplot(numeric_df, diag_kind='kde')
    plt.show()

def feature_engineer(df):
    df["m2_per_room"] = df["area"] / df["rooms"]
    
    df["age_2"] = df["age"] ** 2

    df["log_area"] = np.log1p(df["area"])
    df["log_rooms"] = np.log1p(df["rooms"])
    df["log_age"] = np.log1p(df["age"])
    
    df["city"] = df.apply(lambda row: get_city(row["lat"], row["lon"]), axis=1)
    df["distance_to_center_km"] = df.apply(lambda row: 
        haversine(row["lat"], row["lon"], *city_centers[row["city"]]), axis=1)

    df["rooms_age_interaction"] = df["rooms"] * df["age"]
    df["rooms_distance_interaction"] = df["rooms"] * df["distance_to_center_km"]
    df["age_distance_interaction"] = df["age"] * df["distance_to_center_km"]

    df["area_2"] = df["area"] ** 2
    df["area_3"] = df["area"] ** 3
    df["rooms_2"] = df["rooms"] ** 2
    df["distance_to_center_2"] = df["distance_to_center_km"] ** 2

    df["rooms_per_m2"] = df["rooms"] / df["area"]
    df["distance_per_age"] = df["distance_to_center_km"] / (df["age"])

    df["is_new"] = (df["age"] < 5).astype(int)
    df["is_big"] = (df["area"] > df["area"].median()).astype(int)
    df["is_near_center"] = (df["distance_to_center_km"] < df["distance_to_center_km"].median()).astype(int)

    df = df.drop(columns=["city"])
    
    return df

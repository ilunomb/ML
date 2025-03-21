import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# City center coordinates
city_centers = {
    "New York": (40.7128, -74.0060),
    "Buenos Aires": (-34.6037, -58.3816)
}

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

def feature_engineer(df):
    # Feature 1: Area per Room
    df["m2_per_room"] = df["area"] / df["rooms"]
    
    # Feature 2: Quadratic Age
    df["age_2"] = df["age"] ** 2

    # Feature 3: Log Transformations (to reduce skewness)
    df["log_area"] = np.log1p(df["area"])
    df["log_rooms"] = np.log1p(df["rooms"])
    df["log_age"] = np.log1p(df["age"])
    
    # Feature 4: City and Distance to Center
    df["city"] = df.apply(lambda row: get_city(row["lat"], row["lon"]), axis=1)
    df["distance_to_center_km"] = df.apply(lambda row: 
        haversine(row["lat"], row["lon"], *city_centers[row["city"]]), axis=1)
    
    # Feature 5: Interaction Features
    df["rooms_age_interaction"] = df["rooms"] * df["age"]
    df["rooms_distance_interaction"] = df["rooms"] * df["distance_to_center_km"]
    df["age_distance_interaction"] = df["age"] * df["distance_to_center_km"]

    # Feature 6: Polynomial Features for Non-Linear Effects
    df["area_2"] = df["area"] ** 2
    df["area_3"] = df["area"] ** 3
    df["rooms_2"] = df["rooms"] ** 2
    df["distance_to_center_2"] = df["distance_to_center_km"] ** 2

    # Feature 7: Density and Ratios
    df["rooms_per_m2"] = df["rooms"] / df["area"]
    df["distance_per_age"] = df["distance_to_center_km"] / (df["age"] + 1)  # Avoid div by 0

    # Feature 8: Binary Indicators
    df["is_new"] = (df["age"] < 5).astype(int)  # If the house is new (<5 years)
    df["is_big"] = (df["area"] > df["area"].median()).astype(int)  # Above-median area
    df["is_near_center"] = (df["distance_to_center_km"] < df["distance_to_center_km"].median()).astype(int)

    # Drop City Column (Only Used for Distance Calculation)
    df = df.drop(columns=["city"])
    
    return df

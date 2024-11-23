


import pandas as pd
from geo import get_gps_lat_long_from_easting_northing, lat_long_to_xyz
import matplotlib.pyplot as plt
import seaborn as sns
import numpy





def df_to_lat_long_zero_origin(df):
    # Check if required columns are present
    if 'easting' not in df.columns or 'northing' not in df.columns:
        raise ValueError("DataFrame must contain 'easting' and 'northing' columns")

    # Convert easting and northing to latitude and longitude
    df['latitude'], df['longitude'] = get_gps_lat_long_from_easting_northing(df['easting'], df['northing'])
    
    # Ensure no NaN values in latitude and longitude
    if df['latitude'].isnull().any() or df['longitude'].isnull().any():
        raise ValueError("Conversion to latitude and longitude resulted in NaN values")

    # Shift origin to zero
    df['latitude'] = df['latitude'] - df['latitude'].min()
    df['longitude'] = df['longitude'] - df['longitude'].min()

    # Scale the data
    df['latitude_scaled'] = df['latitude'] / df['latitude'].max()
    df['longitude_scaled'] = df['longitude'] / df['longitude'].max()

    # Drop intermediate latitude and longitude columns
    df.drop(['latitude', 'longitude'], axis=1, inplace=True)
    
    return df
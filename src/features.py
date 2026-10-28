
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


def calc_sqft_basement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculates the 'sqft_basement' column as 'sqft_living - sqft_above'.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
    pd.DataFrame: The DataFrame with 'sqft_basement' recalculated.
    """
    df_copy = df.copy() 
    df_copy['sqft_basement'] = df_copy['sqft_living'] - df_copy['sqft_above']
    
    return df_copy



def calc_last_known_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the last known change in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
    pd.DataFrame: The DataFrame with the last known change calculated.
    """
    df_lastknownchange = df.copy()
    # Create an empty list to store the derived values.
    last_known_change = []

# Inspect the `yr_renovated` value for each row.
    for idx, yr_re in df_lastknownchange["yr_renovated"].items():
    # If `yr_renovated` is 0 or missing, use the original build year instead.
        if str(yr_re) == "nan" or yr_re == 0.0:
            last_known_change.append(df_lastknownchange["yr_built"][idx])
    # Otherwise, keep the renovation year in the new list.
        else:
            last_known_change.append(int(yr_re))

    df_lastknownchange["last_known_change"] = last_known_change
    # Drop the original `yr_renovated` and `yr_built` columns.
    df_lastknownchange.drop("yr_renovated", axis=1, inplace=True)
    df_lastknownchange.drop("yr_built", axis=1, inplace=True)
    return df_lastknownchange

CENTER_LAT = 47.62774
CENTER_LONG = -122.24194
REFERENCE_LAT = 47.6219
EARTH_RADIUS_KM = 6378

def dist(long, lat, ref_long, ref_lat):
    """Distance in km between locations and reference locations (works on arrays)."""
    delta_long = long - ref_long
    delta_lat = lat - ref_lat
    delta_long_corr = delta_long * np.cos(np.radians(ref_lat))
    return np.sqrt(delta_long_corr ** 2 + delta_lat ** 2) * 2 * np.pi * 6378 / 360

def calc_wealth_center_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the distance from each house to the center of wealth.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
    pd.DataFrame: The DataFrame with 'wealth_center_distance' calculated.
    """
    df_copy = df.copy()
    delta_lat = np.absolute(CENTER_LAT - df_copy["lat"])
    delta_long = np.absolute(CENTER_LONG - df_copy["long"])
    df_copy["center_distance"] = (
        np.sqrt((delta_long * np.cos(np.radians(REFERENCE_LAT))) ** 2 + delta_lat ** 2)
        * 2 * np.pi * EARTH_RADIUS_KM / 360
    )
    return df_copy

    
class WaterfrontDistance(BaseEstimator, TransformerMixin):
    """Add 'waterfront_distance': km from each house to the waterfront."""
    def fit(self, X, y=None):
        water = X[X["waterfront"] == 1]
        self.water_long_ = water["long"].to_numpy()
        self.water_lat_ = water["lat"].to_numpy()
        return self  
       
       
    def transform(self, X):
        X = X.copy()
        X["center_distance"] = dist(X["long"], X["lat"], -122.24194, 47.62774)
        distances = dist(X["long"].to_numpy()[:, None], X["lat"].to_numpy()[:, None],
                         self.water_long_[None, :], self.water_lat_[None, :])
        X["waterfront_distance"] = distances.min(axis=1)
        return X




import pandas as pd

def remove_bad_data(df: pd.DataFrame, max_bedrooms: int) -> pd.DataFrame:
    """
    Removes rows from the DataFrame where 'bedrooms' is less than or equal to 0 or greater than or equal to 10.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
    pd.DataFrame: The cleaned DataFrame with invalid bedroom counts removed.
    """
    df_copy = df.copy()
    # Remove rows where 'bedrooms' is less than or equal to 0 or greater than or equal to 10
    cleaned_df = df_copy[(df_copy['bedrooms'] > 0) & (df_copy['bedrooms'] <= max_bedrooms)]
    
    return cleaned_df


def fix_date_format(df: pd.DataFrame, date_col: list[str]) -> pd.DataFrame:
    """
    Converts specified date columns in the DataFrame to datetime format.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be cleaned.
    date_col (list[str]): List of column names to be converted to datetime format.

    Returns:
    pd.DataFrame: The DataFrame with specified date columns converted to datetime format.
    """
    df_datefix = df.copy() 
    for col in date_col:
        df_datefix[col] = pd.to_datetime(df_datefix[col])
    
    return df_datefix


def fix_str_format(df: pd.DataFrame, str_col: list[str]) -> pd.DataFrame:
    """
    Converts specified string columns in the DataFrame to string format.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be cleaned.
    str_col (list[str]): List of column names to be converted to string format.

    Returns:
    pd.DataFrame: The DataFrame with specified string columns converted to string format.
    """
    df_strfix = df.copy() 
    for col in str_col:
        df_strfix[col] = df_strfix[col].astype(str)
    
    return df_strfix
import pandas as pd
import numpy as np


def handle_outliers(df: pd.DataFrame, how: str = 'standard', numeric_cols: list = None, remove_outliers: bool = False) -> pd.DataFrame:
    """
    This function takes a pandas DataFrame as input and returns a cleaned DataFrame
    with the outliers handled according to the specified method.

    Parameters:
    -----------
    dataset : pandas.DataFrame
        The input dataset.
    how : str, optional (default='standard')
        The method to handle outliers. Possible values are:
        - 'standard': Remove observations with values outside the range Q1-1.5IQR and Q3+1.5IQR.
        - 'winsorize': Winsorize the outliers by replacing them with the nearest non-outlier value.
    numeric_cols : list, optional (default=['energy_100g', 'fat_100g', 'carbohydrates_100g', 'proteins_100g'])
        The list of numeric columns to handle outliers.
    remove_outliers : bool, optional (default=False)
        Whether to remove outliers from the dataset or not. If True, the outliers will be removed and the cleaned dataset
        without outliers will be returned. If False, the original dataset with outliers handled according to the specified
        method will be returned.

    Returns:
    --------
    pandas.DataFrame
        The cleaned dataset with outliers handled.
    """
    # Set default columns for scaling if not provided
    if numeric_cols is None:
        numeric_cols = ['energy_100g', 'fat_100g', 'carbohydrates_100g', 'proteins_100g']

    # Replace missing values with the mean of each column
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    if how == 'standard':
        # Calculate quartiles for each numeric column
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1

        # Identify observations with a value outside the range Q1-IQR and Q3+IQR
        outliers = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
        if remove_outliers:
            return outliers
        else:
            # Replace outliers with NaN
            df[numeric_cols] = df[numeric_cols].mask((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR)))
            return df
    elif how == 'winsorize':
        # Winsorize outliers for each numeric column
        for col in numeric_cols:
            q_low = df[col].quantile(0.01)
            q_high = df[col].quantile(0.99)
            if remove_outliers:
                df = df[(df[col] >= q_low) & (df[col] <= q_high)]
            else:
                df[col] = df[col]
        return df
    

def outliers_nutrimentscols(df, g_per_100g_features):
    """
    Filter rows from the DataFrame based on values in the specified columns.

    This function removes rows from the DataFrame where any value in a column specified
    by the g_per_100g_features argument is greater than 100 or less than 0.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        g_per_100g_features (list): A list of column names to filter by.

    Returns:
        pandas.DataFrame: A new DataFrame with rows removed where any value in a column
        specified by g_per_100g_features is greater than 100 or less than 0.
    """
    df = df[~((df[g_per_100g_features] > 100) | (df[g_per_100g_features] < 0)).any(axis=1)]
    return df


def outliers_nutrimentscols_others(df):
    """
    Filters a DataFrame based on certain conditions.

    This function removes rows where values from some columns that
    cannot be greater than other columns.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be filtered.

    Returns:
        pandas.DataFrame: The filtered DataFrame.
    """
    # Delete rows where the nutrient type value cannot be greater than the nutrient total.
    df = df[~(df['saturated-fat_100g'] > df['fat_100g'])
            | (df['sodium_100g'] > df['salt_100g'])
            | (df['added-sugars_100g'] > df['sugars_100g'])
            | (df['sugars_100g'] > df['carbohydrates_100g'])]
    
    # Delete outliers from others nutrients cols with different unit
    df = df[~(df['energy_100g'] > 3700)]
    df = df[~(df['energy-from-fat_100g'] > 3700)]
    df = df[~(df['energy-kcal_100g'] > 900)]
    df = df[~((df['ph_100g'] < 0) | (df['ph_100g'] > 14))]
    
    return df
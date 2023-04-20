import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def handle_outliers_100g(df: pd.DataFrame, cols_to_exclude: list = None) -> pd.DataFrame:
    """
    This function takes a pandas DataFrame as input and returns a cleaned DataFrame
    with the outliers handled according to the specified method.

    Outliers are identified as values < 0 and > 100 for columns ending with '_100g',
    with the exception of the columns specified in the `cols_to_exclude` parameter,
    which are excluded from this condition.

    Outliers are replaced with the median value for the corresponding column.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset.
    cols_to_exclude : list, optional (default=None)
        A list of column names to exclude from the outlier condition.

    Returns:
    --------
    pandas.DataFrame
        The cleaned dataset with outliers handled.
    """

    if cols_to_exclude is None:
        cols_to_exclude = ['energy-kj_100g', 'energy-kcal_100g', 'ph_100g', 'carbon-footprint_100g',
                           'nutrition-score-fr_100g', 'nutrition-score-uk_100g']

    # Replace outliers in columns ending with '_100g'
    for col in df.filter(regex='_100g$', axis=1).columns:
        if col not in cols_to_exclude:
            median = df[col].median()
            df[col] = df[col].apply(lambda x: median if x < 0 or x > 100 else x)

    return df



def handle_outliers_others(df, columns, method = 'nan', k=1.5, sklearn_method=False):
    """
    Detects and handles outliers in a pandas DataFrame using Interquartile Range or
    IsolationForest detection method.
    If IQR method, mean or median calculated on all values in column.
    If IsolationForest method, mean or median calculated on all non-outlier values in column.

    Keyword arguments:
    ------------------
        df: pandas.DataFrame
            The input DataFrame.
        columns: list of str
            The list of column names to handle outliers for.
        method: str, optional (default 'nan').
            The method to use for handling outliers.
            Available methods are 'nan', mean', 'median', and 'drop'
        k: float, optional (default 1.5)
            The multiplier for the IQR range.
        sklearn_method: bool, optional (default False)
            Will use IQR method to detect and remove outliers if set False,
            else will use IsolationForest method.

    Returns:
    --------
        pandas.DataFrame
            The DataFrame with outliers handled.

    Author:
    -------
        Joëlle Sabourdy
    """
    df_outliers = df.copy()

    # Replace missing values --> drop
    df_outliers.dropna(subset=columns, inplace=True)

    if sklearn_method == True:
        for i in columns:
            # Extract values to process
            X = df_outliers[[i]].values
        
            # Define and fit the IsolationForest model
            model = IsolationForest()
            model.fit(X)
        
            # Compute the outlier scores and classify outliers
            outlier_scores = model.decision_function(X)
            outliers_condition = model.predict(X) == -1
            
            # handle outliers
                # replace outliers with NaN value
            if method == 'nan':
                df_outliers.loc[outliers_condition, i] = np.nan
                # drop rows with outliers
            elif method == 'drop':
                df_outliers = df_outliers.loc[~outliers_condition, :]
                 # replace outliers with median value
            elif method == 'median':
                median = df_outliers.loc[~outliers_condition, i].median()
                df_outliers.loc[outliers_condition, i] = median
                 # replace outliers with mean value
            elif method == 'mean':
                mean = df_outliers.loc[~outliers_condition, i].mean()
                df_outliers.loc[outliers_condition, i] = mean
            else:
                raise ValueError("Invalid method. Allowed methods: 'nan','drop', 'median', 'mean'.")
        
    else:
        # InterQuartile Range method
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
    
        # Calculate lower and upper bounds for outliers detection
        min = Q1 - k * IQR
        max = Q3 + k * IQR
    
        # conditions
        outliers_condition = ((df_outliers < min) | (df_outliers > max))
        
        # handle outliers
            # replace outliers with NaN value
        if method == 'nan':
            df_outliers = df_outliers.mask(outliers_condition)
            # drop rows with outliers
        elif method == 'drop':
            df_outliers = df_outliers[~outliers_condition.any(axis=1)]
            # replace outliers with median value
        elif method == 'median':
            median = df_outliers[columns].median()
            df_outliers = df_outliers.mask(outliers_condition, median, axis=1)
            # replace outliers with mean value
        elif method == 'mean':
            mean = df_outliers[columns].mean()
            df_outliers = df_outliers.mask(outliers_condition, mean, axis=1)
        else:
            raise ValueError("Invalid method. Allowed methods: 'nan','drop', 'median', 'mean'.")
            
    return df_outliers



def handle_outliers_100g_others(df: pd.DataFrame, method: str = 'median', cols_to_exclude: list = []) -> pd.DataFrame:
    """
    This function takes a pandas DataFrame as input and returns a cleaned DataFrame
    with the outliers handled according to the specified method.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset.
    method : str, optional (default='median')
        The method to handle outliers. Possible values are:
        - 'median': Replace outliers by the median of each column.
        - 'remove': Remove observations with outliers.
    cols_to_exclude : list, optional (default=[])
        The list of columns to exclude from the outlier handling.

    Returns:
    --------
    pandas.DataFrame
        The cleaned dataset with outliers handled.
    """
    # Set default columns to handle outliers if not provided
    cols_to_handle = [col for col in df.columns if col.endswith('_100g') and col not in cols_to_exclude]
    
    # Replace missing values with the median of each column
    df[cols_to_handle] = df[cols_to_handle].fillna(df[cols_to_handle].median())

    # Identify observations with a value outside the range Q1-1.5IQR and Q3+1.5IQR for each column
    outliers = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in cols_to_handle:
        if col in ['energy-kj_100g', 'energy-kcal_100g', 'ph_100g', 'carbon-footprint_100g', 'nutrition-score-fr_100g', 'nutrition-score-uk_100g'] :
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - df[col].quantile(0.25)
            outliers[col] = (df[col] < 0) | (df[col] > (Q3 + 1.5 * IQR))
        else:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers[col] = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)) | (df[col] < 0)

    # Handle outliers by method
    if method == 'median':
        for col in cols_to_handle:
            if col in cols_to_exclude:
                continue
            median = df.loc[~outliers[col], col].median()
            df[col] = np.where(outliers[col], median, df[col])
    elif method == 'remove':
        df = df[~outliers.any(axis=1)]
    else:
        raise ValueError("Invalid 'method' argument. Allowed values are 'median' and 'remove'.")
    
    return df

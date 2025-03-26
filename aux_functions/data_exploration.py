# Inspeccionar una columna de un dataframe
import pandas as pd
import numpy as np

def inspect_column(df: pd.DataFrame, column_name: str, max_unique: int = 50) -> pd.DataFrame:
    """
    Returns a DataFrame with two columns ("Property" and "Value"). Each row contains
    a piece of information about the specified column, such as data type, null counts,
    duplicates, descriptive statistics, and a limited set of unique values.
    """
    col_data = df[column_name]
    data_type = col_data.dtype
    null_count = col_data.isnull().sum()
    desc_series = col_data.describe(include='all')

    if not isinstance(desc_series, pd.Series):
        desc_series = desc_series.iloc[0]

    desc_dict = {}

    for key, val in desc_series.items():
        desc_dict[f"{key}"] = val

    unique_vals = col_data.dropna().unique().round(2)

    if np.issubdtype(col_data.dtype, np.number):
        unique_vals = np.sort(unique_vals)

    unique_vals_limited = unique_vals[:max_unique].tolist()

    info_dict = {
        "column_name": column_name,
        "dtype": str(data_type),
        "null_count": null_count,
        "unique_values": unique_vals_limited,
        "total_unique_values": len(unique_vals),
        "skew": df[column_name].skew(),
    }
    info_dict.update(desc_dict)

    df_result = pd.DataFrame(
        list(info_dict.items()),
        columns=["Property", "Value"]
    )

    return df_result


# Inspeccionar todas las columnas de un dataframe
def inspect_all_columns(df: pd.DataFrame, max_unique: int = 50) -> pd.DataFrame:
    """
    Creates a single DataFrame where each row corresponds to one column of 'df'
    and each column of this resulting DataFrame is a property (dtype, null_count,
    descriptive stats, etc.).
    """
    rows = []

    for col in df.columns:
        col_data = df[col]
        data_type = col_data.dtype
        null_count = col_data.isnull().sum()

        desc_series = col_data.describe(include='all')

        if not isinstance(desc_series, pd.Series):
            desc_series = desc_series.iloc[0]

        desc_dict = dict(desc_series)

        unique_vals = col_data.dropna().unique()

        if np.issubdtype(data_type, np.number):
            unique_vals = np.sort(unique_vals)
            unique_vals = np.round(unique_vals, 2)

        unique_vals_limited = unique_vals[:max_unique].tolist()

        row_dict = {
            "dtype": str(data_type),
            "null_count": null_count,
            "unique_values": unique_vals_limited,
            "total_unique_values": len(unique_vals),
            "skew": col_data.skew() if np.issubdtype(data_type, np.number) else None,
        }
        row_dict.update(desc_dict)

        rows.append(row_dict)

    df_result = pd.DataFrame(rows, index=df.columns)
    df_result.index.name = "column_name"

    return df_result


# Verificar columnas con las mismas filas nulas
def rows_same_nulls(df: pd.DataFrame):
    """
    Checks for columns with the same null rows to be deleted.
    :param df: dataframe containing columns of interest
    """
    missing_groups = {}
    for col in df.columns:
        missing_indices = frozenset(df.index[df[col].isnull()])
        missing_groups.setdefault(missing_indices, []).append(col)

    for missing_set, columns in missing_groups.items():
        if len(columns) > 1:
            print("Total rows with missing values: ", len(missing_set))
            print("Variables with identical missing rows: ", len(columns))
            print("\n".join(columns))
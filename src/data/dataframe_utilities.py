import pandas as pd

def StandardizeColumnNames(df):
    """
    Standardizes column names
    """

    df.columns = map(str.lower, df.columns)
    df.columns = df.columns.str.replace(', ', '_')
    df.columns = df.columns.str.replace('-', '_')
    df.columns = df.columns.str.replace('/', '_')
    df.columns = df.columns.str.replace('(', '_')
    df.columns = df.columns.str.replace(')', '_')
    df.columns = df.columns.str.replace(' ', '_')

    return df

# reads df from csv and does preprocessing
import pandas as pd


def read_df_from_csv(file):
    time_col = 'time'
    df = pd.read_csv(file, index_col=[0])
    df[time_col] = pd.to_datetime(df[time_col])
    df['id'] = df['id'].astype("string")
    df.sort_values(by=time_col, inplace=True)
    return df

def resample_df(df, rule, time_col):
    df_time = df.set_index([time_col])
    return df_time.resample(rule).agg(['min', 'max', 'mean', 'std'])

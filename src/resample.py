def resample_df(df, rule, time_col, sample_col):
    df_time = df.set_index([time_col])
    return df_time[[sample_col]].resample(rule).agg(['min', 'max', 'mean', 'std'])

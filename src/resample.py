def resample_df(df, rule, time_col, sample_col):
    df_time = df.set_index([time_col])
    return df_time[[sample_col]].resample(rule).agg(['min', 'max', 'mean', 'std'])


# column arguments can be str or tuple for multiindex columns, e.g ('iob', 'mean')
def z_score_normalise(df, col_to_normalise, new_col):
    new_df = df.copy()
    column_to_normalise = df[col_to_normalise]
    new_df[new_col] = (column_to_normalise - column_to_normalise.mean()) / column_to_normalise.std()
    return new_df

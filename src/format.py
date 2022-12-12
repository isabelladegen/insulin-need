# formats read records into useful data formats
import pandas as pd

from src.read import ReadRecord


# takes a list of ReadRecords and creates one flat dataframe adding zip_id as id column
# only keeps_cols specified, all if set to None which is default
def as_flat_dataframe(records: [ReadRecord], drop_na: bool = False, keep_cols=None):
    result = None
    for record in records:
        # get df with id column
        df = record.df_with_id(keep_cols=keep_cols)
        if df is None:
            continue

        # concat to resulting df
        if result is None:
            result = df
        else:
            result = pd.concat([result, df])

    # drop nan
    if drop_na:
        result = result.dropna()
    # reindex from 0 - no of rows
    result.reset_index(inplace=True, drop=True)
    return result

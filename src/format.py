# formats read records into useful data formats
import pandas as pd

from src.read import ReadRecord


# takes a list of ReadRecords and creates one flat dataframe adding zip_id as id column
def as_flat_dataframe(records: [ReadRecord], drop_na: bool = False):
    result = None
    for record in records:
        # add id column
        if record.df is None:
            continue
        df = record.df.copy()
        df.insert(loc=0, column='id', value=record.zip_id)

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

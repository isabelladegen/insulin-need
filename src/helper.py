# various convenience methods

# calculates a df of all the different read records
import dataclasses

import pandas as pd


def dataframe_of_read_record_stats(read_records: []):
    resultdict = list(map(dataclasses.asdict, read_records))
    data = {'zip_id': list(map(lambda x: x.get('zip_id'), resultdict)),
            'rows': list(map(lambda x: x.get('number_of_rows'), resultdict)),
            'rows_without_nan': list(map(lambda x: x.get('number_of_rows_without_nan'), resultdict)),
            'rows_with_nan': list(map(lambda x: x.get('number_of_rows_with_nan'), resultdict)),
            'earliest_date': list(map(lambda x: x.get('earliest_date'), resultdict)),
            'newest_date': list(map(lambda x: x.get('newest_date'), resultdict)),
            'is_android_upload': list(map(lambda x: x.get('is_android_upload'), resultdict))
            }
    return pd.DataFrame(data)


# various convenience methods

# calculates a df of all the different read records
import dataclasses
import glob
from pathlib import Path

import pandas as pd

from src.configurations import Configuration, Resampling


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


def files_for_id(folder, zip_id):
    return glob.glob(str(Path(folder, zip_id).resolve()) + "/*.csv")


def bg_file_path_for(folder, zip_id):
    files = files_for_id(folder, zip_id)
    return Path(list(filter(lambda x: Configuration().bg_file in x, files))[0])


def device_status_file_path_for(folder, zip_id):
    files = files_for_id(folder, zip_id)
    return Path(list(filter(lambda x: Configuration().device_file in x, files))[0])


def preprocessed_file_for(folder, zip_id: str, sampling: Resampling):
    files = files_for_id(folder, zip_id)
    name = sampling.csv_file_name()
    files_matching_name = list(filter(lambda x: name in x, files))
    return Path(files_matching_name[0]) if files_matching_name else None

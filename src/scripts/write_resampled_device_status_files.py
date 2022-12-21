import os
from glob import glob
from os.path import normpath, basename
from pathlib import Path

import pandas as pd

from src.configurations import Configuration, Irregular, Daily, Hourly
from src.helper import preprocessed_file_for
from src.read_preprocessed_df import ReadPreprocessedDataFrame
from src.resampling import ResampleDataFrame


def main():
    # reads irregular sampled file (create first!) and writes daily and hourly sampled files per id and as flat file
    config = Configuration()
    per_id_folder = config.perid_data_folder
    flat_file_folder = config.data_folder
    irregular = Irregular()
    hourly = Hourly()
    daily = Daily()

    missing_zipids = []
    big_hourly_df = None
    big_daily_df = None

    # write hourly and daily dfs per id
    zip_id_dirs = glob(os.path.join(per_id_folder, "*", ""))
    zip_ids = [basename(normpath(path_str)) for path_str in zip_id_dirs]
    for zip_id in zip_ids:
        # check that irregular file exists otherwise print error
        file = preprocessed_file_for(config.perid_data_folder, zip_id, irregular)
        if file is None:
            missing_zipids.append(zip_id)
            print("No irregular sampled file for zip id: " + zip_id)
            continue

        # read the irregular file into df
        df = ReadPreprocessedDataFrame(sampling=irregular, zip_id=zip_id).df

        # resample to hourly and daily df
        resampler = ResampleDataFrame(df)
        daily_df = resampler.resample_to(daily)
        hourly_df = resampler.resample_to(hourly)

        # write pre id
        daily_resampled_file_name = Path(per_id_folder, zip_id, daily.csv_file_name())
        daily_df.to_csv(daily_resampled_file_name)
        hourly_resampled_file_name = Path(per_id_folder, zip_id, hourly.csv_file_name())
        hourly_df.to_csv(hourly_resampled_file_name)

        # add to overall dataframe
        if big_hourly_df is None:
            big_hourly_df = hourly_df
            big_daily_df = daily_df
        else:
            big_hourly_df = pd.concat([big_hourly_df, hourly_df])
            big_daily_df = pd.concat([big_daily_df, daily_df])

    # reset index for big dfs
    big_hourly_df.reset_index(inplace=True, drop=True)
    big_daily_df.reset_index(inplace=True, drop=True)

    # write flat_file dfs
    daily_resampled_file_name = Path(flat_file_folder, daily.csv_file_name())
    big_daily_df.to_csv(daily_resampled_file_name)
    hourly_resampled_file_name = Path(flat_file_folder, hourly.csv_file_name())
    big_hourly_df.to_csv(hourly_resampled_file_name)

    print('Number of zip ids without irregular device status files: ' + str(len(missing_zipids)))


if __name__ == "__main__":
    main()

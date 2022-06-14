# reads BG entries into a list of ReadRecords
import glob
import zipfile
from dataclasses import dataclass
from io import TextIOWrapper
from itertools import groupby
from operator import itemgetter
from pathlib import Path
import logging
import pandas as pd

from src.configurations import Configuration


# Data object that keeps the information from reading each data zip file
@dataclass
class ReadRecord:
    zip_id: str = None  # patient id
    df: pd.DataFrame = None  # dataframe
    has_no_files: bool = False  # true if the specified file to read was empty or did not exist
    number_of_entries_files: int = 0  # number of entries files found
    number_of_rows: int = 0  # number of rows in total
    number_of_rows_without_nan: int = 0
    number_of_rows_with_nan: int = 0
    earliest_date: str = ''  # oldest date in series
    newest_date: str = ''  # newest date in series
    is_android_upload: bool = False  # set to True if from android upload, False otherwise

    # helper method to set read records for no bg files
    def zero_files(self):
        self.has_no_files = True

    def add(self, df):
        if self.df is None:
            self.df = df
        else:
            self.df = pd.concat([self.df, df])

    def calculate_stats(self):
        if self.has_no_files:
            return
        if self.df is None:
            return
        self.number_of_rows = self.df.shape[0]
        self.number_of_rows_without_nan = self.df.dropna().shape[0]
        self.number_of_rows_with_nan = self.df.shape[0] - self.df.dropna().shape[0]
        self.earliest_date = str(self.df.time.min())
        self.newest_date = str(self.df.time.max())


# reads all BG files from each zip files without extracting the zip
def read_all_bg(config: Configuration):
    read_records = read_all(config, read_bg_from_zip)

    # read android data
    results = read_all_android_aps_files(config)
    return read_records + results


# reads all device status files into a list of read records
def read_all_device_status(config):
    return read_all(config, read_device_status_from_zip)


# reads all files using function
def read_all(config, function):
    data = config.data_dir
    # get all zip files in folder
    filepaths = glob.glob(str(data) + "/*.zip")
    read_records = []
    for file in filepaths:
        # Android read below
        if file.endswith(config.android_aps_zip):
            continue
        read_record = function(file, config)
        read_records.append(read_record)
    return read_records


# reads BGs into df from the entries csv file in the given zip file without extracting the zip
def read_bg_from_zip(file_name, config):
    return read_zip_file(config, file_name, is_a_bg_csv_file, read_entries_file_into_df)


# generic zip file read method
def read_zip_file(config, file_name, file_check_function, read_file_into_df_function):
    read_record = ReadRecord()
    read_record.zip_id = Path(file_name).stem
    # find bg files in the zip file
    with zipfile.ZipFile(file_name, mode="r") as archive:
        files_and_folders = archive.namelist()

        # finds all the .csv files that match the file check function
        files_to_read = [x for x in files_and_folders if file_check_function(config, read_record.zip_id, x)]

        # check number of files
        number_of_files = len(files_to_read)
        # stop reading if there are no files
        if number_of_files == 0:
            read_record.zero_files()
            return read_record
        read_record.number_of_entries_files = number_of_files

        # read all the entries files into dataframes
        for file in files_to_read:
            info = archive.getinfo(file)

            # skip files that are zero size, but log them
            if info.file_size == 0:
                logging.info('Found empty file: ' + file + ' for id: ' + read_record.zip_id)
                continue

            # read entries into pandas dataframe
            read_file_into_df_function(archive, file, read_record, config)

        # calculate some information from the dataframe
        read_record.calculate_stats()
        return read_record


# reads BG data from entries file into df and adds it to read_record, config is there for consistency
def read_entries_file_into_df(archive, file, read_record, config):
    with archive.open(file, mode="r") as bg_file:
        df = pd.read_csv(TextIOWrapper(bg_file, encoding="utf-8"),
                         header=None,
                         # parse_dates=[0],
                         # date_parser=lambda col: pd.to_datetime(col, utc=True),
                         dtype={
                             'time': str,
                             'bg': pd.Float64Dtype()
                         },
                         names=['time', 'bg'],
                         na_values=[' null', '', " "])
        convert_problem_timestamps(df, 'time')
        read_record.add(df)


# reads device status file into df and adds it to read_record
def read_device_status_file_into_df(archive, file, read_record, config):
    specific_cols_dic = config.device_status_col_type
    columns_to_read = None
    if specific_cols_dic:  # only analyse headers if we're reading specific columns
        columns_to_read = specific_cols_dic.keys()
        # analyze headers and skip any file that's not data during closed looping or that's from the loop
        with archive.open(file, mode="r") as header_context:
            header = pd.read_csv(TextIOWrapper(header_context, encoding="utf-8"), nrows=0)
            actual_headers = header.columns
            missing_headers = [ele for ele in columns_to_read if ele not in list(actual_headers)]
            if missing_headers:
                if not any("enacted" in h for h in actual_headers):
                    return  # this is not a device status file from a looping period

                if not any("openaps" in h for h in actual_headers):
                    return  # this is likely a loop file and won't have bolus information in the file, skip for now
    with archive.open(file, mode="r") as file_context:
        io_wrapper = TextIOWrapper(file_context, encoding="utf-8")
        # only read files when looping
        # if 'openaps/enacted/deliverAt' in header.columns:
        if columns_to_read:  # if cols is not None read only the columns as specified in the config file
            df = pd.read_csv(io_wrapper,
                             usecols=lambda c: c in set(columns_to_read),
                             dtype=specific_cols_dic,
                             )
        else:
            df = pd.read_csv(io_wrapper)
        time = 'created_at'
        df[time] = pd.to_datetime(df[time])  # time hear is created_at
        to_datetime_if_exists(df, 'pump/status/timestamp')
        to_datetime_if_exists(df, 'openaps/enacted/deliverAt')
        to_datetime_if_exists(df, 'openaps/enacted/timestamp')
        to_datetime_if_exists(df, 'openaps/iob/timestamp')
        to_datetime_if_exists(df, 'openaps/iob/lastBolusTime')
        df.rename(columns={time: 'time'}, errors="raise", inplace=True)
        read_record.add(df)


def to_datetime_if_exists(df, column, unit=None):
    if column in df.columns:
        if unit is not None:
            df[column] = pd.to_datetime(df[column], unit=unit, errors='coerce')
        else:
            df[column] = pd.to_datetime(df[column], errors='coerce')


# reads android bg data
def read_all_android_aps_files(config):
    data_dir = config.data_dir
    android_zip = config.android_aps_zip
    android_file = Path(data_dir + '/' + android_zip)

    # find files in the zip file
    with zipfile.ZipFile(android_file, mode="r") as archive:
        # find high level folders to read data from -> one ReadRecord per high level folder
        all_docs = archive.namelist()
        all_docs.remove('/')  # ignore root
        files = {item.split('/')[0] for item in all_docs}

        records = []
        # read BG from each folder
        for file in files:
            read_record = ReadRecord()
            read_record.zip_id = file
            read_record.is_android_upload = True

            # find all files for that zip_id
            files_for_zip_id = [doc for doc in all_docs if file in doc]

            # find all bg files
            bg_files = [doc for doc in files_for_zip_id if doc.endswith(config.bg_csv_file_android)]
            read_record.number_of_entries_files = len(bg_files)
            if not bg_files:
                read_record.has_no_files = True

            # read bg files into df
            for bg_file in bg_files:
                # upload_info = bg_file.replace(config.bg_csv_file_android, config.android_upload_info)
                with archive.open(bg_file, mode="r") as open_bg_file:
                    df = pd.read_csv(TextIOWrapper(open_bg_file, encoding="utf-8"),
                                     header=None,
                                     parse_dates=['time'],
                                     date_parser=lambda col: pd.to_datetime(col, unit='ms'),
                                     dtype={
                                         'time': str,
                                         'bg': pd.Float64Dtype()
                                     },
                                     names=['time', 'bg'],
                                     na_values=[' null', '', " "])
                    read_record.add(df)

            read_record.calculate_stats()
            records.append(read_record)
    return records


# checks if a file from zip namelist is a bg csv file
def is_a_bg_csv_file(config, patient_id, file_path):
    # file starts with patient id and _entries
    startswith = Path(file_path).name.startswith(patient_id + config.bg_csv_file_start)

    # has right file ending
    endswith = file_path.endswith(config.bg_csv_file_extension)
    return startswith and endswith


# checks if a file from zip namelist is a bg csv file
def is_a_device_status_csv_file(config, patient_id, file_path):
    # file starts with patient id and _entries
    startswith = Path(file_path).name.startswith(patient_id + config.device_status_csv_file_start)

    # has right file ending
    endswith = file_path.endswith(config.device_status_csv_file_extension)
    return startswith and endswith


# reads a device status file
def read_device_status_from_zip(file, config):
    return read_zip_file(config, file, is_a_device_status_csv_file, read_device_status_file_into_df)


# deals with non-compatible AM/PM and timezones timestamps so that all times can be converted to pandas timestamps
# modifies df
def convert_problem_timestamps(df: pd.DataFrame, column: str):
    try:
        # replace tricky timezone strings with easier to translate ones
        df.replace(' EDT ', ' UTC+4 ', regex=True, inplace=True)
        df.replace(' EST ', ' UTC+5 ', regex=True, inplace=True)
        df.replace(' CDT ', ' UTC+5 ', regex=True, inplace=True)
        df.replace(' vorm.', ' AM', regex=True, inplace=True)  # German AM
        df.replace(' nachm.', ' PM', regex=True, inplace=True)  # German PM

        # find all the problematic time stamps
        problem_idx = list(df.index[df[column].str.endswith(' PM') | df[column].str.endswith(' AM')])
        # group consecutive problem indexes
        grouped_problem_idx = groupby(enumerate(problem_idx), lambda ix: ix[0] - ix[1])
        for key, group in grouped_problem_idx:
            # list of consecutive problems
            ls = list(map(itemgetter(1), group))

            # find the best entry that has timezone information to use
            last_idx = ls[-1]
            first_idx = ls[0]
            # check that there is an item after the problematic one that has a timezone
            if last_idx + 1 <= (df.shape[0] - 1):
                with_tz = df[column].iloc[last_idx + 1]
                known_tz = pd.to_datetime(with_tz).tz
            elif first_idx != 0:  # check that there is an item before the problematic on that has a timezone
                with_tz = df[column].iloc[first_idx - 1]
                known_tz = pd.to_datetime(with_tz).tz
            else:
                raise ValueError('No time zone information found')

            # convert all timestamps in the list using the last_tz timezone and move it to UTC
            for i in ls:
                time_string = df[column].iloc[i]
                #  remove redundant PM from time string with hours > 12
                if time_string.endswith(' PM') and pd.to_datetime(time_string.replace(' PM', '')).hour > 12:
                    time_string = time_string.replace(' PM', '')
                df[column].iat[i] = pd.to_datetime(time_string).tz_localize(known_tz).tz_convert(tz='UTC')

        # convert whole column into utc timestamps
        df[column] = pd.to_datetime(df[column], utc=True, errors='coerce')  # errors coerce will insert NaT

    except ValueError as e:
        print(e)

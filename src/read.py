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

from src.configurations import Configuration, GeneralisedCols, OpenAPSConfigs


# Data object that keeps the information from reading each data zip file
@dataclass
class ReadRecord:
    zip_id: str = None  # patient id
    is_android_upload: bool = False  # set to True if from android upload, False otherwise
    system: str = None  # can be used for system specific files to indicate the system
    df: pd.DataFrame = None  # dataframe
    has_no_files: bool = False  # true if the specified file to read was empty or did not exist

    # calculated fields
    number_of_entries_files: int = 0  # number of entries files found
    number_of_rows: int = 0  # number of rows in total
    number_of_rows_without_nan: int = 0
    number_of_rows_with_nan: int = 0
    earliest_date: str = ''  # oldest date in series
    newest_date: str = ''  # newest date in series

    # helper method to set read records if there are no files
    def zero_files(self):
        self.has_no_files = True

    # return its own dataframe with the id added
    # keep cols is a list is None keep all column, otherwise only specified column
    def df_with_id(self, keep_cols=None):
        if self.df is None:
            return None
        if keep_cols is None:
            keep_cols = self.df.columns

        missing_cols = [col for col in keep_cols if col not in self.df.columns]
        if missing_cols:
            print(f"Columns not in file for zip {self.zip_id}: {missing_cols}")
            self.df[missing_cols] = None

        result = self.df[keep_cols].copy()
        result.dropna(how='all', inplace=True)  # drop row if all columns are empty
        result.drop_duplicates(inplace=True, ignore_index=True)
        result.insert(loc=0, column=GeneralisedCols.id, value=self.zip_id)
        result[GeneralisedCols.id] = result[GeneralisedCols.id].astype("string")
        if self.system is not None:
            result.insert(loc=0, column=GeneralisedCols.system, value=self.system)
            if self.system == OpenAPSConfigs.system_name:
                result.rename(columns={OpenAPSConfigs.iob: GeneralisedCols.iob,
                                       OpenAPSConfigs.cob: GeneralisedCols.cob,
                                       OpenAPSConfigs.bg: GeneralisedCols.bg,
                                       OpenAPSConfigs.datetime: GeneralisedCols.datetime}, inplace=True)
        return result

    def add(self, df):
        if self.df is None:
            self.df = df
        else:
            self.df = pd.concat([self.df, df])

    def calculate_stats(self, time_col_name='time'):
        if self.has_no_files:
            return
        if self.df is None:
            return
        self.number_of_rows = self.df.shape[0]
        self.number_of_rows_without_nan = self.df.dropna().shape[0]
        self.number_of_rows_with_nan = self.df.shape[0] - self.df.dropna().shape[0]
        self.earliest_date = str(self.df[time_col_name].min())
        self.newest_date = str(self.df[time_col_name].max())


# reads flat device data csv and does preprocessing
# allows path for file to read
def read_flat_device_status_df_from_file(file: Path, config: Configuration):
    return read_device_status_file_and_convert_date(headers_in_file(file), config, file)


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
        time_col_name_for_stats = 'time'
        if 'device_status' in read_file_into_df_function.__name__:
            time_col_name_for_stats = 'created_at'
        read_record.calculate_stats(time_col_name_for_stats)
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
    read_record.system = OpenAPSConfigs.system_name  # TODO set to appropriate system once others read too
    specific_cols_dic = config.device_status_col_type
    if specific_cols_dic:  # preprocess reading
        with archive.open(file, mode="r") as header_context:
            text_io_wrapper = TextIOWrapper(header_context, encoding="utf-8")
            actual_headers = headers_in_file(text_io_wrapper)
            missing_headers = [ele for ele in (specific_cols_dic.keys()) if ele not in list(actual_headers)]
            if missing_headers:
                if not any("enacted" in h for h in actual_headers):
                    return  # this is not a device status file from a looping period

                if not any("openaps" in h for h in actual_headers):
                    return  # this is likely a loop file and won't have bolus information in the file, skip for now
        # read file for those headers
        with archive.open(file, mode="r") as file_context:
            file_to_read = TextIOWrapper(file_context, encoding="utf-8")
            df = read_device_status_file_and_convert_date(actual_headers, config, file_to_read)
    else:  # read file into one big dat file no encoding
        with archive.open(file, mode="r") as file_context:
            io_wrapper = TextIOWrapper(file_context, encoding="utf-8")
            df = pd.read_csv(io_wrapper)
    read_record.add(df)


def headers_in_file(file):
    header = pd.read_csv(file, nrows=0)
    return header.columns


# reads OpenAPS device status file
def read_device_status_file_and_convert_date(actual_headers, config, file_to_read):
    time_cols = [k for k in config.time_cols() if k in actual_headers]  # check columns that are in this file
    df = pd.read_csv(file_to_read,
                     usecols=lambda c: c in set(config.device_status_col_type.keys()),
                     dtype=config.device_status_col_type,
                     date_parser=lambda c: pd.to_datetime(c, utc=True, errors="ignore"),
                     parse_dates=time_cols  # if it does not work it will be an object
                     )
    convert_left_over_time_cols(df, time_cols)
    return df


def try_to_convert_int_to_datetime(value: str):
    if pd.isnull(value):
        return value  # just return nils
    if value.isdigit():
        result = pd.to_datetime(value, unit='ms', errors='coerce')
        if pd.isnull(result):  # didn't work -> wrong unit try another
            result = pd.to_datetime(value, unit='s', errors='coerce')
            if pd.isnull(result):  # didn't work again
                print("Couldn't parse integer time stamp with value" + value)
                return value  # return original value
        return result
    return value  # keep original value


def convert_left_over_time_cols(df, cols_to_convert: []):
    for col in cols_to_convert:
        sub_df = df[col]
        col_type = str(sub_df.dtypes).lower()
        if col_type.startswith('datetime'):
            continue  # already converted nothing needs to be done
        if sub_df.count() == 0:
            continue  # there's no data in this column
        try:  # changing whole column to int -> if it's an int it might be an epoch time stamp
            sub_df.astype(int)
            result = pd.to_datetime(sub_df, unit='ms', errors='coerce')
            if result.count() == 0:  # didn't work -> wrong unit try another
                result = pd.to_datetime(sub_df, unit='s', errors='coerce')
                if result.count() == 0:  # didn't work again
                    print("Couldn't parse integer time stamp for col " + col)
            else:  # conversion worked store in time column
                df[col] = result
                continue
        except ValueError:  # not an int try if date with varied timezone
            try:  # change individual values to int
                df[col] = sub_df.apply(try_to_convert_int_to_datetime)  # parse the odd into to a datetime
                df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
            except ValueError:
                print("whatever, we couldn't convert a time value  for col " + col)


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

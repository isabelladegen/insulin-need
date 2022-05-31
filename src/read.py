import glob
import zipfile
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
import logging
import pandas as pd
from src.configurations import Configuration


# Data object that keeps the information from reading each data zip file
@dataclass
class ReadRecord:
    zip_id: str = None  # patient id
    bg_df: pd.DataFrame = None  # dataframe of time and value
    has_no_BG_entries: bool = False  # true if the entries files were empty or non-existent
    number_of_entries_files: int = 0  # number of entries files found

    # helper method to set read records for no bg files
    def zero_bg_files(self):
        self.has_no_BG_entries = True

    def add(self, df):
        if self.bg_df is None:
            self.bg_df = df
        else:
            self.bg_df = pd.concat([self.bg_df, df])


# reads all BG files from each zip files without extracting the zip
def read_all_bg(config: Configuration):
    data = config.data_dir

    # get all zip files in folder
    filepaths = glob.glob(str(data) + "/*.zip")
    read_records = []
    for file in filepaths:
        # Don't read aps zip just yet
        if file.endswith(config.android_aps_zip):
            continue
        read_record = read_bg_from_zip(file, config)
        read_records.append(read_record)
    return read_records


# reads BGs into df from the entries csv file in the given zip file without extracting the zip
def read_bg_from_zip(file_name, config):
    read_record = ReadRecord()
    read_record.zip_id = Path(file_name).stem

    # find bg files in the zip file
    with zipfile.ZipFile(file_name, mode="r") as archive:
        files_and_folders = archive.namelist()

        # finds all the .csv files that contain BG readings in the zip
        bg_entries_files = [x for x in files_and_folders if is_a_bg_csv_file(config, read_record.zip_id, x)]

        # check number of bg entries files
        number_of_bg_entries_files = len(bg_entries_files)
        # stop reading if there are no entries files for BG
        if number_of_bg_entries_files == 0:
            read_record.zero_bg_files()
            return read_record
        read_record.number_of_entries_files = number_of_bg_entries_files

        # read all the entries files into dataframes
        for file in bg_entries_files:
            info = archive.getinfo(file)

            # skip files that are zero size, but log them
            if info.file_size == 0:
                logging.info('Found empty entries file: ' + file + ' for id: ' + read_record.zip_id)
                continue

            # read entries into pandas dataframe
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
            read_record.add(df)

        return read_record


# checks if a file from zip namelist is a bg csv file
def is_a_bg_csv_file(config, patient_id, file_path):
    # file starts with patient id and _entries
    startswith = Path(file_path).name.startswith(patient_id + config.bg_csv_file_start)

    # has right file ending
    endswith = file_path.endswith(config.bg_csv_file_extension)
    return startswith and endswith

import glob
import zipfile
from io import TextIOWrapper
from pathlib import Path
import logging
import pandas as pd
from src.configurations import Configuration


# reads all BG files from each zip files without extracting the zip
def read_all_bg(config: Configuration):
    data = config.data_dir

    # get all zip files in folder
    filepaths = glob.glob(str(data) + "/*.zip")
    dfs = []
    for file in filepaths:
        # Don't read aps zip just yet
        if file.endswith(config.android_aps_zip):
            continue
        dfs.append(read_bg_from_zip(file, config))
    return dfs


# reads BGs into df from the entries csv file in the given zip file without extracting the zip
def read_bg_from_zip(file_name, config):
    patient_id = Path(file_name).stem

    # find bg files in the zip file
    with zipfile.ZipFile(file_name, mode="r") as archive:
        files_and_folders = archive.namelist()

        # finds all the .csv files that contain BG readings in the zip
        bg_entries_files = [x for x in files_and_folders if
                            is_a_bg_csv_file(config, patient_id, x)]

        # log warning if there is more than one file
        if len(bg_entries_files) != 1:
            logging.warning('More entries files than expected for id: ' + patient_id + ' Files found: ' + ','.join(
                bg_entries_files))

        # read all these files into dataframes
        df = None
        for file in bg_entries_files:
            info = archive.getinfo(file)

            # skip files that are zero size, but log them
            if info.file_size is 0:
                logging.info('Found empty entries file: ' + file + ' for id: ' + patient_id)
                continue

            # read entries into pandas dataframe
            with archive.open(file, mode="r") as bg_file:
                df = pd.read_csv(TextIOWrapper(bg_file, encoding="utf-8"), header=None, names=['time', 'bg'])
                # add patient id to df
        if df is None:
            logging.warning("No entries file for zip file: " + file_name)
        return df, patient_id


# checks if a file from zip namelist is a bg csv file
def is_a_bg_csv_file(config, patient_id, file_path):
    # file starts with patient id and _entries
    startswith = Path(file_path).name.startswith(patient_id + config.bg_csv_file_start)

    # has right file ending
    endswith = file_path.endswith(config.bg_csv_file_extension)
    return startswith and endswith

import zipfile
from io import TextIOWrapper
from pathlib import Path
import logging
import pandas as pd


def read_bg_values_from(file_name, config):
    patient_id = Path(file_name).stem

    # find bg files in the zip file
    with zipfile.ZipFile(file_name, mode="r") as archive:
        files_and_folders = archive.namelist()

        # finds all the .csv files that contain BG readings in the zip
        bg_entries_files = [x for x in files_and_folders if x.endswith(patient_id + config.bg_csv_file_extension)]

        # log warning if there is more than one file
        if len(bg_entries_files) != 1:
            logging.warning('More entries files than expected for id: ' + patient_id + ' Files found: ' + ','.join(
                bg_entries_files))

        # read all these files into dataframes
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
            df['id'] = patient_id
        return df

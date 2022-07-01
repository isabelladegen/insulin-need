import os
from pathlib import Path

from src.format import as_flat_dataframe


def write_read_record(records, as_flat_file, folder, file_name):
    if as_flat_file:
        # turn read records into a flat dataframe
        df = as_flat_dataframe(records, False)

        file = Path(folder, file_name)
        df.to_csv(file)
    else:
        # create folder
        for record in records:
            df = record.df_with_id()
            if df is None:
                continue

            file = Path(folder, record.zip_id, file_name)
            # create folders if not exist
            file.parent.mkdir(parents=True, exist_ok=True)
            # write df
            df.to_csv(file)

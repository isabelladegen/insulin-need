from pathlib import Path

from src.format import as_flat_dataframe


def write_read_record(records, as_flat_file, folder, file_name):
    df = as_flat_dataframe(records, False)
    file = Path(folder, file_name)
    df.to_csv(file)

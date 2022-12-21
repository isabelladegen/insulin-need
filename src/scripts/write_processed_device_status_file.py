from src.configurations import Configuration, Irregular
from src.preprocess import dedub_device_status_dataframes
from src.read import read_all_device_status
from src.write import write_read_record


# set as_flat_file to True if you want to save one big flat bg csv or set it to false if you want a bg_csv per id
# change keep_columns if you want to keep different columns in the resulting file, default IOB, COB, BG plus time and id
def main():
    config = Configuration()
    as_flat_file = config.as_flat_file
    folder = config.data_folder if as_flat_file else config.perid_data_folder
    result = read_all_device_status(config)
    de_dub_result = dedub_device_status_dataframes(result)

    # write irregular
    write_read_record(de_dub_result, as_flat_file, folder, Irregular.csv_file_name(),
                      keep_cols=config.keep_columns)


if __name__ == "__main__":
    main()

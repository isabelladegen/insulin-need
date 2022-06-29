from src.configurations import Configuration
from src.preprocess import dedub_device_status_dataframes
from src.read import read_all_device_status
from src.write import write_read_record


def main():
    print("Read original Device Status OpenAPS data and save as dedubed flat data frame to csv")
    config = Configuration()
    result = read_all_device_status(config)
    de_dub_result = dedub_device_status_dataframes(result)
    write_read_record(de_dub_result, True, '../../data', 'device_status_116_df_dedubed.csv')


if __name__ == "__main__":
    main()

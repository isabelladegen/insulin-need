from pathlib import Path

from src.configurations import Configuration
from src.format import as_flat_dataframe
from src.preprocess import dedub_device_status_dataframes
from src.read import read_all_device_status


def main():
    print("Read original Device Status OpenAPS data and save as dedubed flat data frame to csv")
    config = Configuration()
    result = read_all_device_status(config)
    de_dub_result = dedub_device_status_dataframes(result)
    df = as_flat_dataframe(de_dub_result, False)
    file = Path('../../data/device_status_116_df_dedubed.csv')
    df.to_csv(file)


if __name__ == "__main__":
    main()

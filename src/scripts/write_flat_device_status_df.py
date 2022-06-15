from pathlib import Path

from src.configurations import Configuration
from src.format import as_flat_dataframe
from src.read import read_all_device_status


def main():
    print("Read original Device Status OpenAPS data and save as flat data frame to csv")
    config = Configuration()
    result = read_all_device_status(config)
    df = as_flat_dataframe(result, False)
    file = Path('../../data/device_status_116_df.csv')
    df.to_csv(file)


if __name__ == "__main__":
    main()

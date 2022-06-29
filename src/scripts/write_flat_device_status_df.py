from src.configurations import Configuration
from src.read import read_all_device_status
from src.write import write_read_record


def main():
    print("Read original Device Status OpenAPS data and save as flat data frame to csv")
    config = Configuration()
    result = read_all_device_status(config)
    write_read_record(result, True, '../../data', 'device_status_116_df.csv')


if __name__ == "__main__":
    main()

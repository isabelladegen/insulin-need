from src.configurations import Configuration
from src.read import read_all_bg
from src.write import write_read_record


def main():
    print("Read original OpenAPS data and save BG data frame to csv")
    config = Configuration()
    result = read_all_bg(config)
    write_read_record(result, True, '../../data', 'bg_df.csv')


if __name__ == "__main__":
    main()

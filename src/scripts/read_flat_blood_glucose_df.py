from pathlib import Path

from src.configurations import Configuration
from src.format import as_flat_dataframe
from src.read import read_all_bg


def main():
    print("Read original OpenAPS data and save BG data frame to csv")
    config = Configuration()
    result = read_all_bg(config)
    df = as_flat_dataframe(result)
    file = Path('../../data/bg_df.csv')
    df.to_csv(file)


if __name__ == "__main__":
    main()

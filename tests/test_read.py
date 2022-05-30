import glob
from pathlib import Path

from hamcrest import *
from src.configurations import TestConfiguration
from src.read import read_bg_from_zip, read_all_bg, is_a_bg_csv_file

config = TestConfiguration()
test_data_dir = config.data_dir
# get all zip files in folder
filepaths = glob.glob(str(test_data_dir) + "/*.zip")


def test_reads_bg_from_given_zip_file():
    path = filepaths[1]
    df, id = read_bg_from_zip(path, config)
    assert_that(df, is_not(empty()))
    assert_that(Path(path).stem, is_(id))


def test_reads_all_peoples_files():
    dfs = read_all_bg(config)
    assert_that(len(dfs), is_(145))


def test_is_a_bg_csv_file():
    example_name = 'direct-sharing-31/84984656_entries.json_csv/84984656_entries.json.csv'
    assert_that(is_a_bg_csv_file(config, '84984656', example_name), is_(True))

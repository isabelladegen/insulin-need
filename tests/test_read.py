import glob
from pathlib import Path

from hamcrest import *
from src.configurations import TestConfiguration
from src.read import read_bg_from_zip, read_all_bg, is_a_bg_csv_file

config = TestConfiguration()


def test_reads_bg_from_given_zip_file():
    test_data_dir = config.data_dir
    # get all zip files in folder
    filepaths = glob.glob(str(test_data_dir) + "/*.zip")
    path = filepaths[1]
    result = read_bg_from_zip(path, config)
    assert_that(result, is_not(empty()))
    assert_that(Path(path).stem, is_(result.zip_id))


def test_reads_all_peoples_files():
    result = read_all_bg(config)
    assert_that(len(result), is_(145))


def test_is_a_bg_csv_file():
    example_name = 'direct-sharing-31/84984656_entries.json_csv/84984656_entries.json.csv'
    assert_that(is_a_bg_csv_file(config, '84984656', example_name), is_(True))

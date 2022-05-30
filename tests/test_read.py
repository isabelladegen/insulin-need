import glob
from hamcrest import *
from src.configurations import TestConfiguration
from src.read import read_bg_values_from

config = TestConfiguration()
test_data_dir = config.data_dir
# get all zip files in folder
filepaths = glob.glob(str(test_data_dir) + "/*.zip")


def test_reads_zip_file():
    path = filepaths[1]
    df = read_bg_values_from(path, config)
    assert_that(df, is_not(empty()))

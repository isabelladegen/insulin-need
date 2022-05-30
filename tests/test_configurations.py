from hamcrest import *

from src.configurations import Configuration, TestConfiguration
from os import path


def test_sets_data_dir_to_valid_directory():
    data_dir = Configuration().data_dir
    assert_that(path.exists(data_dir))


def test_sets_test_data_dir_to_valid_directory():
    data_dir = TestConfiguration().data_dir
    assert_that(path.exists(data_dir))

from hamcrest import *

from src.configurations import Configuration, TestConfiguration


def test_sets_data_dir():
    assert_that(Configuration().data_dir, is_not(empty()))
    assert_that(TestConfiguration().data_dir, is_not(empty()))

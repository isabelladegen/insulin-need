from hamcrest import *

from src.configurations import Configuration, TestConfiguration, OpenAPSConfigs
from os import path


def test_sets_data_dir_to_valid_directory():
    data_dir = Configuration().data_dir
    assert_that(path.exists(data_dir))


def test_sets_test_data_dir_to_valid_directory():
    data_dir = TestConfiguration().data_dir
    assert_that(path.exists(data_dir))


def test_returns_all_enacted_cols():
    cols = TestConfiguration().enacted_cols()
    assert_that(cols, has_length(13))
    assert_that(cols, has_item('openaps/enacted/IOB'))


def test_returns_all_iob_cols():
    cols = TestConfiguration().iob_cols()
    assert_that(cols, has_length(9))
    assert_that(cols, has_item('openaps/iob/lastTemp/rate'))


def test_returns_all_pump_cols():
    cols = TestConfiguration().pump_cols()
    assert_that(cols, has_length(7))
    assert_that(cols, has_item('pump/status/bolusing'))


def test_returns_all_time_cols():
    cols = TestConfiguration().time_cols()
    assert_that(cols, has_length(8))
    assert_that(cols, has_item('openaps/iob/timestamp'))
    assert_that(cols, has_item('openaps/iob/lastBolusTime'))


def test_returns_cols_to_keep():
    cols = TestConfiguration().keep_columns

    assert_that(cols, has_item(OpenAPSConfigs.cob))
    assert_that(cols, has_item(OpenAPSConfigs.iob))
    assert_that(cols, has_item(OpenAPSConfigs.bg))
    assert_that(cols, has_item(OpenAPSConfigs.datetime))
